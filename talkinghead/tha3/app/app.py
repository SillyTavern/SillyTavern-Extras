"""THA3 live mode for SillyTavern-extras.

This implements the live animation backend and serves the API. For usage, see `server.py`.

If you want to play around with THA3 expressions in a standalone app, see `manual_poser.py`.
"""

# TODO: talkinghead live mode:
#  - remove rest of the IFacialMocap stuff (we can run on pure THA3)
#  - fix animation logic, currently a mess
#  - talking animation is broken, fix mouth randomizer
#  - see which version of the sway animation is better
#    - should have body sway, too
#  - improve idle animations
#    - cosine schedule?
#  - add option to server.py to load with float32 or float16, as desired
#  - PNG sending efficiency?

import atexit
import io
import logging
import os
import random
import sys
import time
import numpy as np
import threading
from typing import Dict, List, NoReturn, Union

import PIL

import torch

from flask import Flask, Response
from flask_cors import CORS

from tha3.mocap.ifacialmocap_pose import create_default_ifacialmocap_pose
from tha3.mocap.ifacialmocap_pose_converter import IFacialMocapPoseConverter
from tha3.mocap.ifacialmocap_poser_converter_25 import create_ifacialmocap_pose_converter
from tha3.poser.modes.load_poser import load_poser
from tha3.poser.poser import Poser
from tha3.util import (torch_linear_to_srgb, resize_PIL_image,
                       extract_PIL_image_from_filelike, extract_pytorch_image_from_PIL_image)
from tha3.app.util import load_emotion_presets, to_talkinghead_image, FpsStatistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
# TODO: we could move many of these into TalkingheadManager, and just keep a reference to that as global.
global_source_image = None
global_result_image = None
global_reload = None
is_talking_override = False
is_talking = False
global_timer_paused = False
emotion = "neutral"
lasttransitionedPose = "NotInit"
inMotion = False
fps = 0
current_pose = None
global_basedir = "talkinghead"

# Flask setup
app = Flask(__name__)
CORS(app)

# --------------------------------------------------------------------------------
# API

def setEmotion(_emotion: Dict[str, float]) -> None:
    """Set the current emotion of the character based on sentiment analysis results.

    Currently, we pick the emotion with the highest confidence score.

    _emotion: result of sentiment analysis: {emotion0: confidence0, ...}
    """
    global emotion

    highest_score = float('-inf')
    highest_label = None

    for item in _emotion:
        if item['score'] > highest_score:
            highest_score = item['score']
            highest_label = item['label']

    logger.debug(f"applying {emotion}")
    emotion = highest_label

def unload() -> str:
    global global_timer_paused
    global_timer_paused = True
    logger.debug("unload: animation paused")
    return "Animation Paused"

def start_talking() -> str:
    global is_talking_override
    is_talking_override = True
    logger.debug("start talking")
    return "started"

def stop_talking() -> str:
    global is_talking_override
    is_talking_override = False
    logger.debug("stop talking")
    return "stopped"

def result_feed() -> Response:
    def generate():
        while True:
            if global_result_image is not None:
                try:
                    rgb_image = global_result_image[:, :, [2, 1, 0]]  # Swap B and R channels
                    pil_image = PIL.Image.fromarray(np.uint8(rgb_image))  # Convert to PIL Image
                    if global_result_image.shape[2] == 4:  # Check if there is an alpha channel present
                        alpha_channel = global_result_image[:, :, 3]  # Extract alpha channel
                        pil_image.putalpha(PIL.Image.fromarray(np.uint8(alpha_channel)))  # Set alpha channel in the PIL Image
                    buffer = io.BytesIO()  # Save as PNG with RGBA mode
                    pil_image.save(buffer, format='PNG')
                    image_bytes = buffer.getvalue()
                except Exception as exc:
                    logger.error(f"Error when trying to write image: {exc}")
                yield (b'--frame\r\n'  # Send the PNG image (last available in case of error)
                       b'Content-Type: image/png\r\n\r\n' + image_bytes + b'\r\n')
            else:
                time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# TODO: the input is a flask.request.file.stream; what's the type of that?
def talkinghead_load_file(stream) -> str:
    global global_reload
    global global_timer_paused
    logger.debug("talkinghead_load_file: loading new input image from stream")

    try:
        global_timer_paused = True
        pil_image = PIL.Image.open(stream)  # Load the image using PIL.Image.open
        img_data = io.BytesIO()  # Create a copy of the image data in memory using BytesIO
        pil_image.save(img_data, format='PNG')
        global_reload = PIL.Image.open(io.BytesIO(img_data.getvalue()))  # Set the global_reload to a copy of the image data
        global_timer_paused = False
    except PIL.Image.UnidentifiedImageError:
        logger.warning("Could not load input image from stream, loading blank")
        full_path = os.path.join(os.getcwd(), os.path.normpath(os.path.join(global_basedir, "tha3", "images", "inital.png")))
        TalkingheadManager.load_image(full_path)
        global_timer_paused = True
    return 'OK'

def launch(device: str, model: str) -> Union[None, NoReturn]:
    """Launch the talking head plugin (live mode).

    If the plugin fails to load, the process exits.

    device: "cpu" or "cuda"
    model: one of the folder names inside "talkinghead/tha3/models/"
    """
    global initAMI  # TODO: initAREYOU? See if we still need this - the idea seems to be to stop animation until the first image is loaded.
    initAMI = True

    try:
        poser = load_poser(model, device, modelsdir=os.path.join(global_basedir, "tha3", "models"))
        pose_converter = create_ifacialmocap_pose_converter()  # creates a list of 45

        manager = TalkingheadManager(poser, pose_converter, device)

        # Load character image
        full_path = os.path.join(os.getcwd(), os.path.normpath(os.path.join(global_basedir, "tha3", "images", "inital.png")))
        manager.load_image(full_path)
        manager.start()

    except RuntimeError as exc:
        logger.error(exc)
        sys.exit()

# --------------------------------------------------------------------------------
# Internal stuff

def convert_linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    """RGBA (linear) -> RGBA (SRGB), preserving the alpha channel."""
    rgb_image = torch_linear_to_srgb(image[0:3, :, :])
    return torch.cat([rgb_image, image[3:4, :, :]], dim=0)

class TalkingheadManager:
    """uWu Waifu"""

    def __init__(self, poser: Poser, pose_converter: IFacialMocapPoseConverter, device: torch.device):
        self.pose_converter = pose_converter
        self.poser = poser
        self.device = device

        self.last_blink_timestamp = 0  # TODO: Great idea! We should actually use this.
        self.is_blinked = False  # TODO: what was this for?
        self.targets = {"head_y_index": 0}
        self.progress = {"head_y_index": 0}
        self.direction = {"head_y_index": 1}
        self.originals = {"head_y_index": 0}  # TODO: what was this for?
        self.forward = {"head_y_index": True}  # Direction of interpolation
        self.start_values = {"head_y_index": 0}

        self.fps_statistics = FpsStatistics()

        self.ifacialmocap_pose = create_default_ifacialmocap_pose()
        self.torch_source_image = None
        self.last_update_time = None
        self.last_report_time = None

        self.emotions, self.emotion_names = load_emotion_presets(os.path.join("talkinghead", "emotions"))

    def start(self) -> None:
        """Start the talkinghead update thread."""
        self._terminated = False
        def update_manager():
            while not self._terminated:
                # TODO: add a configurable FPS limiter (take a parameter in `__init__`; populate it from cli args in `server.py`)
                #   - should sleep for `max(eps, frame_target_ms - render_average_ms)`, where `eps = 0.01`, so that the next frame is ready in time
                #     (get render_average_ms from FPS counter; sanity check for nonsense value)
                self.update_result_image_bitmap()
                time.sleep(0.01)
        self.scheduler = threading.Thread(target=update_manager, daemon=True)
        self.scheduler.start()
        atexit.register(self.exit)

    def exit(self) -> None:
        """Terminate the talkinghead update thread."""
        self._terminated = True

    def random_generate_value(self, min: int, max: int, origin_value: float) -> float:
        x = origin_value + random.choice(list(range(min, max, 1))) / 2500.0  # TODO: WTF, "list"? Should just generate a random number directly.
        x = max(0.0, min(x, 1.0))  # clamp (not the manga studio)
        return x

    # def animationTalking(self):
    #     global is_talking
    #     current_pose = self.ifacialmocap_pose
    #
    #     # NOTE: randomize mouth
    #     for blendshape_name in mocap_constants.BLENDSHAPE_NAMES:
    #         if "jawOpen" in blendshape_name:
    #             if is_talking or is_talking_override:
    #                 current_pose[blendshape_name] = self.random_generate_value(-5000, 5000, abs(1 - current_pose[blendshape_name]))
    #             else:
    #                 current_pose[blendshape_name] = 0
    #
    #     return current_pose
    #
    # def animationHeadMove(self):
    #     current_pose = self.ifacialmocap_pose
    #
    #     for key in [mocap_constants.HEAD_BONE_Y]:  # can add more to this list if needed
    #         current_pose[key] = self.random_generate_value(-20, 20, current_pose[key])
    #
    #     return current_pose
    #
    # def animationBlink(self):
    #     current_pose = self.ifacialmocap_pose
    #
    #     if random.random() <= 0.03:
    #         current_pose["eyeBlinkRight"] = 1
    #         current_pose["eyeBlinkLeft"] = 1
    #     else:
    #         current_pose["eyeBlinkRight"] = 0
    #         current_pose["eyeBlinkLeft"] = 0
    #
    #     return current_pose

    def addNamestoConvert(pose):
        # TODO: What are the unknown keys?
        index_to_name = {
            0: 'eyebrow_troubled_left_index',
            1: 'eyebrow_troubled_right_index',
            2: 'eyebrow_angry_left_index',
            3: 'eyebrow_angry_right_index',
            4: 'unknown1',  # COMBACK TO UNK
            5: 'unknown2',  # COMBACK TO UNK
            6: 'eyebrow_raised_left_index',
            7: 'eyebrow_raised_right_index',
            8: 'eyebrow_happy_left_index',
            9: 'eyebrow_happy_right_index',
            10: 'unknown3',  # COMBACK TO UNK
            11: 'unknown4',  # COMBACK TO UNK
            12: 'wink_left_index',
            13: 'wink_right_index',
            14: 'eye_happy_wink_left_index',
            15: 'eye_happy_wink_right_index',
            16: 'eye_surprised_left_index',
            17: 'eye_surprised_right_index',
            18: 'unknown5',  # COMBACK TO UNK
            19: 'unknown6',  # COMBACK TO UNK
            20: 'unknown7',  # COMBACK TO UNK
            21: 'unknown8',  # COMBACK TO UNK
            22: 'eye_raised_lower_eyelid_left_index',
            23: 'eye_raised_lower_eyelid_right_index',
            24: 'iris_small_left_index',
            25: 'iris_small_right_index',
            26: 'mouth_aaa_index',
            27: 'mouth_iii_index',
            28: 'mouth_ooo_index',
            29: 'unknown9a',  # COMBACK TO UNK
            30: 'mouth_ooo_index2',
            31: 'unknown9',  # COMBACK TO UNK
            32: 'unknown10',  # COMBACK TO UNK
            33: 'unknown11',  # COMBACK TO UNK
            34: 'mouth_raised_corner_left_index',
            35: 'mouth_raised_corner_right_index',
            36: 'unknown12',  # COMBACK TO UNK
            37: 'iris_rotation_x_index',
            38: 'iris_rotation_y_index',
            39: 'head_x_index',
            40: 'head_y_index',
            41: 'neck_z_index',
            42: 'body_y_index',
            43: 'body_z_index',
            44: 'breathing_index'
        }

        output = []

        for index, value in enumerate(pose):
            name = index_to_name.get(index, "Unknown")
            output.append(f"{name}: {value}")

        return output

    def animateToEmotion(self, current_pose_list: List[str], target_pose_dict: Dict[str, float]) -> List[str]:
        transitionPose = []

        # Loop through the current_pose_list
        for item in current_pose_list:
            index, value = item.split(': ')

            # Always take the value from target_pose_dict if the key exists
            if index in target_pose_dict and index != "breathing_index":
                transitionPose.append(f"{index}: {target_pose_dict[index]}")
            else:
                transitionPose.append(item)

        # Ensure that the number of elements in transitionPose matches with current_pose_list
        assert len(transitionPose) == len(current_pose_list)

        return transitionPose

    # def animationMain(self):
    #     self.ifacialmocap_pose = self.animationBlink()
    #     self.ifacialmocap_pose = self.animationHeadMove()
    #     self.ifacialmocap_pose = self.animationTalking()
    #     return self.ifacialmocap_pose

    def dict_to_tensor(self, d):
        if isinstance(d, dict):
            return torch.tensor(list(d.values()))
        elif isinstance(d, list):
            return torch.tensor(d)
        else:
            raise ValueError("Unsupported data type passed to dict_to_tensor.")

    def update_ifacialmocap_pose(self, ifacialmocap_pose, emotion_pose):
        # Update Values - The following values are in emotion_pose but not defined in ifacialmocap_pose
        # eye_happy_wink_left_index, eye_happy_wink_right_index
        # eye_surprised_left_index, eye_surprised_right_index
        # eye_relaxed_left_index, eye_relaxed_right_index
        # eye_unimpressed
        # eye_raised_lower_eyelid_left_index, eye_raised_lower_eyelid_right_index
        # mouth_uuu_index
        # mouth_eee_index
        # mouth_ooo_index
        # mouth_delta
        # mouth_smirk
        # body_y_index
        # body_z_index
        # breathing_index

        ifacialmocap_pose['browDownLeft'] = emotion_pose['eyebrow_troubled_left_index']
        ifacialmocap_pose['browDownRight'] = emotion_pose['eyebrow_troubled_right_index']
        ifacialmocap_pose['browOuterUpLeft'] = emotion_pose['eyebrow_angry_left_index']
        ifacialmocap_pose['browOuterUpRight'] = emotion_pose['eyebrow_angry_right_index']
        ifacialmocap_pose['browInnerUp'] = emotion_pose['eyebrow_happy_left_index']
        ifacialmocap_pose['browInnerUp'] += emotion_pose['eyebrow_happy_right_index']
        ifacialmocap_pose['browDownLeft'] = emotion_pose['eyebrow_raised_left_index']
        ifacialmocap_pose['browDownRight'] = emotion_pose['eyebrow_raised_right_index']
        ifacialmocap_pose['browDownLeft'] += emotion_pose['eyebrow_lowered_left_index']
        ifacialmocap_pose['browDownRight'] += emotion_pose['eyebrow_lowered_right_index']
        ifacialmocap_pose['browDownLeft'] += emotion_pose['eyebrow_serious_left_index']
        ifacialmocap_pose['browDownRight'] += emotion_pose['eyebrow_serious_right_index']

        # Update eye values
        ifacialmocap_pose['eyeWideLeft'] = emotion_pose['eye_surprised_left_index']
        ifacialmocap_pose['eyeWideRight'] = emotion_pose['eye_surprised_right_index']

        # Update eye blink (though we will overwrite it later)
        ifacialmocap_pose['eyeBlinkLeft'] = emotion_pose['eye_wink_left_index']
        ifacialmocap_pose['eyeBlinkRight'] = emotion_pose['eye_wink_right_index']

        # Update iris rotation values
        ifacialmocap_pose['eyeLookInLeft'] = -emotion_pose['iris_rotation_y_index']
        ifacialmocap_pose['eyeLookOutLeft'] = emotion_pose['iris_rotation_y_index']
        ifacialmocap_pose['eyeLookInRight'] = emotion_pose['iris_rotation_y_index']
        ifacialmocap_pose['eyeLookOutRight'] = -emotion_pose['iris_rotation_y_index']
        ifacialmocap_pose['eyeLookUpLeft'] = emotion_pose['iris_rotation_x_index']
        ifacialmocap_pose['eyeLookDownLeft'] = -emotion_pose['iris_rotation_x_index']
        ifacialmocap_pose['eyeLookUpRight'] = emotion_pose['iris_rotation_x_index']
        ifacialmocap_pose['eyeLookDownRight'] = -emotion_pose['iris_rotation_x_index']

        # Update iris size values
        ifacialmocap_pose['irisWideLeft'] = emotion_pose['iris_small_left_index']
        ifacialmocap_pose['irisWideRight'] = emotion_pose['iris_small_right_index']

        # Update head rotation values
        ifacialmocap_pose['headBoneX'] = -emotion_pose['head_x_index'] * 15.0
        ifacialmocap_pose['headBoneY'] = -emotion_pose['head_y_index'] * 10.0
        ifacialmocap_pose['headBoneZ'] = emotion_pose['neck_z_index'] * 15.0

        # Update mouth values
        ifacialmocap_pose['mouthSmileLeft'] = emotion_pose['mouth_aaa_index']
        ifacialmocap_pose['mouthSmileRight'] = emotion_pose['mouth_aaa_index']
        ifacialmocap_pose['mouthFrownLeft'] = emotion_pose['mouth_lowered_corner_left_index']
        ifacialmocap_pose['mouthFrownRight'] = emotion_pose['mouth_lowered_corner_right_index']
        ifacialmocap_pose['mouthPressLeft'] = emotion_pose['mouth_raised_corner_left_index']
        ifacialmocap_pose['mouthPressRight'] = emotion_pose['mouth_raised_corner_right_index']

        return ifacialmocap_pose

    def update_blinking_pose(self, transitionedPose):
        PARTS = ['wink_left_index', 'wink_right_index']
        updated_list = []

        should_blink = random.random() <= 0.03  # Determine if there should be a blink

        for item in transitionedPose:
            key, value = item.split(': ')
            if key in PARTS:
                # If there should be a blink, set value to 1; otherwise, use the provided value
                new_value = 1 if should_blink else float(value)
                updated_list.append(f"{key}: {new_value}")
            else:
                updated_list.append(item)

        return updated_list

    def update_talking_pose(self, transitionedPose):
        MOUTHPARTS = ['mouth_aaa_index']

        updated_list = []

        for item in transitionedPose:
            key, value = item.split(': ')

            if key in MOUTHPARTS and is_talking_override:
                new_value = self.random_generate_value(-5000, 5000, abs(1 - float(value)))
                updated_list.append(f"{key}: {new_value}")
            else:
                updated_list.append(item)

        return updated_list

    def update_sway_pose_good(self, transitionedPose):  # TODO: good? why is there a bad one, too? keep only one!
        MOVEPARTS = ['head_y_index']
        updated_list = []

        # logger.debug(f"{self.start_values}, {self.targets}, {self.progress}, {self.direction}")

        for item in transitionedPose:
            key, value = item.split(': ')

            if key in MOVEPARTS:
                current_value = float(value)

                # If progress reaches 1 or 0
                if self.progress[key] >= 1 or self.progress[key] <= 0:
                    # Reverse direction
                    self.direction[key] *= -1

                    # If direction is now forward, set a new target and store starting value
                    if self.direction[key] == 1:
                        self.start_values[key] = current_value
                        self.targets[key] = current_value + random.uniform(-1, 1)
                        self.progress[key] = 0  # Reset progress when setting a new target

                # Linearly interpolate between start and target values
                new_value = self.start_values[key] + self.progress[key] * (self.targets[key] - self.start_values[key])
                new_value = min(max(new_value, -1), 1)  # clip to bounds (just in case)

                # Update progress based on direction
                self.progress[key] += 0.02 * self.direction[key]

                updated_list.append(f"{key}: {new_value}")
            else:
                updated_list.append(item)

        return updated_list

    def update_sway_pose(self, transitionedPose):
        MOVEPARTS = ['head_y_index']
        updated_list = []

        # logger.debug(f"{self.start_values}, {self.targets}, {self.progress}, {self.direction}")

        for item in transitionedPose:
            key, value = item.split(': ')

            if key in MOVEPARTS:
                current_value = float(value)

                # Linearly interpolate between start and target values
                new_value = self.start_values[key] + self.progress[key] * (self.targets[key] - self.start_values[key])
                new_value = min(max(new_value, -1), 1)  # clip to bounds (just in case)

                # Check if we've reached the target or start value
                is_close_to_target = abs(new_value - self.targets[key]) < 0.04
                is_close_to_start = abs(new_value - self.start_values[key]) < 0.04

                if (self.direction[key] == 1 and is_close_to_target) or (self.direction[key] == -1 and is_close_to_start):
                    # Reverse direction
                    self.direction[key] *= -1

                    # If direction is now forward, set a new target and store starting value
                    if self.direction[key] == 1:
                        self.start_values[key] = new_value
                        self.targets[key] = current_value + random.uniform(-0.6, 0.6)
                        self.progress[key] = 0  # Reset progress when setting a new target

                # Update progress based on direction
                self.progress[key] += 0.04 * self.direction[key]

                updated_list.append(f"{key}: {new_value}")
            else:
                updated_list.append(item)

        return updated_list

    def update_transition_pose(self, last_transition_pose_s, transition_pose_s):
        global inMotion
        inMotion = True

        # Create dictionaries from the lists for easier comparison
        last_transition_dict = {}
        for item in last_transition_pose_s:
            key = item.split(': ')[0]
            value = float(item.split(': ')[1])
            if key == 'unknown':
                key += f"_{list(last_transition_dict.values()).count(value)}"
            last_transition_dict[key] = value

        transition_dict = {}
        for item in transition_pose_s:
            key = item.split(': ')[0]
            value = float(item.split(': ')[1])
            if key == 'unknown':
                key += f"_{list(transition_dict.values()).count(value)}"
            transition_dict[key] = value

        updated_last_transition_pose = []

        for key, last_value in last_transition_dict.items():
            # If the key exists in transition_dict, increment its value by 0.4 and clip it to the target
            if key in transition_dict:

                # If the key is 'wink_left_index' or 'wink_right_index', set the value directly dont animate blinks
                if key in ['wink_left_index', 'wink_right_index']:  # BLINK FIX
                    last_value = transition_dict[key]

                # For all other keys, increment its value by 0.1 of the delta and clip it to the target
                else:
                    delta = transition_dict[key] - last_value
                    last_value += delta * 0.1

            # Reconstruct the string and append it to the updated list
            updated_last_transition_pose.append(f"{key}: {last_value}")

        # If any value is less than the target, set inMotion to True
        # TODO/FIXME: inMotion is not actually used by anything else
        if any(last_transition_dict[k] < transition_dict[k] for k in last_transition_dict if k in transition_dict):
            inMotion = True
        else:
            inMotion = False

        return updated_last_transition_pose

    def update_result_image_bitmap(self) -> None:
        """Render an animation frame."""

        global global_timer_paused
        global initAMI
        global global_result_image
        global fps
        global current_pose
        global lasttransitionedPose

        if global_timer_paused:
            return

        try:
            if global_reload is not None:
                TalkingheadManager.load_image(self, file_path=None)  # call load_image function here
                return
            if self.torch_source_image is None:
                return

            # # OLD METHOD
            # ifacialmocap_pose = self.animationMain()  # GET ANIMATION CHANGES
            # current_posesaved = self.pose_converter.convert(ifacialmocap_pose)
            # combined_posesaved = current_posesaved

            # NEW METHOD
            # CREATES THE DEFAULT POSE AND STORES OBJ IN STRING
            # ifacialmocap_pose = self.animationMain()  # DISABLE FOR TESTING!!!!!!!!!!!!!!!!!!!!!!!!
            ifacialmocap_pose = self.ifacialmocap_pose
            # logger.debug(f"ifacialmocap_pose: {ifacialmocap_pose}")

            # GET EMOTION SETTING
            emotion_pose = self.emotions[emotion]
            # logger.debug(f"emotion_pose: {emotion_pose}")

            # MERGE EMOTION SETTING WITH CURRENT OUTPUT
            # NOTE: This is a mutating method that overwrites the original `ifacialmocap_pose`.
            updated_pose = self.update_ifacialmocap_pose(ifacialmocap_pose, emotion_pose)
            # logger.debug(f"updated_pose: {updated_pose}")

            # CONVERT RESULT TO FORMAT NN CAN USE
            current_pose = self.pose_converter.convert(updated_pose)
            # logger.debug(f"current_pose: {current_pose}")

            # SEND THROUGH CONVERT
            current_pose = self.pose_converter.convert(ifacialmocap_pose)
            # logger.debug(f"current_pose2: {current_pose}")

            # ADD LABELS/NAMES TO THE POSE
            names_current_pose = TalkingheadManager.addNamestoConvert(current_pose)
            # logger.debug(f"current pose: {names_current_pose}")

            # GET THE EMOTION VALUES again for some reason
            emotion_pose2 = self.emotions[emotion]
            # logger.debug(f"target pose: {emotion_pose2}")

            # APPLY VALUES TO THE POSE AGAIN?? This needs to overwrite the values
            transitionedPose = self.animateToEmotion(names_current_pose, emotion_pose2)
            # logger.debug(f"combine pose: {transitionedPose}")

            # smooth animate
            # logger.debug(f"LAST VALUES: {lasttransitionedPose}")
            # logger.debug(f"TARGET VALUES: {transitionedPose}")

            if lasttransitionedPose != "NotInit":
                transitionedPose = self.update_transition_pose(lasttransitionedPose, transitionedPose)
                # logger.debug(f"smoothed: {transitionedPose}")

            # Animate blinking
            transitionedPose = self.update_blinking_pose(transitionedPose)

            # Animate Head Sway
            transitionedPose = self.update_sway_pose(transitionedPose)

            # Animate Talking
            transitionedPose = self.update_talking_pose(transitionedPose)

            # reformat the data correctly
            parsed_data = []
            for item in transitionedPose:
                key, value_str = item.split(': ')
                value = float(value_str)
                parsed_data.append((key, value))
            tranisitiondPosenew = [value for _, value in parsed_data]

            # not sure what this is for TBH   # TODO: let's get rid of it then
            ifacialmocap_pose = tranisitiondPosenew

            # pose = torch.tensor(tranisitiondPosenew, device=self.device, dtype=self.poser.get_dtype())
            pose = self.dict_to_tensor(tranisitiondPosenew).to(device=self.device, dtype=self.poser.get_dtype())  # TODO: a WHAT to a WHAT? Optimize this!

            with torch.no_grad():
                output_image = self.poser.pose(self.torch_source_image, pose)[0].float()
                output_image = convert_linear_to_srgb((output_image + 1.0) / 2.0)

                c, h, w = output_image.shape
                output_image = (255.0 * torch.transpose(output_image.reshape(c, h * w), 0, 1)).reshape(h, w, c).byte()

            numpy_image = output_image.detach().cpu().numpy()
            numpy_image_bgra = numpy_image[:, :, [2, 1, 0, 3]]  # Convert color channels from RGB to BGR and keep alpha channel
            global_result_image = numpy_image_bgra

            time_now = time.time_ns()
            if self.last_update_time is not None:
                elapsed_time = time_now - self.last_update_time
                fps = 1.0 / (elapsed_time / 10**9)

                if self.torch_source_image is not None:
                    self.fps_statistics.add_fps(fps)
            self.last_update_time = time_now

            if initAMI:  # If the models are just now initalized stop animation to save
                global_timer_paused = True
                initAMI = False

            if self.last_report_time is None or time_now - self.last_report_time > 5e9:
                trimmed_fps = round(self.fps_statistics.get_average_fps(), 1)
                logger.info("update_result_image_bitmap: FPS: {:.1f}".format(trimmed_fps))
                self.last_report_time = time_now

            # Store current pose to use as last pose on next loop
            lasttransitionedPose = transitionedPose

        except KeyboardInterrupt:
            pass

    def load_image(self, file_path=None) -> None:
        """Load the image file at `file_path`.

        Except, if `global_reload is not None`, use the global reload image data instead.
        """
        global global_source_image
        global global_reload

        if global_reload is not None:
            file_path = "global_reload"

        try:
            if file_path == "global_reload":
                pil_image = global_reload
            else:
                pil_image = resize_PIL_image(
                    extract_PIL_image_from_filelike(file_path),
                    (self.poser.get_image_size(), self.poser.get_image_size()))

            w, h = pil_image.size

            if pil_image.size != (512, 512):
                logger.info("Resizing Char Card to work")
                pil_image = to_talkinghead_image(pil_image)

            w, h = pil_image.size

            if pil_image.mode != 'RGBA':
                logger.error("load_image: image must have alpha channel")
                self.torch_source_image = None
            else:
                self.torch_source_image = extract_pytorch_image_from_PIL_image(pil_image) \
                    .to(self.device).to(self.poser.get_dtype())

            global_source_image = self.torch_source_image

        except Exception as exc:
            logger.error(f"load_image: {exc}")

        finally:
            global_reload = None
