"""THA3 live mode for SillyTavern-extras.

This is the animation engine, running on top of the THA3 posing engine.
This module implements the live animation backend and serves the API. For usage, see `server.py`.

If you want to play around with THA3 expressions in a standalone app, see `manual_poser.py`.
"""

__all__ = ["set_emotion_from_classification", "set_emotion",
           "unload",
           "start_talking", "stop_talking",
           "result_feed",
           "talkinghead_load_file",
           "launch"]

import atexit
import io
import json
import logging
import math
import os
import random
import sys
import time
import numpy as np
import threading
from typing import Any, Dict, List, NoReturn, Optional, Union

import PIL

import torch

from flask import Flask, Response
from flask_cors import CORS

from tha3.poser.modes.load_poser import load_poser
from tha3.poser.poser import Poser
from tha3.util import (torch_linear_to_srgb, resize_PIL_image,
                       extract_PIL_image_from_filelike, extract_pytorch_image_from_PIL_image)
from tha3.app.postprocessor import Postprocessor
from tha3.app.util import posedict_keys, posedict_key_to_index, load_emotion_presets, posedict_to_pose, to_talkinghead_image, RunningAverage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
# Global variables

# Default configuration for the animator, loaded when the plugin is launched.
# Doubles as the authoritative documentation of the animator settings (beside the animation driver docstrings and the actual source code).
animator_defaults = {"target_fps": 25,  # Desired output frames per second. Note this only affects smoothness of the output (if hardware allows).
                                        # The speed at which the animation evolves is based on wall time. Snapshots are rendered at the target FPS,
                                        # or if the hardware is too slow to reach the target FPS, then as often as hardware allows.
                                        # For smooth animation, make the FPS lower than what your hardware could produce, so that some compute
                                        # remains untapped, available to smooth over the occasional hiccup from other running programs.
                     "crop_left": 0.0,  # in units where the image width is 2.0
                     "crop_right": 0.0,  # in units where the image width is 2.0
                     "crop_top": 0.0,  # in units where the image height is 2.0
                     "crop_bottom": 0.0,  # in units where the image height is 2.0
                     "pose_interpolator_step": 0.1,  # 0 < this <= 1; at each frame at a reference of 25 FPS; FPS-corrected automatically; see `interpolate_pose`.

                     "blink_interval_min": 2.0,  # seconds, lower limit for random minimum time until next blink is allowed.
                     "blink_interval_max": 5.0,  # seconds, upper limit for random minimum time until next blink is allowed.
                     "blink_probability": 0.03,  # At each frame at a reference of 25 FPS; FPS-corrected automatically.
                     "blink_confusion_duration": 10.0,  # seconds, upon entering "confusion" emotion, during which blinking quickly in succession is allowed.

                     "talking_fps": 12,  # How often to re-randomize mouth during talking animation.
                                         # Early 2000s anime used ~12 FPS as the fastest actual framerate of new cels (not counting camera panning effects and such).
                     "talking_morph": "mouth_aaa_index",  # which mouth-open morph to use for talking; for available values, see `posedict_keys`

                     "sway_morphs": ["head_x_index", "head_y_index", "neck_z_index", "body_y_index", "body_z_index"],  # which morphs to sway; see `posedict_keys`
                     "sway_interval_min": 5.0,  # seconds, lower limit for random time interval until randomizing new sway pose.
                     "sway_interval_max": 10.0,  # seconds, upper limit for random time interval until randomizing new sway pose.
                     "sway_macro_strength": 0.6,  # [0, 1], in sway pose, max abs deviation from emotion pose target morph value for each sway morph,
                                                  # but also max deviation from center. The emotion pose itself may use higher values; in such cases,
                                                  # sway will only occur toward the center. See `compute_sway_target_pose` for details.
                     "sway_micro_strength": 0.02,  # [0, 1], max abs random noise added each frame. No limiting other than a clamp of final pose to [-1, 1].

                     "breathing_cycle_duration": 4.0,  # seconds, for a full breathing cycle.

                     "postprocessor_chain": []}  # Pixel-space glitch artistry settings; see `postprocessor.py`.

talkinghead_basedir = "talkinghead"

global_animator_instance = None
_animator_output_lock = threading.Lock()  # protect from concurrent access to `result_image` and the `new_frame_available` flag.
global_encoder_instance = None
global_latest_frame_sent = None

# These need to be written to by the API functions.
#
# Since the plugin might not have been started yet at that time (so the animator instance might not exist),
# it's better to keep this state in module-level globals rather than in attributes of the animator.
animation_running = False  # used in initial bootup state, and while loading a new image
current_emotion = "neutral"
is_talking = False
global_reload_image = None

target_fps = 25  # value overridden by `load_animator_settings` at animator startup

# --------------------------------------------------------------------------------
# API

# Flask setup
app = Flask(__name__)
CORS(app)

def set_emotion_from_classification(emotion_scores: List[Dict[str, Union[str, float]]]) -> str:
    """Set the current emotion of the character based on sentiment analysis results.

    Currently, we pick the emotion with the highest confidence score.

    `emotion_scores`: results from classify module: [{"label": emotion0, "score": confidence0}, ...]

    Return a status message for passing over HTTP.
    """
    highest_score = float("-inf")
    highest_label = None
    for item in emotion_scores:
        if item["score"] > highest_score:
            highest_score = item["score"]
            highest_label = item["label"]
    logger.info(f"set_emotion_from_classification: winning score: {highest_label} = {highest_score}")
    return set_emotion(highest_label)

def set_emotion(emotion: str) -> str:
    """Set the current emotion of the character.

    Return a status message for passing over HTTP.
    """
    global current_emotion

    if emotion not in global_animator_instance.emotions:
        logger.warning(f"set_emotion: specified emotion '{emotion}' does not exist, selecting 'neutral'")
        emotion = "neutral"

    logger.info(f"set_emotion: applying emotion {emotion}")
    current_emotion = emotion
    return f"emotion set to {emotion}"

def unload() -> str:
    """Stop animation.

    Return a status message for passing over HTTP.
    """
    global animation_running
    animation_running = False
    logger.info("unload: animation paused")
    return "animation paused"

def start_talking() -> str:
    """Start talking animation.

    Return a status message for passing over HTTP.
    """
    global is_talking
    is_talking = True
    logger.debug("start_talking called")
    return "talking started"

def stop_talking() -> str:
    """Stop talking animation.

    Return a status message for passing over HTTP.
    """
    global is_talking
    is_talking = False
    logger.debug("stop_talking called")
    return "talking stopped"

# There are three tasks we must do each frame:
#
#   1) Render an animation frame
#   2) Encode the new animation frame for network transport
#   3) Send the animation frame over the network
#
# Instead of running serially:
#
#   [render1][encode1][send1] [render2][encode2][send2]
# ------------------------------------------------------> time
#
# we get better throughput by parallelizing and interleaving:
#
#   [render1] [render2] [render3] [render4] [render5]
#             [encode1] [encode2] [encode3] [encode4]
#                       [send1]   [send2]   [send3]
# ----------------------------------------------------> time
#
# Despite the global interpreter lock, this increases throughput, as well as improves the timing of the network send
# since the network thread only needs to care about getting the send timing right.
#
# Either there's enough waiting for I/O for the split between render and encode to make a difference, or it's the fact
# that much of the compute-heavy work in both of those is performed inside C libraries that release the GIL (Torch,
# and the PNG encoder in Pillow, respectively).
#
# This is a simplified picture. Some important details:
#
#   - At startup:
#     - The animator renders the first frame on its own.
#     - The encoder waits for the animator to publish a frame, and then starts normal operation.
#     - The network thread waits for the encoder to publish a frame, and then starts normal operation.
#   - In normal operation (after startup):
#     - The animator waits until the encoder has consumed the previous published frame. Then it proceeds to render and publish a new frame.
#       - This communication is handled through the flag `animator.new_frame_available`.
#     - The network thread does its own thing on a regular schedule, based on the desired target FPS.
#       - However, the network thread publishes metadata on which frame is the latest that has been sent over the network at least once.
#         This is stored as an `id` (i.e. memory address) in `global_latest_frame_sent`.
#       - If the target FPS is too high for the animator and/or encoder to keep up with, the network thread re-sends
#         the latest frame published by the encoder as many times as necessary, to keep the network output at the target FPS
#         regardless of render/encode speed. This handles the case of hardware slower than the target FPS.
#       - On localhost, the network send is very fast, under 0.15 ms.
#     - The encoder uses the metadata to wait until the latest encoded frame has been sent at least once before publishing a new frame.
#       This ensures that no more frames are generated than are actually sent, and syncs also the animator (because the animator is
#       rate-limited by the encoder consuming its frames). This handles the case of hardware faster than the target FPS.
#     - When the animator and encoder are fast enough to keep up with the target FPS, generally when frame N is being sent,
#       frame N+1 is being encoded (or is already encoded, and waiting for frame N to be sent), and frame N+2 is being rendered.
#
def result_feed() -> Response:
    """Return a Flask `Response` that repeatedly yields the current image as 'image/png'."""
    def generate():
        global global_latest_frame_sent

        last_frame_send_complete_time = None
        last_report_time = None
        send_duration_sec = 0.0
        send_duration_statistics = RunningAverage()

        while True:
            # Send the latest available animation frame.
            # Important: grab reference to `image_bytes` only once, since it will be atomically updated without a lock.
            image_bytes = global_encoder_instance.image_bytes
            if image_bytes is not None:
                # How often should we send?
                #  - Excessive spamming can DoS the SillyTavern GUI, so there needs to be a rate limit.
                #  - OTOH, we must constantly send something, or the GUI will lock up waiting.
                # Therefore, send at a target FPS that yields a nice-looking animation.
                frame_duration_target_sec = 1 / target_fps
                if last_frame_send_complete_time is not None:
                    time_now = time.time_ns()
                    this_frame_elapsed_sec = (time_now - last_frame_send_complete_time) / 10**9
                    # The 2* is a fudge factor. It doesn't matter if the frame is a bit too early, but we don't want it to be late.
                    time_until_frame_deadline = frame_duration_target_sec - this_frame_elapsed_sec - 2 * send_duration_sec
                else:
                    time_until_frame_deadline = 0.0  # nothing rendered yet

                if time_until_frame_deadline <= 0.0:
                    time_now = time.time_ns()
                    yield (b"--frame\r\n"
                           b"Content-Type: image/png\r\n\r\n" + image_bytes + b"\r\n")
                    global_latest_frame_sent = id(image_bytes)  # atomic update, no need for lock
                    send_duration_sec = (time.time_ns() - time_now) / 10**9  # about 0.12 ms on localhost (compress_level=1 or 6, doesn't matter)
                    # print(f"send {send_duration_sec:0.6g}s")  # DEBUG

                    # Update the FPS counter, measuring the time between network sends.
                    time_now = time.time_ns()
                    if last_frame_send_complete_time is not None:
                        this_frame_elapsed_sec = (time_now - last_frame_send_complete_time) / 10**9
                        send_duration_statistics.add_datapoint(this_frame_elapsed_sec)
                    last_frame_send_complete_time = time_now
                else:
                    time.sleep(time_until_frame_deadline)

                # Log the FPS counter in 5-second intervals.
                time_now = time.time_ns()
                if animation_running and (last_report_time is None or time_now - last_report_time > 5e9):
                    avg_send_sec = send_duration_statistics.average()
                    msec = round(1000 * avg_send_sec, 1)
                    target_msec = round(1000 * frame_duration_target_sec, 1)
                    fps = round(1 / avg_send_sec, 1) if avg_send_sec > 0.0 else 0.0
                    logger.info(f"output: {msec:.1f}ms [{fps:.1f} FPS]; target {target_msec:.1f}ms [{target_fps:.1f} FPS]")
                    last_report_time = time_now

            else:  # first frame not yet available
                time.sleep(0.1)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

# TODO: the input is a flask.request.file.stream; what's the type of that?
def talkinghead_load_file(stream) -> str:
    """Load image from stream and start animation."""
    global global_reload_image
    global animation_running
    logger.info("talkinghead_load_file: loading new input image from stream")

    try:
        animation_running = False  # pause animation while loading a new image
        pil_image = PIL.Image.open(stream)  # Load the image using PIL.Image.open
        img_data = io.BytesIO()  # Create a copy of the image data in memory using BytesIO
        pil_image.save(img_data, format="PNG")
        global_reload_image = PIL.Image.open(io.BytesIO(img_data.getvalue()))  # Set the global_reload_image to a copy of the image data
    except PIL.Image.UnidentifiedImageError:
        logger.warning("Could not load input image from stream, loading blank")
        full_path = os.path.join(os.getcwd(), os.path.join(talkinghead_basedir, "tha3", "images", "inital.png"))
        global_reload_image = PIL.Image.open(full_path)
    finally:
        animation_running = True
    return "OK"

def launch(device: str, model: str) -> Union[None, NoReturn]:
    """Launch the talking head plugin (live mode).

    If the plugin fails to load, the process exits.

    device: "cpu" or "cuda"
    model: one of the folder names inside "talkinghead/tha3/models/"
    """
    global global_animator_instance
    global global_encoder_instance

    try:
        # If the animator already exists, clean it up first
        if global_animator_instance is not None:
            logger.info(f"launch: relaunching on device {device} with model {model}")
            global_animator_instance.exit()
            global_animator_instance = None
            global_encoder_instance.exit()
            global_encoder_instance = None

        logger.info("launch: loading the THA3 posing engine")
        poser = load_poser(model, device, modelsdir=os.path.join(talkinghead_basedir, "tha3", "models"))
        global_animator_instance = Animator(poser, device)
        global_encoder_instance = Encoder()

        # Load initial blank character image
        full_path = os.path.join(os.getcwd(), os.path.join(talkinghead_basedir, "tha3", "images", "inital.png"))
        global_animator_instance.load_image(full_path)

        global_animator_instance.start()
        global_encoder_instance.start()

    except RuntimeError as exc:
        logger.error(exc)
        sys.exit()

# --------------------------------------------------------------------------------
# Internal stuff

def convert_linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    """RGBA (linear) -> RGBA (SRGB), preserving the alpha channel."""
    rgb_image = torch_linear_to_srgb(image[0:3, :, :])
    return torch.cat([rgb_image, image[3:4, :, :]], dim=0)


class Animator:
    """uWu Waifu"""

    def __init__(self, poser: Poser, device: torch.device):
        self.poser = poser
        self.device = device

        self.postprocessor = Postprocessor(device)
        self.render_duration_statistics = RunningAverage()
        self.animator_thread = None

        self.source_image: Optional[torch.tensor] = None
        self.result_image: Optional[np.array] = None
        self.new_frame_available = False
        self.last_report_time = None

        self.reset_animation_state()
        self.load_emotion_templates()
        self.load_animator_settings()

    # --------------------------------------------------------------------------------
    # Management

    def start(self) -> None:
        """Start the animation thread."""
        self._terminated = False
        def animator_update():
            while not self._terminated:
                try:
                    self.render_animation_frame()
                except Exception as exc:
                    logger.error(exc)
                    raise  # let the animator stop so we won't spam the log
                time.sleep(0.01)  # rate-limit the renderer to 100 FPS maximum (this could be adjusted later)
        self.animator_thread = threading.Thread(target=animator_update, daemon=True)
        self.animator_thread.start()
        atexit.register(self.exit)

    def exit(self) -> None:
        """Terminate the animation thread.

        Called automatically when the process exits.
        """
        self._terminated = True
        self.animator_thread.join()
        self.animator_thread = None

    def reset_animation_state(self):
        """Reset character state trackers for all animation drivers."""
        self.current_pose = None

        self.last_emotion = None
        self.last_emotion_change_timestamp = None

        self.last_sway_target_timestamp = None
        self.last_sway_target_pose = None
        self.last_microsway_timestamp = None
        self.sway_interval = None

        self.last_blink_timestamp = None
        self.blink_interval = None

        self.last_talking_timestamp = None
        self.last_talking_target_value = None
        self.was_talking = False

        self.breathing_epoch = time.time_ns()

    def load_emotion_templates(self, emotions: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        """Load emotion templates.

        `emotions`: `{emotion0: {morph0: value0, ...}, ...}`
                    Optional dict of custom emotion templates.

                    If not given, this loads the templates from the emotion JSON files
                    in `talkinghead/emotions/`.

                    If given:
                      - Each emotion NOT supplied is populated from the defaults.
                      - In each emotion that IS supplied, each morph that is NOT mentioned
                        is implicitly set to zero (due to how `apply_emotion_to_pose` works).

                    For an example JSON file containing a suitable dictionary, see `talkinghead/emotions/_defaults.json`.

                    For available morph names, see `posedict_keys` in `talkinghead/tha3/app/util.py`.

                    For some more detail, see `talkinghead/tha3/poser/modes/pose_parameters.py`.
                    "Arity 2" means `posedict_keys` has separate left/right morphs.

                    If still in doubt, see the GUI panel implementations in `talkinghead/tha3/app/manual_poser.py`.
        """
        # Load defaults as a base
        self.emotions, self.emotion_names = load_emotion_presets(os.path.join("talkinghead", "emotions"))

        # Then override defaults, and add any new custom emotions
        if emotions is not None:
            logger.info(f"load_emotion_templates: loading user-specified templates for emotions {list(sorted(emotions.keys()))}")

            self.emotions.update(emotions)

            emotion_names = set(self.emotion_names)
            emotion_names.update(emotions.keys())
            self.emotion_names = list(sorted(emotion_names))
        else:
            logger.info("load_emotion_templates: loaded default emotion templates")

    def load_animator_settings(self, settings: Optional[Dict[str, Any]] = None) -> None:
        """Load animator settings.

        `settings`: `{setting0: value0, ...}`
                    Optional dict of settings. The type and semantics of each value depends on each
                    particular setting.

        For available settings, see `animator_defaults` in `talkinghead/tha3/app/app.py`.

        Particularly for the setting `"postprocessor_chain"` (pixel-space glitch artistry),
        see `talkinghead/tha3/app/postprocessor.py`.
        """
        global target_fps

        if settings is None:
            settings = {}

        logger.info(f"load_animator_settings: user settings: {settings}")

        # Load server-side settings (`talkinghead/animator.json`)
        try:
            animator_config_path = os.path.join(talkinghead_basedir, "animator.json")
            with open(animator_config_path, "r") as json_file:
                server_settings = json.load(json_file)
        except Exception as exc:
            logger.info(f"load_animator_settings: skipping server settings, reason: {exc}")
            server_settings = {}

        # Let's define some helpers:
        def drop_unrecognized(settings: Dict[str, Any], context: str) -> None:  # DANGER: MUTATING FUNCTION
            unknown_fields = [field for field in settings if field not in animator_defaults]
            if unknown_fields:
                logger.warning(f"load_animator_settings: in {context}: this server did not recognize the following settings, ignoring them: {unknown_fields}")
            for field in unknown_fields:
                settings.pop(field)
            assert all(field in animator_defaults for field in settings)  # contract: only known settings remaining

        def typecheck(settings: Dict[str, Any], context: str) -> None:  # DANGER: MUTATING FUNCTION
            for field, default_value in animator_defaults.items():
                type_match = (int, float) if isinstance(default_value, (int, float)) else type(default_value)
                if field in settings and not isinstance(settings[field], type_match):
                    logger.warning(f"load_animator_settings: in {context}: incorrect type for '{field}': got {type(settings[field])} with value '{settings[field]}', expected {type_match}")
                    settings.pop(field)  # (safe; this is not the collection we are iterating over)

        def aggregate(settings: Dict[str, Any], fallback_settings: Dict[str, Any], fallback_context: str) -> None:  # DANGER: MUTATING FUNCTION
            for field, default_value in fallback_settings.items():
                if field not in settings:
                    logger.info(f"load_animator_settings: filling in '{field}' from {fallback_context}")
                    settings[field] = default_value

        # Now our settings loading strategy is as simple as:
        settings = dict(settings)  # copy to avoid modifying the original, since we'll pop some stuff.
        if settings:
            drop_unrecognized(settings, context="user settings")
            typecheck(settings, context="user settings")
        if server_settings:
            drop_unrecognized(server_settings, context="server settings")
            typecheck(server_settings, context="server settings")
        # both `settings` and `server_settings` are fully valid at this point
        aggregate(settings, fallback_settings=server_settings, fallback_context="server settings")  # first fill in from server-side settings
        aggregate(settings, fallback_settings=animator_defaults, fallback_context="built-in defaults")  # then fill in from hardcoded defaults

        logger.info(f"load_animator_settings: final settings (filled in as necessary): {settings}")

        # Some settings must be applied explicitly.
        logger.debug(f"load_animator_settings: Setting new target FPS = {settings['target_fps']}")
        target_fps = settings.pop("target_fps")  # global variable, controls the network send rate.

        logger.debug("load_animator_settings: Sending new effect chain to postprocessor")
        self.postprocessor.chain = settings.pop("postprocessor_chain")  # ...and that's where the postprocessor reads its filter settings from.

        # The rest of the settings we can just store in an attribute, and let the animation drivers read them from there.
        self._settings = settings

    def load_image(self, file_path=None) -> None:
        """Load the image file at `file_path`, and replace the current character with it.

        Except, if `global_reload_image is not None`, use the global reload image data instead.
        In that case `file_path` is not used.

        When done, this always sets `global_reload_image` to `None`.
        """
        global global_reload_image

        try:
            if global_reload_image is not None:
                pil_image = global_reload_image
            else:
                pil_image = resize_PIL_image(
                    extract_PIL_image_from_filelike(file_path),
                    (self.poser.get_image_size(), self.poser.get_image_size()))

            w, h = pil_image.size

            if pil_image.size != (512, 512):
                logger.info("Resizing Char Card to work")
                pil_image = to_talkinghead_image(pil_image)

            w, h = pil_image.size

            if pil_image.mode != "RGBA":
                logger.error("load_image: image must have alpha channel")
                self.source_image = None
            else:
                self.source_image = extract_pytorch_image_from_PIL_image(pil_image) \
                    .to(self.device).to(self.poser.get_dtype())

        except Exception as exc:
            logger.error(f"load_image: {exc}")

        finally:
            global_reload_image = None

    # --------------------------------------------------------------------------------
    # Animation drivers

    def apply_emotion_to_pose(self, emotion_posedict: Dict[str, float], pose: List[float]) -> List[float]:
        """Copy all morphs except breathing from `emotion_posedict` to `pose`.

        If a morph does not exist in `emotion_posedict`, its value is copied from the original `pose`.

        Return the modified pose.
        """
        new_pose = list(pose)  # copy
        for idx, key in enumerate(posedict_keys):
            if key in emotion_posedict and key != "breathing_index":
                new_pose[idx] = emotion_posedict[key]
        return new_pose

    def animate_blinking(self, pose: List[float]) -> List[float]:
        """Eye blinking animation driver.

        Relevant `self._settings` keys:

        `"blink_interval_min"`: float, seconds, lower limit for random minimum time until next blink is allowed.
        `"blink_interval_max"`: float, seconds, upper limit for random minimum time until next blink is allowed.
        `"blink_probability"`: float, at each frame at a reference of 25 FPS. FPS-corrected automatically.
        `"blink_confusion_duration"`: float, seconds, upon entering "confusion" emotion, during which blinking
                                      quickly in succession is allowed.

        Return the modified pose.
        """
        # Compute FPS-corrected blink probability
        CALIBRATION_FPS = 25
        p_orig = self._settings["blink_probability"]  # blink probability per frame at CALIBRATION_FPS
        avg_render_sec = self.render_duration_statistics.average()
        if avg_render_sec > 0:
            avg_render_fps = 1 / avg_render_sec
            # Even if render completes faster, the `talkinghead` output is rate-limited to `target_fps` at most.
            avg_render_fps = min(avg_render_fps, target_fps)
        else:  # No statistics available yet; let's assume we're running at `target_fps`.
            avg_render_fps = target_fps
        # We give an independent trial for each of `n` "normalized frames" elapsed at `CALIBRATION_FPS` during one actual frame at `avg_render_fps`.
        # Note direction: rendering faster (higher FPS) means less likely to blink per frame, to obtain the same blink density per unit of wall time.
        n = CALIBRATION_FPS / avg_render_fps
        # If at least one of the normalized frames wants to blink, then the actual frame should blink.
        # Doesn't matter that `n` isn't an integer, since the power function over the reals is continuous and we just want a reasonable scaling here.
        p_scaled = 1.0 - (1.0 - p_orig)**n
        should_blink = (random.random() <= p_scaled)

        debug_fps = round(avg_render_fps, 1)
        logger.debug(f"animate_blinking: p @ {CALIBRATION_FPS} FPS = {p_orig}, scaled p @ {debug_fps:.1f} FPS = {p_scaled:0.6g}")

        # Prevent blinking too fast in succession.
        time_now = time.time_ns()
        if self.blink_interval is not None:
            # ...except when the "confusion" emotion has been entered recently.
            seconds_since_last_emotion_change = (time_now - self.last_emotion_change_timestamp) / 10**9
            if current_emotion == "confusion" and seconds_since_last_emotion_change < self._settings["blink_confusion_duration"]:
                pass
            else:
                seconds_since_last_blink = (time_now - self.last_blink_timestamp) / 10**9
                if seconds_since_last_blink < self.blink_interval:
                    should_blink = False

        if not should_blink:
            return pose

        # If there should be a blink, set the wink morphs to 1.
        new_pose = list(pose)  # copy
        for morph_name in ["eye_wink_left_index", "eye_wink_right_index"]:
            idx = posedict_key_to_index[morph_name]
            new_pose[idx] = 1.0

        # Typical for humans is 12...20 times per minute, i.e. 5...3 seconds interval.
        self.last_blink_timestamp = time_now
        self.blink_interval = random.uniform(self._settings["blink_interval_min"],
                                             self._settings["blink_interval_max"])  # seconds; duration of this blink before the next one can begin

        return new_pose

    def animate_talking(self, pose: List[float], target_pose: List[float]) -> List[float]:
        """Talking animation driver.

        Relevant `self._settings` keys:

        `"talking_fps"`: float, how often to re-randomize mouth during talking animation.
                         Early 2000s anime used ~12 FPS as the fastest actual framerate of
                         new cels (not counting camera panning effects and such).
        `"talking_morph"`: str, see `posedict_keys` for available values.
                           Which morph to use for opening and closing the mouth during talking.
                           Any other morphs in the mouth-open group are set to zero while
                           talking is in progress.

        Works by randomizing the mouth-open state in regular intervals.

        When talking ends, the mouth immediately snaps to its position in the target pose
        (to avoid a slow, unnatural closing, since most expressions have the mouth closed).

        Return the modified pose.
        """
        MOUTH_OPEN_MORPHS = ["mouth_aaa_index", "mouth_iii_index", "mouth_uuu_index", "mouth_eee_index", "mouth_ooo_index", "mouth_delta"]
        talking_morph = self._settings["talking_morph"]

        if not is_talking:
            try:
                if self.was_talking:  # when talking ends, snap mouth to target immediately
                    new_pose = list(pose)  # copy
                    for key in MOUTH_OPEN_MORPHS:
                        idx = posedict_key_to_index[key]
                        new_pose[idx] = target_pose[idx]
                    return new_pose
                return pose  # most common case: do nothing (not talking, and wasn't talking during previous frame)
            finally:  # reset state *after* processing
                self.last_talking_target_value = None
                self.last_talking_timestamp = None
                self.was_talking = False
        assert is_talking

        # With 25 FPS (or faster) output, randomizing the mouth every frame looks too fast.
        # Determine whether enough wall time has passed to randomize a new mouth position.
        TARGET_SEC = 1 / self._settings["talking_fps"]  # rate of "actual new cels" in talking animation
        time_now = time.time_ns()
        update_mouth = False
        if self.last_talking_timestamp is None:
            update_mouth = True
        else:
            time_elapsed_sec = (time_now - self.last_talking_timestamp) / 10**9
            if time_elapsed_sec >= TARGET_SEC:
                update_mouth = True

        # Apply the mouth open morph
        new_pose = list(pose)  # copy
        idx = posedict_key_to_index[talking_morph]
        if self.last_talking_target_value is None or update_mouth:
            # Randomize new mouth position
            x = pose[idx]
            x = abs(1.0 - x) + random.uniform(-2.0, 2.0)
            x = max(0.0, min(x, 1.0))  # clamp (not the manga studio)
            self.last_talking_target_value = x
            self.last_talking_timestamp = time_now
        else:
            # Keep the mouth at its latest randomized position (this overrides the interpolator that would pull the mouth toward the target emotion pose)
            x = self.last_talking_target_value
        new_pose[idx] = x

        # Zero out other morphs that affect mouth open/closed state.
        for key in MOUTH_OPEN_MORPHS:
            if key == talking_morph:
                continue
            idx = posedict_key_to_index[key]
            new_pose[idx] = 0.0

        self.was_talking = True
        return new_pose

    def compute_sway_target_pose(self, original_target_pose: List[float]) -> List[float]:
        """History-free sway animation driver.

        `original_target_pose`: emotion pose to modify with a randomized sway target

        Relevant `self._settings` keys:

        `"sway_morphs"`: List[str], which morphs can sway. By default, this is all geometric transformations,
                         but disabling some can be useful for some characters (such as robots).
                         For available values, see `posedict_keys`.
        `"sway_interval_min"`: float, seconds, lower limit for random time interval until randomizing new sway pose.
        `"sway_interval_max"`: float, seconds, upper limit for random time interval until randomizing new sway pose.
                               Note the limits are ignored when `original_target_pose` changes (then immediately refreshing
                               the sway pose), because an emotion pose may affect the geometric transformations, too.
        `"sway_macro_strength"`: float, [0, 1]. In sway pose, max abs deviation from emotion pose target morph value
                                 for each sway morph, but also max deviation from center. The `original_target_pose`
                                 itself may use higher values; in such cases, sway will only occur toward the center.
                                 See the source code of this function for the exact details.
        `"sway_micro_strength"`: float, [0, 1]. Max abs random noise to sway target pose, added each frame, to make
                                 the animation look less robotic. No limiting other than a clamp of final pose to [-1, 1].

        The sway target pose is randomized again when necessary; this takes care of caching internally.

        Return the modified pose.
        """
        # We just modify the target pose, and let the ODE integrator (`interpolate_pose`) do the actual animation.
        # - This way we don't need to track start state, progress, etc.
        # - This also makes the animation nonlinear automatically: a saturating exponential trajectory toward the target.
        #   - If we want a smooth start toward a target pose/morph, we can e.g. save the timestamp when the animation began, and then ramp the rate of change,
        #     beginning at zero and (some time later, as measured from the timestamp) ending at the original, non-ramped value. The ODE itself takes care of
        #     slowing down when we approach the target state.

        # As documented in the original THA tech reports, on the pose axes, zero is centered, and 1.0 = 15 degrees.
        random_max = self._settings["sway_macro_strength"]  # max sway magnitude from center position of each morph
        noise_max = self._settings["sway_micro_strength"]  # amount of dynamic noise (re-generated every frame), added on top of the sway target, no clamping except to [-1, 1]
        SWAYPARTS = self._settings["sway_morphs"]  # some characters might not sway on all axes (e.g. a robot)

        def macrosway() -> List[float]:  # this handles caching and everything
            time_now = time.time_ns()
            should_pick_new_sway_target = True
            if current_emotion == self.last_emotion:
                if self.sway_interval is not None:  # have we created a swayed pose at least once?
                    seconds_since_last_sway_target = (time_now - self.last_sway_target_timestamp) / 10**9
                    if seconds_since_last_sway_target < self.sway_interval:
                        should_pick_new_sway_target = False
            # else, emotion has changed, invalidating the old sway target, because it is based on the old emotion (since emotions may affect the pose too).

            if not should_pick_new_sway_target:
                if self.last_sway_target_pose is not None:  # When keeping the same sway target, return the cached sway pose if we have one.
                    return self.last_sway_target_pose
                else:  # Should not happen, but let's be robust.
                    return original_target_pose

            new_target_pose = list(original_target_pose)  # copy
            for key in SWAYPARTS:
                idx = posedict_key_to_index[key]
                target_value = original_target_pose[idx]

                # Determine the random range so that the swayed target always stays within `[-random_max, random_max]`, regardless of `target_value`.
                # TODO: This is a simple zeroth-order solution that just cuts the random range.
                #       Would be nicer to *gradually* decrease the available random range on the "outside" as the target value gets further from the origin.
                random_upper = max(0, random_max - target_value)  # e.g. if target_value = 0.2, then random_upper = 0.4  => max possible = 0.6 = random_max
                random_lower = min(0, -random_max - target_value)  # e.g. if target_value = -0.2, then random_lower = -0.4  => min possible = -0.6 = -random_max
                random_value = random.uniform(random_lower, random_upper)

                new_target_pose[idx] = target_value + random_value

            self.last_sway_target_pose = new_target_pose
            self.last_sway_target_timestamp = time_now
            self.sway_interval = random.uniform(self._settings["sway_interval_min"],
                                                self._settings["sway_interval_max"])  # seconds; duration of this sway target before randomizing new one
            return new_target_pose

        # Add dynamic noise (re-generated at 25 FPS) to the target to make the animation look less robotic, especially once we are near the target pose.
        def add_microsway() -> None:  # DANGER: MUTATING FUNCTION
            CALIBRATION_FPS = 25  # FPS at which randomizing a new microsway target looks good
            time_now = time.time_ns()
            should_microsway = True
            if self.last_microsway_timestamp is not None:
                seconds_since_last_microsway = (time_now - self.last_microsway_timestamp) / 10**9
                if seconds_since_last_microsway < 1 / CALIBRATION_FPS:
                    should_microsway = False

            if should_microsway:
                for key in SWAYPARTS:
                    idx = posedict_key_to_index[key]
                    x = new_target_pose[idx] + random.uniform(-noise_max, noise_max)
                    x = max(-1.0, min(x, 1.0))
                    new_target_pose[idx] = x
                self.last_microsway_timestamp = time_now

        new_target_pose = macrosway()
        add_microsway()
        return new_target_pose

    def animate_breathing(self, pose: List[float]) -> List[float]:
        """Breathing animation driver.

        Relevant `self._settings` keys:

        `"breathing_cycle_duration"`: seconds. Duration of one full breathing cycle.

        Return the modified pose.
        """
        breathing_cycle_duration = self._settings["breathing_cycle_duration"]  # seconds

        time_now = time.time_ns()
        t = (time_now - self.breathing_epoch) / 10**9  # seconds since breathing-epoch
        cycle_pos = t / breathing_cycle_duration  # number of cycles since breathing-epoch
        if cycle_pos > 1.0:  # prevent loss of accuracy in long sessions
            self.breathing_epoch = time_now  # TODO: be more accurate here, should sync to a whole cycle
        cycle_pos = cycle_pos - float(int(cycle_pos))  # fractional part

        new_pose = list(pose)  # copy
        idx = posedict_key_to_index["breathing_index"]
        new_pose[idx] = math.sin(cycle_pos * math.pi)**2  # 0 ... 1 ... 0, smoothly, with slow start and end, fast middle
        return new_pose

    def interpolate_pose(self, pose: List[float], target_pose: List[float]) -> List[float]:
        """Interpolate from current `pose` toward `target_pose`.

        Relevant `self._settings` keys:

        `"pose_interpolator_step"`: [0, 1]; how far toward `target_pose` to interpolate in one frame,
                                            assuming a reference of 25 FPS. This is FPS-corrected automatically.
                                            0 is fully `pose`, 1 is fully `target_pose`.

        This is a kind of history-free rate-based formulation, which needs only the current and target poses, and
        the step size; there is no need to keep track of e.g. the initial pose or the progress along the trajectory.

        Note that looping back the output as `pose`, while keeping `target_pose` constant, causes the current pose
        to approach `target_pose` on a saturating trajectory. This is because `step` is the fraction of the *current*
        difference between `pose` and `target_pose`, which obviously becomes smaller after each repeat.

        This is a feature, not a bug!
        """
        # The `step` parameter is calibrated against animation at 25 FPS, so we must scale it appropriately, taking
        # into account the actual FPS.
        #
        # How to do this requires some explanation. Numericist hat on. Let's do a quick back-of-the-envelope calculation.
        # This pose interpolator is essentially a solver for the first-order ODE:
        #
        #   u' = f(u, t)
        #
        # Consider the most common case, where the target pose remains constant over several animation frames.
        # Furthermore, consider just one morph (they all behave similarly). Then our ODE is Newton's law of cooling:
        #
        #   u' = -β [u - u∞]
        #
        # where `u = u(t)` is the temperature, `u∞` is the constant temperature of the external environment,
        # and `β > 0` is a material-dependent cooling coefficient.
        #
        # But instead of numerical simulation at a constant timestep size, as would be typical in computational science,
        # we instead read off points off the analytical solution curve. The `step` parameter is *not* the timestep size;
        # instead, it controls the relative distance along the *u* axis that should be covered in one simulation step,
        # so it is actually related to the cooling coefficient β.
        #
        # (How exactly: write the left-hand side as `[unew - uold] / Δt + O([Δt]²)`, drop the error term, and decide
        #  whether to use `uold` (forward Euler) or `unew` (backward Euler) as `u` on the right-hand side. Then compare
        #  to our update formula. But those details don't matter here.)
        #
        # To match the notation in the rest of this code, let us denote the temperature (actually pose morph value) as `x`
        # (instead of `u`). And to keep notation shorter, let `β := step` (although it's not exactly the `β` of the
        # continuous-in-time case above).
        #
        # To scale the animation speed linearly with regard to FPS, we must invert the relation between simulation step
        # number `n` and the solution value `x`. For an initial value `x0`, a constant target value `x∞`, and constant
        # step `β ∈ (0, 1]`, the pose interpolator produces the sequence:
        #
        #   x1 = x0 + β [x∞ - x0] = [1 - β] x0 + β x∞
        #   x2 = x1 + β [x∞ - x1] = [1 - β] x1 + β x∞
        #   x3 = x2 + β [x∞ - x2] = [1 - β] x2 + β x∞
        #   ...
        #
        # Note that with exact arithmetic, if `β < 1`, the final value is only reached in the limit `n → ∞`.
        # For floating point, this is not the case. Eventually the increment becomes small enough that when
        # it is added, nothing happens. After sufficiently many steps, in practice `x` will stop just slightly
        # short of `x∞` (on the side it approached the target from).
        #
        # (For performance reasons, when approaching zero, one may need to beware of denormals, because those
        #  are usually implemented in (slow!) software on modern CPUs. So especially if the target is zero,
        #  it is useful to have some very small cutoff (inside the normal floating-point range) after which
        #  we make `x` instantly jump to the target value.)
        #
        # Inserting the definition of `x1` to the formula for `x2`, we can express `x2` in terms of `x0` and `x∞`:
        #
        #   x2 = [1 - β] ([1 - β] x0 + β x∞) + β x∞
        #      = [1 - β]² x0 + [1 - β] β x∞ + β x∞
        #      = [1 - β]² x0 + [[1 - β] + 1] β x∞
        #
        # Then inserting this to the formula for `x3`:
        #
        #   x3 = [1 - β] ([1 - β]² x0 + [[1 - β] + 1] β x∞) + β x∞
        #      = [1 - β]³ x0 + [1 - β]² β x∞ + [1 - β] β x∞ + β x∞
        #
        # To simplify notation, define:
        #
        #   α := 1 - β
        #
        # We have:
        #
        #   x1 = α  x0 + [1 - α] x∞
        #   x2 = α² x0 + [1 - α] [1 + α] x∞
        #      = α² x0 + [1 - α²] x∞
        #   x3 = α³ x0 + [1 - α] [1 + α + α²] x∞
        #      = α³ x0 + [1 - α³] x∞
        #
        # This suggests that the general pattern is (as can be proven by induction on `n`):
        #
        #   xn = α**n x0 + [1 - α**n] x∞
        #
        # This allows us to determine `x` as a function of simulation step number `n`. Now the scaling question becomes:
        # if we want to reach a given value `xn` by some given step `n_scaled` (instead of the original step `n`),
        # how must we change the step size `β` (or equivalently, the parameter `α`)?
        #
        # To simplify further, observe:
        #
        #   x1 = α x0 + [1 - α] [[x∞ - x0] + x0]
        #      = [α + [1 - α]] x0 + [1 - α] [x∞ - x0]
        #      = x0 + [1 - α] [x∞ - x0]
        #
        # Rearranging yields:
        #
        #   [x1 - x0] / [x∞ - x0] = 1 - α
        #
        # which gives us the relative distance from `x0` to `x∞` that is covered in one step. This isn't yet much
        # to write home about (it's essentially just a rearrangement of the definition of `x1`), but next, let's
        # treat `x2` the same way:
        #
        #   x2 = α² x0 + [1 - α] [1 + α] [[x∞ - x0] + x0]
        #      = [α² x0 + [1 - α²] x0] + [1 - α²] [x∞ - x0]
        #      = [α² + 1 - α²] x0 + [1 - α²] [x∞ - x0]
        #      = x0 + [1 - α²] [x∞ - x0]
        #
        # We obtain
        #
        #   [x2 - x0] / [x∞ - x0] = 1 - α²
        #
        # which is the relative distance, from the original `x0` toward the final `x∞`, that is covered in two steps
        # using the original step size `β = 1 - α`. Next up, `x3`:
        #
        #   x3 = α³ x0 + [1 - α³] [[x∞ - x0] + x0]
        #      = α³ x0 + [1 - α³] [x∞ - x0] + [1 - α³] x0
        #      = x0 + [1 - α³] [x∞ - x0]
        #
        # Rearranging,
        #
        #   [x3 - x0] / [x∞ - x0] = 1 - α³
        #
        # which is the relative distance covered in three steps. Hence, we have:
        #
        #   xrel := [xn - x0] / [x∞ - x0] = 1 - α**n
        #
        # so that
        #
        #   α**n = 1 - xrel              (**)
        #
        # and (taking the natural logarithm of both sides)
        #
        #   n log α = log [1 - xrel]
        #
        # Finally,
        #
        #   n = [log [1 - xrel]] / [log α]
        #
        # Given `α`, this gives the `n` where the interpolator has covered the fraction `xrel` of the original distance.
        # On the other hand, we can also solve (**) for `α`:
        #
        #   α = (1 - xrel)**(1 / n)
        #
        # which, given desired `n`, gives us the `α` that makes the interpolator cover the fraction `xrel` of the original distance in `n` steps.
        #
        CALIBRATION_FPS = 25  # FPS for which the default value `step` was calibrated
        xrel = 0.5  # just some convenient value
        step = self._settings["pose_interpolator_step"]
        alpha_orig = 1.0 - step
        if 0 < alpha_orig < 1:
            avg_render_sec = self.render_duration_statistics.average()
            if avg_render_sec > 0:
                avg_render_fps = 1 / avg_render_sec
                # Even if render completes faster, the `talkinghead` output is rate-limited to `target_fps` at most.
                avg_render_fps = min(avg_render_fps, target_fps)
            else:  # No statistics available yet; let's assume we're running at `target_fps`.
                avg_render_fps = target_fps

            # For a constant target pose and original `α`, compute the number of animation frames to cover `xrel` of distance from initial pose to final pose.
            n_orig = math.log(1.0 - xrel) / math.log(alpha_orig)
            # Compute the scaled `n`. Note the direction: we need a smaller `n` (fewer animation frames) if the render runs slower than the calibration FPS.
            n_scaled = (avg_render_fps / CALIBRATION_FPS) * n_orig
            # Then compute the `α` that reaches `xrel` distance in `n_scaled` animation frames.
            alpha_scaled = (1.0 - xrel)**(1 / n_scaled)
        else:  # avoid some divisions by zero at the extremes
            alpha_scaled = alpha_orig
        step_scaled = 1.0 - alpha_scaled

        debug_fps = round(avg_render_fps, 1)
        logger.debug(f"interpolate_pose: step @ {CALIBRATION_FPS} FPS = {step}, scaled step @ {debug_fps:.1f} FPS = {step_scaled:0.6g}")

        # NOTE: This overwrites blinking, talking, and breathing, but that doesn't matter, because we apply this first.
        # The other animation drivers then modify our result.
        EPSILON = 1e-8
        new_pose = list(pose)  # copy
        for idx, key in enumerate(posedict_keys):
            # # We now animate blinking *after* interpolating the pose, so when blinking, the eyes close instantly.
            # # This modification would make the blink also end instantly.
            # if key in ["eye_wink_left_index", "eye_wink_right_index"]:
            #     new_pose[idx] = target_pose[idx]
            # else:
            #     ...

            delta = target_pose[idx] - pose[idx]
            new_pose[idx] = pose[idx] + step_scaled * delta

            # Prevent denormal floats (which are really slow); important when running on CPU and approaching zero.
            # Our ϵ is really big compared to denormals; but there's no point in continuing to compute ever smaller
            # differences in the animated value when it has already almost (and visually, completely) reached the target.
            if abs(new_pose[idx] - target_pose[idx]) < EPSILON:
                new_pose[idx] = target_pose[idx]
        return new_pose

    # --------------------------------------------------------------------------------
    # Animation logic

    def render_animation_frame(self) -> None:
        """Render an animation frame.

        If the previous rendered frame has not been retrieved yet, do nothing.
        """
        if not animation_running:
            return

        # If no one has retrieved the latest rendered frame yet, do not render a new one.
        if self.new_frame_available:
            return

        if global_reload_image is not None:
            self.load_image()
        if self.source_image is None:
            return

        time_render_start = time.time_ns()

        if self.current_pose is None:  # initialize character pose at plugin startup
            self.current_pose = posedict_to_pose(self.emotions[current_emotion])

        emotion_posedict = self.emotions[current_emotion]
        if current_emotion != self.last_emotion:  # some animation drivers need to know when the emotion last changed
            self.last_emotion_change_timestamp = time_render_start

        target_pose = self.apply_emotion_to_pose(emotion_posedict, self.current_pose)
        target_pose = self.compute_sway_target_pose(target_pose)

        self.current_pose = self.interpolate_pose(self.current_pose, target_pose)
        self.current_pose = self.animate_blinking(self.current_pose)
        self.current_pose = self.animate_talking(self.current_pose, target_pose)
        self.current_pose = self.animate_breathing(self.current_pose)

        # Update this last so that animation drivers have access to the old emotion, too.
        self.last_emotion = current_emotion

        pose = torch.tensor(self.current_pose, device=self.device, dtype=self.poser.get_dtype())

        with torch.no_grad():
            # - [0]: model's output index for the full result image
            # - model's data range is [-1, +1], linear intensity ("gamma encoded")
            output_image = self.poser.pose(self.source_image, pose)[0].float()

            # A simple crop filter, for removing empty space around character.
            # Apply this first so that the postprocessor has fewer pixels to process.
            c, h, w = output_image.shape
            x1 = int((self._settings["crop_left"] / 2.0) * w)
            x2 = int((1 - (self._settings["crop_right"] / 2.0)) * w)
            y1 = int((self._settings["crop_top"] / 2.0) * h)
            y2 = int((1 - (self._settings["crop_bottom"] / 2.0)) * h)
            output_image = output_image[:, y1:y2, x1:x2]

            # [-1, 1] -> [0, 1]
            # output_image = (output_image + 1.0) / 2.0
            output_image.add_(1.0)
            output_image.mul_(0.5)

            self.postprocessor.render_into(output_image)  # apply pixel-space glitch artistry
            output_image = convert_linear_to_srgb(output_image)  # apply gamma correction

            # convert [c, h, w] float -> [h, w, c] uint8
            c, h, w = output_image.shape
            output_image = torch.transpose(output_image.reshape(c, h * w), 0, 1).reshape(h, w, c)
            output_image = (255.0 * output_image).byte()

            output_image_numpy = output_image.detach().cpu().numpy()

        # Update FPS counter, measuring animation frame render time only.
        #
        # This says how fast the renderer *can* run on the current hardware;
        # note we don't actually render more frames than the client consumes.
        time_now = time.time_ns()
        if self.source_image is not None:
            render_elapsed_sec = (time_now - time_render_start) / 10**9
            # remove the average per-frame postprocessing time, to measure render time only
            render_elapsed_sec -= self.postprocessor.render_duration_statistics.average()
            self.render_duration_statistics.add_datapoint(render_elapsed_sec)

        # Set the new rendered frame as the output image, and mark the frame as ready for consumption.
        with _animator_output_lock:
            self.result_image = output_image_numpy  # atomic replace
            self.new_frame_available = True

        # Log the FPS counter in 5-second intervals.
        if animation_running and (self.last_report_time is None or time_now - self.last_report_time > 5e9):
            avg_render_sec = self.render_duration_statistics.average()
            msec = round(1000 * avg_render_sec, 1)
            fps = round(1 / avg_render_sec, 1) if avg_render_sec > 0.0 else 0.0
            logger.info(f"render: {msec:.1f}ms [{fps} FPS available]")
            self.last_report_time = time_now


class Encoder:
    """Network transport encoder.

    We read each frame from the animator as it becomes ready, and keep it available in `self.image_bytes`
    until the next frame arrives. The `self.image_bytes` buffer is replaced atomically, so this needs no lock
    (you always get the latest available frame at the time you access `image_bytes`).
    """

    def __init__(self) -> None:
        self.image_bytes = None
        self.encoder_thread = None

    def start(self) -> None:
        """Start the output encoder thread."""
        self._terminated = False
        def encoder_update():
            last_report_time = None
            encode_duration_statistics = RunningAverage()
            wait_duration_statistics = RunningAverage()

            while not self._terminated:
                # Retrieve a new frame from the animator if available.
                have_new_frame = False
                time_encode_start = time.time_ns()
                with _animator_output_lock:
                    if global_animator_instance.new_frame_available:
                        image_rgba = global_animator_instance.result_image
                        global_animator_instance.new_frame_available = False  # animation frame consumed; start rendering the next one
                        have_new_frame = True  # This flag is needed so we can release the animator lock as early as possible.

                # If a new frame arrived, pack it for sending (only once for each new frame).
                if have_new_frame:
                    try:
                        pil_image = PIL.Image.fromarray(np.uint8(image_rgba[:, :, :3]))
                        if image_rgba.shape[2] == 4:
                            alpha_channel = image_rgba[:, :, 3]
                            pil_image.putalpha(PIL.Image.fromarray(np.uint8(alpha_channel)))

                        # Save as PNG with RGBA mode. Use the fastest compression level available.
                        #
                        # On an i7-12700H @ 2.3 GHz (laptop optimized for low fan noise):
                        #  - `compress_level=1` (fastest), about 20 ms
                        #  - `compress_level=6` (default), about 40 ms (!) - too slow!
                        #  - `compress_level=9` (smallest size), about 120 ms
                        #
                        # time_now = time.time_ns()
                        buffer = io.BytesIO()
                        pil_image.save(buffer, format="PNG", compress_level=1)
                        image_bytes = buffer.getvalue()
                        # pack_duration_sec = (time.time_ns() - time_now) / 10**9

                        # We now have a new encoded frame; but first, sync with network send.
                        # This prevents from rendering/encoding more frames than are actually sent.
                        previous_frame = self.image_bytes
                        if previous_frame is not None:
                            time_wait_start = time.time_ns()
                            # Wait in 1ms increments until the previous encoded frame has been sent
                            while global_latest_frame_sent != id(previous_frame) and not self._terminated:
                                time.sleep(0.001)
                            time_now = time.time_ns()
                            wait_elapsed_sec = (time_now - time_wait_start) / 10**9
                        else:
                            wait_elapsed_sec = 0.0

                        self.image_bytes = image_bytes  # atomic replace so no need for a lock
                    except Exception as exc:
                        logger.error(exc)
                        raise  # let the encoder stop so we won't spam the log

                    # Update FPS counter.
                    time_now = time.time_ns()
                    walltime_elapsed_sec = (time_now - time_encode_start) / 10**9
                    encode_elapsed_sec = walltime_elapsed_sec - wait_elapsed_sec
                    encode_duration_statistics.add_datapoint(encode_elapsed_sec)
                    wait_duration_statistics.add_datapoint(wait_elapsed_sec)

                # Log the FPS counter in 5-second intervals.
                time_now = time.time_ns()
                if animation_running and (last_report_time is None or time_now - last_report_time > 5e9):
                    avg_encode_sec = encode_duration_statistics.average()
                    msec = round(1000 * avg_encode_sec, 1)
                    avg_wait_sec = wait_duration_statistics.average()
                    wait_msec = round(1000 * avg_wait_sec, 1)
                    fps = round(1 / avg_encode_sec, 1) if avg_encode_sec > 0.0 else 0.0
                    logger.info(f"encode: {msec:.1f}ms [{fps} FPS available]; send sync wait {wait_msec:.1f}ms")
                    last_report_time = time_now

                time.sleep(0.01)  # rate-limit the encoder to 100 FPS maximum (this could be adjusted later)
        self.encoder_thread = threading.Thread(target=encoder_update, daemon=True)
        self.encoder_thread.start()
        atexit.register(self.exit)

    def exit(self) -> None:
        """Terminate the output encoder thread.

        Called automatically when the process exits.
        """
        self._terminated = True
        self.encoder_thread.join()
        self.encoder_thread = None
