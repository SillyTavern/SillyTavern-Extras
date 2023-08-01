import argparse
import cv2
import os
import random
import requests
import sys
import threading
import time
import torch
import torch.nn.functional as F
import wx
from PIL import Image
from torchvision import transforms
from flask import Flask, render_template, Response, send_file, request
from flask_cors import CORS
from io import BytesIO

sys.path.append(os.getcwd())
from tha3.mocap.ifacialmocap_constants import *
from tha3.mocap.ifacialmocap_pose import create_default_ifacialmocap_pose
from tha3.mocap.ifacialmocap_pose_converter import IFacialMocapPoseConverter
from tha3.mocap.ifacialmocap_poser_converter_25 import create_ifacialmocap_pose_converter
from tha3.poser.modes.load_poser import load_poser
from tha3.poser.poser import Poser
from tha3.util import (
    torch_linear_to_srgb, resize_PIL_image, extract_PIL_image_from_filelike,
    extract_pytorch_image_from_PIL_image
)
from typing import Optional

# Add the current working directory to the system path
sys.path.append(os.getcwd())

# Global Variables
global_source_image = None
global_source_image_path = None
global_result_image = None
global_reload = None
is_talking_override = False
is_talking = False

# Flask setup
app = Flask(__name__)
CORS(app)

def start_talking():
    global is_talking_override
    is_talking_override = True
    #return send_file(global_source_image_path, mimetype='image/png')
    return "started"

def stop_talking():
    global is_talking_override
    is_talking_override = False
    return "stopped"

def result_feed():
    def generate():
        while True:
            if global_result_image is not None:

                try:
                    # Encode the numpy array to PNG
                    _, buffer = cv2.imencode('.png', global_result_image)
                except Exception as e:
                    print(f"Error when trying to write image: {e}")

                # Send the PNG image
                yield (b'--frame\r\n'
                       b'Content-Type: image/png\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                time.sleep(0.1)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def live2d_load_url(url):
    img = None
    global global_source_image
    global global_reload
    response = requests.get(url)
    try:
        img = Image.open(BytesIO(response.content))
    except Image.UnidentifiedImageError:
        print(f"Could not identify image from URL: {url}")
    global_reload = img
    return 'OK'

def live2d_load_file(stream):
    img = None
    global global_source_image
    global global_reload
    try:
        img = Image.open(stream)
    except Image.UnidentifiedImageError:
        print(f"Could not load image from file")
    global_reload = img
    return 'OK'

def convert_linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    rgb_image = torch_linear_to_srgb(image[0:3, :, :])
    return torch.cat([rgb_image, image[3:4, :, :]], dim=0)

def launch_gui(device, model):
    parser = argparse.ArgumentParser(description='uWu Waifu')

    # Add other parser arguments here

    args, unknown = parser.parse_known_args()

    try:
        poser = load_poser(model, device)
        pose_converter = create_ifacialmocap_pose_converter()

        app = wx.App()
        main_frame = MainFrame(poser, pose_converter, device)
        main_frame.SetSize((750, 600))

        #Lload default image (you can pass args.char if required)
        full_path = os.path.join(os.getcwd(), "live2d\\tha3\\images\\lambda_00.png")
        main_frame.load_image(None, full_path)

        #main_frame.Show(True)
        main_frame.capture_timer.Start(100)
        main_frame.animation_timer.Start(100)
        app.MainLoop()

    except RuntimeError as e:
        print(e)
        sys.exit()

class FpsStatistics:
    def __init__(self):
        self.count = 100
        self.fps = []

    def add_fps(self, fps):
        self.fps.append(fps)
        while len(self.fps) > self.count:
            del self.fps[0]

    def get_average_fps(self):
        if len(self.fps) == 0:
            return 0.0
        else:
            return sum(self.fps) / len(self.fps)

class MainFrame(wx.Frame):
    def __init__(self, poser: Poser, pose_converter: IFacialMocapPoseConverter, device: torch.device):
        super().__init__(None, wx.ID_ANY, "uWu Waifu")
        self.pose_converter = pose_converter
        self.poser = poser
        self.device = device

        self.image_load_counter = 0
        self.custom_background_image = None  # Add this line

        self.sliders = {}
        self.ifacialmocap_pose = create_default_ifacialmocap_pose()
        self.source_image_bitmap = wx.Bitmap(self.poser.get_image_size(), self.poser.get_image_size())
        self.result_image_bitmap = wx.Bitmap(self.poser.get_image_size(), self.poser.get_image_size())
        self.wx_source_image = None
        self.torch_source_image = None
        self.last_pose = None
        self.fps_statistics = FpsStatistics()
        self.last_update_time = None

        self.create_ui()

        self.create_timers()
        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.update_source_image_bitmap()
        self.update_result_image_bitmap()

    def create_timers(self):
        self.capture_timer = wx.Timer(self, wx.ID_ANY)
        self.Bind(wx.EVT_TIMER, self.update_capture_panel, id=self.capture_timer.GetId())
        self.animation_timer = wx.Timer(self, wx.ID_ANY)
        self.Bind(wx.EVT_TIMER, self.update_result_image_bitmap, id=self.animation_timer.GetId())

    def on_close(self, event: wx.Event):
        # Stop the timers
        self.animation_timer.Stop()
        self.capture_timer.Stop()

        # Destroy the windows
        self.Destroy()
        event.Skip()

    def on_start_capture(self, event: wx.Event):
        message_dialog = wx.MessageDialog(self, "", "Error!", wx.OK)
        message_dialog.ShowModal()
        message_dialog.Destroy()
        return

    def random_generate_value(self, min, max, origin_value):
        random_value = random.choice(list(range(min, max, 1))) / 2500.0
        randomized = origin_value + random_value
        if randomized > 1.0:
            randomized = 1.0
        if randomized < 0:
            randomized = 0
        return randomized

    def random_generate_pose(self):
        global is_talking
        current_pose = self.ifacialmocap_pose

        # NOTE: randomize mouth
        for blendshape_name in BLENDSHAPE_NAMES:
            if "jawOpen" in blendshape_name:
                if is_talking or is_talking_override:
                    current_pose[blendshape_name] = self.random_generate_value(-5000, 5000, abs(1 - current_pose[blendshape_name]))
                else:
                    current_pose[blendshape_name] = 0

        # NOTE: randomize head and eye bones
        #for key in [HEAD_BONE_Y, LEFT_EYE_BONE_X, LEFT_EYE_BONE_Y, LEFT_EYE_BONE_Z, RIGHT_EYE_BONE_X, RIGHT_EYE_BONE_Y]:
            #current_pose[key] = self.random_generate_value(-20, 20, current_pose[key])

        #Make her blink
        if random.random() <= 0.03:
            current_pose["eyeBlinkRight"] = 1
            current_pose["eyeBlinkLeft"] = 1
        else:
            current_pose["eyeBlinkRight"] = 0
            current_pose["eyeBlinkLeft"] = 0


        return current_pose    #print(current_pose)

    def read_ifacialmocap_pose(self):
        if not self.animation_timer.IsRunning():
            return self.ifacialmocap_pose
        self.ifacialmocap_pose =  self.random_generate_pose()
        return self.ifacialmocap_pose

    def on_erase_background(self, event: wx.Event):
        pass

    def create_animation_panel(self, parent):
        self.animation_panel = wx.Panel(parent, style=wx.RAISED_BORDER)
        self.animation_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.animation_panel.SetSizer(self.animation_panel_sizer)
        self.animation_panel.SetAutoLayout(1)

        image_size = self.poser.get_image_size()

        # Left Column (Image)
        self.animation_left_panel = wx.Panel(self.animation_panel, style=wx.SIMPLE_BORDER)
        self.animation_left_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.animation_left_panel.SetSizer(self.animation_left_panel_sizer)
        self.animation_left_panel.SetAutoLayout(1)
        self.animation_panel_sizer.Add(self.animation_left_panel, 1, wx.EXPAND)

        self.result_image_panel = wx.Panel(self.animation_left_panel, size=(image_size, image_size),
                                           style=wx.SIMPLE_BORDER)
        self.result_image_panel.Bind(wx.EVT_PAINT, self.paint_result_image_panel)
        self.result_image_panel.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase_background)
        self.result_image_panel.Bind(wx.EVT_LEFT_DOWN, self.load_image)
        self.animation_left_panel_sizer.Add(self.result_image_panel, 1, wx.EXPAND)

        separator = wx.StaticLine(self.animation_left_panel, -1, size=(256, 1))
        self.animation_left_panel_sizer.Add(separator, 0, wx.EXPAND)

        self.fps_text = wx.StaticText(self.animation_left_panel, label="")
        self.animation_left_panel_sizer.Add(self.fps_text, wx.SizerFlags().Border())


        self.animation_left_panel_sizer.Fit(self.animation_left_panel)

        # Right Column (Sliders)

        self.animation_right_panel = wx.Panel(self.animation_panel, style=wx.SIMPLE_BORDER)
        self.animation_right_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.animation_right_panel.SetSizer(self.animation_right_panel_sizer)
        self.animation_right_panel.SetAutoLayout(1)
        self.animation_panel_sizer.Add(self.animation_right_panel, 1, wx.EXPAND)

        separator = wx.StaticLine(self.animation_right_panel, -1, size=(256, 5))
        self.animation_right_panel_sizer.Add(separator, 0, wx.EXPAND)

        background_text = wx.StaticText(self.animation_right_panel, label="--- Background ---", style=wx.ALIGN_CENTER)
        self.animation_right_panel_sizer.Add(background_text, 0, wx.EXPAND)

        self.output_background_choice = wx.Choice(
            self.animation_right_panel,
            choices=[
                "TRANSPARENT",
                "GREEN",
                "BLUE",
                "BLACK",
                "WHITE",
                "LOADED",
                "CUSTOM"
            ]
        )
        self.output_background_choice.SetSelection(0)
        self.animation_right_panel_sizer.Add(self.output_background_choice, 0, wx.EXPAND)


        #self.pose_converter.init_pose_converter_panel(self.animation_panel) # this changes sliders to breathing on

        #sliders go here


        blendshape_groups = {
            'Eyes': ['eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookDownLeft', 'eyeLookUpLeft', 'eyeWideLeft', 'eyeWideRight'],
            'Mouth': ['mouthFrownLeft'],
            'Cheek': ['cheekSquintLeft', 'cheekSquintRight', 'cheekPuff'],
            'Brow': ['browDownLeft', 'browOuterUpLeft', 'browDownRight', 'browOuterUpRight', 'browInnerUp'],
            'Eyelash': ['mouthSmileLeft'],
            'Nose': ['noseSneerLeft', 'noseSneerRight'],
            'Misc': ['tongueOut']
        }

        for group_name, variables in blendshape_groups.items():
            collapsible_pane = wx.CollapsiblePane(self.animation_right_panel, label=group_name, style=wx.CP_DEFAULT_STYLE | wx.CP_NO_TLW_RESIZE)
            collapsible_pane.Bind(wx.EVT_COLLAPSIBLEPANE_CHANGED, self.on_pane_changed)
            self.animation_right_panel_sizer.Add(collapsible_pane, 0, wx.EXPAND)
            pane_sizer = wx.BoxSizer(wx.VERTICAL)
            collapsible_pane.GetPane().SetSizer(pane_sizer)

            for variable in variables:
                variable_label = wx.StaticText(collapsible_pane.GetPane(), label=variable)

                # Multiply min and max values by 100 for the slider
                slider = wx.Slider(
                    collapsible_pane.GetPane(),
                    value=0,
                    minValue=0,
                    maxValue=100,
                    size=(150, -1),  # Set the width to 150 and height to default
                    style=wx.SL_HORIZONTAL | wx.SL_LABELS
                )

                slider.SetName(variable)
                slider.Bind(wx.EVT_SLIDER, self.on_slider_change)
                self.sliders[slider.GetId()] = slider

                pane_sizer.Add(variable_label, 0, wx.ALIGN_CENTER | wx.ALL, 5)
                pane_sizer.Add(slider, 0, wx.EXPAND)






        self.animation_right_panel_sizer.Fit(self.animation_right_panel)
        self.animation_panel_sizer.Fit(self.animation_panel)

    def on_pane_changed(self, event):
        # Update the layout when a collapsible pane is expanded or collapsed
        self.animation_right_panel.Layout()

    def on_slider_change(self, event):
        slider = event.GetEventObject()
        value = slider.GetValue() / 100.0  # Divide by 100 to get the actual float value
        #print(value)
        slider_name = slider.GetName()
        self.ifacialmocap_pose[slider_name] = value

    def create_ui(self):
        #MAke the UI Elements
        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.main_sizer)
        self.SetAutoLayout(1)

        self.capture_pose_lock = threading.Lock()

        #Main panel with JPS
        self.create_animation_panel(self)
        self.main_sizer.Add(self.animation_panel, wx.SizerFlags(0).Expand().Border(wx.ALL, 5))

    def update_capture_panel(self, event: wx.Event):
        data = self.ifacialmocap_pose
        for rotation_name in ROTATION_NAMES:
            value = data[rotation_name]

    @staticmethod
    def convert_to_100(x):
        return int(max(0.0, min(1.0, x)) * 100)

    def paint_source_image_panel(self, event: wx.Event):
        wx.BufferedPaintDC(self.source_image_panel, self.source_image_bitmap)

    def update_source_image_bitmap(self):
        dc = wx.MemoryDC()
        dc.SelectObject(self.source_image_bitmap)
        if self.wx_source_image is None:
            self.draw_nothing_yet_string(dc)
        else:
            dc.Clear()
            dc.DrawBitmap(self.wx_source_image, 0, 0, True)
        del dc

    def draw_nothing_yet_string(self, dc):
        dc.Clear()
        font = wx.Font(wx.FontInfo(14).Family(wx.FONTFAMILY_SWISS))
        dc.SetFont(font)
        w, h = dc.GetTextExtent("Nothing yet!")
        dc.DrawText("Nothing yet!", (self.poser.get_image_size() - w) // 2, (self.poser.get_image_size() - h) // 2)

    def paint_result_image_panel(self, event: wx.Event):
        wx.BufferedPaintDC(self.result_image_panel, self.result_image_bitmap)

    def update_result_image_bitmap(self, event: Optional[wx.Event] = None):

        global global_result_image  # Declare global_source_image as a global variable
        global global_reload

        if global_reload is not None:
            #print("Global Reload the Image")
            MainFrame.load_image(self, event=None, file_path=None)  # call load_image function here
            return



        ifacialmocap_pose = self.read_ifacialmocap_pose()
        current_pose = self.pose_converter.convert(ifacialmocap_pose)
        if self.last_pose is not None and self.last_pose == current_pose:
            return
        self.last_pose = current_pose

        if self.torch_source_image is None:
            dc = wx.MemoryDC()
            dc.SelectObject(self.result_image_bitmap)
            self.draw_nothing_yet_string(dc)
            del dc
            return

        pose = torch.tensor(current_pose, device=self.device, dtype=self.poser.get_dtype())


        with torch.no_grad():
            output_image = self.poser.pose(self.torch_source_image, pose)[0].float()
            output_image = convert_linear_to_srgb((output_image + 1.0) / 2.0)

            background_choice = self.output_background_choice.GetSelection()
            if background_choice == 6:  # Custom background
                self.image_load_counter += 1  # Increment the counter
                if self.image_load_counter <= 1:  # Only open the file dialog if the counter is 5 or less
                    file_dialog = wx.FileDialog(self, "Choose a background image", "", "", "*.png", wx.FD_OPEN)
                    if file_dialog.ShowModal() == wx.ID_OK:
                        background_image_path = file_dialog.GetPath()
                            # Load the image and convert it to a torch tensor
                        pil_image = Image.open(background_image_path).convert("RGBA")
                        tensor_image = transforms.ToTensor()(pil_image).to(self.device)
                            # Resize the image to match the output image size
                        tensor_image = F.interpolate(tensor_image.unsqueeze(0), size=output_image.shape[1:], mode="bilinear").squeeze(0)
                        self.custom_background_image = tensor_image  # Store the custom background image
                        self.output_background_choice.SetSelection(5)
                    else:
                            # If the user cancelled the dialog or didn't choose a file, reset the choice to "TRANSPARENT"
                        self.output_background_choice.SetSelection(5)
                else:
                        # Use the stored custom background image
                    output_image = self.blend_with_background(output_image, self.custom_background_image)


            else:  # Predefined colors
                self.image_load_counter = 0
                if background_choice == 0:  # Transparent
                    pass
                elif background_choice == 1:  # Green
                    background = torch.zeros(4, output_image.shape[1], output_image.shape[2], device=self.device)
                    background[3, :, :] = 1.0  # set alpha to 1.0
                    background[1, :, :] = 1.0
                    output_image = self.blend_with_background(output_image, background)
                elif background_choice == 2:  # Blue
                    background = torch.zeros(4, output_image.shape[1], output_image.shape[2], device=self.device)
                    background[3, :, :] = 1.0  # set alpha to 1.0
                    background[2, :, :] = 1.0
                    output_image = self.blend_with_background(output_image, background)
                elif background_choice == 3:  # Black
                    background = torch.zeros(4, output_image.shape[1], output_image.shape[2], device=self.device)
                    background[3, :, :] = 1.0  # set alpha to 1.0
                    output_image = self.blend_with_background(output_image, background)
                elif background_choice == 4:   # White
                    background = torch.zeros(4, output_image.shape[1], output_image.shape[2], device=self.device)
                    background[3, :, :] = 1.0  # set alpha to 1.0
                    background[0:3, :, :] = 1.0
                    output_image = self.blend_with_background(output_image, background)
                elif background_choice == 5:  # Saved Image
                    output_image = self.blend_with_background(output_image, self.custom_background_image)
                else:
                    pass



            c, h, w = output_image.shape
            output_image = (255.0 * torch.transpose(output_image.reshape(c, h * w), 0, 1)).reshape(h, w, c).byte()


        numpy_image = output_image.detach().cpu().numpy()
        wx_image = wx.ImageFromBuffer(numpy_image.shape[0],
                                      numpy_image.shape[1],
                                      numpy_image[:, :, 0:3].tobytes(),
                                      numpy_image[:, :, 3].tobytes())
        wx_bitmap = wx_image.ConvertToBitmap()

        dc = wx.MemoryDC()
        dc.SelectObject(self.result_image_bitmap)
        dc.Clear()
        dc.DrawBitmap(wx_bitmap,
                      (self.poser.get_image_size() - numpy_image.shape[0]) // 2,
                      (self.poser.get_image_size() - numpy_image.shape[1]) // 2, True)


        # Assuming numpy_image has shape (height, width, 4) and the channels are in RGB order
        # Convert color channels from RGB to BGR and keep alpha channel
        numpy_image_bgra = numpy_image[:, :, [2, 1, 0, 3]]
        #cv2.imwrite('test2.png', numpy_image_bgra)

        global_result_image = numpy_image_bgra


        del dc

        time_now = time.time_ns()
        if self.last_update_time is not None:
            elapsed_time = time_now - self.last_update_time
            fps = 1.0 / (elapsed_time / 10**9)
            if self.torch_source_image is not None:
                self.fps_statistics.add_fps(fps)
            self.fps_text.SetLabelText("FPS = %0.2f" % self.fps_statistics.get_average_fps())
        self.last_update_time = time_now

        self.Refresh()

    def blend_with_background(self, numpy_image, background):
        if background is not None:
            alpha = numpy_image[3:4, :, :]
            color = numpy_image[0:3, :, :]
            new_color = color * alpha + (1.0 - alpha) * background[0:3, :, :]
            return torch.cat([new_color, background[3:4, :, :]], dim=0)
        else:
            return numpy_image

    def resize_image(image, size=(512, 512)):
        image.thumbnail(size, Image.LANCZOS) # Step 1: Resize the image to maintain the aspect ratio with the larger dimension being 512 pixels
        new_image = Image.new("RGBA", size)  # Step 2: Create a new image of size 512x512 with transparency
        new_image.paste(image, ((size[0] - image.size[0]) // 2,
                                (size[1] - image.size[1]) // 2))   # Step 3: Paste the resized image into the new image, centered
        return new_image

    def load_image(self, event: wx.Event, file_path=None):

        global global_source_image  # Declare global_source_image as a global variable
        global global_source_image_path  # Declare global_source_image as a global variable
        global global_reload

        if global_reload is not None:
            file_path = "global_reload"

        #if file_path is None and global_reload is not None:

        if file_path is None:
            dir_name = "data/images"
            file_dialog = wx.FileDialog(self, "Choose an image", dir_name, "", "*.png", wx.FD_OPEN)
            if file_dialog.ShowModal() == wx.ID_OK:
                file_path = os.path.join(file_dialog.GetDirectory(), file_dialog.GetFilename())
            file_dialog.Destroy()

        if file_path:
            try:


                if file_path == "global_reload":
                    pil_image = global_reload # use global_reload directly
                    #print("Loading from Var")
                else:
                    pil_image = resize_PIL_image(
                        extract_PIL_image_from_filelike(file_path),
                        (self.poser.get_image_size(), self.poser.get_image_size()))


                w, h = pil_image.size

                if pil_image.size != (512, 512):
                    print("Resizing Char Card to work")
                    pil_image = MainFrame.resize_image(pil_image)

                w, h = pil_image.size





                if pil_image.mode != 'RGBA':
                    self.source_image_string = "Image must have alpha channel!"
                    self.wx_source_image = None
                    self.torch_source_image = None
                else:
                    self.wx_source_image = wx.Bitmap.FromBufferRGBA(w, h, pil_image.convert("RGBA").tobytes())
                    self.torch_source_image = extract_pytorch_image_from_PIL_image(pil_image) \
                        .to(self.device).to(self.poser.get_dtype())

                global_source_image = self.torch_source_image  # Set global_source_image as a global variable

                global_source_image_path = image_path = os.path.join(file_path) #set file path

                self.update_source_image_bitmap()



            except Exception as error:
                print("Error:")
                print(error)
                #message_dialog = wx.MessageDialog(self, "Could not load image " + file_path, "Poser", wx.OK)
                #message_dialog.ShowModal()
                #message_dialog.Destroy()
        global_reload = None #reset the globe load
        #print("Reseting Load Variable")
        self.Refresh()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='uWu Waifu')
    parser.add_argument(
        '--model',
        type=str,
        required=False,
        default='separable_float',
        choices=['standard_float', 'separable_float', 'standard_half', 'separable_half'],
        help='The model to use.'
    )
    parser.add_argument('--char', type=str, required=False, help='The path to the character image.')
    parser.add_argument(
        '--device',
        type=str,
        required=False,
        default='cuda',
        choices=['cpu', 'cuda'],
        help='The device to use for PyTorch ("cuda" for GPU, "cpu" for CPU).'
    )

    args = parser.parse_args()
    # Add the line below to pass the 'args' object to the launch_gui() function
    launch_gui(device=args.device, model=args.model)
