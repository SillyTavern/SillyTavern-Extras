import io
import asyncio
import json
import os
import torch
import gc
from pathlib import Path
import TTS
from TTS.api import TTS
from TTS.utils.manage import ModelManager

from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark

from TTS.tts.configs.tortoise_config import TortoiseConfig
from TTS.tts.models.tortoise import Tortoise

from flask import send_file

tts = None
type = None
multlang = "None"
multspeak = "None"
loadedModel = "None"
spkdirectory = ""
multspeakjson = ""
_gpu = False

def setGPU(flag):
    global _gpu
    _gpu = flag
    return

def model_type(_config_path):
    try:
        with open(_config_path, 'r') as config_file:
            config_data = json.load(config_file)

            # Search for the key "model" and print its value
            if "model" in config_data:
                model_value = config_data["model"]
                return model_value
            else:
                print("ERR: The key 'model' is not present in the config file.")
    except FileNotFoundError:
        print("Config file not found.")
    except json.JSONDecodeError:
        pass
        #print("Invalid JSON format in the config file.")
    except Exception as e:
        pass
        #print("An error occurred:", str(e))

def load_model(_model, _gpu, _progress):
    global tts
    global type
    global loadedModel
    global multlang
    global multspeak
    
    status = None

    print("GPU is set to: ", _gpu)

    _model_directory, _file = os.path.split(_model)

    if _model_directory == "": #make it assign vars correctly if no filename provioded
        _model_directory = _file
        _file = None

    if _model is None:
        status = "ERROR: Invalid model name or path."
    else:
        try:
            if _gpu == True: #Reclaim memory
                    del tts
                    try:
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
        except Exception as e:
            status = str(e)

        _target_directory = ModelManager().output_prefix  # models location
        _modified_speaker_id = _model_directory.replace("\\", "--")

        if _file != None:
            _model_path = os.path.join(_target_directory, _modified_speaker_id, _file)
        else:
            _model_path = os.path.join(_target_directory, _modified_speaker_id)

        _config_path = os.path.join(_target_directory, _modified_speaker_id, "config.json")

        if model_type(_config_path) == "tortoise":
            print("Loading Tortoise...")
            print("_model", _model)
            print("Tortoise not supported at the moment im tired of working on this")
            #_loadtortoisemodel = _model_directory.replace("--", "/")
            #print("_loadtortoisemodel", _loadtortoisemodel)

            #config = TortoiseConfig()
            #model = Tortoise.init_from_config(config)
            #model.load_checkpoint(config, checkpoint_dir="C:/Users/jsviv/AppData/Local/tts/tts_models--en--multi-dataset--tortoise-v2", eval=False)

            #tts = TTS(_loadtortoisemodel)
            #tts = TTS(model_name="tts_models/en/multi-dataset/tortoise-v2", progress_bar=True, gpu=True)

            #loadedModel = _model
            #print("loaded model", loadedModel)

        if model_type(_config_path) == "bark":
            print("Loading Bark...")
            _loadbarkmodel = _model_directory.replace("--", "/")
            tts = TTS(_loadbarkmodel, gpu=_gpu)
            loadedModel = _model

        _loadertypes = ["tortoise", "bark"]
        if model_type(_config_path) not in _loadertypes:
            try:
                print("Loading ", model_type(_config_path))
                print("Load Line:", _model_path, _progress, _gpu)
                tts = TTS(model_path=_model_path, config_path=_config_path, progress_bar=_progress, gpu=_gpu)
                status = "Loaded"
                loadedModel = _model
            except Exception as e:
                print("An exception occurred while loading VITS:", str(e))
                print("Continuing with other parts of the code...")
        else:
            pass

        type = model_type(_config_path)
        print("Type: ", type)

    if status is None:
        status = "Unknown error occurred"
    if type is None:
        type = "Unknown"

    return status

def is_multi_speaker_model():
    global multspeak
    global type
    global spkdirectory
    global multspeakjson
    global tts

    if tts is None:
        multspeak = "None"
        return multspeak
    try:


        if type == "bark":
            _target_directory = ModelManager().output_prefix
            # Convert _target_directory to a string and remove the trailing backslash if present
            _target_directory_str = str(_target_directory)
            if _target_directory_str.endswith("\\"):
                _target_directory_str = _target_directory_str[:-1]

            spkdirectory = os.path.join(_target_directory_str, "bark_v0", "speakers")

            subfolder_names = [folder for folder in os.listdir(spkdirectory) if os.path.isdir(os.path.join(spkdirectory, folder))]

            subfolder_names.insert(0, "random") # Add "Random" as the first element in the subfolder_names list

            unique_names = list(dict.fromkeys(subfolder_names))
            multspeak = json.dumps({index: name for index, name in enumerate(unique_names)})
            #print(multspeak)
        else:

            value = tts.speakers
            if value is not None:
                unique_speakers = list(dict.fromkeys(value))
                speaker_dict = {index: value for index, value in enumerate(unique_speakers)}
                multspeak = json.dumps(speaker_dict)
                #print(multspeak)
            else:
                multspeak = "None"


    except Exception as e:
        print("Error:", e)
        multspeak = "None"
    multspeakjson = multspeak
    return multspeak #return name and ID in named json

def is_multi_lang_model():
    global multlang
    global tts
    if tts is None:
        multlang = "None"
        return multlang
    try:
        value = tts.languages
        if value is not None:
            unique_lang = list(dict.fromkeys(value))# Remove duplicate values and preserve the order
            lang_dict = {index: value for index, value in enumerate(unique_lang)} # Create a dictionary with indices as keys and values as keys
            multlang = json.dumps(lang_dict)  # Convert the dictionary to JSON format
            #print(multlang)
        else:
            multlang = "None"
    except Exception as e:
        print("Error:", e)
        multlang = "None"

    return multlang

def get_coqui_models(): #DROPDOWN MODELS
    manager = ModelManager()
    model_folder = manager.output_prefix

    cwd = os.path.dirname(os.path.realpath(__file__))
    target_directory = model_folder

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    os.chdir(target_directory)
    folder_list = [
        folder for folder in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, folder)) and "--" in folder and "vocoder" not in folder.lower() and "voice_conversion_models" not in folder.lower()
    ]


    file_paths = []

    for folder in folder_list:
        _config_path = os.path.join(target_directory, folder, "config.json")
        if model_type(_config_path) == "bark" or model_type(_config_path) == "tortoise":
            file_paths.append(str(Path(folder, '')))
        else:
            for file in os.listdir(os.path.join(target_directory, folder)):
                if file.endswith(('.pt', '.tar', '.pkl', '.pth')) and not file.startswith('.'):
                    file_paths.append(str(Path(folder, file)))

    merged_json = json.dumps(file_paths)

    os.chdir(cwd)
    return merged_json

def coqui_checkmap():
    manager = ModelManager()
    model_folder = manager.output_prefix

    cwd = os.path.dirname(os.path.realpath(__file__))
    target_directory = model_folder

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    os.chdir(target_directory)
    folder_list = [
        folder for folder in os.listdir() if os.path.isdir(os.path.join(target_directory, folder)) and "--" in folder and "vocoder" not in folder.lower()
    ]

    file_paths = []

    for folder in folder_list:
        _config_path = os.path.join(target_directory, folder, "config.json")
        if model_type(_config_path) == "bark" or model_type(_config_path) == "tortoise":
            file_paths.append(str(Path(folder, '')))
        else:
            for file in os.listdir(os.path.join(target_directory, folder)):
                if file.endswith(('.pt', '.tar', '.pkl', '.pth')) and not file.startswith('.'):
                    file_paths.append(str(Path(folder, file)))

    # Convert the list into a list of dictionaries with "id" as the key
    keyed_json_list = [{"id": item} for item in file_paths]

    # Convert the list to a JSON string with indentation
    keyed_json_string = json.dumps(keyed_json_list, indent=2)

    # Replace double backslashes with single backslashes
    #keyed_json_string = keyed_json_string.replace("\\\\", "\\")

    os.chdir(cwd)

    return keyed_json_string

def get_coqui_download_models(): #Avail voices list
    formatted_list = []
    #voices_list = json.loads(get_coqui_downloaded())
    voices_list = TTS.list_models()

    for model in voices_list:
        split_model = model.split('/')
        formatted_list.append({
            "type": split_model[0], #type
            "lang": split_model[1], #lang
            "id-only": split_model[2], #id
            "name-only": split_model[3], #name
            "id": split_model[0] + '/' + split_model[1] + "/" + split_model[2] + "/" + split_model[3], #combined id and name tts_models/bn/custom/vits-male
        })

    json_data = json.dumps(formatted_list, indent=4)
    return json_data

def coqui_modeldownload(_modeldownload): #Avail voices function
    global _gpu
    print(_modeldownload)
    try:
        tts = TTS(model_name=_modeldownload, progress_bar=True, gpu=_gpu)
        status = "True"
    except:
        status = "False"
    return status

def coqui_tts(text, speaker_id, mspker_id, style_wav, language_id):
    global type
    global multlang
    global multspeak
    global loadedModel
    global spkdirectory
    global multspeakjson
    global _gpu

    try:
        # Splitting the string to get speaker_id and the rest
        parts = speaker_id.split("[", 1)
        speaker_id = parts[0]
        remainder = parts[1].rstrip("]")
        variables = remainder.split("][")
        # Converting to integers with default values of 0 if conversion fails
        mspker_id = int(variables[0]) if variables[0].isdigit() else 0
        language_id = int(variables[1]) if variables[1].isdigit() else 0
        # multspeak = mspker_id # might break previews
        multlang = language_id
    except Exception:
        pass
        #print("exception 1")

    print("mspker_id: ", mspker_id)
    print("language_id: ", language_id)



    try: #see is values passed in URL
        if language_id is not None:
            float(language_id)
            multlang = float(language_id)
        else:
            pass
    except ValueError:
        pass


    try:
        if mspker_id is not None:
            float(mspker_id)
            multspeak = float(mspker_id)
        else:
            pass
    except ValueError:
        pass


    if loadedModel != speaker_id:
        print("MODEL NOT LOADED!!! Loading... ", loadedModel, speaker_id)
        print("Loading :", speaker_id, "GPU is: ", _gpu)

        load_model(speaker_id, _gpu, True) 


    audio_buffer = io.BytesIO()

    if not isinstance(multspeak, (int, float)) and not isinstance(multlang, (int, float)): #if not a number
        print("Single Model")
        tts.tts_to_file(text, file_path=audio_buffer)
    elif isinstance(multspeak, (int, float)) and not isinstance(multlang, (int, float)):
        print("speaker only")
        if type == "bark" or type == "tortoise":
            try:
                if multspeakjson == "": #failing because multispeakjson not loaded
                    parsed_multspeak = json.loads(is_multi_speaker_model())
                else:
                    parsed_multspeak = json.loads(multspeakjson)

                value_at_key = parsed_multspeak.get(str(mspker_id))
                #print(value_at_key)
                # ♪ In the jungle, the mighty jungle, the lion barks tonight ♪
                #I have a silky smooth voice, and today I will tell you about the exercise regimen of the common sloth.
                if value_at_key == "random":
                    tts.tts_to_file(text, file_path=audio_buffer)
                else:
                    print("using speaker ", value_at_key)
                    tts.tts_to_file(text, file_path=audio_buffer, voice_dir=spkdirectory, speaker=value_at_key)
            except Exception as e:
                print("An error occurred:", str(e))
        else:
            tts.tts_to_file(text, speaker=tts.speakers[int(mspker_id)], file_path=audio_buffer)
    elif not isinstance(multspeak, (int, float)) and isinstance(multlang, (int, float)):
        print("lang only")
        tts.tts_to_file(text, language=tts.languages[int(language_id)], file_path=audio_buffer)
    else:
        print("spk and lang")
        tts.tts_to_file(text, speaker=tts.speakers[int(mspker_id)], language=tts.languages[int(language_id)], file_path=audio_buffer)

    audio_buffer.seek(0)
    response = send_file(audio_buffer, mimetype="audio/wav")

    #reset for next dynamic tts
    multlang = None
    multspeak = None
    return response
