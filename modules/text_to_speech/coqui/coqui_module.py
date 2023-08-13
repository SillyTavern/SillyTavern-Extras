"""
Coqui module for SillyTavern Extras

Authors:
    - Pyrater (https://github.com/pyrater)
    - Tony Ribeiro (https://github.com/Tony-sama)

Models are saved into user cache folder: "C:/Users/<username>/AppData/Local/tts"

References:
    - Code adapted from:
        - Coqui TTS https://tts.readthedocs.io/en/latest/
        - Audio-webui: https://github.com/gitmylo/audio-webui
"""
from flask import abort, request, send_file, jsonify
import json
import os
import io
from TTS.api import TTS
from TTS.utils.manage import ModelManager


DEBUG_PREFIX = "<Coqui-TTS module>"
OUTPUT_PATH = "data/tmp/coqui_output.wav"

gpu = False


def coqui_get_api_models():
    """
    Return supported models in the following format: [language][dataset][name] = TTS_string_id
    Example:
    {
        "en": {
            "ljspeech": {
                "tacotron2-DDC": "tts_models/en/ljspeech/tacotron2-DDC",
                "glow-tts": "tts_models/en/ljspeech/glow-tts",
                "vits": "tts_models/en/ljspeech/vits"
            },
            "vctk": {
                "vits": "tts_models/en/vctk/vits"
            }
        },
        "ja": {
            "kokoro": {
                "tacotron2-DDC": "tts_models/ja/kokoro/tacotron2-DDC"
            }
        }
    }
    """

    try:
        models = {}

        model_selection = ["your_tts", "vits", "jenny", "glow-tts", "tacotron2-DDC"]
        language_selection = ["multilingual", "en", "fr", "es", "ja"]

        for i in TTS.list_models():
            tokens = i.split("/")
            language = tokens[1]
            dataset = tokens[2]
            name = tokens[3]

            if language not in language_selection:
                continue

            if name not in model_selection:
                continue

            if language not in models:
                models[language] = {}

            if dataset not in models[language]:
                models[language][dataset] = {}
            
            models[language][dataset][name] = i

        response = json.dumps(models)
        return response
    
    except Exception as e:
        print(e)
        abort(500, DEBUG_PREFIX + " Exception occurs while trying to get list of TTS available")


def coqui_check_model_state():
    """
        Check if the requested model is installed on the server machine
    """
    try:
        model_state = "absent"
        request_json = request.get_json()
        model_id = request_json["model_id"]
        
        print(DEBUG_PREFIX,"Search for model", model_id)

        coqui_models_folder = ModelManager().output_prefix  # models location
        installed_models = os.listdir(coqui_models_folder)

        model_folder_exists = False
        model_folder = None

        for i in installed_models:
            if model_id == i.replace("--","/"):
                model_folder_exists = True
                model_folder = i
                print(DEBUG_PREFIX,"Folder found:",model_folder)

        # Check failed download
        if model_folder_exists:
            content = os.listdir(os.path.join(coqui_models_folder,model_folder))
            print(DEBUG_PREFIX,"Checking content:",content)
            for i in content:
                if i == model_folder+".zip":
                    print("Corrupt installed found, model download must have failed previously")
                    model_state = "corrupted"
                    break

            if model_state != "corrupted":
                model_state = "installed"

        response = json.dumps({"model_state":model_state})
        return response

    except Exception as e:
        print(e)
        abort(500, DEBUG_PREFIX + " Exception occurs while trying to search for installed model")

def coqui_install_model():
    """
        Install requested model is installed on the server machine
    """
    try:
        model_installed = False
        request_json = request.get_json()
        model_id = request_json["model_id"]
        
        print(DEBUG_PREFIX,"Search for model", model_id)

        coqui_models_folder = ModelManager().output_prefix  # models location
        installed_models = os.listdir(coqui_models_folder)

        for i in installed_models:
            if model_id == i.replace("--","/"):
                model_installed = True

        response = json.dumps({"model_installed":model_installed})
        return response

    except Exception as e:
        print(e)
        abort(500, DEBUG_PREFIX + " Exception occurs while trying to search for installed model")

def coqui_get_local_models():
    """
    Return user local models list in the following format: [language][dataset][name] = TTS_string_id
    """

    abort(500, DEBUG_PREFIX + " Not implemented yet")

def coqui_get_model_settings():
    """
    Process request model and return available speakers
        - expected request: {
            model_id: string
        }
    """
    try:
        request_json = request.get_json()
        #print(request_json)

        print(DEBUG_PREFIX,"Received get_speakers request for model", request_json["model_id"])
        
        model_id = request_json["model_id"]
        model_languages = []
        model_speakers = []

        print(DEBUG_PREFIX,"Loading tts model", model_id,"\n - using", ("GPU" if gpu else "CPU"))

        tts = TTS(model_name=model_id, progress_bar=True, gpu=gpu)

        if tts.is_multi_lingual:
            model_languages = tts.languages

        if tts.is_multi_speaker:
            model_speakers = tts.speakers

        response = json.dumps({"languages":model_languages, "speakers":model_speakers})
        print(DEBUG_PREFIX,"Model settings: ", response)
        return response
    
    
    except Exception as e:
        print(e)
        abort(500, DEBUG_PREFIX + " Exception occurs while trying to get model speakers")



def coqui_process_text():
    """
    Process request text with the loaded RVC model
        - expected request: {
            "text": text,
            "model_id": voiceId,
            "language": language,
            "speaker": speaker
        }

        - model_id formats:
            - model_type/language/dataset/model_name
            - model_type/language/dataset/model_name[spearker_id]
            - model_type/language/dataset/model_name[spearker_id][language_id]
        - examples:
            - tts_models/ja/kokoro/tacotron2-DDC
            - tts_models/en/vctk/vits[0]
            - tts_models/multilingual/multi-dataset/your_tts[2][1]
    """
    global gpu

    try:
        request_json = request.get_json()
        #print(request_json)

        print(DEBUG_PREFIX,"Received TTS request for ", request_json)
        
        text = request_json["text"]
        model_name = request_json["model_id"]
        language = None
        speaker =  None

        if request_json["language"] != "none":
            language = request_json["language"]
        
        if request_json["speaker"] != "none":
            speaker = request_json["speaker"]

        print(DEBUG_PREFIX,"Loading tts model", model_name, "\n - speaker: ",speaker,"\n - language: ",language, "\n - using",("GPU" if gpu else "CPU"))

        tts = TTS(model_name=model_name, progress_bar=True, gpu=gpu)

        if tts.is_multi_speaker:
            if speaker is None:
                abort(400, DEBUG_PREFIX + " Requested model "+model_name+" is multi-speaker but no speaker provided")

        if tts.is_multi_lingual:
            if speaker is None:
                abort(400, DEBUG_PREFIX + " Requested model "+model_name+" is multi-lingual but no language provided")

        tts.tts_to_file(text=text, file_path=OUTPUT_PATH, speaker=speaker, language=language)

        print(DEBUG_PREFIX, "Success, saved to",OUTPUT_PATH)
        
        # Return the output_audio_path object as a response
        response = send_file(OUTPUT_PATH, mimetype="audio/x-wav")
        return response
    
    except Exception as e:
        print(e)
        abort(500, DEBUG_PREFIX + " Exception occurs while trying to process request "+str(request_json))