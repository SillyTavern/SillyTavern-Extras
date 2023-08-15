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
import json
import os
import io
import shutil

from flask import abort, request, send_file, jsonify

from TTS.api import TTS
from TTS.utils.manage import ModelManager


DEBUG_PREFIX = "<Coqui-TTS module>"
audio_buffer = io.BytesIO()

gpu_mode = False
is_downloading = False

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
    global gpu_mode
    global is_downloading

    try:
        model_installed = False
        request_json = request.get_json()
        model_id = request_json["model_id"]
        action = request_json["action"]

        print(DEBUG_PREFIX,"Received request",action,"for model",model_id)
        
        if (is_downloading):
            print(DEBUG_PREFIX,"Rejected, already downloading a model")
            return json.dumps({"status":"downloading"})
        
        coqui_models_folder = ModelManager().output_prefix  # models location
        installed_models = os.listdir(coqui_models_folder)
        model_path = None

        print(DEBUG_PREFIX,"Found",len(installed_models),"models in",coqui_models_folder)

        for i in installed_models:
            if model_id == i.replace("--","/"):
                model_installed = True
                model_path = os.path.join(coqui_models_folder,i)

        if model_installed:
            print(DEBUG_PREFIX,"model found:", model_id)
        else:
            print(DEBUG_PREFIX,"model not found")

        if action == "download":
            if model_installed:
                abort(500, DEBUG_PREFIX + "Bad request, model already installed.")

            is_downloading = True
            TTS(model_name=model_id, progress_bar=True, gpu=gpu_mode)
            is_downloading = False

        if action == "repare":
            if not model_installed:
                abort(500, DEBUG_PREFIX + " bad request: requesting repare of model not installed")


            print(DEBUG_PREFIX,"Deleting corrupted model folder:",model_path)
            shutil.rmtree(model_path, ignore_errors=True)

            is_downloading = True
            TTS(model_name=model_id, progress_bar=True, gpu=gpu_mode)
            is_downloading = False

        response = json.dumps({"status":"done"})
        return response

    except Exception as e:
        is_downloading = False
        print(e)
        abort(500, DEBUG_PREFIX + " Exception occurs while trying to search for installed model")

def coqui_get_local_models():
    """
    Return user local models list in the following format: [language][dataset][name] = TTS_string_id
    """

    abort(500, DEBUG_PREFIX + " Not implemented yet")


def coqui_generate_tts():
    """
    Process request text with the loaded RVC model
        - expected request: {
            "text": text,
            "model_id": voiceId,
            "language_id": language,
            "speaker_id": speaker
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
    global gpu_mode
    global is_downloading
    global audio_buffer

    try:
        request_json = request.get_json()
        #print(request_json)

        print(DEBUG_PREFIX,"Received TTS request for ", request_json)

        if (is_downloading):
            print(DEBUG_PREFIX,"Rejected, currently downloading a model, cannot perform TTS")
            abort(500, DEBUG_PREFIX + " Requested TTS while downloading a model")
        
        text = request_json["text"]
        model_name = request_json["model_id"]
        language_id = None
        speaker_id =  None

        if request_json["language_id"] != "none":
            language_id = request_json["language_id"]
        
        if request_json["speaker_id"] != "none":
            speaker_id = request_json["speaker_id"]

        print(DEBUG_PREFIX,"Loading tts \n- model", model_name, "\n - speaker_id: ",speaker_id,"\n - language_id: ",language_id, "\n - using",("GPU" if gpu_mode else "CPU"))

        is_downloading = True
        tts = TTS(model_name=model_name, progress_bar=True, gpu=gpu_mode)
        is_downloading = False

        if tts.is_multi_lingual:
            if language_id is None:
                abort(400, DEBUG_PREFIX + " Requested model "+model_name+" is multi-lingual but no language id provided")
            language_id = tts.languages[int(language_id)]

        if tts.is_multi_speaker:
            if speaker_id is None:
                abort(400, DEBUG_PREFIX + " Requested model "+model_name+" is multi-speaker but no speaker id provided")
            speaker_id =tts.speakers[int(speaker_id)]

        tts.tts_to_file(text=text, file_path=audio_buffer, speaker=speaker_id, language=language_id)

        print(DEBUG_PREFIX, "Success, saved to",audio_buffer)
        
        # Return the output_audio_path object as a response
        response = send_file(audio_buffer, mimetype="audio/x-wav")
        audio_buffer = io.BytesIO()
        
        return response

    except Exception as e:
        print(e)
        abort(500, DEBUG_PREFIX + " Exception occurs while trying to process request "+str(request_json))