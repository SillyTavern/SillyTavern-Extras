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


DEBUG_PREFIX = "<Coqui-TTS module>"
OUTPUT_PATH = "data/tmp/coqui_output.wav"

gpu = False


def coqui_supported_models():
    """
    Return supported models in the following format: [language][dataset][name] = TTS_string_id
    Example:
    {
        "multilingual": {
            "multi-dataset": {
                "your_tts": "tts_models/multilingual/multi-dataset/your_tts"
            }
        },
        "en": {
            "ljspeech": {
                "tacotron2-DDC": "tts_models/en/ljspeech/tacotron2-DDC",
                "vits": "tts_models/en/ljspeech/vits"
            },
            "jenny": {
                "jenny": "tts_models/en/jenny/jenny"
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
            
            models[language][name+"/"+dataset] = i

        response = json.dumps(models,indent=4)
        return response
    
    except Exception as e:
        print(e)
        abort(500, DEBUG_PREFIX + " Exception occurs while trying to get list of TTS available")


def coqui_process_text():
    """
    Process request text with the loaded RVC model
    - expected request: {
        text: string,
        voiceId: string
    }

    voiceId formats:
        - model_type/language/dataset/model_name
        - model_type/language/dataset/model_name[spearker_id]
        - model_type/language/dataset/model_name[spearker_id][language_id]
    examples:
        - tts_models/ja/kokoro/tacotron2-DDC
        - tts_models/en/vctk/vits[0]
        - tts_models/multilingual/multi-dataset/your_tts[2][1]
    """
    global gpu

    try:
        request_json = request.get_json()
        #print(request_json)

        print(DEBUG_PREFIX,"Received TTS request for voiceId", request_json["voiceId"], "with text:\n",request_json["text"])
        
        text = request_json["text"]
        tokens = [i.strip("]") for i in request_json["voiceId"].split("[")]

        print(tokens)

        model_name = tokens[0]
        speaker = None
        language = None

        if len(tokens) > 1:
            speaker = tokens[1]
        if len(tokens) > 2:
            language = tokens[2]

        print(DEBUG_PREFIX,"Loading tts model", model_name, "\n - speaker: ",speaker,"\n - language: ",language, "\n - using",("GPU" if gpu else "CPU"))

        tts = TTS(model_name=model_name, progress_bar=True, gpu=gpu)

        if tts.is_multi_speaker:
            if speaker is None:
                abort(400, DEBUG_PREFIX + " Requested model "+model_name+" is multi-speaker but no speaker provided")

            speaker = tts.speakers[int(speaker)]

        if tts.is_multi_lingual:
            if speaker is None:
                abort(400, DEBUG_PREFIX + " Requested model "+model_name+" is multi-lingual but no language provided")

            language = tts.languages[int(language)]

        tts.tts_to_file(text=text, file_path=OUTPUT_PATH, speaker=speaker, language=language)

        print(DEBUG_PREFIX, "Success, saved to",OUTPUT_PATH)
        
        # Return the output_audio_path object as a response
        response = send_file(OUTPUT_PATH, mimetype="audio/x-wav")
        return response
    
    except Exception as e:
        print(e)
        abort(500, DEBUG_PREFIX + " Exception occurs while trying to process request "+str(request_json))