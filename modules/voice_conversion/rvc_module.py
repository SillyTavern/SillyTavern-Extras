"""
RVC module for SillyTavern Extras

Authors:
    - Tony Ribeiro (https://github.com/Tony-sama)

Models are saved into local data folder: "data/models/hubert" and "data/models/rvc"

References:
    - Code adapted from:
        - RVC-webui: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
        - Audio-webui: https://github.com/gitmylo/audio-webui
"""
from flask import abort, request, send_file, jsonify
import json
import wave
import modules.voice_conversion.rvc.rvc as rvc
from scipy.io import wavfile
import os
import io
import logging

DEBUG_PREFIX = "<RVC module>"

def rvc_process_audio():
    """
    Process request audio file with the loaded RVC model
    """
    try:
        file = request.files.get('AudioFile')
        print(DEBUG_PREFIX, "received:", file)
        
        # Create new instances of io.BytesIO() for each request
        input_audio_path = io.BytesIO()
        output_audio_path = io.BytesIO()
        
        file.save(input_audio_path)
        input_audio_path.seek(0)
        
        parameters = json.loads(request.form["json"])
        
        print(DEBUG_PREFIX, "Received audio conversion request with model", parameters["modelName"])
        model_path = "data/models/rvc/" + parameters["modelName"] + ".pth"
        print(DEBUG_PREFIX, "loading", model_path)
        rvc.load_rvc(model_path)

        file_index_path = "data/models/rvc/" + parameters["modelName"] + ".index"
        print(DEBUG_PREFIX, "Check for index file", file_index_path)
        if not os.path.isfile(file_index_path):
            file_index_path = ""
            print(DEBUG_PREFIX, "no index file found, proceeding without index")
        
        info, (tgt_sr, wav_opt) = rvc.vc_single(
            sid=0,
            input_audio_path=input_audio_path,
            f0_up_key=parameters["pitchOffset"],
            f0_file=None,
            f0_method=parameters["pitchExtraction"],
            file_index=file_index_path,
            file_index2="",
            index_rate=parameters["indexRate"],
            filter_radius=parameters["filterRadius"] // 2 * 2 + 1, # Need to be odd number
            resample_sr=0,
            rms_mix_rate=1,
            protect=parameters["protect"],
            crepe_hop_length=128)
        
        #print(info) # DBG

        #out_path = os.path.join("data/", "rvc_output.wav")
        wavfile.write(output_audio_path, tgt_sr, wav_opt)
        output_audio_path.seek(0)  # Reset cursor position
        
        print(DEBUG_PREFIX, "Audio converted using RVC model:", rvc.rvc_model_name)
        
        # Return the output_audio_path object as a response
        response = send_file(output_audio_path, mimetype="audio/x-wav")
        return response

    except Exception as e:
        print(e)
        abort(500, DEBUG_PREFIX + " Exception occurs while processing audio")