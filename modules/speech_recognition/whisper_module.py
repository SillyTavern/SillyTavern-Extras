"""
Speech-to-text module based on Whisper for SillyTavern Extras
    - Whisper github: https://github.com/openai/whisper

Authors:
    - Tony Ribeiro (https://github.com/Tony-sama)

Models are saved into user cache folder, example: C:/Users/toto/.cache/whisper

References:
    - Code adapted from:
        - whisper github: https://github.com/openai/whisper
        - oobabooga text-generation-webui github: https://github.com/oobabooga/text-generation-webui
"""
from flask import jsonify, abort, request

import whisper

DEBUG_PREFIX = "<stt whisper module>"
RECORDING_FILE_PATH = "stt_test.wav"

model = None

def load_model(file_path=None):
    """
    Load given vosk model from file or default to en-us model.
    Download model to user cache folder, example: C:/Users/toto/.cache/vosk
    """

    if file_path is None:
        return whisper.load_model("base.en")
    else:
        return whisper.load_model(file_path)
    
def process_audio():
    """
    Transcript request audio file to text using Whisper
    """

    if model is None:
        print(DEBUG_PREFIX,"Whisper model not initialized yet.")
        return ""

    try:    
        file = request.files.get('AudioFile')
        file.save(RECORDING_FILE_PATH)
          
        result = model.transcribe(RECORDING_FILE_PATH)
        transcript = result["text"]
        print(DEBUG_PREFIX, "Transcripted from audio file (whisper):", transcript)

        return jsonify({"transcript": transcript})

    except Exception as e: # No exception observed during test but we never know
        print(e)
        abort(500, DEBUG_PREFIX+" Exception occurs while processing audio")