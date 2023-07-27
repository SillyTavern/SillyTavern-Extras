"""
Speech-to-text module based on Vosk for SillyTavern Extras
    - Vosk website: https://alphacephei.com/vosk/
    - Vosk api: https://github.com/alphacep/vosk-api

Authors:
    - Tony Ribeiro (https://github.com/Tony-sama)

Models are saved into user cache folder, example: C:/Users/toto/.cache/vosk

References:
    - Code adapted from: https://github.com/alphacep/vosk-api/blob/master/python/example/test_simple.py
"""
from flask import jsonify, abort, request

import wave
from vosk import Model, KaldiRecognizer, SetLogLevel
import soundfile

DEBUG_PREFIX = "<stt vosk module>"
RECORDING_FILE_PATH = "stt_test.wav"

model = None

SetLogLevel(-1)

def load_model(file_path=None):
    """
    Load given vosk model from file or default to en-us model.
    Download model to user cache folder, example: C:/Users/toto/.cache/vosk
    """

    if file_path is None:
        return Model(lang="en-us")
    else:
        return Model(file_path)

def process_audio():
    """
    Transcript request audio file to text using Whisper
    """

    if model is None:
        print(DEBUG_PREFIX,"Vosk model not initialized yet.")
        return ""

    try:    
        file = request.files.get('AudioFile')
        file.save(RECORDING_FILE_PATH)

        # Read and rewrite the file with soundfile
        data, samplerate = soundfile.read(RECORDING_FILE_PATH)
        soundfile.write(RECORDING_FILE_PATH, data, samplerate)

        wf = wave.open(RECORDING_FILE_PATH, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print("Audio file must be WAV format mono PCM.")
            abort(500, DEBUG_PREFIX+" Audio file must be WAV format mono PCM.")

        rec = KaldiRecognizer(model, wf.getframerate())
        #rec.SetWords(True)
        #rec.SetPartialWords(True)

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                break
        
        transcript = rec.Result()[14:-3]
        print(DEBUG_PREFIX, "Transcripted from request audio file:", transcript)
        return jsonify({"transcript": transcript})

    except Exception as e: # No exception observed during test but we never know
        print(e)
        abort(500, DEBUG_PREFIX+" Exception occurs while processing audio")