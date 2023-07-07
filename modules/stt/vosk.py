"""
Speech-to-text module based on Vosk for SillyTavern Extras
    - Vosk website: https://alphacephei.com/vosk/
    - Vosk api: https://github.com/alphacep/vosk-api

Authors:
    - Tony Ribeiro (https://github.com/Tony-sama)

Models are saved into user cache foler, example: C:/Users/toto/.cache/vosk

References:
    - Code adapted from: https://github.com/alphacep/vosk-api/blob/master/python/example/test_microphone.py
"""
from flask import jsonify, abort

import queue
import sys
import sounddevice as sd

from vosk import Model, KaldiRecognizer

q = queue.Queue()

device_info = sd.query_devices(0, "input") # TODO: check if better way than just using id 0
# soundfile expects an int, sounddevice provides a float:
samplerate = int(device_info["default_samplerate"])

model = None

def load_model(file_path=None):
    """
    Load given vosk model from file or default to en-us model.
    Download model to user cache folder, example: C:/Users/toto/.cache/vosk
    """
    if file_path is None:
        return Model(lang="en-us")
    else:
        return Model(file_path)

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def record_mic():
    """
    Continuously record from mic and transcript voice.
    Return the transcript once no more voice is detected.
    """
    if model is None:
        print("Vosk model not initialized yet.")
        return ""
    try:
        with sd.RawInputStream(samplerate=samplerate, blocksize = 8000, device=0, dtype="int16", channels=1, callback=callback):

            rec = KaldiRecognizer(model, samplerate)
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    transcript = rec.Result()[14:-3]
                    print("Transcripted from microphone: ", transcript)
                    return jsonify({"transcript": transcript})
                #else:
                #    print(rec.PartialResult())

    except Exception as e: # No exception observed during test but we never know
        print(e)
        abort(500, "Exception occurs while recording with vosk-stt module")