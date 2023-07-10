"""
Speech-to-text module based on Vosk for SillyTavern Extras
    - Vosk website: https://alphacephei.com/vosk/
    - Vosk api: https://github.com/alphacep/vosk-api

Authors:
    - Tony Ribeiro (https://github.com/Tony-sama)

Models are saved into user cache folder, example: C:/Users/toto/.cache/vosk

References:
    - Code adapted from: https://github.com/alphacep/vosk-api/blob/master/python/example/test_microphone.py
"""
from flask import jsonify, abort

import queue
import sys
import sounddevice as sd
import soundfile as sf

from vosk import Model, KaldiRecognizer

DEBUG_PREFIX = "<stt vosk module>"

model = None
device = None

def load_model(file_path=None):
    """
    Load given vosk model from file or default to en-us model.
    Download model to user cache folder, example: C:/Users/toto/.cache/vosk
    """

    if file_path is None:
        return Model(lang="en-us")
    else:
        return Model(file_path)

def record_mic():
    """
    Continuously record from mic and transcript voice.
    Return the transcript once no more voice is detected.
    """
    if model is None:
        print("<stt-vosk-module> Vosk model not initialized yet.")
        return ""
    
    q = queue.Queue()

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))

    try:
        device_info = sd.query_devices(device, "input")
        # soundfile expects an int, sounddevice provides a float:
        samplerate = int(device_info["default_samplerate"])

        print(DEBUG_PREFIX, "Start recording from:", device_info["name"], "with samplerate", samplerate)

        with sd.RawInputStream(samplerate=samplerate, blocksize = 8000, device=device, dtype="int16", channels=1, callback=callback):

            rec = KaldiRecognizer(model, samplerate)
            full_recording = bytearray()
            while True:
                data = q.get()
                full_recording.extend(data)

                if rec.AcceptWaveform(data):
                    # Extract transcript string
                    transcript = rec.Result()[14:-3]
                    print(DEBUG_PREFIX, "Transcripted from microphone (streaming):", transcript)

                    # ----------------------------------
                    # DEBUG: save recording to wav file
                    # ----------------------------------
                    output_file = convert_bytearray_to_wav_ndarray(input_bytearray=full_recording, sampling_rate=samplerate)
                    sf.write(file=DEBUG_OUTPUT_FILE, data=output_file, samplerate=samplerate)
                    print(DEBUG_PREFIX, "Recorded message saved to", DEBUG_OUTPUT_FILE)
                    # ----------------------------------

                    return jsonify({"transcript": transcript})
                #else:
                #    print(rec.PartialResult())

    except Exception as e: # No exception observed during test but we never know
        print(e)
        abort(500, DEBUG_PREFIX+" Exception occurs while recording")

# ----------------------------------
# DEBUG: For checking audio quality
# ----------------------------------
import io
import numpy as np
from scipy.io.wavfile import write

def convert_bytearray_to_wav_ndarray(input_bytearray: bytes, sampling_rate=16000):
    """
    Convert a bytearray to wav format to output in a file for quality check debuging
    """
    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    write(byte_io, sampling_rate, np.frombuffer(input_bytearray, dtype=np.int16))
    output_wav = byte_io.read()
    output, _ = sf.read(io.BytesIO(output_wav))
    return output

DEBUG_OUTPUT_FILE = "stt_test.wav"
# ----------------------------------