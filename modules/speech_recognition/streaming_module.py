"""
Speech-to-text module based on Vosk and Whisper for SillyTavern Extras
    - Vosk website: https://alphacephei.com/vosk/
    - Vosk api: https://github.com/alphacep/vosk-api
    - Whisper github: https://github.com/openai/whisper

Authors:
    - Tony Ribeiro (https://github.com/Tony-sama)

Models are saved into user cache folder, example: C:/Users/toto/.cache/whisper and C:/Users/toto/.cache/vosk

References:
    - Code adapted from:
        - whisper github: https://github.com/openai/whisper
        - oobabooga text-generation-webui github: https://github.com/oobabooga/text-generation-webui
        - vosk github: https://github.com/alphacep/vosk-api/blob/master/python/example/test_microphone.py
"""
from flask import jsonify, abort

import queue
import sys
import sounddevice as sd
import soundfile as sf
import io
import numpy as np
from scipy.io.wavfile import write

import vosk
import whisper

DEBUG_PREFIX = "<stt streaming module>"
RECORDING_FILE_PATH = "stt_test.wav"

whisper_model = None
vosk_model = None
device = None

def load_model(file_path=None):
    """
    Load given vosk model from file or default to en-us model.
    Download model to user cache folder, example: C:/Users/toto/.cache/vosk
    """

    if file_path is None:
        return (whisper.load_model("base.en"), vosk.Model(lang="en-us"))
    else:
        return (whisper.load_model(file_path), vosk.Model(lang="en-us"))
    
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

def record_and_transcript():
    """
    Continuously record from mic and transcript voice.
    Return the transcript once no more voice is detected.
    """
    if whisper_model is None:
        print(DEBUG_PREFIX,"Whisper model not initialized yet.")
        return ""
    
    q = queue.Queue()
    stream_errors = list()

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
            stream_errors.append(status)
        q.put(bytes(indata))

    try:
        device_info = sd.query_devices(device, "input")
        # soundfile expects an int, sounddevice provides a float:
        samplerate = int(device_info["default_samplerate"])

        print(DEBUG_PREFIX, "Start recording from:", device_info["name"], "with samplerate", samplerate)

        with sd.RawInputStream(samplerate=samplerate, blocksize = 8000, device=device, dtype="int16", channels=1, callback=callback):

            rec = vosk.KaldiRecognizer(vosk_model, samplerate)
            full_recording = bytearray()
            while True:
                data = q.get()
                if len(stream_errors) > 0:
                    raise Exception(DEBUG_PREFIX+" Stream errors: "+str(stream_errors))
                
                full_recording.extend(data)

                if rec.AcceptWaveform(data):
                    # Extract transcript string
                    transcript = rec.Result()[14:-3]
                    print(DEBUG_PREFIX, "Transcripted from microphone stream (vosk):", transcript)

                    # ----------------------------------
                    # DEBUG: save recording to wav file
                    # ----------------------------------
                    output_file = convert_bytearray_to_wav_ndarray(input_bytearray=full_recording, sampling_rate=samplerate)
                    sf.write(file=RECORDING_FILE_PATH, data=output_file, samplerate=samplerate)
                    print(DEBUG_PREFIX, "Recorded message saved to", RECORDING_FILE_PATH)
                    
                    # Whisper HACK
                    result = whisper_model.transcribe(RECORDING_FILE_PATH)
                    transcript = result["text"]
                    print(DEBUG_PREFIX, "Transcripted from audio file (whisper):", transcript)
                    # ----------------------------------

                    return jsonify({"transcript": transcript})
                #else:
                #    print(rec.PartialResult())

    except Exception as e: # No exception observed during test but we never know
        print(e)
        abort(500, DEBUG_PREFIX+" Exception occurs while recording")