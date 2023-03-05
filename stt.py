import speech_recognition as sr
import whisper

english = True


def init_stt(model="base", english=True, energy=300, pause=0.8, dynamic_energy=False):
    if model != "large" and english:
        model = model + ".en"
    audio_model = whisper.load_model(model)
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = dynamic_energy
    return r, audio_model


r, audio_model = init_stt()
