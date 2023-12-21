from functools import wraps
from flask import (
    Flask,
    jsonify,
    request,
    Response,
    render_template_string,
    abort,
    send_from_directory,
    send_file,
)
from flask_cors import CORS
from flask_compress import Compress
import markdown
import argparse
from transformers import AutoTokenizer, pipeline
from transformers import AutoModelForSeq2SeqLM
import unicodedata
import torch
import time
import os
import gc
import sys
import secrets
from PIL import Image
import base64
from io import BytesIO
from random import randint
import webuiapi
import hashlib
from constants import *
from colorama import Fore, Style, init as colorama_init

colorama_init()

if sys.hexversion < 0x030b0000:
    print(f"{Fore.BLUE}{Style.BRIGHT}Python 3.11 or newer is recommended to run this program.{Style.RESET_ALL}")
    time.sleep(2)

class SplitArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(
            namespace, self.dest, values.replace('"', "").replace("'", "").split(",")
        )

#Setting Root Folders for Silero Generations so it is compatible with STSL, should not effect regular runs. - Rolyat
parent_dir = os.path.dirname(os.path.abspath(__file__))
SILERO_SAMPLES_PATH = os.path.join(parent_dir, "tts_samples")
SILERO_SAMPLE_TEXT = os.path.join(parent_dir)

# Create directories if they don't exist
if not os.path.exists(SILERO_SAMPLES_PATH):
    os.makedirs(SILERO_SAMPLES_PATH)
if not os.path.exists(SILERO_SAMPLE_TEXT):
    os.makedirs(SILERO_SAMPLE_TEXT)

# Script arguments
parser = argparse.ArgumentParser(
    prog="SillyTavern Extras", description="Web API for transformers models"
)
parser.add_argument(
    "--port", type=int, help="Specify the port on which the application is hosted"
)
parser.add_argument(
    "--listen", action="store_true", help="Host the app on the local network"
)
parser.add_argument(
    "--share", action="store_true", help="Share the app on CloudFlare tunnel"
)
parser.add_argument("--cpu", action="store_true", help="Run the models on the CPU")
parser.add_argument("--cuda", action="store_false", dest="cpu", help="Run the models on the GPU")
parser.add_argument("--cuda-device", help="Specify the CUDA device to use")
parser.add_argument("--mps", "--apple", "--m1", "--m2", action="store_false", dest="cpu", help="Run the models on Apple Silicon")
parser.set_defaults(cpu=True)
parser.add_argument("--summarization-model", help="Load a custom summarization model")
parser.add_argument(
    "--classification-model", help="Load a custom text classification model"
)
parser.add_argument("--captioning-model", help="Load a custom captioning model")
parser.add_argument("--embedding-model", help="Load a custom text embedding model")
parser.add_argument("--chroma-host", help="Host IP for a remote ChromaDB instance")
parser.add_argument("--chroma-port", help="HTTP port for a remote ChromaDB instance (defaults to 8000)")
parser.add_argument("--chroma-folder", help="Path for chromadb persistence folder", default='.chroma_db')
parser.add_argument('--chroma-persist', help="ChromaDB persistence", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument(
    "--secure", action="store_true", help="Enforces the use of an API key"
)
parser.add_argument("--talkinghead-gpu", action="store_true", help="Run the talkinghead animation on the GPU (CPU is default)")
parser.add_argument(
    "--talkinghead-model", type=str, help="The THA3 model to use. 'float' models are fp32, 'half' are fp16. 'auto' (default) picks fp16 for GPU and fp32 for CPU.",
    required=False, default="auto",
    choices=["auto", "standard_float", "separable_float", "standard_half", "separable_half"],
)
parser.add_argument(
    "--talkinghead-models", type=str, help="If THA3 models are not yet installed, use the given HuggingFace repository to install them.",
    default="OktayAlpk/talking-head-anime-3"
)

parser.add_argument("--coqui-gpu", action="store_true", help="Run the voice models on the GPU (CPU is default)")
parser.add_argument("--coqui-models", help="Install given Coqui-api TTS model at launch (comma separated list, last one will be loaded at start)")

parser.add_argument("--max-content-length", help="Set the max")
parser.add_argument("--rvc-save-file", action="store_true", help="Save the last rvc input/output audio file into data/tmp/ folder (for research)")

parser.add_argument("--stt-vosk-model-path", help="Load a custom vosk speech-to-text model")
parser.add_argument("--stt-whisper-model-path", help="Load a custom vosk speech-to-text model")
sd_group = parser.add_mutually_exclusive_group()

local_sd = parser.add_argument_group("sd-local")
local_sd.add_argument("--sd-model", help="Load a custom SD image generation model")
local_sd.add_argument("--sd-cpu", help="Force the SD pipeline to run on the CPU", action="store_true")

remote_sd = parser.add_argument_group("sd-remote")
remote_sd.add_argument(
    "--sd-remote", action="store_true", help="Use a remote backend for SD"
)
remote_sd.add_argument(
    "--sd-remote-host", type=str, help="Specify the host of the remote SD backend"
)
remote_sd.add_argument(
    "--sd-remote-port", type=int, help="Specify the port of the remote SD backend"
)
remote_sd.add_argument(
    "--sd-remote-ssl", action="store_true", help="Use SSL for the remote SD backend"
)
remote_sd.add_argument(
    "--sd-remote-auth",
    type=str,
    help="Specify the username:password for the remote SD backend (if required)",
)

parser.add_argument(
    "--enable-modules",
    action=SplitArgs,
    default=[],
    help="Override a list of enabled modules",
)

args = parser.parse_args()

port = args.port if args.port else 5100
host = "0.0.0.0" if args.listen else "localhost"
summarization_model = (
    args.summarization_model
    if args.summarization_model
    else DEFAULT_SUMMARIZATION_MODEL
)
classification_model = (
    args.classification_model
    if args.classification_model
    else DEFAULT_CLASSIFICATION_MODEL
)
captioning_model = (
    args.captioning_model if args.captioning_model else DEFAULT_CAPTIONING_MODEL
)
embedding_model = (
    args.embedding_model if args.embedding_model else DEFAULT_EMBEDDING_MODEL
)

sd_use_remote = False if args.sd_model else True
sd_model = args.sd_model if args.sd_model else DEFAULT_SD_MODEL
sd_remote_host = args.sd_remote_host if args.sd_remote_host else DEFAULT_REMOTE_SD_HOST
sd_remote_port = args.sd_remote_port if args.sd_remote_port else DEFAULT_REMOTE_SD_PORT
sd_remote_ssl = args.sd_remote_ssl
sd_remote_auth = args.sd_remote_auth

modules = (
    args.enable_modules if args.enable_modules and len(args.enable_modules) > 0 else []
)

if len(modules) == 0:
    print(
        f"{Fore.RED}{Style.BRIGHT}You did not select any modules to run! Choose them by adding an --enable-modules option"
    )
    print(f"Example: --enable-modules=caption,summarize{Style.RESET_ALL}")

# Models init
cuda_device = DEFAULT_CUDA_DEVICE if not args.cuda_device else args.cuda_device
device_string = cuda_device if torch.cuda.is_available() and not args.cpu else 'mps' if torch.backends.mps.is_available() and not args.cpu else 'cpu'
device = torch.device(device_string)
torch_dtype = torch.float32 if device_string != cuda_device  else torch.float16

if not torch.cuda.is_available() and not args.cpu:
    print(f"{Fore.YELLOW}{Style.BRIGHT}torch-cuda is not supported on this device.{Style.RESET_ALL}")
    if not torch.backends.mps.is_available() and not args.cpu:
        print(f"{Fore.YELLOW}{Style.BRIGHT}torch-mps is not supported on this device.{Style.RESET_ALL}")


print(f"{Fore.GREEN}{Style.BRIGHT}Using torch device: {device_string}{Style.RESET_ALL}")

if "talkinghead" in modules:
    # Install the THA3 models if needed
    talkinghead_models_dir = os.path.join(os.getcwd(), "talkinghead", "tha3", "models")
    if not os.path.exists(talkinghead_models_dir):
        # API:
        #   https://huggingface.co/docs/huggingface_hub/en/guides/download
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "You need to install huggingface_hub to install talkinghead models automatically. "
                "See https://pypi.org/project/huggingface-hub/ for installation."
            )
        os.makedirs(talkinghead_models_dir, exist_ok=True)
        print(f"THA3 models not yet installed. Installing from {args.talkinghead_models} into talkinghead/tha3/models.")
        # TODO: I'd prefer to install with symlinks, but how about Windows users?
        snapshot_download(repo_id=args.talkinghead_models, local_dir=talkinghead_models_dir, local_dir_use_symlinks=False)

    import sys
    import threading
    mode = "cuda" if args.talkinghead_gpu else "cpu"
    model = args.talkinghead_model
    if model == "auto":  # default
        # FP16 boosts the rendering performance by ~1.5x, but is only supported on GPU.
        model = "separable_half" if args.talkinghead_gpu else "separable_float"
    print(f"Initializing talkinghead pipeline in {mode} mode with model {model}....")
    talkinghead_path = os.path.abspath(os.path.join(os.getcwd(), "talkinghead"))
    sys.path.append(talkinghead_path) # Add the path to the 'tha3' module to the sys.path list

    try:
        import talkinghead.tha3.app.app as talkinghead
        def launch_talkinghead():
            # mode: choices='The device to use for PyTorch ("cuda" for GPU, "cpu" for CPU).'
            # model: choices=['standard_float', 'separable_float', 'standard_half', 'separable_half'],
            talkinghead.launch(mode, model)
        talkinghead_thread = threading.Thread(target=launch_talkinghead)
        talkinghead_thread.daemon = True  # Set the thread as a daemon thread
        talkinghead_thread.start()

    except ModuleNotFoundError:
        print("Error: Could not import the 'talkinghead' module.")

if "caption" in modules:
    print("Initializing an image captioning model...")
    captioning_pipeline = pipeline('image-to-text', model=captioning_model, device=device_string, torch_dtype=torch_dtype)

if "summarize" in modules:
    print("Initializing a text summarization model...")
    summarization_pipeline = pipeline('summarization', model=summarization_model, device=device_string, torch_dtype=torch_dtype)

if "sd" in modules and not sd_use_remote:
    from diffusers import StableDiffusionPipeline
    from diffusers import EulerAncestralDiscreteScheduler

    print("Initializing Stable Diffusion pipeline...")
    sd_device_string = cuda_device if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    sd_device = torch.device(sd_device_string)
    sd_torch_dtype = torch.float32 if sd_device_string != cuda_device else torch.float16
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        sd_model, custom_pipeline="lpw_stable_diffusion", torch_dtype=sd_torch_dtype
    ).to(sd_device)
    sd_pipe.safety_checker = lambda images, clip_input: (images, False)
    sd_pipe.enable_attention_slicing()
    # pipe.scheduler = KarrasVeScheduler.from_config(pipe.scheduler.config)
    sd_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        sd_pipe.scheduler.config
    )
elif "sd" in modules and sd_use_remote:
    print("Initializing Stable Diffusion connection")
    try:
        sd_remote = webuiapi.WebUIApi(
            host=sd_remote_host, port=sd_remote_port, use_https=sd_remote_ssl
        )
        if sd_remote_auth:
            username, password = sd_remote_auth.split(":")
            sd_remote.set_auth(username, password)
        sd_remote.util_wait_for_ready()
    except Exception as e:
        # remote sd from modules
        print(
            f"{Fore.RED}{Style.BRIGHT}Could not connect to remote SD backend at http{'s' if sd_remote_ssl else ''}://{sd_remote_host}:{sd_remote_port}! Disabling SD module...{Style.RESET_ALL}"
        )
        modules.remove("sd")

if "tts" in modules:
    print("tts module is deprecated. Please use silero-tts instead.")
    modules.remove("tts")
    modules.append("silero-tts")


if "silero-tts" in modules:
    if not os.path.exists(SILERO_SAMPLES_PATH):
        os.makedirs(SILERO_SAMPLES_PATH)
    print("Initializing Silero TTS server")
    from silero_api_server import tts

    tts_service = tts.SileroTtsService(SILERO_SAMPLES_PATH)
    if len(os.listdir(SILERO_SAMPLES_PATH)) == 0:
        print("Generating Silero TTS samples...")
        tts_service.update_sample_text(SILERO_SAMPLE_TEXT)
        tts_service.generate_samples()

if "edge-tts" in modules:
    print("Initializing Edge TTS client")
    import tts_edge as edge


if "chromadb" in modules:
    print("Initializing ChromaDB")
    import chromadb
    import posthog
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    from sentence_transformers import SentenceTransformer

    # Assume that the user wants in-memory unless a host is specified
    # Also disable chromadb telemetry
    posthog.capture = lambda *args, **kwargs: None
    if args.chroma_host is None:
        if args.chroma_persist:
            chromadb_client = chromadb.PersistentClient(path=args.chroma_folder, settings=Settings(anonymized_telemetry=False))
            print(f"ChromaDB is running in-memory with persistence. Persistence is stored in {args.chroma_folder}. Can be cleared by deleting the folder or purging db.")
        else:
            chromadb_client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
            print(f"ChromaDB is running in-memory without persistence.")
    else:
        chroma_port=(
            args.chroma_port if args.chroma_port else DEFAULT_CHROMA_PORT
        )
        chromadb_client = chromadb.HttpClient(host=args.chroma_host, port=chroma_port, settings=Settings(anonymized_telemetry=False))
        print(f"ChromaDB is remotely configured at {args.chroma_host}:{chroma_port}")

    chromadb_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(embedding_model, device=device_string)

    # Check if the db is connected and running, otherwise tell the user
    try:
        chromadb_client.heartbeat()
        print("Successfully pinged ChromaDB! Your client is successfully connected.")
    except:
        print("Could not ping ChromaDB! If you are running remotely, please check your host and port!")

# Flask init
app = Flask(__name__)
CORS(app)  # allow cross-domain requests
Compress(app) # compress responses
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

max_content_length = (
    args.max_content_length
    if args.max_content_length
    else None)

if max_content_length is not None:
    print("Setting MAX_CONTENT_LENGTH to",max_content_length,"Mb")
    app.config["MAX_CONTENT_LENGTH"] = int(max_content_length) * 1024 * 1024

if "classify" in modules:
    import modules.classify.classify_module as classify_module
    classify_module.init_text_emotion_classifier(classification_model, device, torch_dtype)

if "vosk-stt" in modules:
    print("Initializing Vosk speech-recognition (from ST request file)")
    vosk_model_path = (
    args.stt_vosk_model_path
    if args.stt_vosk_model_path
    else None)

    import modules.speech_recognition.vosk_module as vosk_module

    vosk_module.model = vosk_module.load_model(file_path=vosk_model_path)
    app.add_url_rule("/api/speech-recognition/vosk/process-audio", view_func=vosk_module.process_audio, methods=["POST"])

if "whisper-stt" in modules:
    print("Initializing Whisper speech-recognition (from ST request file)")
    whisper_model_path = (
    args.stt_whisper_model_path
    if args.stt_whisper_model_path
    else None)

    import modules.speech_recognition.whisper_module as whisper_module

    whisper_module.model = whisper_module.load_model(file_path=whisper_model_path)
    app.add_url_rule("/api/speech-recognition/whisper/process-audio", view_func=whisper_module.process_audio, methods=["POST"])

if "streaming-stt" in modules:
    print("Initializing vosk/whisper speech-recognition (from extras server microphone)")
    whisper_model_path = (
    args.stt_whisper_model_path
    if args.stt_whisper_model_path
    else None)

    import modules.speech_recognition.streaming_module as streaming_module

    streaming_module.whisper_model, streaming_module.vosk_model = streaming_module.load_model(file_path=whisper_model_path)
    app.add_url_rule("/api/speech-recognition/streaming/record-and-transcript", view_func=streaming_module.record_and_transcript, methods=["POST"])

if "rvc" in modules:
    print("Initializing RVC voice conversion (from ST request file)")
    print("Increasing server upload limit")
    rvc_save_file = (
    args.rvc_save_file
    if args.rvc_save_file
    else False)

    if rvc_save_file:
        print("RVC saving file option detected, input/output audio will be savec into data/tmp/ folder")

    import sys
    sys.path.insert(0,'modules/voice_conversion')

    import modules.voice_conversion.rvc_module as rvc_module
    rvc_module.save_file = rvc_save_file

    if "classify" in modules:
        rvc_module.classification_mode = True

    rvc_module.fix_model_install()
    app.add_url_rule("/api/voice-conversion/rvc/get-models-list", view_func=rvc_module.rvc_get_models_list, methods=["POST"])
    app.add_url_rule("/api/voice-conversion/rvc/upload-models", view_func=rvc_module.rvc_upload_models, methods=["POST"])
    app.add_url_rule("/api/voice-conversion/rvc/process-audio", view_func=rvc_module.rvc_process_audio, methods=["POST"])


if "coqui-tts" in modules:
    mode = "GPU" if args.coqui_gpu else "CPU"
    print("Initializing Coqui TTS client in " + mode + " mode")
    import modules.text_to_speech.coqui.coqui_module as coqui_module

    if mode == "GPU":
        coqui_module.gpu_mode = True

    coqui_models = (
    args.coqui_models
    if args.coqui_models
    else None
    )

    if coqui_models is not None:
        coqui_models = coqui_models.split(",")
        for i in coqui_models:
            if not coqui_module.install_model(i):
                raise ValueError("Coqui model loading failed, most likely a wrong model name in --coqui-models argument, check log above to see which one")

    # Coqui-api models
    app.add_url_rule("/api/text-to-speech/coqui/coqui-api/check-model-state", view_func=coqui_module.coqui_check_model_state, methods=["POST"])
    app.add_url_rule("/api/text-to-speech/coqui/coqui-api/install-model", view_func=coqui_module.coqui_install_model, methods=["POST"])

    # Users models
    app.add_url_rule("/api/text-to-speech/coqui/local/get-models", view_func=coqui_module.coqui_get_local_models, methods=["POST"])

    # Handle both coqui-api/users models
    app.add_url_rule("/api/text-to-speech/coqui/generate-tts", view_func=coqui_module.coqui_generate_tts, methods=["POST"])

def require_module(name):
    def wrapper(fn):
        @wraps(fn)
        def decorated_view(*args, **kwargs):
            if name not in modules:
                abort(403, "Module is disabled by config")
            return fn(*args, **kwargs)

        return decorated_view

    return wrapper


# AI stuff
def classify_text(text: str) -> list:
    return classify_module.classify_text_emotion(text)


def caption_image(raw_image: Image) -> str:
    caption = captioning_pipeline(raw_image.convert("RGB"))[0]['generated_text']
    return caption


def summarize_chunks(text: str) -> str:
    try:
        return summarize(text)
    except IndexError:
        print(
            "Sequence length too large for model, cutting text in half and calling again"
        )
        return summarize_chunks(
            text[: (len(text) // 2)]
        ) + summarize_chunks(text[(len(text) // 2) :])


def summarize(text: str) -> str:
    summary = normalize_string(summarization_pipeline(text)[0]['summary_text'])
    return summary


def normalize_string(input: str) -> str:
    output = " ".join(unicodedata.normalize("NFKC", input).strip().split())
    return output


def generate_image(data: dict) -> Image:
    prompt = normalize_string(f'{data["prompt_prefix"]} {data["prompt"]}')

    if sd_use_remote:
        image = sd_remote.txt2img(
            prompt=prompt,
            negative_prompt=data["negative_prompt"],
            sampler_name=data["sampler"],
            steps=data["steps"],
            cfg_scale=data["scale"],
            width=data["width"],
            height=data["height"],
            restore_faces=data["restore_faces"],
            enable_hr=data["enable_hr"],
            save_images=True,
            send_images=True,
            do_not_save_grid=False,
            do_not_save_samples=False,
        ).image
    else:
        image = sd_pipe(
            prompt=prompt,
            negative_prompt=data["negative_prompt"],
            num_inference_steps=data["steps"],
            guidance_scale=data["scale"],
            width=data["width"],
            height=data["height"],
        ).images[0]

    image.save("./debug.png")
    return image


def image_to_base64(image: Image, quality: int = 75) -> str:
    buffer = BytesIO()
    image.convert("RGB")
    image.save(buffer, format="JPEG", quality=quality)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

ignore_auth = []
# Reads an API key from an already existing file. If that file doesn't exist, create it.
if args.secure:
    try:
        with open("api_key.txt", "r") as txt:
            api_key = txt.read().replace('\n', '')
    except:
        api_key = secrets.token_hex(5)
        with open("api_key.txt", "w") as txt:
            txt.write(api_key)

    print(f"{Fore.YELLOW}{Style.BRIGHT}Your API key is {api_key}{Style.RESET_ALL}")
elif args.share and args.secure != True:
    print(f"{Fore.RED}{Style.BRIGHT}WARNING: This instance is publicly exposed without an API key! It is highly recommended to restart with the \"--secure\" argument!{Style.RESET_ALL}")
else:
    print(f"{Fore.YELLOW}{Style.BRIGHT}No API key given because you are running locally.{Style.RESET_ALL}")


def is_authorize_ignored(request):
    view_func = app.view_functions.get(request.endpoint)

    if view_func is not None:
        if view_func in ignore_auth:
            return True
    return False


@app.before_request
def before_request():
    # Request time measuring
    request.start_time = time.time()

    # Checks if an API key is present and valid, otherwise return unauthorized
    # The options check is required so CORS doesn't get angry
    try:
        if request.method != 'OPTIONS' and args.secure and is_authorize_ignored(request) == False and getattr(request.authorization, 'token', '') != api_key:
            print(f"{Fore.RED}{Style.NORMAL}WARNING: Unauthorized API key access from {request.remote_addr}{Style.RESET_ALL}")
            response = jsonify({ 'error': '401: Invalid API key' })
            response.status_code = 401
            return response
    except Exception as e:
        print(f"API key check error: {e}")
        return "401 Unauthorized\n{}\n\n".format(e), 401


@app.after_request
def after_request(response):
    duration = time.time() - request.start_time
    response.headers["X-Request-Duration"] = str(duration)
    return response


@app.route("/", methods=["GET"])
def index():
    with open("./README.md", "r", encoding="utf8") as f:
        content = f.read()
    return render_template_string(markdown.markdown(content, extensions=["tables"]))


@app.route("/api/extensions", methods=["GET"])
def get_extensions():
    extensions = dict(
        {
            "extensions": [
                {
                    "name": "not-supported",
                    "metadata": {
                        "display_name": """<span style="white-space:break-spaces;">Extensions serving using Extensions API is no longer supported. Please update the mod from: <a href="https://github.com/Cohee1207/SillyTavern">https://github.com/Cohee1207/SillyTavern</a></span>""",
                        "requires": [],
                        "assets": [],
                    },
                }
            ]
        }
    )
    return jsonify(extensions)


@app.route("/api/caption", methods=["POST"])
@require_module("caption")
def api_caption():
    data = request.get_json()

    if "image" not in data or not isinstance(data["image"], str):
        abort(400, '"image" is required')

    image = Image.open(BytesIO(base64.b64decode(data["image"])))
    image = image.convert("RGB")
    image.thumbnail((512, 512))
    caption = caption_image(image)
    thumbnail = image_to_base64(image)
    print("Caption:", caption, sep="\n")
    gc.collect()
    return jsonify({"caption": caption, "thumbnail": thumbnail})


@app.route("/api/summarize", methods=["POST"])
@require_module("summarize")
def api_summarize():
    data = request.get_json()

    if "text" not in data or not isinstance(data["text"], str):
        abort(400, '"text" is required')

    print("Summary input:", data["text"], sep="\n")
    summary = summarize_chunks(data["text"])
    print("Summary output:", summary, sep="\n")
    gc.collect()
    return jsonify({"summary": summary})


@app.route("/api/classify", methods=["POST"])
@require_module("classify")
def api_classify():
    data = request.get_json()

    if "text" not in data or not isinstance(data["text"], str):
        abort(400, '"text" is required')

    print("Classification input:", data["text"], sep="\n")
    classification = classify_text(data["text"])
    print("Classification output:", classification, sep="\n")
    gc.collect()
    if "talkinghead" in modules: #send emotion to talkinghead
        talkinghead.setEmotion(classification)
    return jsonify({"classification": classification})


@app.route("/api/classify/labels", methods=["GET"])
@require_module("classify")
def api_classify_labels():
    classification = classify_text("")
    labels = [x["label"] for x in classification]
    if "talkinghead" in modules:
        labels.append('talkinghead')  # Add 'talkinghead' to the labels list
    return jsonify({"labels": labels})

@app.route("/api/talkinghead/load", methods=["POST"])
def live_load():
    file = request.files['file']
    # convert stream to bytes and pass to talkinghead_load
    return talkinghead.talkinghead_load_file(file.stream)

@app.route('/api/talkinghead/unload')
def live_unload():
    return talkinghead.unload()

@app.route('/api/talkinghead/start_talking')
def start_talking():
    return talkinghead.start_talking()

@app.route('/api/talkinghead/stop_talking')
def stop_talking():
    return talkinghead.stop_talking()

@app.route('/api/talkinghead/result_feed')
def result_feed():
    return talkinghead.result_feed()

@app.route("/api/image", methods=["POST"])
@require_module("sd")
def api_image():
    required_fields = {
        "prompt": str,
    }

    optional_fields = {
        "steps": 30,
        "scale": 6,
        "sampler": "DDIM",
        "width": 512,
        "height": 512,
        "restore_faces": False,
        "enable_hr": False,
        "prompt_prefix": PROMPT_PREFIX,
        "negative_prompt": NEGATIVE_PROMPT,
    }

    data = request.get_json()

    # Check required fields
    for field, field_type in required_fields.items():
        if field not in data or not isinstance(data[field], field_type):
            abort(400, f'"{field}" is required')

    # Set optional fields to default values if not provided
    for field, default_value in optional_fields.items():
        type_match = (
            (int, float)
            if isinstance(default_value, (int, float))
            else type(default_value)
        )
        if field not in data or not isinstance(data[field], type_match):
            data[field] = default_value

    try:
        print("SD inputs:", data, sep="\n")
        image = generate_image(data)
        base64image = image_to_base64(image, quality=90)
        return jsonify({"image": base64image})
    except RuntimeError as e:
        abort(400, str(e))


@app.route("/api/image/model", methods=["POST"])
@require_module("sd")
def api_image_model_set():
    data = request.get_json()

    if not sd_use_remote:
        abort(400, "Changing model for local sd is not supported.")
    if "model" not in data or not isinstance(data["model"], str):
        abort(400, '"model" is required')

    old_model = sd_remote.util_get_current_model()
    sd_remote.util_set_model(data["model"], find_closest=False)
    # sd_remote.util_set_model(data['model'])
    sd_remote.util_wait_for_ready()
    new_model = sd_remote.util_get_current_model()

    return jsonify({"previous_model": old_model, "current_model": new_model})


@app.route("/api/image/model", methods=["GET"])
@require_module("sd")
def api_image_model_get():
    model = sd_model

    if sd_use_remote:
        model = sd_remote.util_get_current_model()

    return jsonify({"model": model})


@app.route("/api/image/models", methods=["GET"])
@require_module("sd")
def api_image_models():
    models = [sd_model]

    if sd_use_remote:
        models = sd_remote.util_get_model_names()

    return jsonify({"models": models})


@app.route("/api/image/samplers", methods=["GET"])
@require_module("sd")
def api_image_samplers():
    samplers = ["Euler a"]

    if sd_use_remote:
        samplers = [sampler["name"] for sampler in sd_remote.get_samplers()]

    return jsonify({"samplers": samplers})


@app.route("/api/modules", methods=["GET"])
def get_modules():
    return jsonify({"modules": modules})


@app.route("/api/tts/speakers", methods=["GET"])
@require_module("silero-tts")
def tts_speakers():
    voices = [
        {
            "name": speaker,
            "voice_id": speaker,
            "preview_url": f"{str(request.url_root)}api/tts/sample/{speaker}",
        }
        for speaker in tts_service.get_speakers()
    ]
    return jsonify(voices)

# Added fix for Silero not working as new files were unable to be created if one already existed. - Rolyat 7/7/23
@app.route("/api/tts/generate", methods=["POST"])
@require_module("silero-tts")
def tts_generate():
    voice = request.get_json()
    if "text" not in voice or not isinstance(voice["text"], str):
        abort(400, '"text" is required')
    if "speaker" not in voice or not isinstance(voice["speaker"], str):
        abort(400, '"speaker" is required')
    # Remove asterisks
    voice["text"] = voice["text"].replace("*", "")
    try:
        # Remove the destination file if it already exists
        if os.path.exists('test.wav'):
            os.remove('test.wav')

        audio = tts_service.generate(voice["speaker"], voice["text"])
        audio_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(audio))

        os.rename(audio, audio_file_path)
        return send_file(audio_file_path, mimetype="audio/x-wav")
    except Exception as e:
        print(e)
        abort(500, voice["speaker"])


@app.route("/api/tts/sample/<speaker>", methods=["GET"])
@require_module("silero-tts")
def tts_play_sample(speaker: str):
    return send_from_directory(SILERO_SAMPLES_PATH, f"{speaker}.wav")


@app.route("/api/edge-tts/list", methods=["GET"])
@require_module("edge-tts")
def edge_tts_list():
    voices = edge.get_voices()
    return jsonify(voices)


@app.route("/api/edge-tts/generate", methods=["POST"])
@require_module("edge-tts")
def edge_tts_generate():
    data = request.get_json()
    if "text" not in data or not isinstance(data["text"], str):
        abort(400, '"text" is required')
    if "voice" not in data or not isinstance(data["voice"], str):
        abort(400, '"voice" is required')
    if "rate" in data and isinstance(data['rate'], int):
        rate = data['rate']
    else:
        rate = 0
    # Remove asterisks
    data["text"] = data["text"].replace("*", "")
    try:
        audio = edge.generate_audio(text=data["text"], voice=data["voice"], rate=rate)
        return Response(audio, mimetype="audio/mpeg")
    except Exception as e:
        print(e)
        abort(500, data["voice"])


@app.route("/api/chromadb", methods=["POST"])
@require_module("chromadb")
def chromadb_add_messages():
    data = request.get_json()
    if "chat_id" not in data or not isinstance(data["chat_id"], str):
        abort(400, '"chat_id" is required')
    if "messages" not in data or not isinstance(data["messages"], list):
        abort(400, '"messages" is required')

    chat_id_md5 = hashlib.md5(data["chat_id"].encode()).hexdigest()
    collection = chromadb_client.get_or_create_collection(
        name=f"chat-{chat_id_md5}", embedding_function=chromadb_embed_fn
    )

    documents = [m["content"] for m in data["messages"]]
    ids = [m["id"] for m in data["messages"]]
    metadatas = [
        {"role": m["role"], "date": m["date"], "meta": m.get("meta", "")}
        for m in data["messages"]
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    return jsonify({"count": len(ids)})


@app.route("/api/chromadb/purge", methods=["POST"])
@require_module("chromadb")
def chromadb_purge():
    data = request.get_json()
    if "chat_id" not in data or not isinstance(data["chat_id"], str):
        abort(400, '"chat_id" is required')

    chat_id_md5 = hashlib.md5(data["chat_id"].encode()).hexdigest()

    try:
        chromadb_client.delete_collection(f"chat-{chat_id_md5}")
        print(f"Collection chat-{chat_id_md5} deleted")
    except ValueError:
        print(f"Collection chat-{chat_id_md5} does not exist, skipping deletion")

    return 'Ok', 200


@app.route("/api/chromadb/query", methods=["POST"])
@require_module("chromadb")
def chromadb_query():
    data = request.get_json()
    if "chat_id" not in data or not isinstance(data["chat_id"], str):
        abort(400, '"chat_id" is required')
    if "query" not in data or not isinstance(data["query"], str):
        abort(400, '"query" is required')

    if "n_results" not in data or not isinstance(data["n_results"], int):
        n_results = 1
    else:
        n_results = data["n_results"]

    chat_id_md5 = hashlib.md5(data["chat_id"].encode()).hexdigest()
    collection = chromadb_client.get_or_create_collection(
        name=f"chat-{chat_id_md5}", embedding_function=chromadb_embed_fn
    )

    if collection.count() == 0:
        print(f"Queried empty/missing collection for {repr(data['chat_id'])}.")
        return jsonify([])


    n_results = min(collection.count(), n_results)
    query_result = collection.query(
        query_texts=[data["query"]],
        n_results=n_results,
    )

    documents = query_result["documents"][0]
    ids = query_result["ids"][0]
    metadatas = query_result["metadatas"][0]
    distances = query_result["distances"][0]

    messages = [
        {
            "id": ids[i],
            "date": metadatas[i]["date"],
            "role": metadatas[i]["role"],
            "meta": metadatas[i]["meta"],
            "content": documents[i],
            "distance": distances[i],
        }
        for i in range(len(ids))
    ]

    return jsonify(messages)

@app.route("/api/chromadb/multiquery", methods=["POST"])
@require_module("chromadb")
def chromadb_multiquery():
    data = request.get_json()
    if "chat_list" not in data or not isinstance(data["chat_list"], list):
        abort(400, '"chat_list" is required and should be a list')
    if "query" not in data or not isinstance(data["query"], str):
        abort(400, '"query" is required')

    if "n_results" not in data or not isinstance(data["n_results"], int):
        n_results = 1
    else:
        n_results = data["n_results"]

    messages = []

    for chat_id in data["chat_list"]:
        if not isinstance(chat_id, str):
            continue

        try:
            chat_id_md5 = hashlib.md5(chat_id.encode()).hexdigest()
            collection = chromadb_client.get_collection(
                name=f"chat-{chat_id_md5}", embedding_function=chromadb_embed_fn
            )

            # Skip this chat if the collection is empty
            if collection.count() == 0:
                continue

            n_results_per_chat = min(collection.count(), n_results)
            query_result = collection.query(
                query_texts=[data["query"]],
                n_results=n_results_per_chat,
            )
            documents = query_result["documents"][0]
            ids = query_result["ids"][0]
            metadatas = query_result["metadatas"][0]
            distances = query_result["distances"][0]

            chat_messages = [
                {
                    "id": ids[i],
                    "date": metadatas[i]["date"],
                    "role": metadatas[i]["role"],
                    "meta": metadatas[i]["meta"],
                    "content": documents[i],
                    "distance": distances[i],
                }
                for i in range(len(ids))
            ]

            messages.extend(chat_messages)
        except Exception as e:
            print(e)

    #remove duplicate msgs, filter down to the right number
    seen = set()
    messages = [d for d in messages if not (d['content'] in seen or seen.add(d['content']))]
    messages = sorted(messages, key=lambda x: x['distance'])[0:n_results]

    return jsonify(messages)


@app.route("/api/chromadb/export", methods=["POST"])
@require_module("chromadb")
def chromadb_export():
    data = request.get_json()
    if "chat_id" not in data or not isinstance(data["chat_id"], str):
        abort(400, '"chat_id" is required')

    chat_id_md5 = hashlib.md5(data["chat_id"].encode()).hexdigest()
    try:
        collection = chromadb_client.get_collection(
            name=f"chat-{chat_id_md5}", embedding_function=chromadb_embed_fn
        )
    except Exception as e:
        print(e)
        abort(400, "Chat collection not found in chromadb")

    collection_content = collection.get()
    documents = collection_content.get('documents', [])
    ids = collection_content.get('ids', [])
    metadatas = collection_content.get('metadatas', [])

    unsorted_content = [
        {
            "id": ids[i],
            "metadata": metadatas[i],
            "document": documents[i],
        }
        for i in range(len(ids))
    ]

    sorted_content = sorted(unsorted_content, key=lambda x: x['metadata']['date'])

    export = {
        "chat_id": data["chat_id"],
        "content": sorted_content
    }

    return jsonify(export)

@app.route("/api/chromadb/import", methods=["POST"])
@require_module("chromadb")
def chromadb_import():
    data = request.get_json()
    content = data['content']
    if "chat_id" not in data or not isinstance(data["chat_id"], str):
        abort(400, '"chat_id" is required')

    chat_id_md5 = hashlib.md5(data["chat_id"].encode()).hexdigest()
    collection = chromadb_client.get_or_create_collection(
        name=f"chat-{chat_id_md5}", embedding_function=chromadb_embed_fn
    )

    documents = [item['document'] for item in content]
    metadatas = [item['metadata'] for item in content]
    ids = [item['id'] for item in content]


    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Imported {len(ids)} (total {collection.count()}) content entries into {repr(data['chat_id'])}")

    return jsonify({"count": len(ids)})


@app.route("/api/websearch", methods=["POST"])
@require_module("websearch")
def api_websearch():
    data = request.get_json()
    if "query" not in data or not isinstance(data["query"], str):
        abort(400, '"query" is required')

    query = data["query"]
    engine = data["engine"] if "engine" in data else "google"
    import modules.websearch.script as websearch

    if engine == "duckduckgo":
        results = websearch.search_duckduckgo(query)
    else:
        results = websearch.search_google(query)

    return jsonify({"results": results[0], "links": results[1]})


if args.share:
    from flask_cloudflared import _run_cloudflared
    import inspect

    sig = inspect.signature(_run_cloudflared)
    sum = sum(
        1
        for param in sig.parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD
    )
    if sum > 1:
        metrics_port = randint(8100, 9000)
        cloudflare = _run_cloudflared(port, metrics_port)
    else:
        cloudflare = _run_cloudflared(port)
    print(f"{Fore.GREEN}{Style.NORMAL}Running on: {cloudflare}{Style.RESET_ALL}")

ignore_auth.append(tts_play_sample)
ignore_auth.append(result_feed)
app.run(host=host, port=port)
