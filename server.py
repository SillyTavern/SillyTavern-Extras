from functools import wraps
from flask import Flask, jsonify, request, render_template_string, abort
from flask_cors import CORS
import markdown
import argparse
from transformers import AutoTokenizer, AutoProcessor, pipeline
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import BlipForConditionalGeneration, GPT2Tokenizer
import unicodedata
import torch
import time
from PIL import Image
import base64
from io import BytesIO
from random import randint
import webuiapi
from colorama import Fore, Style, init as colorama_init

colorama_init()


# Constants
# Also try: 'Qiliang/bart-large-cnn-samsum-ElectrifAi_v10'
DEFAULT_SUMMARIZATION_MODEL = 'Qiliang/bart-large-cnn-samsum-ChatGPT_v3'
# Also try: 'joeddav/distilbert-base-uncased-go-emotions-student'
DEFAULT_CLASSIFICATION_MODEL = 'bhadresh-savani/distilbert-base-uncased-emotion'
# Also try: 'Salesforce/blip-image-captioning-base'
DEFAULT_CAPTIONING_MODEL = 'Salesforce/blip-image-captioning-large'
DEFAULT_KEYPHRASE_MODEL = 'ml6team/keyphrase-extraction-distilbert-inspec'
DEFAULT_PROMPT_MODEL = 'FredZhang7/anime-anything-promptgen-v2'
DEFAULT_SD_MODEL = "ckpt/anything-v4.5-vae-swapped"
DEFAULT_REMOTE_SD_HOST = "127.0.0.1"
DEFAULT_REMOTE_SD_PORT = 7860
#ALL_MODULES = ['caption', 'summarize', 'classify', 'keywords', 'prompt', 'sd']
DEFAULT_SUMMARIZE_PARAMS = {
    'temperature': 1.0,
    'repetition_penalty': 1.0,
    'max_length': 500,
    'min_length': 200,
    'length_penalty': 1.5,
    'bad_words': ["\n", '"', "*", "[", "]", "{", "}", ":", "(", ")", "<", ">", "Â"]
}

class SplitArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.replace('"', '').replace("'", '').split(','))

# Script arguments
parser = argparse.ArgumentParser(
    prog='TavernAI Extras', description='Web API for transformers models')
parser.add_argument('--port', type=int,
                    help="Specify the port on which the application is hosted")
parser.add_argument('--listen', action='store_true',
                    help="Host the app on the local network")
parser.add_argument('--share', action='store_true',
                    help="Share the app on CloudFlare tunnel")
parser.add_argument('--cpu', action='store_true',
                    help="Run the models on the CPU")
parser.add_argument('--summarization-model',
                    help="Load a custom summarization model")
parser.add_argument('--classification-model',
                    help="Load a custom text classification model")
parser.add_argument('--captioning-model',
                    help="Load a custom captioning model")
parser.add_argument('--keyphrase-model',
                    help="Load a custom keyphrase extraction model")
parser.add_argument('--prompt-model',
                    help="Load a custom prompt generation model")

sd_group = parser.add_mutually_exclusive_group()
local_sd = sd_group.add_argument_group('sd-local')
local_sd.add_argument('--sd-model',
                    help="Load a custom SD image generation model")
local_sd.add_argument('--sd-cpu',
                    help="Force the SD pipeline to run on the CPU")
remote_sd = sd_group.add_argument_group('sd-remote')
remote_sd.add_argument('--sd-remote', action='store_true',
                    help="Use a remote backend for SD")
remote_sd.add_argument('--sd-remote-host', type=str,
                    help="Specify the host of the remote SD backend")
remote_sd.add_argument('--sd-remote-port', type=int,
                    help="Specify the port of the remote SD backend")
remote_sd.add_argument('--sd-remote-ssl', action='store_true',
                    help="Use SSL for the remote SD backend")
remote_sd.add_argument('--sd-remote-auth', type=str,
                    help="Specify the username:password for the remote SD backend (if required)")

parser.add_argument('--enable-modules', action=SplitArgs, default=[],
                    help="Override a list of enabled modules")

args = parser.parse_args()

port = args.port if args.port else 5100
host = '0.0.0.0' if args.listen else 'localhost'
summarization_model = args.summarization_model if args.summarization_model else DEFAULT_SUMMARIZATION_MODEL
classification_model = args.classification_model if args.classification_model else DEFAULT_CLASSIFICATION_MODEL
captioning_model = args.captioning_model if args.captioning_model else DEFAULT_CAPTIONING_MODEL
keyphrase_model = args.keyphrase_model if args.keyphrase_model else DEFAULT_KEYPHRASE_MODEL
prompt_model = args.prompt_model if args.prompt_model else DEFAULT_PROMPT_MODEL

sd_use_remote = False if args.sd_model else True
sd_model = args.sd_model if args.sd_model else DEFAULT_SD_MODEL
sd_remote_host = args.sd_remote_host if args.sd_remote_host else DEFAULT_REMOTE_SD_HOST
sd_remote_port = args.sd_remote_port if args.sd_remote_port else DEFAULT_REMOTE_SD_PORT
sd_remote_ssl = args.sd_remote_ssl
sd_remote_auth = args.sd_remote_auth

modules = args.enable_modules if args.enable_modules and len(args.enable_modules) > 0 else []

if len(modules) == 0:
    print(f'{Fore.RED}{Style.BRIGHT}You did not select any modules to run! Choose them by adding an --enable-modules option')
    print(f'Example: --enable-modules=caption,summarize{Style.RESET_ALL}')

# Models init
device_string = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
device = torch.device(device_string)
torch_dtype = torch.float32 if device_string == "cpu" else torch.float16

if 'caption' in modules:
    print('Initializing an image captioning model...')
    captioning_processor = AutoProcessor.from_pretrained(captioning_model)
    if 'blip' in captioning_model:
        captioning_transformer = BlipForConditionalGeneration.from_pretrained(captioning_model, torch_dtype=torch_dtype).to(device)
    else:
        captioning_transformer = AutoModelForCausalLM.from_pretrained(captioning_model, torch_dtype=torch_dtype).to(device)

if 'summarize' in modules:
    print('Initializing a text summarization model...')
    summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model)
    summarization_transformer = AutoModelForSeq2SeqLM.from_pretrained(summarization_model, torch_dtype=torch_dtype).to(device)

if 'classify' in modules:
    print('Initializing a sentiment classification pipeline...')
    classification_pipe = pipeline("text-classification", model=classification_model, top_k=None, device=device, torch_dtype=torch_dtype)

if 'keywords' in modules:
    print('Initializing a keyword extraction pipeline...')
    import pipelines as pipelines
    keyphrase_pipe = pipelines.KeyphraseExtractionPipeline(keyphrase_model)

if 'prompt' in modules:
    print('Initializing a prompt generator')
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    gpt_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    gpt_model = AutoModelForCausalLM.from_pretrained(prompt_model)
    prompt_generator = pipeline('text-generation', model=gpt_model, tokenizer=gpt_tokenizer)

if 'sd' in modules and not sd_use_remote:
    from diffusers import StableDiffusionPipeline
    from diffusers import EulerAncestralDiscreteScheduler
    print('Initializing Stable Diffusion pipeline')
    sd_device_string = "cuda" if torch.cuda.is_available() and not args.sd_cpu else "cpu"
    sd_device = torch.device(sd_device_string)
    sd_torch_dtype = torch.float32 if sd_device_string == "cpu" else torch.float16
    sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model, custom_pipeline="lpw_stable_diffusion", torch_dtype=sd_torch_dtype).to(sd_device)
    sd_pipe.safety_checker = lambda images, clip_input: (images, False)
    sd_pipe.enable_attention_slicing()
    # pipe.scheduler = KarrasVeScheduler.from_config(pipe.scheduler.config)
    sd_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(sd_pipe.scheduler.config)
elif 'sd' in modules and sd_use_remote:
    print('Initializing Stable Diffusion connection')
    try:
        sd_remote = webuiapi.WebUIApi(host=sd_remote_host, port=sd_remote_port, use_https=sd_remote_ssl)
        if sd_remote_auth:
            username, password = sd_remote_auth.split(':')
            sd_remote.set_auth(username, password)
        sd_remote.util_wait_for_ready()
    except Exception as e:
        # remote sd from modules
        print(f"{Fore.RED}{Style.BRIGHT}Could not connect to remote SD backend at http{'s' if sd_remote_ssl else ''}://{sd_remote_host}:{sd_remote_port}! Disabling SD module...{Style.RESET_ALL}")
        modules.remove('sd')

prompt_prefix = "best quality, absurdres, "
neg_prompt = """lowres, bad anatomy, error body, error hair, error arm,
error hands, bad hands, error fingers, bad fingers, missing fingers
error legs, bad legs, multiple legs, missing legs, error lighting,
error shadow, error reflection, text, error, extra digit, fewer digits,
cropped, worst quality, low quality, normal quality, jpeg artifacts,
signature, watermark, username, blurry"""


# list of key phrases to be looking for in text (unused for now)
indicator_list = ['female', 'girl', 'male', 'boy', 'woman', 'man', 'hair', 'eyes', 'skin', 'wears',
                  'appearance', 'costume', 'clothes', 'body', 'tall', 'short', 'chubby', 'thin',
                  'expression', 'angry', 'sad', 'blush', 'smile', 'happy', 'depressed', 'long',
                  'cold', 'breasts', 'chest', 'tail', 'ears', 'fur', 'race', 'species', 'wearing',
                  'shoes', 'boots', 'shirt', 'panties', 'bra', 'skirt', 'dress', 'kimono', 'wings', 'horns',
                  'pants', 'shorts', 'leggins', 'sandals', 'hat', 'glasses', 'sweater', 'hoodie', 'sweatshirt']

# Flask init
app = Flask(__name__)
CORS(app) # allow cross-domain requests
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024


def require_module(name):
    def wrapper(fn):
        @wraps(fn)
        def decorated_view(*args, **kwargs):
            if name not in modules:
                abort(403, 'Module is disabled by config')
            return fn(*args, **kwargs)
        return decorated_view
    return wrapper


# AI stuff
def classify_text(text: str) -> list:
    output = classification_pipe(text)[0]
    return sorted(output, key=lambda x: x['score'], reverse=True)


def caption_image(raw_image: Image, max_new_tokens: int = 20) -> str:
    inputs = captioning_processor(raw_image.convert('RGB'), return_tensors="pt").to(device, torch_dtype)
    outputs = captioning_transformer.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = captioning_processor.decode(outputs[0], skip_special_tokens=True)
    return caption


def summarize(text: str, params: dict) -> str:
    # Tokenize input
    inputs = summarization_tokenizer(text, return_tensors="pt").to(device)
    token_count = len(inputs[0])

    bad_words_ids = [
        summarization_tokenizer(bad_word, add_special_tokens=False).input_ids
        for bad_word in params['bad_words']
    ]
    summary_ids = summarization_transformer.generate(
        inputs["input_ids"],
        num_beams=2,
        min_length=min(token_count, int(params['min_length'])),
        max_length=max(token_count, int(params['max_length'])),
        repetition_penalty=float(params['repetition_penalty']),
        temperature=float(params['temperature']),
        length_penalty=float(params['length_penalty']),
        bad_words_ids=bad_words_ids,
    )
    summary = summarization_tokenizer.batch_decode(
        summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    summary = normalize_string(summary)
    return summary


def normalize_string(input: str) -> str:
    output = " ".join(unicodedata.normalize("NFKC", input).strip().split())
    return output


def extract_keywords(text: str) -> list:
    punctuation = '(){}[]\n\r<>'
    trans = str.maketrans(punctuation, ' '*len(punctuation))
    text = text.translate(trans)
    text = normalize_string(text)
    return list(keyphrase_pipe(text))


def generate_prompt(keywords: list, length: int = 100, num: int = 4) -> str:
    prompt = ', '.join(keywords)
    outs = prompt_generator(prompt, max_length=length, num_return_sequences=num, do_sample=True,
                            repetition_penalty=1.2, temperature=0.7, top_k=4, early_stopping=True)
    return [out['generated_text'] for out in outs]


def generate_image(input: str, steps: int = 30, scale: int = 6, sampler: str = 'DDIM', model: str = None) -> Image:
    prompt = normalize_string(f'{prompt_prefix}{input}')
    print(prompt)

    if sd_use_remote:
        if model is not None and model != sd_remote.util_get_current_model():
            sd_remote.util_set_model(model, find_closest=False)
            sd_remote.util_wait_for_ready()

        image = sd_remote.txt2img(
            prompt=prompt,
            negative_prompt=neg_prompt,
            sampler_name=sampler,
            steps=steps,
            cfg_scale=scale,
        ).image
    else:
        image = sd_pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=steps,
            guidance_scale=scale,
        ).images[0]

    image.save("./debug.png")
    return image


def image_to_base64(image: Image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8") 
    return img_str


@app.before_request
# Request time measuring
def before_request():
    request.start_time = time.time()


@app.after_request
def after_request(response):
    duration = time.time() - request.start_time
    response.headers['X-Request-Duration'] = str(duration)
    return response


@app.route('/', methods=['GET'])
def index():
    with open('./README.md', 'r', encoding='utf8') as f:
        content = f.read()
    return render_template_string(markdown.markdown(content, extensions=['tables']))


@app.route('/api/extensions', methods=['GET'])
def get_extensions():
    extensions = dict({
        'extensions': [
            {
                'name': 'not-supported',
                'metadata': {
                        "display_name": """<span style="white-space:break-spaces;">Extensions serving using Extensions API is no longer supported. Please update the mod from: <a href="https://github.com/SillyLossy/TavernAI">https://github.com/SillyLossy/TavernAI</a></span>""",
                        "requires": [],
                        "assets": []
                }
            }
        ]
    })
    return jsonify(extensions)


@app.route('/api/caption', methods=['POST'])
@require_module('caption')
def api_caption():
    data = request.get_json()

    if 'image' not in data or not isinstance(data['image'], str):
        abort(400, '"image" is required')

    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    caption = caption_image(image)
    return jsonify({'caption': caption})


@app.route('/api/summarize', methods=['POST'])
@require_module('summarize')
def api_summarize():
    data = request.get_json()

    if 'text' not in data or not isinstance(data['text'], str):
        abort(400, '"text" is required')

    params = DEFAULT_SUMMARIZE_PARAMS.copy()

    if 'params' in data and isinstance(data['params'], dict):
        params.update(data['params'])

    summary = summarize(data['text'], params)
    return jsonify({'summary': summary})


@app.route('/api/classify', methods=['POST'])
@require_module('classify')
def api_classify():
    data = request.get_json()

    if 'text' not in data or not isinstance(data['text'], str):
        abort(400, '"text" is required')

    classification = classify_text(data['text'])
    return jsonify({'classification': classification})


@app.route('/api/classify/labels', methods=['GET'])
@require_module('classify')
def api_classify_labels():
    classification = classify_text('')
    labels = [x['label'] for x in classification]
    return jsonify({'labels': labels})


@app.route('/api/keywords', methods=['POST'])
@require_module('keywords')
def api_keywords():
    data = request.get_json()

    if 'text' not in data or not isinstance(data['text'], str):
        abort(400, '"text" is required')

    keywords = extract_keywords(data['text'])
    return jsonify({'keywords': keywords})


@app.route('/api/prompt', methods=['POST'])
@require_module('prompt')
def api_prompt():
    data = request.get_json()

    if 'text' not in data or not isinstance(data['text'], str):
        abort(400, '"text" is required')

    keywords = extract_keywords(data['text'])

    if 'name' in data and isinstance(data['name'], str):
        keywords.insert(0, data['name'])

    prompts = generate_prompt(keywords)
    return jsonify({'prompts': prompts})


@app.route('/api/image', methods=['POST'])
@require_module('sd')
def api_image():
    data = request.get_json()

    if 'prompt' not in data or not isinstance(data['prompt'], str):
        abort(400, '"prompt" is required')
    if 'steps' not in data or not isinstance(data['steps'], int):
        data['steps'] = 30
    if 'scale' not in data or not isinstance(data['scale'], int):
        data['scale'] = 6
    if 'sampler' not in data or not isinstance(data['sampler'], str):
        data['sampler'] = 'DDIM'
    if 'model' not in data or not isinstance(data['model'], str):
        data['model'] = None

    try:
        image = generate_image(data['prompt'], data['steps'], data['scale'], data['sampler'], data['model'])
        base64image = image_to_base64(image)
        return jsonify({'image': base64image})
    except RuntimeError as e:
        abort(400, str(e))

@app.route('/api/image/model', methods=['POST'])
@require_module('sd')
def api_image_model_set():
    data = request.get_json()

    if not sd_use_remote:
        abort(400, 'Changing model for local sd is not supported.')
    if 'model' not in data or not isinstance(data['model'], str):
        abort(400, '"model" is required')

    old_model = sd_remote.util_get_current_model()
    sd_remote.util_set_model(data['model'], find_closest=False)
    #sd_remote.util_set_model(data['model'])
    sd_remote.util_wait_for_ready()
    new_model = sd_remote.util_get_current_model()

    return jsonify({'previous_model': old_model, 'current_model': new_model})

@app.route('/api/image/model', methods=['GET'])
@require_module('sd')
def api_image_model_get():
    model = sd_model

    if sd_use_remote:
        model = sd_remote.util_get_current_model()

    return jsonify({'model': model})

@app.route('/api/image/models', methods=['GET'])
@require_module('sd')
def api_image_models():
    models = [sd_model]

    if sd_use_remote:
        models = sd_remote.util_get_model_names()
    
    return jsonify({'models': models})

@app.route('/api/image/samplers', methods=['GET'])
@require_module('sd')
def api_image_samplers():
    samplers = ['Euler a']
    
    if sd_use_remote:
        samplers = [sampler['name'] for sampler in sd_remote.get_samplers()]
    
    return jsonify({'samplers': samplers})

@app.route('/api/modules', methods=['GET'])
def get_modules():
    return jsonify({'modules': modules})


if args.share:
    from flask_cloudflared import _run_cloudflared
    import inspect
    sig = inspect.signature(_run_cloudflared)
    sum = sum(1 for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD)
    if sum > 1:
        metrics_port = randint(8100, 9000)
        cloudflare = _run_cloudflared(port, metrics_port)
    else:
        cloudflare = _run_cloudflared(port)
    print("Running on", cloudflare)

app.run(host=host, port=port)
