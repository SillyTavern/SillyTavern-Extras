from flask import Flask, jsonify, request, render_template_string, abort
import markdown
import argparse
from transformers import AutoTokenizer, AutoProcessor, pipeline
from transformers import BlipForConditionalGeneration, BartForConditionalGeneration
from transformers import AutoModelForTokenClassification, TokenClassificationPipeline
from transformers.pipelines import AggregationStrategy
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import unicodedata
import torch
import time
from PIL import Image
import base64
from io import BytesIO
import numpy as np
from diffusers import StableDiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler


# Constants
# Also try: 'Qiliang/bart-large-cnn-samsum-ElectrifAi_v10'
DEFAULT_SUMMARIZATION_MODEL = 'Qiliang/bart-large-cnn-samsum-ChatGPT_v3'
DEFAULT_CLASSIFICATION_MODEL = 'bhadresh-savani/distilbert-base-uncased-emotion'
DEFAULT_CAPTIONING_MODEL = 'Salesforce/blip-image-captioning-base'
DEFAULT_KEYPHRASE_MODEL = 'ml6team/keyphrase-extraction-distilbert-inspec'
DEFAULT_PROMPT_MODEL = 'FredZhang7/anime-anything-promptgen-v2'
DEFAULT_SD_MODEL = "ckpt/anything-v4.5-vae-swapped"
DEFAULT_SUMMARIZE_PARAMS = {
    'temperature': 1.0,
    'repetition_penalty': 1.0,
    'max_length': 500,
    'min_length': 200,
    'length_penalty': 1.5,
    'bad_words': ["\n", '"', "*", "[", "]", "{", "}", ":", "(", ")", "<", ">"]
}

# Script arguments
parser = argparse.ArgumentParser(
    prog='TavernAI Extras', description='Web API for transformers models')
parser.add_argument('--port', type=int,
                    help="Specify the port on which the application is hosted")
parser.add_argument('--listen', action='store_true',
                    help="Hosts the app on the local network")
parser.add_argument('--share', action='store_true',
                    help="Shares the app on CloudFlare tunnel")
parser.add_argument('--cpu', action='store_true',
                    help="Runs the models on the CPU")
parser.add_argument('--summarization-model',
                    help="Load a custom BART summarization model")
parser.add_argument('--classification-model',
                    help="Load a custom BERT text classification model")
parser.add_argument('--captioning-model',
                    help="Load a custom BLIP captioning model")
parser.add_argument('--keyphrase-model',
                    help="Load a custom keyphrase extraction model")
parser.add_argument('--prompt-model',
                    help="Load a custom GPT-2 prompt generation model")
parser.add_argument('--sd-model',
                    help="Load a custom SD image generation model")
parser.add_argument('--sd-cpu',
                    help="Force the SD pipeline to run on the CPU")

args = parser.parse_args()

port = args.port if args.port else 5100
host = '0.0.0.0' if args.listen else 'localhost'
summarization_model = args.summarization_model if args.summarization_model else DEFAULT_SUMMARIZATION_MODEL
classification_model = args.classification_model if args.classification_model else DEFAULT_CLASSIFICATION_MODEL
captioning_model = args.captioning_model if args.captioning_model else DEFAULT_CAPTIONING_MODEL
keyphrase_model = args.keyphrase_model if args.keyphrase_model else DEFAULT_KEYPHRASE_MODEL
prompt_model = args.prompt_model if args.prompt_model else DEFAULT_PROMPT_MODEL
sd_model = args.sd_model if args.sd_model else DEFAULT_SD_MODEL

# Models init
device_string = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
device = torch.device(device_string)
torch_dtype = torch.float32 if device_string == "cpu" else torch.float16

print('Initializing BLIP image captioning model...')
blip_processor = AutoProcessor.from_pretrained(captioning_model)
blip = BlipForConditionalGeneration.from_pretrained(
    captioning_model, torch_dtype=torch_dtype).to(device)

print('Initializing BART text summarization model...')
bart_tokenizer = AutoTokenizer.from_pretrained(summarization_model)
bart = BartForConditionalGeneration.from_pretrained(
    summarization_model, torch_dtype=torch_dtype).to(device)

print('Initializing BERT sentiment classification model...')
bert_classifier = pipeline("text-classification", model=classification_model,
                           top_k=None, device=device, torch_dtype=torch_dtype)

print('Initializing keyword extractor...')


class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE
            if self.model.config.model_type == "roberta"
            else AggregationStrategy.FIRST,
        )
        return np.unique([result.get("word").strip() for result in results])


keyphrase_pipe = KeyphraseExtractionPipeline(keyphrase_model)

print('Initializing GPT prompt generator')
gpt_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
gpt_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
gpt_model = GPT2LMHeadModel.from_pretrained(
    'FredZhang7/anime-anything-promptgen-v2')
prompt_generator = pipeline(
    'text-generation', model=gpt_model, tokenizer=gpt_tokenizer)


print('Initializing Stable Diffusion pipeline')
sd_device_string = "cuda" if torch.cuda.is_available() and not args.sd_cpu else "cpu"
sd_device = torch.device(sd_device_string)
sd_torch_dtype = torch.float32 if sd_device_string == "cpu" else torch.float16
sd_pipe = StableDiffusionPipeline.from_pretrained(
    sd_model,
    custom_pipeline="lpw_stable_diffusion",
    torch_dtype=sd_torch_dtype,
).to(sd_device)
sd_pipe.safety_checker = lambda images, clip_input: (images, False)
sd_pipe.enable_attention_slicing()
# pipe.scheduler = KarrasVeScheduler.from_config(pipe.scheduler.config)
sd_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    sd_pipe.scheduler.config)

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
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024


# AI stuff
def classify_text(text: str) -> list[dict]:
    output = bert_classifier(text)[0]
    return sorted(output, key=lambda x: x['score'], reverse=True)


def caption_image(raw_image: Image, max_new_tokens: int = 20) -> str:
    inputs = blip_processor(raw_image.convert(
        'RGB'), return_tensors="pt").to(device, torch_dtype)
    outputs = blip.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
    return caption


def summarize(text: str, params: dict) -> str:
    # Tokenize input
    inputs = bart_tokenizer(text, return_tensors="pt")
    token_count = len(inputs[0])

    bad_words_ids = [
        bart_tokenizer(bad_word, add_special_tokens=True).input_ids
        for bad_word in params['bad_words']
    ]
    summary_ids = bart.generate(
        inputs["input_ids"],
        num_beams=2,
        min_length=min(token_count, int(params['min_length'])),
        max_length=max(token_count, int(params['max_length'])),
        repetition_penalty=float(params['repetition_penalty']),
        temperature=float(params['temperature']),
        length_penalty=float(params['length_penalty']),
        bad_words_ids=bad_words_ids,
    )
    summary = bart_tokenizer.batch_decode(
        summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    summary = normalize_string(summary)
    return summary


def normalize_string(input: str) -> str:
    output = " ".join(unicodedata.normalize("NFKC", input).strip().split())
    return output


def extract_keywords(text: str) -> list[str]:
    punctuation = '(){}[]\n\r<>'
    trans = str.maketrans(punctuation, ' '*len(punctuation))
    text = text.translate(trans)
    text = normalize_string(text)
    return list(keyphrase_pipe(text))


def generate_prompt(keywords: list[str], length: int = 100, num: int = 4) -> str:
    prompt = ', '.join(keywords)
    outs = prompt_generator(prompt, max_length=length, num_return_sequences=num, do_sample=True,
                            repetition_penalty=1.2, temperature=0.7, top_k=4, early_stopping=True)
    return [out['generated_text'] for out in outs]


def generate_image(input: str, steps: int = 30, scale: int = 6) -> Image:
    prompt = normalize_string(f'{prompt_prefix}{input}')
    print(prompt)

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
    with open('./README.md', 'r') as f:
        content = f.read()
    return render_template_string(markdown.markdown(content, extensions=['tables']))


@app.route('/api/caption', methods=['POST'])
def api_caption():
    data = request.get_json()

    if not 'image' in data or not isinstance(data['image'], str):
        abort(400, '"image" is required')

    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    caption = caption_image(image)
    return jsonify({'caption': caption})


@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    data = request.get_json()

    if not 'text' in data or not isinstance(data['text'], str):
        abort(400, '"text" is required')

    params = DEFAULT_SUMMARIZE_PARAMS.copy()

    if 'params' in data and isinstance(data['params'], dict):
        params.update(data['params'])

    summary = summarize(data['text'], params)[0]
    return jsonify({'summary': summary})


@app.route('/api/classify', methods=['POST'])
def api_classify():
    data = request.get_json()

    if not 'text' in data or not isinstance(data['text'], str):
        abort(400, '"text" is required')

    classification = classify_text(data['text'])
    return jsonify({'classification': classification})


@app.route('/api/keywords', methods=['POST'])
def api_keywords():
    data = request.get_json()

    if not 'text' in data or not isinstance(data['text'], str):
        abort(400, '"text" is required')

    keywords = extract_keywords(data['text'])
    return jsonify({'keywords': keywords})


@app.route('/api/prompt', methods=['POST'])
def api_prompt():
    data = request.get_json()

    if not 'text' in data or not isinstance(data['text'], str):
        abort(400, '"text" is required')

    keywords = extract_keywords(data['text'])

    if 'name' in data or isinstance(data['name'], str):
        keywords.insert(0, data['name'])

    prompts = generate_prompt(keywords)
    return jsonify({'prompts': prompts})


@app.route('/api/image', methods=['POST'])
def api_image():
    data = request.get_json()

    if not 'prompt' in data or not isinstance(data['prompt'], str):
        abort(400, '"prompt" is required')

    image = generate_image(data['prompt'])
    base64image = image_to_base64(image)
    return jsonify({'image': base64image})


if args.share:
    from flask_cloudflared import run_with_cloudflared
    run_with_cloudflared(app)

app.run(host=host, port=port)
