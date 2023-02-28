from flask import Flask, jsonify, request, render_template_string, abort
import markdown
import argparse
from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers import BlipForConditionalGeneration, AutoProcessor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import unicodedata
import torch
import time
from PIL import Image
import base64
from io import BytesIO

# Constants
# Also try: 'Qiliang/bart-large-cnn-samsum-ElectrifAi_v10'
DEFAULT_BART = 'Qiliang/bart-large-cnn-samsum-ChatGPT_v3'
DEFAULT_BERT = 'bhadresh-savani/distilbert-base-uncased-emotion'
DEFAULT_BLIP = 'Salesforce/blip-image-captioning-base'
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
parser.add_argument('--bart-model', help="Load a custom BART model")
parser.add_argument('--bert-model', help="Load a custom BERT model")
parser.add_argument('--blip-model', help="Load a custom BLIP model")

args = parser.parse_args()

if args.port:
    port = args.port
else:
    port = 5100

if args.listen:
    host = '0.0.0.0'
else:
    host = 'localhost'

if args.bart_model:
    bart_model = args.bart_model
else:
    bart_model = DEFAULT_BART

if args.bert_model:
    bert_model = args.bert_model
else:
    bert_model = DEFAULT_BERT

if args.blip_model:
    blip_model = args.blip_model
else:
    blip_model = DEFAULT_BLIP

# Models init

device_string = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
torch_device = torch.device(device_string)
torch_dtype = torch.float32 if device_string == "cpu" else torch.float16

print('Initializing BLIP...')
blip_processor = AutoProcessor.from_pretrained(blip_model)
blip = BlipForConditionalGeneration.from_pretrained(
    blip_model, torch_dtype=torch_dtype).to(torch_device)

print('Initializing BART...')
bart_tokenizer = AutoTokenizer.from_pretrained(bart_model)
bart = BartForConditionalGeneration.from_pretrained(
    bart_model, torch_dtype=torch_dtype).to(torch_device)

print('Initializing BERT...')
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)
bert = AutoModelForSequenceClassification.from_pretrained(
    bert_model, torch_dtype=torch_dtype).to(torch_device)

# Flask init
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024


# AI stuff


def caption_image(raw_image: Image, max_new_tokens: int = 20) -> str:
    inputs = blip_processor(raw_image.convert(
        'RGB'), return_tensors="pt").to(torch_device, torch_dtype)
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
    # Normalize string
    summary = " ".join(unicodedata.normalize("NFKC", summary).strip().split())
    return summary

# Request time measuring


@app.before_request
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

    summary = summarize(data['text'], params)
    return jsonify({'summary': summary})


if args.share:
    from flask_cloudflared import run_with_cloudflared
    run_with_cloudflared(app)

app.run(host=host, port=port)
