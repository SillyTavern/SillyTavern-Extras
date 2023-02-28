from flask import Flask, jsonify, request, flash, abort
import argparse
from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers import BlipForConditionalGeneration, AutoProcessor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import unicodedata
import torch
import json
from PIL import Image
import base64
from io import BytesIO

# Constants
# Also try: 'Qiliang/bart-large-cnn-samsum-ElectrifAi_v10'
DEFAULT_BART = 'Qiliang/bart-large-cnn-samsum-ChatGPT_v3'
DEFAULT_BERT = 'bhadresh-savani/distilbert-base-uncased-emotion'
DEFAULT_BLIP = 'Salesforce/blip-image-captioning-base'
UPLOAD_FOLDER = './uploads'

# Script arguments
parser = argparse.ArgumentParser(
                    prog = 'TavernAI Extras',
                    description = 'Web API for transformers models')
parser.add_argument('--port', type=int, help="Specify the port on which the application is hosted")
parser.add_argument('--listen', action='store_true', help="Hosts the app on the local network")
parser.add_argument('--share', action='store_true', help="Shares the app on cloudflare tunnel")
parser.add_argument('--bart-model', help="Customize a BART model to be used by the app")
parser.add_argument('--bert-model', help="Customize a BERT model to be used by the app")
parser.add_argument('--blip-model', help="Customize a BLIP model to be used by the app")

args = parser.parse_args()

if args.port:
    port = args.port
else:
    port = 5100

if args.share:
    from flask_cloudflared import _run_cloudflared
    cloudflare = _run_cloudflared(port)

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
print('Initializing BLIP...')
blip_processor = AutoProcessor.from_pretrained(blip_model)
blip = BlipForConditionalGeneration.from_pretrained(blip_model, torch_dtype=torch.float32).to("cpu")

print('Initializing BART...')
bart_tokenizer = AutoTokenizer.from_pretrained(bart_model)
bart = BartForConditionalGeneration.from_pretrained(bart_model, torch_dtype=torch.float32).to("cpu")

print('Initializing BERT...')
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)
bert = AutoModelForSequenceClassification.from_pretrained(bert_model)

# Flask init
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# AI stuff
def caption_image(raw_image, max_new_tokens=20):
    inputs = blip_processor(raw_image.convert('RGB'), return_tensors="pt").to("cpu", torch.float32)
    outputs = blip.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
    return caption

@app.route('/', methods=['GET'])
def index():
    return 'I work OK'

@app.route('/api/caption', methods=['POST'])
def api_caption():
    data = request.get_json()
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    caption = caption_image(image)
    return jsonify({ 'caption': caption })

app.run(host=host, port=port)
