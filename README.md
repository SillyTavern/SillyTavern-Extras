# TavernAI - Extras
## What is this
A set of unofficial APIs for various [TavernAI](https://github.com/TavernAI/TavernAI) extensions

## How to run
* Install Python 3.10
* Run `pip install -r requirements.txt`
* Run `python server.py`

## Included functionality
### BLIP model for image captioning
`POST /api/caption`
#### **Input**
```
{ "image": "base64 encoded image" }
```
#### **Output**
```
{ "caption": "caption of the posted image" }
```

### BART model for text summarization
Not implemented yet

### BERT model for text classification
Not implemented yet

## Additional options
| Flag           | Description                                                          |
| -------------- | -------------------------------------------------------------------- |
| `--port`       | Specify the port on which the application is hosted. Default: *5100* |
| `--listen`     | Hosts the app on the local network                                   |
| `--share`      | Shares the app on CloudFlare tunnel                                  |
| `--bart-model` | Load a custom BART model.<br>Expects a HuggingFace model ID.<br>Default: [Qiliang/bart-large-cnn-samsum-ChatGPT_v3](https://huggingface.co/Qiliang/bart-large-cnn-samsum-ChatGPT_v3) |
| `--bert-model` | Load a custom BERT model.<br>Expects a HuggingFace model ID.<br>Default: [bhadresh-savani/distilbert-base-uncased-emotion](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion) |
| `--blip-model` | Load a custom BLIP model.<br>Expects a HuggingFace model Id.<br>Default: [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base) |