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
`POST /api/summarize`
#### **Input**
```
{ "text": "text to be summarize", "params": {} }
```
#### **Output**
```
{ "summary": "summarized text" }
```
#### Optional `params` object for control over summarization:
| Name                  | Default value                                                 |
| --------------------- | ------------------------------------------------------------- |
| `temperature`         | 1.0                                                           |
| `repetition_penalty`  | 1.0                                                           |
| `max_length`          | 500                                                           |
| `min_length`          | 200                                                           |
| `length_penalty`      | 1.5                                                           |
| `bad_words`           | ["\n", '"', "*", "[", "]", "{", "}", ":", "(", ")", "<", ">"] |

### BERT model for text classification
Not implemented yet

## Additional options
| Flag           | Description                                                          |
| -------------- | -------------------------------------------------------------------- |
| `--port`       | Specify the port on which the application is hosted. Default: *5100* |
| `--listen`     | Hosts the app on the local network                                   |
| `--share`      | Shares the app on CloudFlare tunnel                                  |
| `--cpu`        | Run the models on the CPU instead of CUDA                            |
| `--bart-model` | Load a custom BART model.<br>Expects a HuggingFace model ID.<br>Default: [Qiliang/bart-large-cnn-samsum-ChatGPT_v3](https://huggingface.co/Qiliang/bart-large-cnn-samsum-ChatGPT_v3) |
| `--bert-model` | Load a custom BERT model.<br>Expects a HuggingFace model ID.<br>Default: [bhadresh-savani/distilbert-base-uncased-emotion](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion) |
| `--blip-model` | Load a custom BLIP model.<br>Expects a HuggingFace model Id.<br>Default: [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base) |