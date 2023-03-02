# TavernAI - Extras
## What is this
A set of unofficial APIs for various [TavernAI](https://github.com/TavernAI/TavernAI) extensions.

All modules require at least 6 Gb of VRAM to run. With Stable Diffusion disabled, it will probably fit in 4 Gb.
Alternatively, everything could also be run on the CPU.

Try on Colab (runs KoboldAI backend and TavernAI Extras server alongside):  <a target="_blank" href="https://colab.research.google.com/github/SillyLossy/TavernAI-extras/blob/main/colab/GPU.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## How to run
* Install Python 3.10
* Run `pip install -r requirements.txt`
* Run `python server.py`
* Get the API URL. Defaults to `http://localhost:5100` if you run locally.
* Start TavernAI with extensions support: set `enableExtensions` to `true` in [config.conf](https://github.com/SillyLossy/TavernAI/blob/dev/config.conf)
* Navigate to TavernAI settings and put in an API URL and tap "Connect" to load the extensions

<img src="https://user-images.githubusercontent.com/18619528/222469130-84cf5784-7f0d-48b9-bf8d-3851f2c8cea0.png" style="width:500px">

## UI Extensions
| Name             | Description                      | Required [Modules](#modules) | Screenshot |
| ---------------- | ---------------------------------| ---------------------------- | ---------- |
| Image Captioning | Send a cute picture to your bot! | `caption`                    | <img src="https://user-images.githubusercontent.com/18619528/222471170-5c28faca-dd33-4479-a768-2acd92563c4c.png" style="height:200px"> |

## Modules

| Name        | Description                       |
| ----------- | --------------------------------- |
| `caption`   | Image captioning                  |
| `summarize` | Text summarization                |
| `classify`  | Text sentiment classification     |
| `keywords`  | Text key phrases extraction       |
| `prompt`    | SD prompt generation from text    |
| `sd`        | Stable Diffusion image generation |

## API Endpoints
### Get UI extensions list
`GET /api/extensions`
#### **Input**
None
#### **Output**
```
{"extensions":[{"metadata":{"css":"file.css","display_name":"human-friendly name","js":"file.js","requires":["module_id"]},"name":"extension_name"}]}
```

### Get UI extension JS script
`GET /api/script/<name>`
#### **Input**
Extension name in a route
#### **Output**
File content

### Get UI extension CSS stylesheet
`GET /api/style/<name>`
#### **Input**
Extension name in a route
#### **Output**
File content

### Image captioning
`POST /api/caption`
#### **Input**
```
{ "image": "base64 encoded image" }
```
#### **Output**
```
{ "caption": "caption of the posted image" }
```

### Text summarization
`POST /api/summarize`
#### **Input**
```
{ "text": "text to be summarize", "params": {} }
```
#### **Output**
```
{ "summary": "summarized text" }
```
#### Optional: `params` object for control over summarization:
| Name                  | Default value                                                 |
| --------------------- | ------------------------------------------------------------- |
| `temperature`         | 1.0                                                           |
| `repetition_penalty`  | 1.0                                                           |
| `max_length`          | 500                                                           |
| `min_length`          | 200                                                           |
| `length_penalty`      | 1.5                                                           |
| `bad_words`           | ["\n", '"', "*", "[", "]", "{", "}", ":", "(", ")", "<", ">"] |

### Text sentiment classification
`POST /api/classify`
#### **Input**
```
{ "text": "text to classify sentiment of" }
```
#### **Output**
```
{
    "classification": [
        {
            "label": "joy",
            "score": 1.0
        },
        {
            "label": "anger",
            "score": 0.7
        },
        {
            "label": "love",
            "score": 0.6
        },
        {
            "label": "sadness",
            "score": 0.5
        },
        {
            "label": "fear",
            "score": 0.4
        },
        {
            "label": "surprise",
            "score": 0.3
        }
    ]
}
```
> **NOTES**
> 1. Sorted by descending score order
> 2. Six fixed categories
> 3. Value range from 0.0 to 1.0

### Key phrase extraction
`POST /api/keywords`
#### **Input**
```
{ "text": "text to be scanned for key phrases" }
```
#### **Output**
```
{
    "keywords": [
        "array of",
        "extracted",
        "keywords",
    ]
}
```

### Stable Diffusion prompt generation
`POST /api/prompt`
#### **Input**
```
{ "name": "character name (optional)", "text": "textual summary of a character" }
```
#### **Output**
```
{ "prompts": [ "array of generated prompts" ] }
```

### Stable Diffusion image generation
`POST /api/image`
#### **Input**
```
{ "prompt": "prompt to be generated" }
```
#### **Output**
```
{ "image": "base64 encoded image" }
```

## Additional options
| Flag                     | Description                                                            |
| ------------------------ | ---------------------------------------------------------------------- |
| `--port`                 | Specify the port on which the application is hosted. Default: **5100** |
| `--listen`               | Hosts the app on the local network                                     |
| `--share`                | Shares the app on CloudFlare tunnel                                    |
| `--cpu`                  | Run the models on the CPU instead of CUDA                              |
| `--summarization-model`  | Load a custom BART summarization model.<br>Expects a HuggingFace model ID.<br>Default: [Qiliang/bart-large-cnn-samsum-ChatGPT_v3](https://huggingface.co/Qiliang/bart-large-cnn-samsum-ChatGPT_v3) |
| `--classification-model` | Load a custom sentiment classification model.<br>Expects a HuggingFace model ID.<br>Default: [bhadresh-savani/distilbert-base-uncased-emotion](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion) |
| `--captioning-model`     | Load a custom captioning model.<br>Expects a HuggingFace model ID.<br>Default: [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base) |
| `--keyphrase-model`      | Load a custom key phrase extraction model.<br>Expects a HuggingFace model ID.<br>Default: [ml6team/keyphrase-extraction-distilbert-inspec](https://huggingface.co/ml6team/keyphrase-extraction-distilbert-inspec) |
| `--prompt-model`         | Load a custom prompt generation model.<br>Expects a HuggingFace model ID.<br>Default: [FredZhang7/anime-anything-promptgen-v2](https://huggingface.co/FredZhang7/anime-anything-promptgen-v2) |
| `--sd-model`             | Load a custom Stable Diffusion image generation model.<br>Expects a HuggingFace model ID.<br>Default: [ckpt/anything-v4.5-vae-swapped](https://huggingface.co/ckpt/anything-v4.5-vae-swapped)<br>*Must have VAE pre-baked in PyTorch format or the output will look drab!* |
| `--sd-cpu`               | Forces the Stable Diffusion generation pipeline to run on the CPU.<br>**SLOW!** |
| `--enable-modules`       | Override a list of enabled modules. Runs with everything enabled by default.<br>Expects a comma-separated list of module names. See [Modules](#modules) |
