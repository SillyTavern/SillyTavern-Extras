# SillyTavern - Extras
## What is this
A set of APIs for various SillyTavern extensions.

**You need to run the latest version of my TavernAI fork. Grab it here: [Direct link to ZIP](https://github.com/Cohee1207/SillyTavern/archive/refs/heads/main.zip), [Git repository](https://github.com/Cohee1207/SillyTavern)**

All modules require at least 6 Gb of VRAM to run. With Stable Diffusion disabled, it will probably fit in 4 Gb.
Alternatively, everything could also be run on the CPU.

Try on Colab (will give you a link to Extras API):  <a target="_blank" href="https://colab.research.google.com/github/Cohee1207/SillyTavern/blob/main/colab/GPU.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Colab link:
https://colab.research.google.com/github/Cohee1207/SillyTavern/blob/main/colab/GPU.ipynb

## How to run
### :exclamation: **IMPORTANT!**
> Default **requirements.txt** contains only basic packages for text processing


> If you want to use the most advanced features (like Stable Diffusion, TTS), change that to **requirements-complete.txt** in commands below. See [Modules](#modules) section for more details.

> You must specify a list of module names to be run in the `--enable-modules` command (`caption` provided as an example). See [Modules](#modules) section.
### ☁️ Colab
* Open colab link
* Select desired "extra" options and start the cell
* Wait for it to finish
* Get an API URL link from colab output under the `### SillyTavern Extensions LINK ###` title
* Start SillyTavern with extensions support: set `enableExtensions` to `true` in [config.conf](https://github.com/Cohee1207/SillyTavern/blob/dev/config.conf)
* Navigate to SillyTavern extensions menu and put in an API URL and tap "Connect" to load the extensions

### What about mobile/Android/Termux? 🤔

There are some folks in the community having success running Extras on their phones via Ubuntu on Termux. This project wasn't made with mobile support in mind, so this guide is provided strictly for your information only: https://rentry.org/STAI-Termux#downloading-and-running-tai-extras

#### ❗ IMPORTANT!

I will not provide any support for running this on Android. Direct all your questions to the creator of this guide.

### 💻 Locally
#### Option 1 - Conda (recommended) 🐍

**PREREQUISITES**
* Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
* _(Important!) Read how to use Conda: https://conda.io/projects/conda/en/latest/user-guide/getting-started.html_
* Install git: https://git-scm.com/downloads

**EXECUTE THESE COMMANDS ONE BY ONE IN THE _CONDA COMMAND PROMPT_.**

**TYPE/PASTE EACH COMMAND INTO THE PROMPT, HIT ENTER AND WAIT FOR IT TO FINISH!**

* Before the first run, create an environment (let's call it `extras`):
```
conda create -n extras
```
* Now activate the newly created env
```
conda activate extras
```
* Install the required system packages
```
conda install pytorch=2.0.0 torchvision=0.15.0 torchaudio=2.0.0 pytorch-cuda=11.7 git -c pytorch -c nvidia
```
* Clone this repository
```
git clone https://github.com/Cohee1207/SillyTavern-extras
```
* Navigated to the freshly cloned repository
```
cd SillyTavern-extras
```
* Install the project requirements
```
pip install -r requirements.txt
```
* Run the Extensions API server
```
python server.py --enable-modules=caption,summarize,classify
```
* Copy the Extra's server API URL listed in the console window after it finishes loading up. On local installs, this defaults to `http://localhost:5100`.
* Open your SillyTavern [config.conf](https://github.com/Cohee1207/SillyTavern/blob/dev/config.conf) file (located in the base install folder), and look for a line "`const enableExtensions`". Make sure that line has "`= true`", and not "`= false`". 
* Start your SillyTavern server
* Open the Extensions panel (via the 'Stacked Blocks' icon at the top of the page), paste the API URL into the input box, and click "Connect" to connect to the Extras extension server.
* To run again, simply activate the environment and run these commands. Be sure to the additional options for server.py (see below) that your setup requires.
```
conda activate extras
python server.py
```

#### Option 2 - Vanilla 🍦
* Install Python 3.10: https://www.python.org/downloads/release/python-31010/
* Install git: https://git-scm.com/downloads
* Clone the repo:
```
git clone https://github.com/Cohee1207/SillyTavern-extras
cd SillyTavern-extras
```
* Run `python -m pip install -r requirements.txt`
* Run `python server.py --enable-modules=caption,summarize,classify`
* Get the API URL. Defaults to `http://localhost:5100` if you run locally.
* Start SillyTavern with extensions support: set `enableExtensions` to `true` in [config.conf](https://github.com/Cohee1207/SillyTavern/blob/dev/config.conf)
* Navigate to SillyTavern extensions menu and put in an API URL and tap "Connect" to load the extensions

## Modules

| Name        | Description                       | Included in default requirements.txt       |
| ----------- | --------------------------------- | ------ |
| `caption`   | Image captioning                  | ✔️ Yes        |
| `summarize` | Text summarization                | ✔️ Yes    |
| `classify`  | Text sentiment classification     | ✔️ Yes      |
| `keywords`  | Text key phrases extraction       | ✔️ Yes      |
| `prompt`    | SD prompt generation from text    | ✔️ Yes     |
| `sd`        | Stable Diffusion image generation | :x: No (✔️ remote)      |
| `tts`       | [Silero TTS server](https://github.com/ouoertheo/silero-api-server) | :x: No |
| `chromadb`  | Infinity context server           | :x: No |


## Additional options
| Flag                     | Description                                                            |
| ------------------------ | ---------------------------------------------------------------------- |
| `--enable-modules`       | **Required option**. Provide a list of enabled modules.<br>Expects a comma-separated list of module names. See [Modules](#modules)<br>Example: `--enable-modules=caption,sd` |
| `--port`                 | Specify the port on which the application is hosted. Default: **5100** |
| `--listen`               | Host the app on the local network                                      |
| `--share`                | Share the app on CloudFlare tunnel                                     |
| `--cpu`                  | Run the models on the CPU instead of CUDA                              |
| `--summarization-model`  | Load a custom summarization model.<br>Expects a HuggingFace model ID.<br>Default: [Qiliang/bart-large-cnn-samsum-ChatGPT_v3](https://huggingface.co/Qiliang/bart-large-cnn-samsum-ChatGPT_v3) |
| `--classification-model` | Load a custom sentiment classification model.<br>Expects a HuggingFace model ID.<br>Default (6 emotions): [nateraw/bert-base-uncased-emotion](https://huggingface.co/nateraw/bert-base-uncased-emotion)<br>Other solid option is (28 emotions): [joeddav/distilbert-base-uncased-go-emotions-student](https://huggingface.co/joeddav/distilbert-base-uncased-go-emotions-student)<br>For Chinese language: [touch20032003/xuyuan-trial-sentiment-bert-chinese](https://huggingface.co/touch20032003/xuyuan-trial-sentiment-bert-chinese) |
| `--captioning-model`     | Load a custom captioning model.<br>Expects a HuggingFace model ID.<br>Default: [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large) |
| `--keyphrase-model`      | Load a custom key phrase extraction model.<br>Expects a HuggingFace model ID.<br>Default: [ml6team/keyphrase-extraction-distilbert-inspec](https://huggingface.co/ml6team/keyphrase-extraction-distilbert-inspec) |
| `--prompt-model`         | Load a custom prompt generation model.<br>Expects a HuggingFace model ID.<br>Default: [FredZhang7/anime-anything-promptgen-v2](https://huggingface.co/FredZhang7/anime-anything-promptgen-v2) |
| `--embedding-model`      | Load a custom text embedding model.<br>Expects a HuggingFace model ID.<br>Default: [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) |
| `--sd-model`             | Load a custom Stable Diffusion image generation model.<br>Expects a HuggingFace model ID.<br>Default: [ckpt/anything-v4.5-vae-swapped](https://huggingface.co/ckpt/anything-v4.5-vae-swapped)<br>*Must have VAE pre-baked in PyTorch format or the output will look drab!* |
| `--sd-cpu`               | Force the Stable Diffusion generation pipeline to run on the CPU.<br>**SLOW!** |
| `--sd-remote`            | Use a remote SD backend.<br>**Supported APIs: [sd-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)**  |
| `--sd-remote-host`       | Specify the host of the remote SD backend<br>Default: **127.0.0.1** |
| `--sd-remote-port`       | Specify the port of the remote SD backend<br>Default: **7860** |
| `--sd-remote-ssl`        | Use SSL for the remote SD backend<br>Default: **False** |
| `--sd-remote-auth`       | Specify the `username:password` for the remote SD backend (if required) |

## API Endpoints
### Get active list
`GET /api/modules`
#### **Input**
None
#### **Output**
```
{"modules":["caption", "classify", "summarize"]}
```

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
> 2. List of categories defined by the summarization model
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
{ "prompt": "prompt to be generated", "sampler": "DDIM", "steps": 20, "scale": 6, "model": "model_name" }
```
#### **Output**
```
{ "image": "base64 encoded image" }
```
> **NOTES**
> 1. Only the "prompt" parameter is required
> 2. Both "sampler" and "model" parameters only work when using a remote SD backend

### Get available Stable Diffusion models
`GET /api/image/models`
#### **Output**
```
{ "models": [list of all available model names] }
```

### Get available Stable Diffusion samplers
`GET /api/image/samplers`
#### **Output**
```
{ "samplers": [list of all available sampler names] }
```

### Get currently loaded Stable Diffusion model
`GET /api/image/model`
#### **Output**
```
{ "model": "name of the current loaded model" }
```

### Load a Stable Diffusion model (remote)
`POST /api/image/model`
#### **Input**
```
{ "model": "name of the model to load" }
```
#### **Output**
```
{ "previous_model": "name of the previous model", "current_model": "name of the newly loaded model" }
```

### Generate TTS voice
`POST /api/tts/generate`
#### **Input**
```
{ "speaker": "speaker voice_id", "text": "text to narrate" }
```
#### **Output**
WAV audio file.

### Get TTS voices
`GET /api/tts/speakers`
#### **Output**
```
[
    {
        "name": "en_0",
        "preview_url": "http://127.0.0.1:5100/api/tts/sample/en_0",
        "voice_id": "en_0"
    }
]
```

### Get TTS voice sample
`GET /api/tts/sample/<voice_id>`
#### **Output**
WAV audio file.

### Add messages to chromadb
`POST /api/chromadb`
#### **Input**
```
{
    "chat_id": "chat1 - 2023-12-31",
    "messages": [
        {
            "id": "633a4bd1-8350-46b5-9ef2-f5d27acdecb7", 
            "date": 1684164339877,
            "role": "user",
            "content": "Hello, AI world!",
            "meta": "this is meta"
        },
        {
            "id": "8a2ed36b-c212-4a1b-84a3-0ffbe0896506", 
            "date": 1684164411759,
            "role": "assistant",
            "content": "Hello, Hooman!"
        },
    ] 
}
```
#### **Output**
```
{ "count": 2 }
```

### Query chromadb
`POST /api/chromadb/query`
#### **Input**
```
{
    "chat_id": "chat1 - 2023-12-31",
    "query": "Hello",
    "n_results": 2,
}
```
#### **Output**
```
[
    {
        "id": "633a4bd1-8350-46b5-9ef2-f5d27acdecb7", 
        "date": 1684164339877,
        "role": "user",
        "content": "Hello, AI world!",
        "distance": 0.31,
        "meta": "this is meta"
    },
    {
        "id": "8a2ed36b-c212-4a1b-84a3-0ffbe0896506", 
        "date": 1684164411759,
        "role": "assistant",
        "content": "Hello, Hooman!",
        "distance": 0.29
    },
]
```

### Delete the messages from chromadb
`POST /api/chromadb/purge`
#### **Input**
```
{ "chat_id": "chat1 - 2023-04-12" }
```
