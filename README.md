# SillyTavern - Extras

# Recent news

* We're migrating SillyTavern - Extras to Python 3.11, some of the modules new will be incompatible with old Python 3.10 installs. To migrate using conda, please remove the old environment using `conda remove --name extras --all` and reinstall using the instructions below.

## What is this
A set of APIs for various SillyTavern extensions.

**You need to run the latest version of SillyTavern. Grab it here: [How to install](https://docs.sillytavern.app/installation/windows/), [Git repository](https://github.com/SillyTavern/SillyTavern)**

All modules, except for Stable Diffusion, run on the CPU by default. However, they can alternatively be configured to use CUDA (with `--cuda` command line option). When running all modules simultaneously, you can expect a usage of approximately 6 GB of RAM. Loading Stable Diffusion adds an additional couple of GB to the memory usage.

Try on Colab (will give you a link to Extras API):  <a target="_blank" href="https://colab.research.google.com/github/SillyTavern/SillyTavern/blob/release/colab/GPU.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Colab link:
https://colab.research.google.com/github/SillyTavern/SillyTavern/blob/release/colab/GPU.ipynb

Documentation:
https://docs.sillytavern.app/

## How to run
### :exclamation: **IMPORTANT!**
 Default **requirements.txt** contains only basic packages for text processing

If you want to use the most advanced features (like Stable Diffusion, TTS), change that to **requirements-complete.txt** in commands below. See [Modules](#modules) section for more details.

If you run on Apple Silicon (M1/M2), use the **requirements-silicon.txt** file instead.

### Getting an error when installing from requirements-complete.txt?

> ERROR: Could not build wheels for hnswlib, which is required to install pyproject.toml-based projects

Installing chromadb package requires one of the following:

1. Have Visual C++ build tools installed: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Installing hnswlib from conda: `conda install -c conda-forge hnswlib`

### Missing modules reported by SillyTavern extensions menu?

You must specify a list of module names to be run in the `--enable-modules` command (`caption` provided as an example). See [Modules](#modules) section.

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

We will NOT provide any support for running this on Android. Direct all your questions to the creator of this guide.

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
* Install Python 3.11
```
conda install python=3.11
```
* Install the required system packages
```
conda install git
```
* Clone this repository
```
git clone https://github.com/SillyTavern/SillyTavern-extras
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
* Open your SillyTavern config.conf file (located in the base install folder), and look for a line "`const enableExtensions`". Make sure that line has "`= true`", and not "`= false`".
* Start your SillyTavern server
* Open the Extensions panel (via the 'Stacked Blocks' icon at the top of the page), paste the API URL into the input box, and click "Connect" to connect to the Extras extension server.
* To run again, simply activate the environment and run these commands. Be sure to the additional options for server.py (see below) that your setup requires.
```
conda activate extras
python server.py
```

#### Option 2 - Vanilla 🍦
* Install Python 3.11: https://www.python.org/downloads/release/python-3114/
* Install git: https://git-scm.com/downloads
* Clone the repo:
```
git clone https://github.com/SillyTavern/SillyTavern-extras
cd SillyTavern-extras
```
* Run `python -m pip install -r requirements.txt`
* Run `python server.py --enable-modules=caption,summarize,classify`
* Get the API URL. Defaults to `http://localhost:5100` if you run locally.
* Start SillyTavern with extensions support: set `enableExtensions` to `true` in config.conf
* Navigate to SillyTavern extensions menu and put in an API URL and tap "Connect" to load the extensions

## Modules

| Name        | Description                       | Included in default requirements.txt       |
| ----------- | --------------------------------- | ------ |
| `caption`   | Image captioning                  | ✔️ Yes        |
| `summarize` | Text summarization                | ✔️ Yes    |
| `classify`  | Text sentiment classification     | ✔️ Yes      |
| `sd`        | Stable Diffusion image generation | :x: No (✔️ remote)      |
| `silero-tts`       | [Silero TTS server](https://github.com/ouoertheo/silero-api-server) | :x: No |
| `edge-tts` | [Microsoft Edge TTS client](https://github.com/rany2/edge-tts) | ✔️ Yes |
| `chromadb`  | Infinity context server           | :x: No |


## Additional options
| Flag                     | Description                                                            |
| ------------------------ | ---------------------------------------------------------------------- |
| `--enable-modules`       | **Required option**. Provide a list of enabled modules.<br>Expects a comma-separated list of module names. See [Modules](#modules)<br>Example: `--enable-modules=caption,sd` |
| `--port`                 | Specify the port on which the application is hosted. Default: **5100** |
| `--listen`               | Host the app on the local network                                      |
| `--share`                | Share the app on CloudFlare tunnel                                     |
| `--secure`               | Adds API key authentication requirements. Highly recommended when paired with share! |
| `--cpu`                  | Run the models on the CPU instead of CUDA. Enabled by default. |
| `--mps` or `--m1`        | Run the models on Apple Silicon. Only for M1 and M2 processors. |
| `--cuda`                 | Uses CUDA (GPU+VRAM) to run modules if it is available. Otherwise, falls back to using CPU. |
| `--cuda-device`          | Specifies a CUDA device to use. Defaults to `cuda:0` (first available GPU). |
| `--summarization-model`  | Load a custom summarization model.<br>Expects a HuggingFace model ID.<br>Default: [Qiliang/bart-large-cnn-samsum-ChatGPT_v3](https://huggingface.co/Qiliang/bart-large-cnn-samsum-ChatGPT_v3) |
| `--classification-model` | Load a custom sentiment classification model.<br>Expects a HuggingFace model ID.<br>Default (6 emotions): [nateraw/bert-base-uncased-emotion](https://huggingface.co/nateraw/bert-base-uncased-emotion)<br>Other solid option is (28 emotions): [joeddav/distilbert-base-uncased-go-emotions-student](https://huggingface.co/joeddav/distilbert-base-uncased-go-emotions-student)<br>For Chinese language: [touch20032003/xuyuan-trial-sentiment-bert-chinese](https://huggingface.co/touch20032003/xuyuan-trial-sentiment-bert-chinese) |
| `--captioning-model`     | Load a custom captioning model.<br>Expects a HuggingFace model ID.<br>Default: [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large) |
| `--embedding-model`      | Load a custom text embedding model.<br>Expects a HuggingFace model ID.<br>Default: [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) |
| `--chroma-host`          | Specifies a host IP for a remote ChromaDB server. |
| `--chroma-port`          | Specifies an HTTP port for a remote ChromaDB server.<br>Default: `8000` |
| `--sd-model`             | Load a custom Stable Diffusion image generation model.<br>Expects a HuggingFace model ID.<br>Default: [ckpt/anything-v4.5-vae-swapped](https://huggingface.co/ckpt/anything-v4.5-vae-swapped)<br>*Must have VAE pre-baked in PyTorch format or the output will look drab!* |
| `--sd-cpu`               | Force the Stable Diffusion generation pipeline to run on the CPU.<br>**SLOW!** |
| `--sd-remote`            | Use a remote SD backend.<br>**Supported APIs: [sd-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)**  |
| `--sd-remote-host`       | Specify the host of the remote SD backend<br>Default: **127.0.0.1** |
| `--sd-remote-port`       | Specify the port of the remote SD backend<br>Default: **7860** |
| `--sd-remote-ssl`        | Use SSL for the remote SD backend<br>Default: **False** |
| `--sd-remote-auth`       | Specify the `username:password` for the remote SD backend (if required) |

## ChromaDB
ChromaDB is a blazing fast and open source database that is used for long-term memory when chatting with characters. It can be run in-memory or on a local server on your LAN.

NOTE: You should **NOT** run ChromaDB on a cloud server. There are no methods for authentication (yet), so unless you want to expose an unauthenticated ChromaDB to the world, run this on a local server in your LAN.

### In-memory setup

Run the extras server with the `chromadb` module enabled (recommended).

### Remote setup

Use this if you want to use ChromaDB with docker or host it remotely. If you don't know what that means and only want to use ChromaDB with ST on your local device, use the 'in-memory' instructions instead.

Prerequisites: Docker, Docker compose (make sure you're running in rootless mode with the systemd service enabled if on Linux).

Steps:

1. Run `git clone https://github.com/chroma-core/chroma chromadb` and `cd chromadb`
2. Run `docker-compose up -d --build` to build ChromaDB. This may take a long time depending on your system
3. Once the build process is finished, ChromaDB should be running in the background. You can check with the command `docker ps`
4. On your client machine, specify your local server ip in the `--chroma-host` argument (ex. `--chroma-host=192.168.1.10`)


If you are running ChromaDB on the same machine as SillyTavern, you will have to change the port of one of the services. To do this for ChromaDB:

1. Run `docker ps` to get the container ID and then `docker container stop <container ID>`
2. Enter the ChromaDB git repository `cd chromadb`
3. Open `docker-compose.yml` and look for the line starting with `uvicorn chromadb.app:app`
4. Change the `--port` argument to whatever port you want.
5. Look for the `ports` category and change the occurrences of `8000` to whatever port you chose in step 4.
6. Save and exit. Then run `docker-compose up --detach`
7. On your client machine, make sure to specity the `--chroma-port` argument (ex. `--chroma-port=<your-port-here>`) along with the `--chroma-host` argument.

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

### Generate Silero TTS voice
`POST /api/tts/generate`
#### **Input**
```
{ "speaker": "speaker voice_id", "text": "text to narrate" }
```
#### **Output**
WAV audio file.

### Get Silero TTS voices
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

### Get Silero TTS voice sample
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

### Get a list of Edge TTS voices
`GET /api/edge-tts/list`
#### **Output**
```
[{'Name': 'Microsoft Server Speech Text to Speech Voice (af-ZA, AdriNeural)', 'ShortName': 'af-ZA-AdriNeural', 'Gender': 'Female', 'Locale': 'af-ZA', 'SuggestedCodec': 'audio-24khz-48kbitrate-mono-mp3', 'FriendlyName': 'Microsoft Adri Online (Natural) - Afrikaans (South Africa)', 'Status': 'GA', 'VoiceTag': {'ContentCategories': ['General'], 'VoicePersonalities': ['Friendly', 'Positive']}}]
```

### Generate Edge TTS voice
`POST /api/edge-tts/generate`
#### **Input**
```
{ "text": "Text to narrate", "voice": "af-ZA-AdriNeural", "rate": 0 }
```
#### **Output**
MP3 audio file.
