# Constants
DEFAULT_CUDA_DEVICE = "cuda:0"
# Also try: 'Qiliang/bart-large-cnn-samsum-ElectrifAi_v10'
DEFAULT_SUMMARIZATION_MODEL = "Qiliang/bart-large-cnn-samsum-ChatGPT_v3"
# Also try: 'joeddav/distilbert-base-uncased-go-emotions-student'
DEFAULT_CLASSIFICATION_MODEL = "nateraw/bert-base-uncased-emotion"
# Also try: 'Salesforce/blip-image-captioning-base'
DEFAULT_CAPTIONING_MODEL = "Salesforce/blip-image-captioning-large"
DEFAULT_SD_MODEL = "ckpt/anything-v4.5-vae-swapped"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_REMOTE_SD_HOST = "127.0.0.1"
DEFAULT_REMOTE_SD_PORT = 7860
DEFAULT_CHROMA_PORT = 8000
SILERO_SAMPLES_PATH = "tts_samples"
SILERO_SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog"
# ALL_MODULES = ['caption', 'summarize', 'classify', 'keywords', 'prompt', 'sd']
DEFAULT_SUMMARIZE_PARAMS = {
    "temperature": 1.0,
    "repetition_penalty": 1.0,
    "max_length": 500,
    "min_length": 200,
    "length_penalty": 1.5,
    "bad_words": [
        "\n",
        '"',
        "*",
        "[",
        "]",
        "{",
        "}",
        ":",
        "(",
        ")",
        "<",
        ">",
        "Â",
        "The text ends",
        "The story ends",
        "The text is",
        "The story is",
    ],
}

PROMPT_PREFIX = "best quality, absurdres, "
NEGATIVE_PROMPT = """lowres, bad anatomy, error body, error hair, error arm,
error hands, bad hands, error fingers, bad fingers, missing fingers
error legs, bad legs, multiple legs, missing legs, error lighting,
error shadow, error reflection, text, error, extra digit, fewer digits,
cropped, worst quality, low quality, normal quality, jpeg artifacts,
signature, watermark, username, blurry"""
