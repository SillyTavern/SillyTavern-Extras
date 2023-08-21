"""
Classify module for SillyTavern Extras

Authors:
    - Tony Ribeiro (https://github.com/Tony-sama)

Provides classification features for text

References:
"""

import torch
from transformers import pipeline

DEBUG_PREFIX = "<Classify module>"

# Models init
cuda_device = "cuda:0"# if not args.cuda_device else args.cuda_device
device_string = cuda_device if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)
torch_dtype = torch.float32 if device_string != cuda_device  else torch.float16

text_emotion_pipe = None

def init_text_emotion_classifier(model_name: str) -> None:
    global text_emotion_pipe

    print(DEBUG_PREFIX,"Initializing text classification pipeline with model",model_name)
    text_emotion_pipe = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            device=device,
            torch_dtype=torch_dtype,
        )


def classify_text_emotion(text: str) -> list:
    output = text_emotion_pipe(
        text,
        truncation=True,
        max_length=text_emotion_pipe.model.config.max_position_embeddings,
    )[0]
    return sorted(output, key=lambda x: x["score"], reverse=True)