"""
Classify module for SillyTavern Extras

Authors:
    - Tony Ribeiro (https://github.com/Tony-sama)
    - Cohee (https://github.com/Cohee1207)

Provides classification features for text

References:
    - https://huggingface.co/tasks/text-classification
"""

from transformers import pipeline

DEBUG_PREFIX = "<Classify module>"

# Models init

text_emotion_pipe = None

def init_text_emotion_classifier(model_name: str, device: str, torch_dtype: str) -> None:
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
