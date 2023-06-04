import io
import edge_tts
import asyncio


def get_voices():
    voices = asyncio.run(edge_tts.list_voices())
    return voices


async def _iterate_chunks(audio):
    async for chunk in audio.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]


async def _async_generator_to_list(async_gen):
    result = []
    async for item in async_gen:
        result.append(item)
    return result


def generate_audio(text: str, voice: str) -> bytes:
    audio = edge_tts.Communicate(text, voice)
    chunks = asyncio.run(_async_generator_to_list(_iterate_chunks(audio)))
    buffer = io.BytesIO()

    for chunk in chunks:
        buffer.write(chunk)
    
    return buffer.getvalue()
