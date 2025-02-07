import multiprocessing

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

import base64
import io
import time
from concurrent.futures import ProcessPoolExecutor

from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import TextStreamer
from unsloth import FastVisionModel

import asyncio

app = FastAPI(title="OpenAI-compatible API")


class DataModel(BaseModel):
    prompt: str = ""
    image: str = ""


g_trained_model = None
g_trained_tokenizer = None


def init_f():
    global g_trained_model, g_trained_tokenizer
    model_path = "/model"
    g_trained_model, g_trained_tokenizer = FastVisionModel.from_pretrained(
        model_path,
        use_gradient_checkpointing="unsloth"
    )


def generate_answer(image, instruction, model, tokenizer, max_new_tokens=1024, temperature=1.5, min_p=0.1):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to('cuda')

    class CapturingTextStreamer(TextStreamer):
        def __init__(self, tokenizer, skip_prompt=True):
            super().__init__(tokenizer, skip_prompt)
            self.generated_text = []  # Initialize an empty string to store the generated text

        def on_finalized_text(self, text: str, stream_end: bool = False):
            # print(text, flush=True, end="" if not stream_end else None)
            self.generated_text.append(text)  # Append the finalized text to the captured output

        def get_generated_text(self):
            return "".join(self.generated_text)

    text_streamer = CapturingTextStreamer(tokenizer, skip_prompt=True)
    # text_streamer = TextStreamer(tokenizer, skip_prompt = True)

    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=max_new_tokens,
                       use_cache=True, temperature=temperature, min_p=min_p)

    return text_streamer.get_generated_text()


def process_image(payload: DataModel) -> str:
    return generate_answer(to_image(payload.image), payload.prompt, g_trained_model, g_trained_tokenizer)


def to_image(base64_str: str):
    jpeg_image = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, 'utf-8'))))
    buf = io.BytesIO()
    jpeg_image.save(buf, format='PNG')
    return Image.open(buf)


if __name__ == "__main__":
    executor = ProcessPoolExecutor(max_workers=1, initializer=init_f)


@app.post("/v1/chat/completions")
async def chat_completions(payload: DataModel):
    return await asyncio.get_running_loop().run_in_executor(executor, process_image, payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

