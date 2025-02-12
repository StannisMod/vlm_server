import multiprocessing
import os

import click
import torch.cuda

from indexed_pool_executor import IndexedProcessPoolExecutor

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

import base64
import io
from concurrent.futures import ProcessPoolExecutor

from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel

from transformers import TextStreamer

import asyncio

app = FastAPI(title="OpenAI-compatible API")


class DataModel(BaseModel):
    prompt: str = ""
    image: str = ""


g_trained_model = None
g_trained_tokenizer = None
g_device = None


def init_f(i: int,
           cuda_devices: list[int],
           workers_per_gpu: int,
           max_new_tokens: int,
           temperature: float,
           min_p: float):
    global g_trained_model, g_trained_tokenizer, g_device
    global g_max_new_tokens, g_temperature, g_min_p

    g_max_new_tokens = max_new_tokens
    g_temperature = temperature
    g_min_p = min_p

    device_index = i // workers_per_gpu
    g_device = f'cuda:{cuda_devices[device_index]}'

    os.environ['CUDA_VISIBLE_DEVICES'] = f'{device_index}' #  ','.join(list(map(str, range(torch.cuda.device_count()))))

    from unsloth import FastVisionModel

    model_path = "/model"
    g_trained_model, g_trained_tokenizer = FastVisionModel.from_pretrained(
        model_path,
        use_gradient_checkpointing="unsloth",
        device_map=g_device
    )
    g_trained_model = g_trained_model.to(g_device)
    FastVisionModel.for_inference(g_trained_model)


def generate_answer(image, instruction, model, tokenizer, max_new_tokens=1024, temperature=1.5, min_p=0.1):
    global g_device

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
    ).to(g_device)

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

    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=max_new_tokens,
                       use_cache=True, temperature=temperature, min_p=min_p)

    return text_streamer.get_generated_text()


def process_image(payload: DataModel) -> str:
    return generate_answer(to_image(payload.image), payload.prompt, g_trained_model, g_trained_tokenizer,
                           g_max_new_tokens, g_temperature, g_min_p)


def to_image(base64_str: str):
    jpeg_image = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, 'utf-8'))))
    buf = io.BytesIO()
    jpeg_image.save(buf, format='PNG')
    return Image.open(buf)


# host globals
g_cuda_devices: list[str] | None = None
g_workers_per_gpu: int | None = None
g_max_new_tokens: int | None = None
g_temperature: float | None = None
g_min_p: float | None = None
g_executor = None


@click.command()
@click.option('--cuda_devices',
              default=','.join(list(map(str, range(torch.cuda.device_count())))),
              help='Indices of GPUs that server should use. Workers should be spread over it')
@click.option('--workers_per_gpu', default=2, help='Number of workers that should run simultaneously on one GPU')
@click.option('--max_new_tokens', default=1024, help='Model parameter broadcast to all workers')
@click.option('--temperature', default=1.5, help='Model parameter broadcast to all workers')
@click.option('--min_p', default=0.1, help='Model parameter broadcast to all workers')
def main(cuda_devices: str | None,
         workers_per_gpu: str | None,
         max_new_tokens: str | None,
         temperature: str | None,
         min_p: str | None):
    global g_executor

    cuda_devices = '2,5'

    cuda_devices = cuda_devices.split(',')
    workers_per_gpu = int(workers_per_gpu)
    max_new_tokens = int(max_new_tokens)
    temperature = float(temperature)
    min_p = float(min_p)

    g_executor = IndexedProcessPoolExecutor(max_workers=len(cuda_devices) * workers_per_gpu, initializer=init_f,
                                            initargs=(cuda_devices, workers_per_gpu, max_new_tokens, temperature, min_p))

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


@app.post("/v1/chat/completions")
async def chat_completions(payload: DataModel):
    return await asyncio.get_running_loop().run_in_executor(g_executor, process_image, payload)


if __name__ == "__main__":
    main()
