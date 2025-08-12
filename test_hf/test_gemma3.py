from transformers import pipeline
import torch

pipe = pipeline("text-generation", model="google/gemma-3-1b-it", device="cuda", torch_dtype=torch.bfloat16)

messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Write a poem on Hugging Face, the company"},]
        },
    ],
]

outputs = pipe(messages, max_new_tokens=50)
# print(outputs[0]["generated_text"][-1])
print(outputs, flush=True)

