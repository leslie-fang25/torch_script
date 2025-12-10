# trt llm version: 2d2b8bae32b1d65f44873d762eb44fbc38e1336d
from tensorrt_llm import SamplingParams
# from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import (CudaGraphConfig, EagleDecodingConfig,
                                KvCacheConfig, MoeConfig, MTPDecodingConfig,
                                NGramDecodingConfig,
                                TorchCompileConfig)
from tensorrt_llm.sampling_params import SamplingParams, GuidedDecodingParams

def main():

    # Alternatively, use "nvidia/Llama-3.1-8B-Instruct-FP8" to enable FP8 inference.
    print("---- start to create llm ----", flush=True)

    # Simple example
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        disable_overlap_scheduler=True,
        cuda_graph_config=None,
        guided_decoding_backend="xgrammar")

    json_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "pattern": "^[\\w]+$"
            },
            "population": {
                "type": "integer"
            },
        },
        "required": ["name", "population"],
    }
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "Give me the information of the capital of France in the JSON format.",
        },
    ]
    prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    output = llm.generate(
        prompt,
        sampling_params=SamplingParams(max_tokens=256, guided_decoding=GuidedDecodingParams(json=json_schema)),
    )
    print(output.outputs[0].text)    




# The entry point of the program need to be protected for spawning processes.
if __name__ == '__main__':
    main()
