# trt llm version: 2d2b8bae32b1d65f44873d762eb44fbc38e1336d
from tensorrt_llm import SamplingParams
# from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import (CudaGraphConfig, EagleDecodingConfig,
                                KvCacheConfig, MoeConfig, MTPDecodingConfig,
                                NGramDecodingConfig,
                                TorchCompileConfig)

def main():

    prompts = [
        # "Hello, my name is",
        # "The capital of France is",
        "Niuniu is my son, he likes playing fd 2048, and he is now playing it on a 4x4 grid. He wants to know the maximum number of points he can get ",
    ]
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95)

    # Alternatively, use "nvidia/Llama-3.1-8B-Instruct-FP8" to enable FP8 inference.
    print("---- start to create llm ----", flush=True)

    eagle_model_dir = f"/llm-models/Qwen3/qwen3_8b_eagle3"
    target_model_dir = f"/llm-models/Qwen3/Qwen3-8B"

    draft_len = 4
    spec_config = EagleDecodingConfig(max_draft_len=draft_len,
                                        speculative_model_dir=eagle_model_dir,
                                        eagle3_one_model=False)

    # Simple example
    llm = LLM(
        model=target_model_dir,
        disable_overlap_scheduler=True,
        cuda_graph_config=None,
        speculative_config=spec_config)  

    print("\n ---- start llm.generate ---- \n", flush=True)

    outputs = llm.generate(prompts, sampling_params)

    print("---- finish generate ----", flush=True)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# The entry point of the program need to be protected for spawning processes.
if __name__ == '__main__':
    main()
