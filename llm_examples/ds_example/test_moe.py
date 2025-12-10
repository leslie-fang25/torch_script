# trt llm version: 2d2b8bae32b1d65f44873d762eb44fbc38e1336d
from tensorrt_llm import SamplingParams
# from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import (CudaGraphConfig, EagleDecodingConfig,
                                KvCacheConfig, MoeConfig, MTPDecodingConfig,
                                NGramDecodingConfig,
                                TorchCompileConfig)

def main():

    # MODEL_NAME = "deepseek-ai/DeepSeek-V3-Lite"
    # MODEL_PATH = f"{llm_models_root()}/DeepSeek-V3-Lite/bf16"

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



    # kv cache reuse
    MODEL_PATH = "/llm-models/DeepSeek-V3-Lite/bf16"

    # enable_block_reuse = True
    enable_block_reuse = False

    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.75,
        enable_block_reuse=enable_block_reuse)
    # enable_chunked_prefill = False
    enable_chunked_prefill = True
    mtp_config = None
    # mtp_nextn = 2
    # mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
    pytorch_config = dict(
        disable_overlap_scheduler=True,
        cuda_graph_config=None,
        enable_iter_perf_stats = False
    )
    llm = LLM(
        MODEL_PATH,
        **pytorch_config,
        kv_cache_config=kv_cache_config,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_attention_dp=False,
        speculative_config=mtp_config,
        max_num_tokens=32,
    )


    print("\n ---- start llm.generate ---- \n", flush=True)

    outputs = llm.generate(prompts, sampling_params)

    print("---- finish generate ----", flush=True)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


    # prompts2 = [
    #     # "Hello, my name is",
    #     # "The capital of France is",
    #     "Niuniu is my son, he likes playing fd 2048, and he is now playing it on a 4x4 grid. He wants to know the maximum number of points uuuyt ",
    # ]

    # outputs2 = llm.generate(prompts2, sampling_params)
    # print("---- finish generate2 ----", flush=True)

    # # Print the outputs.
    # for output in outputs2:
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# The entry point of the program need to be protected for spawning processes.
if __name__ == '__main__':
    main()
