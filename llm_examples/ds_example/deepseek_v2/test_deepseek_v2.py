from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch._inductor.config as config

config.freezing = True


def run_release():
    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
            trust_remote_code=True,
            # trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
            trust_remote_code=True,
            # trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            _attn_implementation="eager",).to("cuda")

        input_text = "dfasd #write a quick sort algorithm in C++ ??"
        # input_text = "where is me and you and him?"
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        # model.forward = torch.compile(model.forward)

        # print("model is: {}".format(model), flush=True)
        # print(model.__class__.__module__) 
        # import sys
        # import inspect
        # print(inspect.getfile(model.__class__))

        outputs = model.generate(**inputs, max_length=100)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def run_debug():
    with torch.no_grad():
        # local_model_path = "/home/scratch.leslief_sw/pytorch/pytorch_workspace/torch_script/llm_examples/ds_example/deepseek_v2/my_local_model"
        local_model_path = "/home/leslief/my_local_model"


        # def download_local_model(local_path):
        #     from huggingface_hub import snapshot_download

        #     # 将模型下载到当前目录下的 'my_local_model' 文件夹
        #     local_model_path = snapshot_download(
        #         repo_id="deepseek-ai/DeepSeek-Coder-V2-Lite-Base", 
        #         local_dir=local_path,
        #         local_dir_use_symlinks=False  # 关键：确保下载的是实际文件，而不是软链接
        #     )
        #     print(f"模型已下载到: {local_model_path}")

        # # 执行一次，同时docker 环境里面vscode可能没有权限编辑，删除modeling_deepseek.py，创建指向scratchpad 里面的软链接
        # download_local_model(local_model_path)
        # exit(-1)


        tokenizer = AutoTokenizer.from_pretrained(
            local_model_path,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            _attn_implementation="eager",).to("cuda")

        input_text = "dfasd #write a quick sort algorithm in C++ ??"
        # input_text = "where is me and you and him?"
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)


        outputs = model.generate(**inputs, max_length=100)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    run_release()
    # run_debug()
