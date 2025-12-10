import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 这里使用 R1 蒸馏版 8B，它是目前最具性价比的 "Lite" 选择
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# 1. 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 加载模型
# device_map="auto" 会自动利用 GPU
# torch_dtype=torch.bfloat16 是 DeepSeek 推荐的精度
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True  # 某些 DeepSeek 模型结构特殊，必须开启此项
)

# 3. 准备输入
text = "请简要介绍一下 DeepSeek-V3 的 MoE 架构优势。"
messages = [
    {"role": "user", "content": text}
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

# 4. 推理生成
outputs = model.generate(
    inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.6,  # 控制创造性，0.6 比较稳重
    top_p=0.9
)

# 5. 解码输出
response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
print(f"DeepSeek 回答:\n{response}")