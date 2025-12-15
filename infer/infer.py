# coding=utf-8
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ================================================================
# 路径配置
# ================================================================
model_local_path = "./weight/openpangu-7B-v1.1"
lora_weights_path = "lora weight"               # 修改为微调得到的lora权重
test_file = "./data/djsp_test_slow.json"
save_file = "./res/res_quick.txt"  

tokenizer = AutoTokenizer.from_pretrained(
    model_local_path,
    use_fast=False,
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_local_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)

model = PeftModel.from_pretrained(model, lora_weights_path, device_map="auto")
model.eval()

with open(test_file, "r", encoding="utf-8") as f:
    test_data = json.load(f)

fw = open(save_file, "w", encoding="utf-8")

for idx, item in enumerate(test_data):

    # =========================
    # ★ sys_prompt 从文件读取
    # =========================
    sys_prompt = item.get("instruction", "")

    user_input = item["input"]
    gt_output = item["output"]

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_input}
    ]

    # 构造模型输入
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    eos_ids = [tokenizer.eos_token_id]
    if "[unused10]" in tokenizer.get_vocab():
        eos_ids.append(tokenizer.convert_tokens_to_ids("[unused10]"))

    outputs = model.generate(
        **model_inputs,
        max_new_tokens=2048,
        eos_token_id=eos_ids,
        return_dict_in_generate=True,
        do_sample=False
    )

    input_len = model_inputs.input_ids.shape[1]
    gen_tokens = outputs.sequences[:, input_len:]
    raw_output = tokenizer.decode(gen_tokens[0], skip_special_tokens=False)

    thinking = ""
    content = ""

    if "[unused17]" in raw_output:
        if "[unused16]" in raw_output:
            thinking = raw_output.split("[unused16]")[-1].split("[unused17]")[0].strip()

        content = raw_output.split("[unused17]")[-1].split("[unused10]")[0].strip()
    else:
        content = raw_output.strip()

    fw.write(f"==================== {idx} ====================\n")
    fw.write("=== Sys Prompt ===\n")
    fw.write(sys_prompt + "\n\n")

    fw.write("=== Input ===\n")
    fw.write(user_input + "\n\n")

    fw.write("=== Ground Truth Output ===\n")
    fw.write(gt_output + "\n\n")

    fw.write("=== Model Raw Output ===\n")
    fw.write(raw_output + "\n\n")
    fw.write("--------------------------------------------------\n\n")

fw.close()
print(f"推理结束，结果保存在：{save_file}")
