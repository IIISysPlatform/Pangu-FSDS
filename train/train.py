import os
import json
import traceback
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser("DDP LoRA Training")

    # ---- 训练参数 ----
    parser.add_argument("--use_card_num", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad_acc_steps", type=int, default=4)
    # ---- LoRA ----
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    return parser.parse_args()

class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        # 确保目录存在（虽然通常在调用前已创建，但这里也安全处理）
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        self.log = open(filename, "a", encoding="utf8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass  # 必须有



MODEL_PATH = "../weight/openpangu-7B-v1.1"
DATA_PATH = "../dataset/djsp.json"
OUTPUT_DIR = "./djsp_lora_quick_slow"

MAX_LENGTH = 1024
DTYPE = torch.float16


SYSTEM_PREFIX = "[unused9]系统："
USER_PREFIX = "[unused9]用户："
ASSISTANT_PREFIX = "[unused9]助手："
SUFFIX_TOKEN = "[unused10]"

def build_prompt_and_target(instruction: str, input_text: str, output_text: str):
    user_content = instruction
    if input_text.strip():
        user_content = instruction + "\n\n" + input_text
    system_content = "你是一个非常有帮助的助手"
    prompt = (
        f"{SYSTEM_PREFIX}{system_content}{SUFFIX_TOKEN}"
        f"{USER_PREFIX}{user_content}{SUFFIX_TOKEN}"
        f"{ASSISTANT_PREFIX}"
    )
    return {"prompt": prompt, "target": output_text}

class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.items = []

        with open(path, "r", encoding="utf8") as f:
            data = json.load(f)
            for obj in data:
                instruction = obj.get("instruction", obj.get("query", ""))
                input_text = obj.get("input", "")
                output = obj.get("output", obj.get("answer", ""))
                self.items.append((instruction, input_text, output))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        inst, inp, out = self.items[idx]
        return build_prompt_and_target(inst, inp, out)

def collate_fn(batch, tokenizer, max_length):
    prompts = [b["prompt"] for b in batch]
    targets = [b["target"] for b in batch]

    full_texts = [p + t for p, t in zip(prompts, targets)]
    tokenized_full = tokenizer(
        full_texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    tokenized_prompt = tokenizer(
        prompts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = tokenized_full["input_ids"]
    attention_mask = tokenized_full["attention_mask"]
    labels = input_ids.clone()

    for i in range(len(batch)):
        prompt_len = (tokenized_prompt["attention_mask"][i] == 1).sum().item()
        labels[i, :prompt_len] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    torch.npu.set_device(rank)


def cleanup():
    try:
        dist.destroy_process_group()
    except Exception:
        pass

def train(rank, world_size, args):
    # ⭐⭐⭐ 关键修改：日志保存到 ./log/ 目录下 ⭐⭐⭐
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"log_rank{rank}.txt")
    sys.stdout = Logger(log_filename)
    print(f"[Rank {rank}] 日志已重定向到 {log_filename}")

    setup(rank, world_size)
    device = f"npu:{rank}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=DTYPE
    )
    model.to(device)

    print(f"[Rank {rank}] NPU {torch.npu.current_device()}:{torch.npu.get_device_name(torch.npu.current_device())}，总显存：{torch.npu.get_device_properties(torch.npu.current_device()).total_memory/(1024**3):.2f}GB")
    print(model.device, model)
    model.gradient_checkpointing_enable()

    # LoRA
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(model, peft_config)

    model = DDP(model, device_ids=[rank])

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    dataset = SFTDataset(DATA_PATH, tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        collate_fn=lambda x: collate_fn(x, tokenizer, MAX_LENGTH),
        num_workers=0, pin_memory=False
    )

    best_loss = float("inf")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.train()

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type="npu", dtype=DTYPE):
                outputs = model(**batch)

            loss = outputs.loss
            if torch.isnan(loss):
                print(f"[Rank {rank}] WARNING: loss is NaN at step {step}, skipping")
                continue

            (loss / args.grad_acc_steps).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if (step + 1) % args.grad_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            real_loss = loss.item()
            if rank == 0 and real_loss < best_loss:
                best_loss = real_loss
                save_dir = os.path.join(OUTPUT_DIR, "best")
                os.makedirs(save_dir, exist_ok=True)
                model.module.save_pretrained(save_dir)
                print(f"[Rank0] New best loss={best_loss:.4f}, saved to {save_dir}")

            if step % 10 == 0 and rank == 0:
                print(f"[Rank0] Epoch {epoch+1} Step {step} Loss {real_loss:.4f}")

        if rank == 0:
            last_dir = os.path.join(OUTPUT_DIR, "last")
            os.makedirs(last_dir, exist_ok=True)
            model.module.save_pretrained(last_dir)
            print(f"[Rank0] Saved LAST model to {last_dir}")

    cleanup()

if __name__ == "__main__":
    args = parse_args()
    print("==== Training Started ====")  # 主进程输出到终端
    print(vars(args))

    available = torch.npu.device_count()
    if args.use_card_num > available:
        raise ValueError(f"你要求使用 {args.use_card_num} 张卡，但机器只有 {available} 张 NPU")


    if args.use_card_num == 1:
        train(0, 1)
    else:
        mp.spawn(train, args=(args.use_card_num, args), nprocs=args.use_card_num, join=True)
