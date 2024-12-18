# from datasets import load_dataset
import os
import json
import torch
import transformers
import subprocess
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers import LlamaTokenizer
from trl import SFTTrainer
# from datasets import load_dataset
from datasets import Dataset, DatasetDict
import argparse
import glob
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
import torch.nn as nn

def get_arguments():
    parser = argparse.ArgumentParser(description="Train a model with Hugging Face Transformers")
    # Data
    parser.add_argument("--train_path", type=str, default="DS_train.json", help="Training data")
    parser.add_argument("--test_path", type=str, default="DS_train.json", help="Test data")

    # Model
    parser.add_argument("--model_name", type=str, default="/scratch/ahcie-gpu2/llama-models-meta-hf/Llama-2-13b-hf", help="Model name or path")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="lora_alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="lora_alpha")

    # Training
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--output_dir", type=str, default="xxxxxxxx")
    parser.add_argument("--max_seq_length", type=int, default=4096)

    # 解析命令行参数
    args = parser.parse_args()
    return args


def formatting_func(example):
  if example.get("example", "") != "":
      input_prompt = ("### Instruction:\n"
      f"{example['instruction']}\n\n"
      "### Example:\n"
      f"{example['example']}\n\n"
      f"### Input: \n"
      f"{example['context']}\n\n"
      f"### Response: \n"
      f"{example['response']}")
  else:
    input_prompt = ("### Instruction:\n"
      f"{example['instruction']}\n\n"
      f"### Input: \n"
      f"{example['context']}\n\n"
      f"### Response: \n"
      f"{example['response']}")
    # input_prompt = (f"Below is an instruction that describes a task. "
    #   "Write a response that appropriately completes the request.\n\n"
    #   "### Instruction:\n"
    #   f"{example['instruction']}\n\n"
    #   f"### Response:\n"
    #   f"{example['response']}")

  return {"text" : input_prompt}

def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return [json.loads(line) for line in data]

def prepare_data(path):
    data = load_dataset(path)
    data = {key: [dic[key] for dic in data] for key in data[0]}
    del data['category']
    print(data)
    print("1"*100)
    data = Dataset.from_dict(data)
    print("2"*100)
    formatted_data = data.map(formatting_func)
    print( formatted_data)
    # exit()
    return formatted_data


def get_model(args):
    # model_id = '/scratch/ahcie-gpu2/openllama-models/MedLLaMA_13B'

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config, torch_dtype=torch.float16, device_map='auto',
    )
    # Initialize process group for DDP
    # init_process_group(backend='nccl')
    # torch.cuda.set_device(local_rank)
    # Move the model to the appropriate device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # base_model = base_model.to(device)
    # base_model = DDP(base_model, device_ids=[0,1])

    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    # base_model = nn.DataParallel(base_model)

    return base_model, tokenizer


def set_trainer(args,train_data,test_data, model, tokenizer):
    if "mamba-2.8b-hf" in args.model_name:
        qlora_config =  LoraConfig(
            r=8,
            target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
            task_type="CAUSAL_LM",
            bias="none"
        )
    else:
        qlora_config = LoraConfig(
            r=args.lora_rank,
            # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

    supervised_finetuning_trainer = SFTTrainer(
        model,
        train_dataset=train_data,
        eval_dataset=test_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            evaluation_strategy=args.evaluation_strategy,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            output_dir=args.output_dir,
            optim="paged_adamw_8bit",
            fp16=True,
            save_total_limit=1,
            load_best_model_at_end=True,
            # Enable DDP (Distributed Data Parallel)
            ddp_find_unused_parameters=False,
            # Distribute across 2 GPUs
            # device_map="auto",
            # Set the number of GPUs to use
            # num_trainers=2  # for DDP, specify the number of GPUs
        ),
        tokenizer=tokenizer,
        peft_config=qlora_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length
    )

    return supervised_finetuning_trainer


def rename_best(args, new_name='best_model'):
    current_name_pattern = 'checkpoint-*'
    current_folders = glob.glob(os.path.join(args.output_dir, current_name_pattern))

    # Check if there is exactly one matching folder
    if len(current_folders) != 1:
        raise Exception("Expected exactly one model folder, but found {}: {}".format(len(current_folders), current_folders))
    
    best_model_folder = current_folders[0]

    new_folder_path = os.path.join(args.output_dir, new_name)

    if os.path.exists(best_model_folder):
        os.rename(best_model_folder, new_folder_path)


"""
CUDA_VISIBLE_DEVICES=4 nohup  python llama2_13b_train.py >myout.llama2_13b_train_TE 2>&1 &

"""

def main():
    args = get_arguments()

    train_data = prepare_data(args.train_path)
    test_data = prepare_data(args.test_path)

    model, tokenizer = get_model(args)

    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(result.stdout)
    print(os.system('nvidia-smi'))

    supervised_finetuning_trainer = set_trainer(args,train_data,test_data, model, tokenizer)
    print(os.system('nvidia-smi'))
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(result.stdout)
    supervised_finetuning_trainer.train()

    rename_best(args)


if __name__ == "__main__":
    main()