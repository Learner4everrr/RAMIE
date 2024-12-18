from peft import get_peft_model
import torch
import transformers
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer,LlamaTokenizer
from transformers import AutoTokenizer
from peft import PeftModel
import sentencepiece
import accelerate
import argparse
import json
import os

def get_arguments():
    parser = argparse.ArgumentParser(description="Generation")
    # Data
    parser.add_argument("--testfile_path", type=str, default="DS_train.json", help="Test data")

    # Model
    parser.add_argument("--model_name", type=str, default="/scratch/ahcie-gpu2/llama-models-meta-hf/Llama-2-13b-hf", help="Model name or path")

    # Lora path
    parser.add_argument("--lora_path", type=str, default="xxxxxxxx")

    # gen file dir
    parser.add_argument("--gen_dir", type=str, default="xxxxxxxx")



    # 解析命令行参数
    args = parser.parse_args()
    os.makedirs(args.gen_dir, exist_ok=True)
    return args


def load_trained_model(args):

    
    lora_weights=args.lora_path+"/"+"best_model"
    base_model=args.model_name

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    
    try:
        tokenizer = AutoTokenizer.from_pretrained(lora_weights)
    except:
        tokenizer = LlamaTokenizer.from_pretrained(lora_weights)
    # tokenizer=LlamaTokenizer.from_pretrained(lora_weights)  #, config=config, cache_dir="./llamacache"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map='auto')

    # model = get_peft_model(model, lora_config)
    model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
    return model, tokenizer



def make_inference(model, tokenizer, instruction, context = None, example = None):
    if example:
      prompt = ("### Instruction:\n"
      f"{instruction}\n\n"
      "### Example:\n"
      f"{example}\n\n"
      f"### Input: \n"
      f"{context}\n\n"
      f"### Response: \n")
    else:
      prompt = ("### Instruction:\n"
      f"{instruction}\n\n"
      f"### Input: \n"
      f"{context}\n\n"
      f"### Response: \n")
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False,max_length=1500).to("cuda:0")
    # outputs = base_model.generate(**inputs, max_new_tokens=100)
    # display(Markdown((tokenizer.decode(outputs[0], skip_special_tokens=True))))
    # model.zero()
    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300)
        results=(tokenizer.decode(outputs[0], skip_special_tokens=True))
        return results
        # print(results)
        # print("---- NON-INSTRUCT-TUNED-MODEL ----")


def generation(args, model, tokenizer):

    save_file_name = args.gen_dir + '/' + "test_generation.json"
    input_test_file_name=args.testfile_path

    fw=open(save_file_name,"w")
    i=0
    with open(input_test_file_name,"r",encoding="utf-8") as fr:
      for line in fr.readlines():
        line=json.loads(line.strip())
        instruction=line["instruction"]
        sentence=line["context"]
        ground_truth=line["response"]
        example = line.get("example", None)
        predicted=make_inference(model, tokenizer, instruction, sentence,example)
        i=i+1
        print(i)
        
        Dic_={}
        Dic_["sentence"]=sentence
        Dic_["ground_truth"]=ground_truth
        Dic_["predicted"]=predicted

        fw.write(json.dumps(Dic_))
        fw.flush()
        fw.write("\n")

def main():
    args = get_arguments()
    model, tokenizer = load_trained_model(args)
    generation(args, model, tokenizer)
    
    exit()

if __name__=="__main__":
    main()
 



"""
CUDA_VISIBLE_DEVICES=3 nohup  python generation_llama2_13b_3000.py >myout.generation_llama2_13b_3000 2>&1 &

"""