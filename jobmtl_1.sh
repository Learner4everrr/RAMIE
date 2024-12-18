#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00

nvidia-smi

CUDA_VISIBLE_DEVICES=0 python 1_train.py \
    --train_path mtl_LLM/DS_train_randomEx_all.json \
    --test_path mtl_LLM/DS_test_randomEx_all.json \
    --model_name /scratch/ahcie-gpu2/llama-models-meta-hf/Llama-2-13b-hf \
    --lora_rank 64 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --max_steps 20000 \
    --save_steps 4000 \
    --eval_steps 4000 \
    --evaluation_strategy "steps" \
    --output_dir mtl_LLM/randomEx/Llama-2-13b-hf\
    --max_seq_length 4096 \
    > mtl_LLM/result



CUDA_VISIBLE_DEVICES=0 python 2_generation.py \
    --model_name /scratch/ahcie-gpu2/llama-models-meta-hf/Llama-2-13b-hf \
    --testfile_path NER/DS_test_randomEx.json \
    --lora_path mtl_LLM/randomEx/Llama-2-13b-hf \
    --gen_dir mtl_LLM/randomEx/Llama-2-13b-hf/NER \
    > mtl_LLM/result

CUDA_VISIBLE_DEVICES=0 python 3_evaluation.py \
    --generated_file_dir mtl_LLM/randomEx/Llama-2-13b-hf/NER \
    > mtl_LLM/result



CUDA_VISIBLE_DEVICES=0 python 2_generation.py \
    --model_name /scratch/ahcie-gpu2/llama-models-meta-hf/Llama-2-13b-hf \
    --testfile_path relation_extraction/DS_test_randomEx.json \
    --lora_path mtl_LLM/randomEx/Llama-2-13b-hf \
    --gen_dir mtl_LLM/randomEx/Llama-2-13b-hf/relation_extraction \
    > mtl_LLM/result

CUDA_VISIBLE_DEVICES=0 python 3_evaluation.py \
    --generated_file_dir mtl_LLM/randomEx/Llama-2-13b-hf/relation_extraction \
    > mtl_LLM/result



CUDA_VISIBLE_DEVICES=0 python 2_generation.py \
    --model_name /scratch/ahcie-gpu2/llama-models-meta-hf/Llama-2-13b-hf \
    --testfile_path triple_extraction/DS_test_randomEx.json \
    --lora_path mtl_LLM/randomEx/Llama-2-13b-hf \
    --gen_dir mtl_LLM/randomEx/Llama-2-13b-hf/triple_extraction \
    > mtl_LLM/result

CUDA_VISIBLE_DEVICES=0 python 3_evaluation.py \
    --generated_file_dir mtl_LLM/randomEx/Llama-2-13b-hf/triple_extraction \
    > mtl_LLM/result



CUDA_VISIBLE_DEVICES=0 python 2_generation.py \
    --model_name /scratch/ahcie-gpu2/llama-models-meta-hf/Llama-2-13b-hf \
    --testfile_path usage_prediction/DS_test_randomEx.json \
    --lora_path mtl_LLM/randomEx/Llama-2-13b-hf \
    --gen_dir mtl_LLM/randomEx/Llama-2-13b-hf/usage_prediction \
    > mtl_LLM/result

CUDA_VISIBLE_DEVICES=0 python 3_evaluation.py \
    --generated_file_dir mtl_LLM/randomEx/Llama-2-13b-hf/usage_prediction \
    > mtl_LLM/result