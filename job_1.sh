#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00

nvidia-smi

CUDA_VISIBLE_DEVICES=0 python 1_train.py \
    --train_path NER/DS_train_bgelarge.json \
    --test_path NER/DS_test_bgelarge.json \
    --model_name /scratch/ahcie-gpu2/llama-models-meta-hf/Llama-2-13b-hf \
    --lora_rank 64 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --max_steps 5000 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --evaluation_strategy "steps" \
    --output_dir NER/bge-large/6-Llama-2-13b-hf\
    --max_seq_length 4096 \
    > NER/result

CUDA_VISIBLE_DEVICES=0 python 2_generation.py \
    --model_name /scratch/ahcie-gpu2/llama-models-meta-hf/Llama-2-13b-hf \
    --testfile_path NER/DS_test_bgelarge.json \
    --lora_path NER/bge-large/6-Llama-2-13b-hf \
    > NER/result


CUDA_VISIBLE_DEVICES=0 python 3_evaluation.py \
    --generated_file_dir NER/bge-large/6-Llama-2-13b-hf \
    > NER/result



CUDA_VISIBLE_DEVICES=0 python 1_train.py \
    --train_path relation_extraction/DS_train_bgelarge.json \
    --test_path relation_extraction/DS_test_bgelarge.json \
    --model_name /scratch/ahcie-gpu2/llama-models-meta-hf/Llama-2-13b-hf \
    --lora_rank 64 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --max_steps 5000 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --evaluation_strategy "steps" \
    --output_dir relation_extraction/bge-large/6-Llama-2-13b-hf\
    --max_seq_length 4096 \
    > relation_extraction/result

CUDA_VISIBLE_DEVICES=0 python 2_generation.py \
    --model_name /scratch/ahcie-gpu2/llama-models-meta-hf/Llama-2-13b-hf \
    --testfile_path relation_extraction/DS_test_bgelarge.json \
    --lora_path relation_extraction/bge-large/6-Llama-2-13b-hf \
    > relation_extraction/result


CUDA_VISIBLE_DEVICES=0 python 3_evaluation.py \
    --generated_file_dir relation_extraction/bge-large/6-Llama-2-13b-hf \
    > relation_extraction/result


CUDA_VISIBLE_DEVICES=0 python 1_train.py \
    --train_path triple_extraction/DS_train_bgelarge.json \
    --test_path triple_extraction/DS_test_bgelarge.json \
    --model_name /scratch/ahcie-gpu2/llama-models-meta-hf/Llama-2-13b-hf \
    --lora_rank 64 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --max_steps 5000 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --evaluation_strategy "steps" \
    --output_dir triple_extraction/bge-large/6-Llama-2-13b-hf\
    --max_seq_length 4096 \
    > triple_extraction/result

CUDA_VISIBLE_DEVICES=0 python 2_generation.py \
    --model_name /scratch/ahcie-gpu2/llama-models-meta-hf/Llama-2-13b-hf \
    --testfile_path triple_extraction/DS_test_bgelarge.json \
    --lora_path triple_extraction/bge-large/6-Llama-2-13b-hf \
    > triple_extraction/result


CUDA_VISIBLE_DEVICES=0 python 3_evaluation.py \
    --generated_file_dir triple_extraction/bge-large/6-Llama-2-13b-hf \
    > triple_extraction/result

CUDA_VISIBLE_DEVICES=0 python 1_train.py \
    --train_path usage_prediction/DS_train_bgelarge.json \
    --test_path usage_prediction/DS_test_bgelarge.json \
    --model_name /scratch/ahcie-gpu2/llama-models-meta-hf/Llama-2-13b-hf \
    --lora_rank 64 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --max_steps 5000 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --evaluation_strategy "steps" \
    --output_dir usage_prediction/bge-large/6-Llama-2-13b-hf\
    --max_seq_length 4096 \
    > usage_prediction/result

CUDA_VISIBLE_DEVICES=0 python 2_generation.py \
    --model_name /scratch/ahcie-gpu2/llama-models-meta-hf/Llama-2-13b-hf \
    --testfile_path usage_prediction/DS_test_bgelarge.json \
    --lora_path usage_prediction/bge-large/6-Llama-2-13b-hf \
    > usage_prediction/result


CUDA_VISIBLE_DEVICES=0 python 3_evaluation.py \
    --generated_file_dir usage_prediction/bge-large/6-Llama-2-13b-hf \
    > usage_prediction/result