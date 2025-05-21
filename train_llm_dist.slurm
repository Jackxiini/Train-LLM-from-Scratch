#!/bin/bash
#SBATCH --job-name=train_LLM_dist
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.40gb:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

module load gnu10
module load python/3.10.1-5r

cd /scratch/wxi/Train-LLM-from-Scratch

srun torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    llm/train/train_llm_dist.py \
    --train_dataset_path ../data/owt_train_encoded.npy \
    --val_dataset_path ../data/owt_valid_encoded.npy \
    --vocab_path bpe_32k_owt/owt-vocab.json \
    --merges_path bpe_32k_owt/owt-merges.txt \
    --num_iters 100000 \
    --d_model 2048 \
    --num_layers 12 \
    --num_heads 16 \
    --d_ff 5632 \
    --batch_size 512 \
    --context_length 2048 \
    --max_lr 3e-4 \
    --min_lr 3e-5 \
    --warmup_iters 2000 \
    --precision bfloat16 \
    --gradient_max_norm 1.0 \
    --weight_decay 0.1 \
    --validation_interval 1000 \
    --checkpoint_interval 5000 \
    --log_interval 100 \
    --dist_backend nccl
