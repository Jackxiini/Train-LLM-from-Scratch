#!/bin/bash
#SBATCH --job-name=train_LLM
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128GB

##load modules
module load gnu10
module load python/3.10.1-5r

source /scratch/wxi/Train-LLM-from-Scratch/

## Execute script
python llm/train/train_llm.py --num_iters 100000 --batch_size 256 --vocab_size 10000