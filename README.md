# Train LLM from Scratch
This repository provides a modern implementation of a Transformer-based language model trained entirely from scratch. It includes preprocessing pipelines, a tokenizer trainer, and a modern Transformer architecture.
## Setup

### Environment
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.

You can now run any code in the repo using
```sh
uv run <python_file_path>
```

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### Tokenizer Training (BPE)
We adopt a GPT-2-style byte-level tokenizer using BPE (Byte-Pair Encoding). Tokenization is based on regex segmentation. Special tokens (`<|endoftext|>`) are excluded during training.
To train a tokenizer on OpenWebText:
```
uv run llm/train/train_tokenizer.py \
--input_path ../data/owt_train.txt --vocab_size 32000 \
--output_dir bpe_32k_owt --prefix owt \
--encoded_path owt_train_encoded.npy --train-bpe
```
Once the tokenizer is trained, you can encode the validation set:
```
uv run llm/train/train_tokenizer.py \
--input_path ../data/owt_train.txt --vocab_size 32000 \
--output_dir bpe_32k_owt --prefix owt \
--encoded_path owt_valid_encoded.npy
```
Ensure that both `owt_train_encoded.npy` and `owt_valid_encoded.npy` are saved in the `data/` directory.

### Transformer Language Model
The Transformer model in this repository includes the following modern components:

- **Pre-Layer Normalization**  
- **RMSNorm** for lightweight normalization  
- **SwiGLU** feed-forward layers (SiLU activation + Gated Linear Unit)  
- **RoPE (Rotary Positional Embedding)** for relative position encoding  
- **Causal multi-head self-attention**

Additionally, the following training modules are implemented from scratch:

- Cross-entropy loss  
- AdamW optimizer  
- Cosine annealing learning rate scheduler with linear warm-up  
- Gradient clipping for stability

### Train TransformerLM
To start training the Transformer model:
```
uv run llm/train/train_llm.py
```
Note: Training is currently supported with float32 precision only. Use of bfloat16 may result in unstable loss (NaNs).

When trained correctly, this model achieves a validation loss of approximately 1.5783 (at 10000 step) and 1.2806 (at 50000 step) using the default hyperparameters listed below on TinyStories dataset
```
uv run llm/train/train_llm.py \
  --train_dataset_path ../data/TS_train_encoded.npy \
  --val_dataset_path ../data/TS_valid_encoded.npy \
  --vocab_size 50257 \
  --special_tokens '["<|endoftext|>"]' \
  --vocab_path bpe_10k_TSV2/tsv2-vocab.json \
  --merges_path bpe_10k_TSV2/tsv2-merges.txt \
  --d_model 512 \
  --num_layers 4 \
  --num_heads 16 \
  --d_ff 1344 \
  --rope_theta 10000.0 \
  --batch_size 128 \
  --context_length 256 \
  --num_iters 50000 \
  --max_lr 1e-3 \
  --min_lr 1e-5 \
  --warmup_iters 10000 \
  --weight_decay 0.01 \
  --gradient_max_norm 1.0 \
  --beta1 0.9 \
  --beta2 0.95 \
  --checkpoint_interval 1000 \
  --temperature 1.0 \
  --top_k 10 \
  --top_p 0.0 \
```
