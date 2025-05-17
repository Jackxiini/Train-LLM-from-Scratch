# Train LLM from Scratch

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

### Train BPE Tokenizer 
Text is split using a GPT-2-style regex. Special tokens like `<|endoftext|>` are excluded. Due to the speed, tokenizers package is suggested to use. 
Here is an example to train on owt dataset:
```
uv run llm/train/train_tokenizer.py \
--input_path ../data/owt_train.txt --vocab_size 32000 \
--output_dir bpe_32k_owt --prefix owt \
--encoded_path owt_train_encoded.npy --train-bpe
```
If bpe has been trained, `--train-bpe` could be set to False. Validation set is also needed to be processed.
```
uv run llm/train/train_tokenizer.py \
--input_path ../data/owt_train.txt --vocab_size 32000 \
--output_dir bpe_32k_owt --prefix owt \
--encoded_path owt_valid_encoded.npy
```
Put both train and valid npy into data folder.

### Transformer Language Model

