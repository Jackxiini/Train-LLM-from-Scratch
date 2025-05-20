from tokenizers import ByteLevelBPETokenizer
import os
import argparse
import numpy as np
from tqdm import tqdm

#uv run llm/train/train_tokenizer.py --input_path ../data/owt_train.txt --vocab_size 32000 --output_dir bpe_32k_owt --prefix owt --encoded_path owt_train_encoded.npy --train-bpe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",     type=str,
                        default="../data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--vocab_size",     type=int,   default=10000)
    parser.add_argument("--min_frequency",  type=int,   default=2)
    parser.add_argument("--special_tokens", nargs="+",
                        default=["<|endoftext|>"])
    parser.add_argument("--output_dir",     type=str,   default="bpe_10k_TSV2")
    parser.add_argument("--prefix",         type=str,   default="tsv2")
    parser.add_argument("--encoded_path",   type=str,   default="TS_train_encoded.npy")
    parser.add_argument("--chunk_size",     type=int,   default=10000,
                        help="use this many lines to flush the buffer")
    parser.add_argument("--train-bpe",      action="store_true",  default=False,
                        help="if true, train the BPE tokenizer; otherwise, reuse the existing vocab/merges")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    vocab_file  = os.path.join(args.output_dir, f"{args.prefix}-vocab.json")
    merges_file = os.path.join(args.output_dir, f"{args.prefix}-merges.txt")

    # 1. train the BPE tokenizer
    if args.train_bpe:
        print("Training BPE tokenizer...")
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=args.input_path,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=args.special_tokens
        )
        tokenizer.save_model(args.output_dir, prefix=args.prefix)
        print(f"BPE model saved to {args.output_dir}")
    else:
        # make sure the files exist
        if not (os.path.isfile(vocab_file) and os.path.isfile(merges_file)):
            raise FileNotFoundError(f"cannot find {vocab_file} or {merges_file}, please train the BPE tokenizer first")
        print("reusing the existing BPE model")

    # 2. load the tokenizer
    tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
    eot_id = tokenizer.token_to_id(args.special_tokens[0])

    # 3. encode the training data
    all_ids = []
    buffer  = []
    with open(args.input_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(tqdm(f, desc="Encoding lines")):
            text = line.strip()
            if not text:
                continue
            ids = tokenizer.encode(text).ids
            ids.append(eot_id)
            buffer.extend(ids)

            if (lineno + 1) % args.chunk_size == 0:
                all_ids.extend(buffer)
                buffer.clear()

    # take the remaining buffer
    all_ids.extend(buffer)

    # 4. save the encoded data
    print("Saving encoded data...")
    np.save(args.encoded_path, np.array(all_ids, dtype=np.uint16))
    print(f"encoded data saved to {args.encoded_path}")

if __name__ == "__main__":
    main()