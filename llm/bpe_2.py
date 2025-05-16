import re
from collections import defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries


def BPE(input_path: str, vocab_size: int, special_tokens: list[str]):
    ## Usage
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, 3, "<|endoftext|>".encode("utf-8"))
            
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            tokens = re.findall(PAT, chunk)
            print(tokens)

            vocab:dict[int, bytes] = {i:bytes([i]) for i in range(256)}
            merges: list[tuple[int, int], int] = []
            process = [list(i.encode("utf-8")) for i in tokens]
            
            for i in range(vocab_size):
                count = defaultdict(int)
                for j in process:
                    for index1 , index2 in zip(j, j[1:]):
                        count[(index1, index2)] += 1
                pair = max(count, key=count.get)
                #pair = max(count, key=lambda k: (count.get(k), k))
                merges.append(pair)
                index1, index2 = pair
                vocab[i+256] = vocab[index1]+vocab[index2]
                newprocess = []
                for k in process:
                    microprocess = []
                    l = 0
                    while(l<len(k)):
                        if (l<len(k) - 1 and k[l]==pair[0] and k[l+1]==pair[1]):
                            microprocess.append(i+256)
                            l=l+2
                        else:
                            microprocess.append(k[l])
                            l = l+1
                    newprocess.append(microprocess)
                process = newprocess
            
            return vocab, merges