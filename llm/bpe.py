import os
import regex
from collections import Counter
from typing import List, Tuple, BinaryIO
from multiprocessing import Pool, cpu_count



def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes
) -> List[int]:
    # (Function remains unchanged)
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))


def bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d

BYTE_TO_UNICODE = bytes_to_unicode()
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}

# 2. GPT-2's Unicode-level tokenization pattern (regex module)
PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
PAT_REGEX = regex.compile(PAT)

# 3. Special token
SPECIAL_TOKEN = "<|endoftext|>"
SPECIAL_BYTES = SPECIAL_TOKEN.encode('utf-8')


def process_chunk(
    input_path: str,
    start: int,
    end: int,
    special_tokens: list[str] = None,
) -> Counter:
    counts = Counter()
    with open(input_path, 'rb') as f:
        f.seek(start)
        data = f.read(end - start)

    # Protect special tokens by replacing them with a unique marker
    special_bytes = [st.encode("utf-8") for st in (special_tokens or [])]
    for st_bytes in special_bytes:
        marker = b'__SPECIAL_TOKEN__' + st_bytes + b'__'
        data = data.replace(st_bytes, marker)

    text = ''.join(BYTE_TO_UNICODE[b] for b in data)

    for token in PAT_REGEX.findall(text):
        bt = bytes([UNICODE_TO_BYTE[ch] for ch in token])
        # Restore special token if marker is found
        if bt.startswith(b'__SPECIAL_TOKEN__') and bt.endswith(b'__'):
            for st_bytes in special_bytes:
                if bt == b'__SPECIAL_TOKEN__' + st_bytes + b'__':
                    bt = st_bytes
                    break
        counts[(bt,)] += 1
    return counts


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = None,
) -> Tuple[dict[int, bytes], List[Tuple[bytes, bytes]]]:
    if num_processes is None:
        num_processes = cpu_count()
    special_bytes = [st.encode("utf-8") for st in special_tokens]
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_processes, SPECIAL_BYTES)
    tasks = [(input_path, boundaries[i], boundaries[i+1], special_tokens) for i in range(len(boundaries)-1)]
    with Pool(num_processes) as pool:
        counters = pool.starmap(process_chunk, tasks)
    vocab_counter = Counter()
    for c in counters:
        vocab_counter.update(c)
    # Initial vocab: all single bytes (0-255) as tuples, plus special tokens
    initial_vocab = set((bytes([i]),) for i in range(256))
    for st_b in special_bytes:
        initial_vocab.add((st_b,))
    for k in list(vocab_counter.keys()):
        if k not in initial_vocab:
            del vocab_counter[k]
    def get_pairs(v: Counter) -> Counter:
        pairs = Counter()
        for seq, freq in v.items():
            # Skip if sequence contains any special token
            if any(special in seq for special in special_bytes):
                continue
            for i in range(len(seq)-1):
                # Skip if either token in the pair is a special token
                if any(special in (seq[i], seq[i+1]) for special in special_bytes):
                    continue
                pairs[(seq[i], seq[i+1])] += freq
        return pairs
    def merge_vocab(pair: Tuple[bytes, bytes], v: Counter) -> Counter:
        a, b = pair
        merged = a + b
        new_v = Counter()
        for seq, freq in v.items():
            # Skip if sequence contains any special token
            if any(special in seq for special in special_bytes):
                new_v[seq] += freq
                continue
            i = 0
            new_seq = []
            while i < len(seq):
                if i < len(seq)-1 and (seq[i], seq[i+1]) == pair:
                    new_seq.append(merged)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_v[tuple(new_seq)] += freq
        return new_v
    merges: List[Tuple[bytes, bytes]] = []
    while len(vocab_counter) + len(merges) + len(special_tokens) < vocab_size:
        pairs = get_pairs(vocab_counter)
        if not pairs:
            break
        best_freq = max(pairs.values())
        candidates = [p for p, f in pairs.items() if f == best_freq]
        best_pair = max(candidates)
        merges.append(best_pair)
        vocab_counter = merge_vocab(best_pair, vocab_counter)
    final_vocab: dict[int, bytes] = {}
    idx = 0
    for st in special_tokens:
        st_b = st.encode("utf-8")
        final_vocab[idx] = st_b
        idx += 1
    for token_seq in vocab_counter:
        if token_seq in [(sb,) for sb in special_bytes]:
            continue
        if idx >= vocab_size:
            break
        final_vocab[idx] = b"".join(token_seq)
        idx += 1
    return final_vocab, merges