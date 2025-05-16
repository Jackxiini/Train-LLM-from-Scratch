import os
import regex # Use the regex library
import multiprocessing as mp
from typing import List, Tuple, Dict, BinaryIO # Import necessary types
from collections import defaultdict

# Pre-tokenization regex pattern from the assignment
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# find_chunk_boundaries function (from pretokenization_example.py)
# (Assuming this is allowed to be used verbatim or with minor adjustments)
# Needs to be defined here or imported
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0) # Reset file pointer to the beginning

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size # Ensure the last boundary is the file size

    mini_chunk_size = 4096 * 10 # Read ahead by 40k bytes at a time (increased for efficiency)

    # Adjust boundaries to fall at the beginning of the next special token
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start search from boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF before finding special token, boundary is at the end
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                # Adjust boundary to the found position
                chunk_boundaries[bi] = initial_position + found_at
                break
            # Move initial position forward by mini_chunk_size
            initial_position += len(mini_chunk) # Use len(mini_chunk) in case it's less than mini_chunk_size at the end

    # Ensure boundaries are unique and sorted
    return sorted(list(set(chunk_boundaries)))


# --- Worker function to process a single chunk ---
def process_single_chunk(args: Tuple[str, int, int, List[str], str]) -> List[List[bytes]]:
    """
    Processes a single file chunk: reads it, decodes, applies pre-tokenization logic,
    and returns the list of initial token sequences for this chunk.
    """
    file_path, start, end, special_tokens_list_str, pat_regex_pattern = args

    # Re-open the file in the worker process
    try:
        with open(file_path, "rb") as f:
            f.seek(start)
            chunk_bytes = f.read(end - start)

        # Decode the chunk bytes to text
        # Use 'replace' error handling for potentially invalid UTF-8 in raw data
        chunk_text = chunk_bytes.decode("utf-8", errors="replace")

        # --- Implement the two-step pre-tokenization logic for this chunk ---

        initial_token_sequences_for_chunk: List[List[bytes]] = []

        # Step 1: Split chunk_text by special tokens
        # Build the regex pattern for splitting special tokens (need to escape them)
        # This pattern needs to match the string representation in chunk_text
        special_tokens_pattern_str = "|".join(regex.escape(token) for token in special_tokens_list_str)
        # Add capturing group to keep the special tokens in the split result
        split_pattern = f'({special_tokens_pattern_str})'

        # Perform the first split
        segments = regex.split(split_pattern, chunk_text)

        # Step 2: Process each segment (special token or ordinary text)
        for segment in segments:
            if segment in special_tokens_list_str:
                # If it's a special token string, encode it and add as a single-element sequence
                initial_token_sequences_for_chunk.append([segment.encode("utf-8")])
            else:
                # If it's an ordinary text segment, apply the PAT regex
                # Use finditer for efficiency
                pretoken_matches = regex.finditer(pat_regex_pattern, segment)

                # Iterate through the PAT matches in this segment
                for match in pretoken_matches:
                    pt_string = match.group(0)
                    pt_bytes = pt_string.encode("utf-8")

                    # Convert the byte sequence of the pre-token into a list of single bytes
                    single_byte_sequence = [bytes([b]) for b in pt_bytes] # Each byte is a bytes object

                    # Add this list of single bytes as a new sequence
                    initial_token_sequences_for_chunk.append(single_byte_sequence)

        # Return the accumulated list of initial token sequences for this chunk
        return initial_token_sequences_for_chunk

    except Exception as e:
        # Log the error or print it, return None or an empty list to indicate failure
        print(f"Process {mp.current_process().pid} failed to process chunk [{start}, {end}): {e}")
        return [] # Return empty list on error


# --- Main training function using parallel pre-tokenization ---
def train_bpe_model(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:

    # --- Step 1: Find chunk boundaries using find_chunk_boundaries ---
    # Needs the special token as bytes for find_chunk_boundaries
    # Assuming the first special token is the main split token for boundaries
    split_token_for_boundaries_bytes = special_tokens[0].encode("utf-8") if special_tokens else b''

    print(f"File size: {os.path.getsize(input_path)} bytes")

    # Use a context manager to ensure the file is closed after finding boundaries
    with open(input_path, "rb") as f:
        # Determine number of processes (e.g., CPU count)
        num_processes = mp.cpu_count() - 4
        print(f"Targeting {num_processes} chunks for parallel pre-tokenization...")
        # Find chunk boundaries
        boundaries = find_chunk_boundaries(f, num_processes, split_token_for_boundaries_bytes)
        print(f"Found {len(boundaries) - 1} chunks. Boundaries: {boundaries}")


    # --- Step 2: Prepare arguments for worker function ---
    # Create a list of tuples, each tuple is the args for one worker (process_single_chunk)
    # Each worker needs (file_path, start, end, special_tokens_list_str, PAT_regex_pattern)
    # Pass the special tokens as strings and the PAT pattern string
    worker_args = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        worker_args.append((input_path, start, end, special_tokens, PAT))

    # --- Step 3: Execute parallel pre-tokenization using multiprocessing.Pool ---
    print(f"Starting parallel pre-tokenization on {len(worker_args)} chunks with {num_processes} processes...")
    all_pretoken_sequences: List[List[bytes]] = [] # List to collect results from all chunks

    # Use a Pool of workers
    with mp.Pool(processes=num_processes) as pool:
        # Use imap to process results as they are ready (more memory efficient for large outputs)
        # map could also be used but loads all results into memory at once
        # The results from pool.imap are lists of lists of bytes (List[List[bytes]])
        for chunk_result in pool.imap(process_single_chunk, worker_args):
             # Extend the main list with the sequences from this chunk
             if chunk_result is not None: # Handle potential errors from worker
                 all_pretoken_sequences.extend(chunk_result)

    print("Parallel pre-tokenization finished.")
    print(f"Total number of initial token sequences generated: {len(all_pretoken_sequences)}")

    # --- Step 4: BPE Merging Loop (Sequential) ---
    # Now you have the complete list of initial token sequences for the entire corpus
    # stored in all_pretoken_sequences.
    # The BPE merging algorithm operates on this list.

    print("Starting BPE merging loop (sequential)...")

    # TODO: Implement the core BPE merging algorithm here
    # This involves:
    # 1. Initializing vocabulary with base bytes and special tokens.
    # 2. Looping until vocab_size is reached.
    # 3. Inside the loop:
    #    a. COUNT frequencies of all adjacent pairs in all_pretoken_sequences.
    #    b. FIND the pair with the highest frequency (handle ties).
    #    c. CREATE the new merged token (byte concatenation).
    #    d. UPDATE the vocabulary and assign a new ID.
    #    e. RECORD the merge (add the pair (bytes, bytes) to a list).
    #    f. UPDATE all_pretoken_sequences by replacing occurrences of the merged pair.
    # 4. Return the final vocabulary and merges list.


    # --- Initial Vocabulary setup (part of Step 4.1) ---
    vocab:dict[int, bytes] = {i:b"<|endoftext|>" if i == 0 else bytes([i-1]) for i in range(0, 257)}
    merges: list[tuple[int, int], int] = []
    process = all_pretoken_sequences
    
    for i in range(vocab_size-257):
        count = defaultdict(int)
        for j in process:
            for index1 , index2 in zip(j, j[1:]):
                count[(index1, index2)] += 1
        # pair = max(count, key=count.get)
        pair = max(count, key=lambda k: (count.get(k), k))
        merges.append(pair)
        idx1, idx2 = pair
        vocab[i+257] = idx1+idx2
        newprocess = []
        for k in process:
            microprocess = []
            l = 0
            while(l<len(k)):
                if (l<len(k) - 1 and k[l]==pair[0] and k[l+1]==pair[1]):
                    microprocess.append(vocab[i+257])
                    l=l+2
                else:
                    microprocess.append(k[l])
                    l = l+1
            newprocess.append(microprocess)
        process = newprocess
    
    return vocab, merges
            