import os
import multiprocessing
import regex as re

from typing import BinaryIO
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Iterable, Iterator
from collections import Counter


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
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def split_by_special_tokens(text: str, special_tokens: List[str]) -> List[str]:
    """
    Split the text by special tokens, ensuring that the special tokens are not split.
    Example:
        text = "Hello <|endoftext|> World"
        special_tokens = ["<|endoftext|>"]
        Returns: ["Hello ", "<|endoftext|> ", "World"]
    """
    special_tokens_sorted = sorted(special_tokens, key=lambda x: -len(x)) # sort by length descending
    # This is to ensure that longer tokens are matched first

    if not special_tokens_sorted:
        return [text]   # If no special tokens, return the text as a single part
    else:
        pattern = "|".join(re.escape(token) for token in special_tokens_sorted)
        return re.split(f"({pattern})", text)


def pretokenize(text: str, special_tokens: List[str], drop_special_token: bool=True) -> List[bytes]:
    """
    Pretokenize the text into bytes, handling special tokens.
    """
    parts = split_by_special_tokens(text, special_tokens)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens_list = []

    for part in parts:
        if part in special_tokens:
            if not drop_special_token:  # keep special tokens, otherwise ignore
                tokens_list.append(part.encode('utf-8'))
        else:
            matches = re.finditer(PAT, part)
            for match in matches:
                token = match.group(0)
                if token:
                    tokens_list.append(token.encode('utf-8'))
        
    return tokens_list


def pretokenize_chunk(args) -> List[bytes]:
    """
    Pretokenize a chunk of text into bytes, handling special tokens.
    """
    input_path, special_tokens, start, end, drop_special_token = args
    print(f"Processing bytes from {start} to {end}...")
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    
    return pretokenize(chunk, special_tokens, drop_special_token=drop_special_token)


def merge(counts: Dict[Tuple[int, int], int], 
          index_dict: Dict[Tuple[int, int], Set[int]], 
          pretokens: List[bytes], 
          max_pair: Tuple[int, int], 
          new_index: int) -> None:
    """
    Merge the pairs with higest counts and update the counts and index_dict.
    """
    index_set = index_dict[max_pair]

    for i in index_set:
        pretoken = pretokens[i]
        new_pretoken = []

        pos_list = []   # store position of max_pair in each pretoken afte merging
        pos = 0
        

        # Replace the max_pair in the pretoken with new_index
        j = 0
        while j < len(pretoken):
            if j < len(pretoken) - 1 and (pretoken[j], pretoken[j + 1]) == max_pair:
                new_pretoken.append(new_index)
                pos_list.append(pos)
                j += 2
                pos += 1
            else:
                new_pretoken.append(pretoken[j])
                j += 1
                pos += 1
        
        # Update the counts and index_dict
        for pos in pos_list:
            counts[max_pair] -= 1

            if pos > 0:
                if new_pretoken[pos - 1] == new_index:
                    counts[(max_pair[1], max_pair[0])] -= 1
                else:
                    counts[(new_pretoken[pos - 1], max_pair[0])] -= 1
                counts[(new_pretoken[pos - 1], new_index)] += 1
                index_dict[(new_pretoken[pos - 1], new_index)].add(i)
            
            if pos < len(new_pretoken) - 1:
                if new_pretoken[pos + 1] == new_index:
                    counts[(max_pair[1], max_pair[0])] -= 1
                else:
                    counts[(max_pair[1], new_pretoken[pos + 1])] -= 1
                counts[(new_index, new_pretoken[pos + 1])] += 1
                index_dict[(new_index, new_pretoken[pos + 1])].add(i)
        
        pretokens[i] = new_pretoken


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a Byte Pair Encoding (BPE) tokenizer on the given input file.
    Args:
        input_path (str): Path to the input text file.
        vocab_size (int): Desired size of the vocabulary.
        special_tokens (List[str]): List of special tokens to include in the vocabulary.
    Returns:
        Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]: A tuple containing the vocabulary mapping and the list of merges.
    """
    
    # Initialize the vocab
    vocab = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')

    merges = []
    number_merges = vocab_size - len(vocab)

    # Chunk the text file
    num_processes = 4
    chunk_list = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

    # Prepare the chunk list for multiprocessing
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk_list.append((input_path, special_tokens, start, end, True))
    
    # Pretokenze the chunks with multiple processes
    pretokens_list = []
    with multiprocessing.Pool(num_processes) as pool:
        pretokens_list = pool.map(pretokenize_chunk, chunk_list)
    pretokens = [item for sublist in pretokens_list for item in sublist]

    # Merging
    counts = defaultdict(int)   # Store the counts of each pair
    index_dict = defaultdict(set)  # Store pretoken location for each pair
    for j, pretoken in enumerate(pretokens):
        for index1, index2 in zip(pretoken, pretoken[1:]):
            counts[index1, index2] += 1
            index_dict[index1, index2].add(j)
    
    for _ in range(number_merges):
        # Prefer lexicographically lager pairs
        max_pair = max(
            counts.items(),
            key=lambda x: (
                x[1],
                vocab[x[0][0]].decode("utf-8", errors="ignore"),
                vocab[x[0][1]].decode("utf-8", errors="ignore")
            ),
        )[0]
        index1, index2 = max_pair
        new_index = len(vocab)
        vocab[new_index] = vocab[index1] + vocab[index2]
        merges.append((vocab[index1], vocab[index2]))

        merge(counts, index_dict, pretokens, max_pair, new_index)

    return vocab, merges


class BPETokenizer:
    """
    A class to handle Byte Pair Encoding (BPE) tokenization.
    """
    
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] | None = None) -> "BPETokenizer":
        """
        Construct the BPE tokenizer from files.
        """
        raise NotImplementedError("This method is not implemented yet.")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode the input text into a sequence of token IDs.
        """
        vocab_reversed = {v: k for k, v in self.vocab.items()}  # bytes: int
        pretoken_bytes = pretokenize(text, self.special_tokens, drop_special_token=False)
        special_token_bytes = [token.encode('utf-8') for token in self.special_tokens]
        pretokens = [] # List[List[int]]

        # Convert pretokens from bytes to List[int] by vocab
        for pretoken in pretoken_bytes:
            new_pretoken = []
            if pretoken in special_token_bytes:
                new_pretoken.append(vocab_reversed[pretoken])
            else:
                for b in pretoken:
                    index = vocab_reversed[bytes([b])]
                    new_pretoken.append(index)
            
            pretokens.append(new_pretoken)

        # Apply merges
        for i, pretoken in enumerate(pretokens):
            for merge in self.merges:
                new_index = vocab_reversed[merge[0] + merge[1]]
                new_pretoken = []
                j = 0
                while j < len(pretoken):
                    if j < len(pretoken) - 1 and (self.vocab[pretoken[j]], self.vocab[pretoken[j + 1]]) == merge:
                        new_pretoken.append(new_index)
                        j += 2
                    else:
                        new_pretoken.append(pretoken[j])
                        j += 1
                pretoken = new_pretoken # update pretoken with merged result
            
            pretokens[i] = pretoken # update pretokens with final merged result
        
        tokens = [token for pretoken in pretokens for token in pretoken]
        
        return tokens
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: List[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        tokens = [self.vocab[i] for i in ids]
        text = b"".join(tokens).decode("utf-8", errors="replace")
        return text
