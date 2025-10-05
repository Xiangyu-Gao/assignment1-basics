import os
import multiprocessing
import regex as re

from collections import defaultdict
from typing import List, Tuple, Dict, Set, Iterable, Iterator
from collections import Counter
from tqdm import trange
from cs336_basics.tokenizer.utils import find_chunk_boundaries


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def check_and_convert_special_tokens(
    special_tokens: List[str] | List[bytes],
) -> List[bytes]:
    """
    Check if special tokens are in the vocabulary and convert them to bytes.
    """
    if not all(isinstance(token, bytes) for token in special_tokens):
        special_tokens_bytes = [
            token.encode("utf-8") for token in special_tokens if isinstance(token, str)
        ]

    return special_tokens_bytes


def initialize_vocab(special_tokens: List[bytes]) -> Dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}  # ASCII characters
    for i, token in enumerate(special_tokens, start=256):
        vocab[i] = token

    return vocab


def word_to_bytes(word: str) -> List[bytes]:
    """
    Convert a word to bytes list.
    """
    byte_ids = [bytes([b]) for b in word.encode("utf-8")]

    return byte_ids


def pair_counts(
    word_counter: Dict[Tuple[bytes], int],
) -> Dict[Tuple[bytes, bytes], int]:
    """
    Count pairs of bytes in the word counter.
    """
    pairs: Dict[Tuple[bytes, bytes], int] = {}
    for token, freq in word_counter.items():
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            pairs[pair] = pairs.get(pair, 0) + freq

    return pairs


def get_most_frequent_pair(
    pairs: Dict[Tuple[bytes, bytes], int],
) -> Tuple[bytes, bytes]:
    max_freq = max(pairs.values())
    candidates = [pair for pair, freq in pairs.items() if freq == max_freq]
    res = max(candidates)

    return res


def add_pair_to_vocab(
    vocab: Dict[int, bytes], pair: Tuple[bytes, bytes], vocab_inv: Dict[bytes, int]
) -> int:
    """
    Add a new pair to the vocabulary.
    """
    index = len(vocab)
    s = vocab[vocab_inv[pair[0]]] + vocab[vocab_inv[pair[1]]]
    vocab[index] = s
    vocab_inv[vocab[index]] = index

    return index


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
    word_counter = Counter()    # Count the frequency of each word

    for part in parts:
        if part in special_tokens:
            if not drop_special_token:  # keep special tokens, otherwise ignore
                token = tuple(word_to_bytes(part))
                word_counter[token] += 1
        else:
            matches = re.finditer(PAT, part)
            for match in matches:
                word = match.group(0)
                token = tuple(word_to_bytes(word))
                word_counter[token] += 1
        
    return word_counter


def pretokenize_chunk(args) -> List[bytes]:
    """
    Pretokenize a chunk of text into bytes, handling special tokens.
    return world counter
    """
    input_path, special_tokens, start, end, drop_special_token = args
    print(f"Processing bytes from {start} to {end}...")
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    
    return pretokenize(chunk, special_tokens, drop_special_token=drop_special_token)


def merge_pair(
    word_counter: Dict[Tuple[bytes], int], pair: Tuple[bytes, bytes]
) -> Tuple[Dict[Tuple[bytes], int], Dict]:
    """
    Merge a pair of bytes in the word counter.
    """
    new_word_counter = Counter()
    updated_pair_counts = defaultdict(int)

    for token, freq in word_counter.items():
        new_token = []
        i = 0
        while i < len(token):
            if i < len(token) - 1 and (token[i], token[i + 1]) == pair:
                new_token.append(token[i] + token[i + 1])
                i += 2
            else:
                new_token.append(token[i])
                i += 1

        new_word_counter[tuple(new_token)] += freq

        for j in range(len(new_token) - 1):
            new_pair = (new_token[j], new_token[j + 1])
            updated_pair_counts[new_pair] += freq

    return new_word_counter, updated_pair_counts


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
    
    special_tokens_bytes = check_and_convert_special_tokens(special_tokens)
    
    vocab = initialize_vocab(special_tokens_bytes)
    vocab_inv = {v: k for k, v in vocab.items()}
    merges: List[Tuple[bytes, bytes]] = []

    # Chunk the text file
    num_processes = 4
    chunk_list = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

    # Prepare the chunk list for multiprocessing
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk_list.append((input_path, special_tokens, start, end, True))
    
    # Pretokenze the chunks with multiple processes
    word_counter = Counter()
    with multiprocessing.Pool(num_processes) as pool:
        word_counter_list = pool.map(pretokenize_chunk, chunk_list)
    # Combine the counters from all processes
    for counter in word_counter_list:
        word_counter.update(counter)

    # Merging
    pairs_freqs = pair_counts(word_counter)
    
    num_merges = vocab_size - len(vocab)
    
    for _ in trange(num_merges):
        
        most_common_pair = get_most_frequent_pair(pairs_freqs)
        
        new_index = add_pair_to_vocab(vocab, most_common_pair, vocab_inv)

        merges.append(most_common_pair)

        word_counter, pairs_freqs = merge_pair(word_counter, most_common_pair)

    return vocab, merges


class BPETokenizer:
    """
    A class to handle Byte Pair Encoding (BPE) tokenization.
    """

    PAT = PAT

    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.vocab_inv = {v: k for k, v in vocab.items()}  # bytes: int
        self.special_tokens = special_tokens if special_tokens is not None else []

    def _pre_tokenize(self, text: str) -> List[bytes]:
        """
        Pretokenize the text into bytes, handling special tokens.
        """
        parts = split_by_special_tokens(text, self.special_tokens)
        token_list = [] # List[bytes]

        for part in parts:
            if part in self.special_tokens:
                # keep special tokens, otherwise ignore
                token_list.append(part.encode("utf-8"))
            else:
                tokens = re.findall(PAT, part)
                token_list.extend(word_to_bytes(token) for token in tokens)
        return token_list

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] | None = None) -> "BPETokenizer":
        """
        Construct the BPE tokenizer from files.
        """
        raise NotImplementedError("This method is not implemented yet.")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode the input text into a sequence of token IDs using greedy strategy.
        """
        pretoken_bytes = self._pre_tokenize(text)
        token_ids = []

        # Pre-compute a mapping from a merge tuple to its new token ID.
        # This avoids repeated lookups and comparisons.
        merges_map = {
            (self.vocab_inv[pair[0]], self.vocab_inv[pair[1]]): self.vocab_inv.get(pair[0] + pair[1])
            for pair in self.merges
        }

        for pretoken_byte in pretoken_bytes:
            if pretoken_byte in [token.encode('utf-8') for token in self.special_tokens]:
                token_ids.append(self.vocab_inv[pretoken_byte])
                continue
            
            # Convert pre-token bytes to a list of initial token IDs.
            current_ids = [self.vocab_inv[b] for b in pretoken_byte]
            
            # Apply merges iteratively until no more merges are possible.
            while True:
                # Find the first mergeable pair based on the priority defined in self.merges.
                # This is a key optimization: we don't iterate through all merges.
                best_pair_index = -1
                best_pair = None
                
                for i in range(len(current_ids) - 1):
                    pair = (current_ids[i], current_ids[i+1])
                    
                    # Check if this pair is a valid merge and find its priority.
                    merge_bytes = (self.vocab[pair[0]], self.vocab[pair[1]])
                    try:
                        merge_index = self.merges.index(merge_bytes)
                    except ValueError:
                        continue # Not a valid merge

                    if best_pair_index == -1 or merge_index < best_pair_index:
                        best_pair_index = merge_index
                        best_pair = i

                if best_pair is None:
                    break # No more merges to be done on this pre-token

                # Perform the merge at the found position.
                i = best_pair
                new_id = merges_map[(current_ids[i], current_ids[i+1])]
                
                # Reconstruct the list with the merged token.
                current_ids = current_ids[:i] + [new_id] + current_ids[i+2:]
            
            token_ids.extend(current_ids)
        
        return token_ids
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encode lines of text from an iterable using buffered batching.
        This version preserves newlines by assuming the input was split with `splitlines(keepends=True)`.
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
