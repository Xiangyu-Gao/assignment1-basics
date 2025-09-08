import tiktoken

from cs336_basics.BPETokenizer import BPETokenizer


reference_tokenizer = tiktoken.get_encoding("gpt2")
# Use the merges and vocab from the reference tokenizer
bpe_merges = reference_tokenizer._merges
bpe_vocab = reference_tokenizer._subtoken_to_idx
special_tokens = reference_tokenizer.special_tokens

# Initialize your BPETokenizer with the same merges and vocab
tokenizer = BPETokenizer(bpe_vocab, bpe_merges, special_tokens=special_tokens)
import pdb; pdb.set_trace()