import pickle
import numpy as np
import os
from tqdm import tqdm
from cs336_basics.tokenizer.BPETokenizer import BPETokenizer


def run_tokenizer(input_path, tokenizer_info_path, output_path, chunk_size=100):
    """
    Main function to perform tokenization using a single process.
    This version processes and saves tokens in chunks to save memory.

    Args:
        input_path (str): Path to the input text file.
        tokenizer_info_path (str): Path to the pickled tokenizer info file.
        output_path (str): Path where the tokenized binary file will be saved.
        chunk_size (int): The number of lines to process at a time before writing to disk.
    """
    # Read tokenizer info from the pickled file
    try:
        with open(tokenizer_info_path, "rb") as f:
            tokenizer_info = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Tokenizer info file not found at {tokenizer_info_path}")
        return
        
    bpe_vocab = tokenizer_info.get("vocab")
    bpe_merges = tokenizer_info.get("merges")
    special_tokens = tokenizer_info.get("special_tokens", ["<|endoftext|>"])

    # Ensure required tokenizer info is present
    if not bpe_vocab or not bpe_merges:
        print("Error: 'vocab' or 'merges' not found in the tokenizer info file.")
        return

    # Initialize the BPETokenizer with the loaded information
    tokenizer = BPETokenizer(bpe_vocab, bpe_merges, special_tokens)

    # Read all lines from the input file
    print("Reading input file...")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    
    # Remove the output file if it already exists to ensure a clean start
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Existing file at {output_path} removed.")
        
    print(f"Starting single-process tokenization of {len(lines)} lines in chunks of {chunk_size}...")
    
    total_tokens_generated = 0
    # Process lines in chunks
    for i in tqdm(range(0, len(lines), chunk_size), desc="Tokenizing and Saving Chunks"):
        chunk_lines = lines[i:i + chunk_size]
        chunk_tokens = []
        for line in chunk_lines:
            res = tokenizer.encode(line)
            chunk_tokens.extend(res)
        
        # Convert the chunk of tokens to a numpy array and save to file
        tokens = np.array(chunk_tokens, dtype=np.int64)
        
        try:
            with open(output_path, "ab") as f:
                tokens.tofile(f)
            total_tokens_generated += len(tokens)
        except IOError as e:
            print(f"Error: Could not write to output file {output_path}. Reason: {e}")
            return
    
    print(f"\nTokenized data saved to {output_path}")
    print(f"Total tokens generated: {total_tokens_generated:,}")


# Main execution block
if __name__ == "__main__":
    # Define file paths for input, tokenizer info, and output
    # input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    # tokenizer_info_path = "tinystory_vocab.pkl"
    # output_path = "tinystory_val_bpe_tokenized.bin"
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    tokenizer_info_path = "tinystory_vocab.pkl"
    output_path = "tinystory_train_bpe_tokenized.bin"
    
    # Call the main tokenization function
    run_tokenizer(input_path, tokenizer_info_path, output_path)
