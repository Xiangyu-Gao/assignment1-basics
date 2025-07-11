import time
import pickle
from tests.adapters import run_train_bpe

if __name__ == "__main__":
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    output_file_name = "tinystory_vocab.pkl"

    # input_path = "data/owt_train.txt"
    # output_file_name = "owt_vocab.pkl"
    
    # calculate the time and memory taken to run the BPE training
    start_time = time.time()

    # Run the BPE training
    print("Starting BPE training...")
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    
    # Dump the vocabulary and merges to files
    with open(output_file_name, "wb") as vocab_file:
        pickle.dump({"vocab": vocab, "merges": merges}, vocab_file)

    print(f"Time taken: {end_time - start_time:.2f} seconds")

    print(f"The whole vocab set is: {vocab}")

    print(f"The longest vocab is: {max(vocab, key=len)}")

    