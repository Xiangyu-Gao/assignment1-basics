import time
import pickle
import cProfile

from cs336_basics.tokenizer.BPETokenizer import train_bpe


def main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    output_file_name = "tinystory_vocab.pkl"

    # input_path = "data/owt_train.txt"
    # output_file_name = "owt_vocab.pkl"
    
    # calculate the time and memory taken to run the BPE training
    start_time = time.time()

    special_tokens = ["<|endoftext|>"]

    # Run the BPE training
    print("Starting BPE training...")
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=special_tokens,
    )
    end_time = time.time()
    
    # Dump the vocabulary and merges to files
    with open(output_file_name, "wb") as vocab_file:
        pickle.dump(
            {
                "vocab": vocab,
                "merges": merges,
                "special_tokens": special_tokens,
            },
            vocab_file
        )

    print(f"Time taken: {end_time - start_time:.2f} seconds")

    # print(f"The whole vocab set is: {vocab}")

    print(f"The longest vocab is: {max(vocab.values(), key=len)}")


if __name__ == "__main__":
    main()
    # cProfile.run("main()", sort="cumulative")

    

    