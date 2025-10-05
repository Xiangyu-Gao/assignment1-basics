import argparse
import wandb
import logging
import pickle
import torch
import numpy as np

from tqdm import trange

from cs336_basics.modules.transformer_lm import TransformerLM
from cs336_basics.modules.adamw import AdamW
from cs336_basics.modules.data_loading import data_loading
from cs336_basics.modules.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.modules.cross_entropy import cross_entropy_loss
from cs336_basics.modules.learning_rate_schedule import lr_cosine_schedule
from cs336_basics.modules.softmax import softmax


def get_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default="data/tinystory_train_bpe_tokenized.bin", help='Path to training data. default: tinystory_train_bpe_tokenized_tmp.bin')
    parser.add_argument('--eval_data_path', type=str, default="data/tinystory_val_bpe_tokenized.bin", help='Path to evaluation data. default: tinystory_train_bpe_tokenized_tmp.bin')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--context_length', type=int, default=128, help='Context length for the model')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--vocab_path', type=str, default="data/tinystory_vocab.pkl", help='Path to vocabulary file')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of model embeddings')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='Dimension of feedforward network')
    parser.add_argument('--theta', type=float, default=0.7, help='Theta parameter for the model')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW optimizer')
    parser.add_argument('--epoch', type=int, default=10, help='Number of epoches for training')
    parser.add_argument('--max_iters', type=int, default=10000, help='Maximum number of training iterations')
    parser.add_argument('--warmup_iters', type=int, default=1000, help='Number of warmup iterations for learning rate schedule')
    parser.add_argument('--log_interval', type=int, default=100, help='Interval for logging training progress')
    parser.add_argument('--save_interval', type=int, default=1000, help='Interval for saving checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training (e.g., "cpu" or "cuda:0")')

    return parser.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = TransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            num_layers=args.num_layers,
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            theta=args.theta
        )
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        self.loss_fn = cross_entropy_loss
        self.device = args.device
        self.model.to(self.device)
        # Load the vocabulary
        with open(args.vocab_path, 'rb') as f:
            tokenizer_info = pickle.load(f)
    
        self.vocab = tokenizer_info.get("vocab")

    def train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        
        logits = self.model(x_batch)
    
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), y_batch.view(-1))
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, data):
        for iteration in trange(1, self.args.max_iters + 1):
            x_batch, y_batch = data_loading(
                data,
                batch_size=self.args.batch_size,
                context_length=self.args.context_length,
                device=self.device
            )
            loss = self.train_step(x_batch, y_batch)
            
            if iteration % self.args.log_interval == 0:
                logging.info(f"Iteration {iteration}, Loss: {loss:.4f}")
            
            if iteration % self.args.save_interval == 0:
                self.save(f"{self.args.checkpoint_dir}/checkpoint_{iteration}.pt", iteration)
            
            # Update learning rate
            lr = lr_cosine_schedule(
                iteration,
                max_learning_rate=self.args.learning_rate,
                min_learning_rate=1e-5,
                warmup_iters=self.args.warmup_iters,
                cosine_cycle_iters=self.args.max_iters,
            )
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def validate(self, data, max_new_tokens=100, temperature=0.1, top_p=0.9):
        # randomly sample 5 samples from data and use them as prompts
        # generate max_new_tokens for each prompt using nucleus sampling
        # print the prompt and the generated text
        self.model.eval()
       
        with torch.no_grad():
            for _ in range(5):
                x_batch, _ = data_loading(
                        data,
                        batch_size=1,
                        context_length=self.args.context_length,
                        device=self.device
                )
                input_text = ''.join([self.vocab[int(idx)].decode("utf-8", errors="replace") for idx in x_batch[0].cpu().numpy()])
                output_sequence = torch.empty((1, 0), dtype=torch.long, device=self.device)

                # Generate new tokens
                for _ in range(max_new_tokens):
                    # Run the model to get the logits for the next token
                    logits = self.model(x_batch)
                    # Apply the softmax with temperature
                    # the smaller the temperature, the more peaked the distribution
                    # the larger the temperature, the more uniform the distribution
                    logits = logits / temperature  
                    probs = softmax(logits, dim=-1)
                    # sample next token from the distribution using Nucleus sampling
                    sampled_tokens = self.nucleus_sampling(probs[:, -1, :], top_p=top_p)
                    # Append sampled token to input sequence for next iteration
                    x_batch = torch.cat([x_batch[:, :-1], sampled_tokens], dim=1)

                    # Append to output sequence
                    output_sequence = torch.cat([output_sequence, sampled_tokens], dim=1)

                    # Stop if the sampled token is the end-of-text token (assuming it's the 256-th in the vocab)
                    if sampled_tokens.item() == 256:
                        break
                
                # Decode the output sequence to text and print
                output_sequence = output_sequence.cpu().numpy()
                output_text = ''.join([self.vocab[int(idx)].decode("utf-8", errors="replace") for idx in output_sequence[0]])
                print(f"Input text: {input_text}\n")
                print(f"Generated text: {output_text}\n{'-'*40}\n")
        
        return
    
    def nucleus_sampling(self, probs: torch.Tensor, top_p: float) -> torch.Tensor:
        """
        Perform nucleus (top-p) sampling on the given probability distribution.

        Args:
            probs (torch.Tensor): The probability distribution over the vocabulary.
            top_p (float): The cumulative probability threshold for nucleus sampling.

        Returns:
            torch.Tensor: The sampled token indices.
        """
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create a mask for tokens to keep
        mask = cumulative_probs <= top_p
        # Ensure at least one token is kept
        mask[..., 0] = True

        # Set probabilities of filtered tokens to zero
        filtered_probs = sorted_probs * mask.float()
        filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Re-normalize

        # Sample from the filtered distribution
        sampled_indices = torch.multinomial(filtered_probs, num_samples=1)
        sampled_tokens = sorted_indices.gather(-1, sampled_indices)

        return sampled_tokens

    def save(self, path: str, iteration: int):
        save_checkpoint(self.model, self.optimizer, iteration, path)

    def load(self, path: str):
        iteration = load_checkpoint(path, self.model, self.optimizer)
        return iteration


if __name__ == "__main__":
    args = get_args()

    project="LM-from-scratch"
    config = {
        "dataset": "Custom Dataset",
        "epochs": args.epoch,
    }
    
    # Load training and validate data
    train_data = np.memmap(args.train_data_path, dtype=np.int64, mode='r')
    val_data = np.memmap(args.eval_data_path, dtype=np.int64, mode='r')

    trainer = Trainer(args)
   
    with wandb.init(project=project, config=config) as run:
        # Training loop
        for epoch_idx in range(args.epoch):
            print(f"Starting epoch {epoch_idx + 1}/{args.epoch}")

            loss = trainer.train(train_data)

            # Log metrics to W&B
            run.log({"loss": loss})

            # Validation step every 10 epochs
            if (epoch_idx + 1) % 10 == 0:  # Validate every epoch
                print(f"Validating at epoch {epoch_idx + 1}")
                trainer.validate(val_data, max_new_tokens=100, temperature=0.1, top_p=0.9)