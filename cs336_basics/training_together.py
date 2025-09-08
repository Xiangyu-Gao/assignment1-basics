import args
import wandb
import logging
import torch
import torch.nn as nn
import numpy as np

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamWs
from cs336_basics.data_loading import data_loading
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.cross_entropy import cross_entropy_loss
from cs336_basics.learning_rate_schedules import lr_cosine_schedule


def get_args():
    """Parse command line arguments for training configuration."""
    parser = args.get_args()
    parser.add_argument('--train_data_path', type=str, required=True, help='Path to training data (numpy memmap file)')
    parser.add_argument('--eval_data_path', type=str, required=True, help='Path to evaluation data (numpy memmap file)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--context_length', type=int, default=128, help='Context length for the model')
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size')
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
        self.optimizer = AdamWs(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        self.loss_fn = cross_entropy_loss
        self.device = args.device
        self.model.to(self.device)
        self.train_data_path = args.train_data_path
        self.eval_data_path = args.eval_data_path

    def train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        
        logits = self.model(x_batch)
    
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), y_batch.view(-1))
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, data):
        for iteration in range(1, self.args.max_iters + 1):
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
                base_lr=self.args.learning_rate,
                iteration=iteration,
                max_iters=self.args.max_iters,
                warmup_iters=self.args.warmup_iters
            )
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

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

    train_data_path = "/path/to/train/data.npy"
    
    # Load training data
    data = np.memmap(train_data_path, dtype=np.int64, mode='r')

    trainer = Trainer(args)
   
    with wandb.init(project=project, config=config) as run:
        # Training loop
        for epoch_idx in range(args.epoch):
    
            loss = trainer.train(data)

            # Log metrics to W&B
            run.log({"loss": loss})