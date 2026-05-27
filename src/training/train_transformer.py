"""
Training script for Transformer Music Generator (Task 3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import math

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer import TransformerMusicGenerator, compute_perplexity
from config import *


class TokenSequenceDataset(Dataset):
    """Dataset for tokenized music sequences."""
    
    def __init__(self, npy_path: str, max_seq_len: int = 512):
        """
        Initialize dataset.
        
        Args:
            npy_path: Path to tokenized sequences (.npy file)
            max_seq_len: Maximum sequence length
        """
        self.data = np.load(npy_path)
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        
        # Truncate if too long
        if len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]
        
        return torch.tensor(seq, dtype=torch.long)


def collate_fn(batch, pad_token_id=0):
    """
    Collate function to pad sequences to same length.
    
    Args:
        batch: List of sequences
        pad_token_id: Padding token ID
        
    Returns:
        Tuple of (padded_sequences, attention_mask)
    """
    # Find max length in batch
    max_len = max(len(seq) for seq in batch)
    
    # Pad sequences
    padded = []
    masks = []
    
    for seq in batch:
        pad_len = max_len - len(seq)
        padded_seq = F.pad(seq, (0, pad_len), value=pad_token_id)
        mask = torch.cat([torch.ones(len(seq)), torch.zeros(pad_len)])
        
        padded.append(padded_seq)
        masks.append(mask)
    
    return torch.stack(padded), torch.stack(masks).bool()


def train_transformer(train_path: str,
                     val_path: str,
                     vocab_size: int,
                     output_dir: str = MODELS_DIR,
                     epochs: int = TRANSFORMER_EPOCHS,
                     batch_size: int = TRANSFORMER_BATCH_SIZE,
                     learning_rate: float = TRANSFORMER_LEARNING_RATE,
                     d_model: int = 256,
                     n_heads: int = 8,
                     n_layers: int = 6,
                     d_ff: int = 1024,
                     dropout: float = 0.1,
                     max_seq_len: int = 512,
                     pad_token_id: int = 0,
                     device: str = None):
    """
    Train Transformer music generator.
    
    Args:
        train_path: Path to training data (.npy file with token sequences)
        val_path: Path to validation data (.npy file with token sequences)
        vocab_size: Size of token vocabulary
        output_dir: Directory to save model and plots
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feedforward dimension
        dropout: Dropout probability
        max_seq_len: Maximum sequence length
        pad_token_id: Padding token ID
        device: Device to train on (None = auto-detect)
    """
    # Setup device
    if device is None:
        device = DEVICE
    
    print(f"Training on device: {device}")
    
    # Create datasets
    train_dataset = TokenSequenceDataset(train_path, max_seq_len)
    val_dataset = TokenSequenceDataset(val_path, max_seq_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id)
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Vocab size: {vocab_size}")
    
    # Initialize model
    model = TransformerMusicGenerator(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_seq_len=max_seq_len
    ).to(device)
    
    # Multi-GPU support
    if USE_MULTI_GPU and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Optimizer with learning rate warmup
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    
    # Learning rate scheduler
    def lr_lambda(step):
        warmup_steps = 4000
        step = max(step, 1)
        return min(step ** (-0.5), step * warmup_steps ** (-1.5))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training history
    train_losses = []
    val_losses = []
    perplexities = []
    
    print("\n" + "="*60)
    print("Training Transformer Music Generator (Task 3)")
    print("="*60)
    
    global_step = 0
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        
        for batch, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch = batch.to(device)
            mask = mask.to(device)
            
            # Prepare inputs and targets
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            input_mask = mask[:, :-1]
            
            # Forward pass
            logits = model(inputs, src_key_padding_mask=~input_mask)
            
            # Compute loss
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
                ignore_index=pad_token_id
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item() * batch.size(0)
            global_step += 1
        
        train_loss = epoch_loss / len(train_dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch, mask in val_loader:
                batch = batch.to(device)
                mask = mask.to(device)
                
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                input_mask = mask[:, :-1]
                
                logits = model(inputs, src_key_padding_mask=~input_mask)
                
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    targets.reshape(-1),
                    ignore_index=pad_token_id
                )
                
                val_loss += loss.item() * batch.size(0)
        
        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        
        # Compute perplexity
        perplexity = math.exp(val_loss)
        perplexities.append(perplexity)
        
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'transformer_final.pth')
    
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)
    
    print(f"\nModel saved to: {model_path}")
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss', color='#4C72B0')
    ax1.plot(val_losses, label='Validation Loss', color='#DD8452', linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('Transformer Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Perplexity
    ax2.plot(perplexities, label='Validation Perplexity', color='#55A868')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Transformer Perplexity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Task 3: Transformer Training Curves')
    plt.tight_layout()
    
    plot_path = os.path.join(PLOTS_DIR, 'task3_transformer_loss.png')
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"Training curves saved to: {plot_path}")
    
    return model, train_losses, val_losses, perplexities


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Transformer Music Generator')
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data (tokenized sequences)')
    parser.add_argument('--val_data', type=str, required=True,
                       help='Path to validation data (tokenized sequences)')
    parser.add_argument('--vocab_size', type=int, required=True,
                       help='Size of token vocabulary')
    parser.add_argument('--output_dir', type=str, default=MODELS_DIR,
                       help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=TRANSFORMER_EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=TRANSFORMER_BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=TRANSFORMER_LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--d_model', type=int, default=256,
                       help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--max_seq_len', type=int, default=512,
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    train_transformer(
        train_path=args.train_data,
        val_path=args.val_data,
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len
    )
