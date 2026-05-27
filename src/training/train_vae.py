"""
Training script for Variational Autoencoder (Task 2).
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vae import VAE, vae_loss, kl_annealing_schedule
from models.autoencoder import FocalLoss
from config import *


class PianoRollDataset(Dataset):
    """Dataset for piano-roll windows."""
    
    def __init__(self, npy_path: str):
        self.data = np.load(npy_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)


def train_vae(train_path: str,
             val_path: str,
             output_dir: str = MODELS_DIR,
             epochs: int = VAE_EPOCHS,
             batch_size: int = VAE_BATCH_SIZE,
             learning_rate: float = VAE_LEARNING_RATE,
             beta_warmup: float = 0.3,
             device: str = None):
    """
    Train Variational Autoencoder with KL annealing.
    
    Args:
        train_path: Path to training data (.npy file)
        val_path: Path to validation data (.npy file)
        output_dir: Directory to save model and plots
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        beta_warmup: Fraction of epochs for KL warmup
        device: Device to train on (None = auto-detect)
    """
    # Setup device
    if device is None:
        device = DEVICE
    
    print(f"Training on device: {device}")
    
    # Create datasets
    train_dataset = PianoRollDataset(train_path)
    val_dataset = PianoRollDataset(val_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize model
    model = VAE(
        input_size=NUM_PITCHES,
        hidden_size=HIDDEN_SIZE,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    # Multi-GPU support
    if USE_MULTI_GPU and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Loss and optimizer
    recon_criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    train_recon_losses = []
    train_kl_losses = []
    val_losses = []
    
    print("\n" + "="*60)
    print("Training Variational Autoencoder (Task 2)")
    print("="*60)
    
    # Training loop
    for epoch in range(epochs):
        # Compute beta for KL annealing
        beta = kl_annealing_schedule(epoch, epochs, beta_warmup)
        
        # Training
        model.train()
        epoch_recon = 0.0
        epoch_kl = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch = batch.to(device)
            
            # Forward pass
            recon, mu, logvar = model(batch)
            
            # Compute loss
            total_loss, recon_loss, kl_loss = vae_loss(
                recon, batch, mu, logvar, recon_criterion, beta
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_recon += recon_loss.item() * batch.size(0)
            epoch_kl += kl_loss.item() * batch.size(0)
        
        train_recon = epoch_recon / len(train_dataset)
        train_kl = epoch_kl / len(train_dataset)
        train_recon_losses.append(train_recon)
        train_kl_losses.append(train_kl)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                total_loss, _, _ = vae_loss(
                    recon, batch, mu, logvar, recon_criterion, beta
                )
                val_loss += total_loss.item() * batch.size(0)
        
        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1:3d} | β={beta:.2f} | Recon: {train_recon:.4f} | "
              f"KL: {train_kl:.4f} | Val: {val_loss:.4f}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'vae_final.pth')
    
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)
    
    print(f"\nModel saved to: {model_path}")
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reconstruction loss
    ax1.plot(train_recon_losses, label='Train Recon', color='#4C72B0')
    ax1.plot(val_losses, label='Validation Total', color='#DD8452', linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('VAE Reconstruction Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # KL divergence
    ax2.plot(train_kl_losses, label='Train KL', color='#55A868')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('KL Divergence')
    ax2.set_title('VAE KL Divergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Task 2: VAE Training Curves')
    plt.tight_layout()
    
    plot_path = os.path.join(PLOTS_DIR, 'task2_vae_loss.png')
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"Training curves saved to: {plot_path}")
    
    return model, train_recon_losses, train_kl_losses, val_losses


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Variational Autoencoder')
    parser.add_argument('--train_data', type=str, default=TRAIN_PIANOROLL,
                       help='Path to training data')
    parser.add_argument('--val_data', type=str, default=VAL_PIANOROLL,
                       help='Path to validation data')
    parser.add_argument('--output_dir', type=str, default=MODELS_DIR,
                       help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=VAE_EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=VAE_BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=VAE_LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--beta_warmup', type=float, default=0.3,
                       help='Fraction of epochs for KL warmup')
    
    args = parser.parse_args()
    
    train_vae(
        train_path=args.train_data,
        val_path=args.val_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        beta_warmup=args.beta_warmup
    )
