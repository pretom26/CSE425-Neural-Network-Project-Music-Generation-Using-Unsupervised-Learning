"""
Training script for LSTM Autoencoder (Task 1).
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

from models.autoencoder import LSTMAutoencoder, FocalLoss
from config import *


class PianoRollDataset(Dataset):
    """Dataset for piano-roll windows."""
    
    def __init__(self, npy_path: str):
        self.data = np.load(npy_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)


def train_autoencoder(train_path: str,
                     val_path: str,
                     output_dir: str = MODELS_DIR,
                     epochs: int = AE_EPOCHS,
                     batch_size: int = AE_BATCH_SIZE,
                     learning_rate: float = AE_LEARNING_RATE,
                     device: str = None):
    """
    Train LSTM Autoencoder.
    
    Args:
        train_path: Path to training data (.npy file)
        val_path: Path to validation data (.npy file)
        output_dir: Directory to save model and plots
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
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
    model = LSTMAutoencoder(
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
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    train_losses = []
    val_losses = []
    
    print("\n" + "="*60)
    print("Training LSTM Autoencoder (Task 1)")
    print("="*60)
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch = batch.to(device)
            
            # Forward pass
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * batch.size(0)
        
        train_loss = epoch_loss / len(train_dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, _ = model(batch)
                loss = criterion(recon, batch)
                val_loss += loss.item() * batch.size(0)
        
        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'ae_final.pth')
    
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)
    
    print(f"\nModel saved to: {model_path}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='#4C72B0')
    plt.plot(val_losses, label='Validation Loss', color='#DD8452', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Focal Loss')
    plt.title('Task 1: LSTM Autoencoder Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(PLOTS_DIR, 'task1_ae_loss.png')
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"Training curve saved to: {plot_path}")
    
    return model, train_losses, val_losses


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LSTM Autoencoder')
    parser.add_argument('--train_data', type=str, default=TRAIN_PIANOROLL,
                       help='Path to training data')
    parser.add_argument('--val_data', type=str, default=VAL_PIANOROLL,
                       help='Path to validation data')
    parser.add_argument('--output_dir', type=str, default=MODELS_DIR,
                       help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=AE_EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=AE_BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=AE_LEARNING_RATE,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    train_autoencoder(
        train_path=args.train_data,
        val_path=args.val_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
