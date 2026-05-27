"""
Variational Autoencoder for multi-genre music generation (Task 2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .autoencoder import LSTMAutoencoder


class VAE(LSTMAutoencoder):
    """
    Variational Autoencoder extending LSTM Autoencoder.
    
    Key differences from standard autoencoder:
        - Encoder outputs mean (μ) and log-variance (log σ²)
        - Reparameterization trick for sampling: z = μ + σ * ε
        - KL divergence loss term
    """
    
    def __init__(self,
                 input_size: int = 88,
                 hidden_size: int = 256,
                 latent_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.3):
        """
        Initialize VAE.
        
        Args:
            input_size: Number of input features (88 for piano)
            hidden_size: LSTM hidden dimension
            latent_dim: Latent space dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__(input_size, hidden_size, latent_dim, num_layers, dropout)
        
        # Additional layer for log-variance
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            Tuple of (mu, logvar):
                - mu: Mean of shape (batch, latent_dim)
                - logvar: Log-variance of shape (batch, latent_dim)
        """
        # Pass through encoder LSTM
        _, (h_n, _) = self.encoder(x)
        
        # Take final hidden state
        h = h_n[-1]  # (batch, hidden_size)
        
        # Compute mean and log-variance
        mu = self.fc_z(h)  # (batch, latent_dim)
        logvar = self.fc_logvar(h)  # (batch, latent_dim)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, I).
        
        Args:
            mu: Mean tensor of shape (batch, latent_dim)
            logvar: Log-variance tensor of shape (batch, latent_dim)
            
        Returns:
            Sampled latent tensor of shape (batch, latent_dim)
        """
        # Compute standard deviation: σ = exp(0.5 * log σ²)
        std = torch.exp(0.5 * logvar)
        
        # Sample epsilon from standard normal
        eps = torch.randn_like(std)
        
        # Reparameterize: z = μ + σ * ε
        z = mu + std * eps
        
        return z
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            Tuple of (reconstruction, mu, logvar):
                - reconstruction: (batch, seq_len, input_size)
                - mu: (batch, latent_dim)
                - logvar: (batch, latent_dim)
        """
        seq_len = x.size(1)
        
        # Encode to distribution parameters
        mu, logvar = self.encode(x)
        
        # Sample latent code
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon = self.decode(z, seq_len)
        
        return recon, mu, logvar
    
    def generate(self,
                z: torch.Tensor = None,
                batch_size: int = 1,
                seq_len: int = 128,
                threshold: float = 0.35,
                device: str = 'cpu') -> torch.Tensor:
        """
        Generate music from latent code or sample from prior.
        
        Args:
            z: Latent tensor (if None, samples from N(0, I))
            batch_size: Number of samples (if z is None)
            seq_len: Length of sequence to generate
            threshold: Binarization threshold
            device: Device to generate on
            
        Returns:
            Binary piano-roll of shape (batch, seq_len, input_size)
        """
        self.eval()
        with torch.no_grad():
            # Sample from prior if z not provided
            if z is None:
                z = torch.randn(batch_size, self.latent_dim, device=device)
            
            # Decode
            logits = self.decode(z, seq_len)
            
            # Apply sigmoid and threshold
            probs = torch.sigmoid(logits)
            binary = (probs > threshold).float()
            
            return binary
    
    def interpolate(self,
                   z1: torch.Tensor,
                   z2: torch.Tensor,
                   num_steps: int = 8,
                   seq_len: int = 128,
                   threshold: float = 0.35) -> torch.Tensor:
        """
        Interpolate between two latent codes.
        
        Args:
            z1: First latent code of shape (latent_dim,)
            z2: Second latent code of shape (latent_dim,)
            num_steps: Number of interpolation steps
            seq_len: Length of generated sequences
            threshold: Binarization threshold
            
        Returns:
            Interpolated sequences of shape (num_steps, seq_len, input_size)
        """
        self.eval()
        with torch.no_grad():
            # Create interpolation coefficients
            alphas = torch.linspace(0, 1, num_steps, device=z1.device)
            
            # Interpolate: z_α = (1 - α) * z1 + α * z2
            z_interp = []
            for alpha in alphas:
                z = (1 - alpha) * z1 + alpha * z2
                z_interp.append(z)
            
            z_interp = torch.stack(z_interp)  # (num_steps, latent_dim)
            
            # Generate from interpolated codes
            outputs = self.generate(z_interp, seq_len=seq_len, threshold=threshold)
            
            return outputs


def vae_loss(recon: torch.Tensor,
            target: torch.Tensor,
            mu: torch.Tensor,
            logvar: torch.Tensor,
            recon_loss_fn: nn.Module,
            beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss: L = L_recon + β * KL.
    
    Args:
        recon: Reconstructed output
        target: Ground truth target
        mu: Mean of latent distribution
        logvar: Log-variance of latent distribution
        recon_loss_fn: Reconstruction loss function (e.g., FocalLoss)
        beta: KL weighting factor (for β-VAE)
        
    Returns:
        Tuple of (total_loss, recon_loss, kl_loss)
    """
    # Reconstruction loss
    recon_loss = recon_loss_fn(recon, target)
    
    # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
    # Closed form: -0.5 * Σ(1 + log σ² - μ² - σ²)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def kl_annealing_schedule(epoch: int, 
                         total_epochs: int, 
                         warmup_fraction: float = 0.3) -> float:
    """
    Compute β for KL annealing schedule.
    
    Linearly increases β from 0 to 1 over warmup period.
    
    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of training epochs
        warmup_fraction: Fraction of epochs for warmup
        
    Returns:
        Beta value for current epoch
    """
    warmup_epochs = int(total_epochs * warmup_fraction)
    
    if epoch < warmup_epochs:
        return epoch / max(warmup_epochs, 1)
    else:
        return 1.0
