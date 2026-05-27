"""
LSTM Autoencoder for music generation (Task 1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LSTMAutoencoder(nn.Module):
    """
    LSTM-based autoencoder for piano-roll sequences.
    
    Architecture:
        Encoder: LSTM -> Linear (to latent)
        Decoder: Linear (from latent) -> LSTM -> Linear (to output)
    """
    
    def __init__(self,
                 input_size: int = 88,
                 hidden_size: int = 256,
                 latent_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.3):
        """
        Initialize LSTM Autoencoder.
        
        Args:
            input_size: Number of input features (88 for piano)
            hidden_size: LSTM hidden dimension
            latent_dim: Latent space dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Encoder: LSTM + projection to latent
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc_z = nn.Linear(hidden_size, latent_dim)
        
        # Decoder: projection from latent + LSTM + output
        self.decoder_fc = nn.Linear(latent_dim, hidden_size)
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.output_layer = nn.Linear(hidden_size, input_size)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence to latent representation.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            Latent vector of shape (batch, latent_dim)
        """
        # Pass through encoder LSTM
        _, (h_n, _) = self.encoder(x)
        
        # Take final hidden state of last layer
        h = h_n[-1]  # (batch, hidden_size)
        
        # Project to latent space
        z = self.fc_z(h)  # (batch, latent_dim)
        
        return z
    
    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Decode latent representation to output sequence.
        
        Args:
            z: Latent tensor of shape (batch, latent_dim)
            seq_len: Length of output sequence
            
        Returns:
            Output tensor of shape (batch, seq_len, input_size)
        """
        batch_size = z.size(0)
        
        # Project latent to hidden dimension
        h = self.decoder_fc(z)  # (batch, hidden_size)
        
        # Repeat across time steps
        h = h.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_size)
        
        # Pass through decoder LSTM
        dec_out, _ = self.decoder(h)  # (batch, seq_len, hidden_size)
        
        # Project to output dimension (raw logits)
        output = self.output_layer(dec_out)  # (batch, seq_len, input_size)
        
        return output
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode then decode.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            Tuple of (reconstruction, latent):
                - reconstruction: (batch, seq_len, input_size)
                - latent: (batch, latent_dim)
        """
        seq_len = x.size(1)
        
        # Encode
        z = self.encode(x)
        
        # Decode
        recon = self.decode(z, seq_len)
        
        return recon, z
    
    def generate(self, 
                z: torch.Tensor, 
                seq_len: int = 128, 
                threshold: float = 0.35) -> torch.Tensor:
        """
        Generate music from latent code.
        
        Args:
            z: Latent tensor of shape (batch, latent_dim)
            seq_len: Length of sequence to generate
            threshold: Binarization threshold
            
        Returns:
            Binary piano-roll of shape (batch, seq_len, input_size)
        """
        self.eval()
        with torch.no_grad():
            # Decode
            logits = self.decode(z, seq_len)
            
            # Apply sigmoid and threshold
            probs = torch.sigmoid(logits)
            binary = (probs > threshold).float()
            
            return binary
    
    def sample_latent(self, 
                     batch_size: int = 1, 
                     device: str = 'cpu') -> torch.Tensor:
        """
        Sample random latent codes from standard normal distribution.
        
        Args:
            batch_size: Number of samples
            device: Device to create tensor on
            
        Returns:
            Latent tensor of shape (batch_size, latent_dim)
        """
        return torch.randn(batch_size, self.latent_dim, device=device)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in binary piano-roll.
    
    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits of shape (batch, seq_len, input_size)
            targets: Ground truth binary labels of same shape
            
        Returns:
            Scalar loss value
        """
        # Binary cross-entropy with logits
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute pt (probability of correct class)
        pt = torch.exp(-BCE)
        
        # Focal loss formula
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE
        
        return focal_loss.mean()
