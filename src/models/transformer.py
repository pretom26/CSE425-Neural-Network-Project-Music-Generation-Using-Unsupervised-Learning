"""
Transformer-based music generator (Task 3).
Decoder-only architecture for autoregressive generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerMusicGenerator(nn.Module):
    """
    Decoder-only Transformer for music generation.
    
    Architecture similar to GPT: token embedding + positional encoding +
    stacked transformer decoder layers + output projection.
    """
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 1024,
                 dropout: float = 0.1,
                 max_seq_len: int = 512):
        """
        Initialize Transformer generator.
        
        Args:
            vocab_size: Size of token vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feedforward dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int, device: str = 'cpu') -> torch.Tensor:
        """
        Generate causal mask for autoregressive generation.
        
        Args:
            sz: Sequence length
            device: Device to create mask on
            
        Returns:
            Causal mask of shape (sz, sz)
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self,
               src: torch.Tensor,
               src_mask: Optional[torch.Tensor] = None,
               src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer.
        
        Args:
            src: Input token IDs of shape (batch, seq_len)
            src_mask: Causal mask of shape (seq_len, seq_len)
            src_key_padding_mask: Padding mask of shape (batch, seq_len)
            
        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        # Embed tokens
        x = self.token_embedding(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Generate causal mask if not provided
        if src_mask is None:
            src_mask = self.generate_square_subsequent_mask(src.size(1), src.device)
        
        # Pass through transformer decoder
        # Note: TransformerDecoder expects (tgt, memory) but we use self-attention only
        output = self.transformer_decoder(
            tgt=x,
            memory=x,
            tgt_mask=src_mask,
            tgt_key_padding_mask=src_key_padding_mask
        )
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits
    
    @torch.no_grad()
    def generate(self,
                prompt: torch.Tensor,
                max_length: int = 512,
                temperature: float = 1.0,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None,
                eos_token_id: Optional[int] = None) -> torch.Tensor:
        """
        Generate sequence autoregressively.
        
        Args:
            prompt: Initial token IDs of shape (batch, prompt_len)
            max_length: Maximum length to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            eos_token_id: End-of-sequence token ID (stops generation)
            
        Returns:
            Generated token IDs of shape (batch, generated_len)
        """
        self.eval()
        
        batch_size = prompt.size(0)
        device = prompt.device
        
        # Start with prompt
        generated = prompt
        
        for _ in range(max_length - prompt.size(1)):
            # Forward pass
            logits = self.forward(generated)
            
            # Get logits for last position
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated


def compute_perplexity(model: TransformerMusicGenerator,
                      dataloader: torch.utils.data.DataLoader,
                      device: str = 'cpu',
                      pad_token_id: int = 0) -> float:
    """
    Compute perplexity on a dataset.
    
    Args:
        model: Transformer model
        dataloader: DataLoader for evaluation
        device: Device to run on
        pad_token_id: Padding token ID (ignored in loss)
        
    Returns:
        Perplexity value
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask', None)
            else:
                input_ids = batch.to(device)
                attention_mask = None
            
            # Prepare inputs and targets
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            
            # Forward pass
            logits = model(inputs)
            
            # Compute loss
            loss = F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                targets.reshape(-1),
                ignore_index=pad_token_id,
                reduction='sum'
            )
            
            # Count non-padding tokens
            if attention_mask is not None:
                num_tokens = attention_mask[:, 1:].sum().item()
            else:
                num_tokens = (targets != pad_token_id).sum().item()
            
            total_loss += loss.item()
            total_tokens += num_tokens
    
    # Compute perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity
