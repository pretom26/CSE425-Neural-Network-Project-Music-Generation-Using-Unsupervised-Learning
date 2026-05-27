"""
Music generation utilities for all models.
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.autoencoder import LSTMAutoencoder
from models.vae import VAE
from models.transformer import TransformerMusicGenerator
from preprocessing.piano_roll import save_piano_roll_as_midi
from config import *


def generate_from_autoencoder(model: LSTMAutoencoder,
                              num_samples: int = 5,
                              seq_len: int = WINDOW_LEN,
                              threshold: float = GEN_THRESHOLD,
                              device: str = 'cpu',
                              output_dir: str = MIDI_OUTPUT_DIR,
                              prefix: str = 'ae_sample') -> List[str]:
    """
    Generate music samples from trained autoencoder.
    
    Args:
        model: Trained autoencoder model
        num_samples: Number of samples to generate
        seq_len: Length of sequences
        threshold: Binarization threshold
        device: Device to generate on
        output_dir: Output directory for MIDI files
        prefix: Filename prefix
        
    Returns:
        List of generated MIDI file paths
    """
    model.eval()
    model.to(device)
    
    generated_files = []
    
    with torch.no_grad():
        for i in range(num_samples):
            # Sample random latent code
            z = torch.randn(1, model.latent_dim, device=device)
            
            # Generate
            piano_roll = model.generate(z, seq_len=seq_len, threshold=threshold)
            piano_roll = piano_roll.squeeze(0).cpu().numpy()
            
            # Save as MIDI
            output_path = os.path.join(output_dir, f'{prefix}_{i}.mid')
            save_piano_roll_as_midi(piano_roll, output_path, fs=FS)
            generated_files.append(output_path)
            
            print(f"Generated: {output_path}")
    
    return generated_files


def generate_from_vae(model: VAE,
                     num_samples: int = 8,
                     seq_len: int = WINDOW_LEN,
                     threshold: float = GEN_THRESHOLD,
                     device: str = 'cpu',
                     output_dir: str = MIDI_OUTPUT_DIR,
                     prefix: str = 'vae_sample') -> List[str]:
    """
    Generate music samples from trained VAE.
    
    Args:
        model: Trained VAE model
        num_samples: Number of samples to generate
        seq_len: Length of sequences
        threshold: Binarization threshold
        device: Device to generate on
        output_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        List of generated MIDI file paths
    """
    model.eval()
    model.to(device)
    
    generated_files = []
    
    with torch.no_grad():
        for i in range(num_samples):
            # Sample from prior N(0, I)
            z = torch.randn(1, model.latent_dim, device=device)
            
            # Generate
            piano_roll = model.generate(z, seq_len=seq_len, threshold=threshold, device=device)
            piano_roll = piano_roll.squeeze(0).cpu().numpy()
            
            # Save as MIDI
            output_path = os.path.join(output_dir, f'{prefix}_{i}.mid')
            save_piano_roll_as_midi(piano_roll, output_path, fs=FS)
            generated_files.append(output_path)
            
            print(f"Generated: {output_path}")
    
    return generated_files


def generate_vae_interpolation(model: VAE,
                               data_loader: torch.utils.data.DataLoader,
                               num_steps: int = 8,
                               seq_len: int = WINDOW_LEN,
                               threshold: float = GEN_THRESHOLD,
                               device: str = 'cpu',
                               output_dir: str = MIDI_OUTPUT_DIR,
                               prefix: str = 'vae_interp') -> List[str]:
    """
    Generate interpolation between two real pieces.
    
    Args:
        model: Trained VAE model
        data_loader: DataLoader with real data
        num_steps: Number of interpolation steps
        seq_len: Length of sequences
        threshold: Binarization threshold
        device: Device to generate on
        output_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        List of generated MIDI file paths
    """
    model.eval()
    model.to(device)
    
    # Get two real samples
    batch = next(iter(data_loader))
    x1, x2 = batch[0:1].to(device), batch[1:2].to(device)
    
    with torch.no_grad():
        # Encode to get latent codes
        mu1, _ = model.encode(x1)
        mu2, _ = model.encode(x2)
        
        # Interpolate
        outputs = model.interpolate(mu1.squeeze(0), mu2.squeeze(0), 
                                   num_steps=num_steps, seq_len=seq_len, 
                                   threshold=threshold)
    
    # Save interpolated samples
    generated_files = []
    for i, piano_roll in enumerate(outputs):
        piano_roll = piano_roll.cpu().numpy()
        output_path = os.path.join(output_dir, f'{prefix}_{i:02d}.mid')
        save_piano_roll_as_midi(piano_roll, output_path, fs=FS)
        generated_files.append(output_path)
        print(f"Generated interpolation: {output_path}")
    
    return generated_files


def generate_from_transformer(model: TransformerMusicGenerator,
                              tokenizer,
                              num_samples: int = 10,
                              max_length: int = 512,
                              temperature: float = GEN_TEMPERATURE,
                              top_k: int = GEN_TOP_K,
                              device: str = 'cpu',
                              output_dir: str = MIDI_OUTPUT_DIR,
                              prefix: str = 'transformer_sample') -> List[str]:
    """
    Generate music from trained Transformer.
    
    Args:
        model: Trained Transformer model
        tokenizer: MIDI tokenizer
        num_samples: Number of samples to generate
        max_length: Maximum sequence length
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        device: Device to generate on
        output_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        List of generated MIDI file paths
    """
    model.eval()
    model.to(device)
    
    generated_files = []
    
    for i in range(num_samples):
        # Start with BOS token
        prompt = torch.tensor([[tokenizer.bos_token_id]], device=device)
        
        # Generate
        with torch.no_grad():
            generated = model.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode to MIDI
        token_ids = generated.squeeze(0).cpu().tolist()
        output_path = os.path.join(output_dir, f'{prefix}_{i}.mid')
        
        try:
            tokenizer.decode(token_ids, output_path)
            generated_files.append(output_path)
            print(f"Generated: {output_path}")
        except Exception as e:
            print(f"Error decoding sample {i}: {e}")
    
    return generated_files


def generate_baseline_random(num_samples: int = 5,
                            seq_len: int = WINDOW_LEN,
                            density: float = 0.05,
                            output_dir: str = MIDI_OUTPUT_DIR,
                            prefix: str = 'baseline_random') -> List[str]:
    """
    Generate random baseline samples.
    
    Args:
        num_samples: Number of samples
        seq_len: Sequence length
        density: Fraction of active notes
        output_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        List of generated MIDI file paths
    """
    generated_files = []
    
    for i in range(num_samples):
        # Random binary piano-roll
        piano_roll = (np.random.rand(seq_len, NUM_PITCHES) < density).astype(np.float32)
        
        # Save as MIDI
        output_path = os.path.join(output_dir, f'{prefix}_{i}.mid')
        save_piano_roll_as_midi(piano_roll, output_path, fs=FS)
        generated_files.append(output_path)
        
        print(f"Generated random baseline: {output_path}")
    
    return generated_files


def generate_baseline_markov(train_data: np.ndarray,
                            num_samples: int = 5,
                            seq_len: int = WINDOW_LEN,
                            output_dir: str = MIDI_OUTPUT_DIR,
                            prefix: str = 'baseline_markov') -> List[str]:
    """
    Generate Markov chain baseline samples.
    
    Args:
        train_data: Training data array of shape (N, seq_len, 88)
        num_samples: Number of samples
        seq_len: Sequence length
        output_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        List of generated MIDI file paths
    """
    # Build transition matrix
    transition_counts = np.zeros((NUM_PITCHES, NUM_PITCHES))
    
    for sample in train_data:
        for t in range(len(sample) - 1):
            active_now = np.where(sample[t] > 0)[0]
            active_next = np.where(sample[t+1] > 0)[0]
            
            for p1 in active_now:
                for p2 in active_next:
                    transition_counts[p1, p2] += 1
    
    # Normalize to probabilities
    transition_probs = transition_counts / (transition_counts.sum(axis=1, keepdims=True) + 1e-8)
    
    generated_files = []
    
    for i in range(num_samples):
        piano_roll = np.zeros((seq_len, NUM_PITCHES), dtype=np.float32)
        
        # Start with random pitch
        current_pitch = np.random.randint(0, NUM_PITCHES)
        piano_roll[0, current_pitch] = 1
        
        # Generate sequence
        for t in range(1, seq_len):
            # Sample next pitch from transition probabilities
            probs = transition_probs[current_pitch]
            if probs.sum() > 0:
                current_pitch = np.random.choice(NUM_PITCHES, p=probs)
                piano_roll[t, current_pitch] = 1
        
        # Save as MIDI
        output_path = os.path.join(output_dir, f'{prefix}_{i}.mid')
        save_piano_roll_as_midi(piano_roll, output_path, fs=FS)
        generated_files.append(output_path)
        
        print(f"Generated Markov baseline: {output_path}")
    
    return generated_files
