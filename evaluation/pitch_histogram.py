"""
Pitch histogram computation and similarity metrics.
"""

import numpy as np
import pretty_midi
from typing import Optional


def compute_pitch_histogram(midi_path: str, normalize: bool = True) -> np.ndarray:
    """
    Compute pitch class histogram (12 bins for chromatic scale).
    
    Args:
        midi_path: Path to MIDI file
        normalize: Whether to normalize to sum to 1
        
    Returns:
        Histogram array of shape (12,) representing pitch class distribution
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        
        # Count pitch classes (0-11)
        pitch_counts = np.zeros(12)
        
        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    pitch_class = note.pitch % 12
                    pitch_counts[pitch_class] += 1
        
        # Normalize to probability distribution
        if normalize and pitch_counts.sum() > 0:
            pitch_counts = pitch_counts / pitch_counts.sum()
        
        return pitch_counts
    
    except Exception as e:
        print(f"Error computing pitch histogram for {midi_path}: {e}")
        return np.zeros(12)


def pitch_histogram_similarity(gen_midi: str, 
                               ref_midi: str,
                               metric: str = 'l1') -> float:
    """
    Compute similarity between pitch histograms of two MIDI files.
    
    Formula: H(p, q) = Σ|p_i - q_i| for i ∈ [0, 11]
    
    Args:
        gen_midi: Path to generated MIDI file
        ref_midi: Path to reference MIDI file
        metric: Distance metric ('l1' or 'l2')
        
    Returns:
        Distance value (lower = more similar)
        - L1: Range [0, 2], 0 = identical
        - L2: Range [0, √2], 0 = identical
    """
    hist_gen = compute_pitch_histogram(gen_midi, normalize=True)
    hist_ref = compute_pitch_histogram(ref_midi, normalize=True)
    
    if metric == 'l1':
        # Manhattan distance
        return np.sum(np.abs(hist_gen - hist_ref))
    elif metric == 'l2':
        # Euclidean distance
        return np.sqrt(np.sum((hist_gen - hist_ref) ** 2))
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'l1' or 'l2'.")


def compute_pitch_range(midi_path: str) -> tuple:
    """
    Get the pitch range (min, max) used in a MIDI file.
    
    Args:
        midi_path: Path to MIDI file
        
    Returns:
        Tuple of (min_pitch, max_pitch)
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        
        pitches = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                pitches.extend([note.pitch for note in instrument.notes])
        
        if not pitches:
            return (0, 0)
        
        return (min(pitches), max(pitches))
    
    except Exception as e:
        print(f"Error computing pitch range for {midi_path}: {e}")
        return (0, 0)


def compute_pitch_entropy(midi_path: str) -> float:
    """
    Compute entropy of pitch class distribution.
    Higher entropy = more uniform pitch usage.
    
    Args:
        midi_path: Path to MIDI file
        
    Returns:
        Entropy value in bits
    """
    hist = compute_pitch_histogram(midi_path, normalize=True)
    
    # Remove zero probabilities
    hist = hist[hist > 0]
    
    if len(hist) == 0:
        return 0.0
    
    # Compute entropy: H = -Σ p_i * log2(p_i)
    entropy = -np.sum(hist * np.log2(hist))
    
    return entropy
