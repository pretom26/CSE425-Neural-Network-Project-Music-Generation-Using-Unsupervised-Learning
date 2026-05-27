"""
Rhythm diversity scoring and analysis.
"""

import numpy as np
import pretty_midi
from typing import List, Dict
from collections import Counter


def compute_rhythm_diversity(midi_path: str, 
                            resolution: float = 0.05) -> float:
    """
    Compute rhythm diversity score.
    
    Formula: D_rhythm = (# unique durations) / (# total notes)
    
    Args:
        midi_path: Path to MIDI file
        resolution: Quantization resolution in seconds (default: 50ms)
        
    Returns:
        Diversity score in range [0, 1]
        - Higher values indicate more rhythmic variety
        - Lower values indicate repetitive rhythms
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        
        durations = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    duration = note.end - note.start
                    # Quantize to avoid floating point issues
                    quantized = round(duration / resolution) * resolution
                    durations.append(quantized)
        
        if len(durations) == 0:
            return 0.0
        
        # Compute diversity
        unique_durations = len(set(durations))
        total_notes = len(durations)
        
        return unique_durations / total_notes
    
    except Exception as e:
        print(f"Error computing rhythm diversity for {midi_path}: {e}")
        return 0.0


def get_duration_distribution(midi_path: str, 
                              resolution: float = 0.05) -> Dict[float, int]:
    """
    Get the distribution of note durations.
    
    Args:
        midi_path: Path to MIDI file
        resolution: Quantization resolution in seconds
        
    Returns:
        Dictionary mapping duration -> count
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        
        durations = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    duration = note.end - note.start
                    quantized = round(duration / resolution) * resolution
                    durations.append(quantized)
        
        return dict(Counter(durations))
    
    except Exception as e:
        print(f"Error getting duration distribution for {midi_path}: {e}")
        return {}


def compute_inter_onset_intervals(midi_path: str) -> List[float]:
    """
    Compute inter-onset intervals (IOI) between consecutive notes.
    
    Args:
        midi_path: Path to MIDI file
        
    Returns:
        List of inter-onset intervals in seconds
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        
        # Collect all note onsets
        onsets = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                onsets.extend([note.start for note in instrument.notes])
        
        if len(onsets) < 2:
            return []
        
        # Sort onsets
        onsets.sort()
        
        # Compute intervals
        ioi = [onsets[i+1] - onsets[i] for i in range(len(onsets)-1)]
        
        return ioi
    
    except Exception as e:
        print(f"Error computing IOI for {midi_path}: {e}")
        return []


def compute_rhythm_complexity(midi_path: str) -> float:
    """
    Compute rhythm complexity using IOI entropy.
    
    Args:
        midi_path: Path to MIDI file
        
    Returns:
        Complexity score (entropy of IOI distribution)
    """
    ioi = compute_inter_onset_intervals(midi_path)
    
    if len(ioi) == 0:
        return 0.0
    
    # Quantize IOI
    ioi_quantized = [round(x, 2) for x in ioi]
    
    # Compute distribution
    counts = Counter(ioi_quantized)
    total = sum(counts.values())
    probs = np.array([count / total for count in counts.values()])
    
    # Compute entropy
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    
    return entropy


def compute_note_density(midi_path: str) -> float:
    """
    Compute note density (notes per second).
    
    Args:
        midi_path: Path to MIDI file
        
    Returns:
        Notes per second
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        
        total_notes = 0
        for instrument in midi.instruments:
            if not instrument.is_drum:
                total_notes += len(instrument.notes)
        
        duration = midi.get_end_time()
        
        if duration == 0:
            return 0.0
        
        return total_notes / duration
    
    except Exception as e:
        print(f"Error computing note density for {midi_path}: {e}")
        return 0.0
