"""
Evaluation metrics for generated music.
Implements pitch histogram similarity, rhythm diversity, and repetition ratio.
"""

import numpy as np
import pretty_midi
from typing import List, Tuple
from collections import Counter


def compute_pitch_histogram(midi_path: str, normalize: bool = True) -> np.ndarray:
    """
    Compute pitch class histogram (12 bins for chromatic scale).
    
    Args:
        midi_path: Path to MIDI file
        normalize: Whether to normalize to sum to 1
        
    Returns:
        Histogram array of shape (12,)
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        
        # Count pitch classes
        pitch_counts = np.zeros(12)
        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    pitch_class = note.pitch % 12
                    pitch_counts[pitch_class] += 1
        
        # Normalize
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
    Compute similarity between pitch histograms.
    
    Args:
        gen_midi: Path to generated MIDI file
        ref_midi: Path to reference MIDI file
        metric: Distance metric ('l1' or 'l2')
        
    Returns:
        Distance value (lower = more similar)
    """
    hist_gen = compute_pitch_histogram(gen_midi)
    hist_ref = compute_pitch_histogram(ref_midi)
    
    if metric == 'l1':
        return np.sum(np.abs(hist_gen - hist_ref))
    elif metric == 'l2':
        return np.sqrt(np.sum((hist_gen - hist_ref) ** 2))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_rhythm_diversity(midi_path: str, 
                            resolution: float = 0.05) -> float:
    """
    Compute rhythm diversity score.
    
    Measures the variety of note durations.
    
    Args:
        midi_path: Path to MIDI file
        resolution: Quantization resolution in seconds
        
    Returns:
        Diversity score (# unique durations / # total notes)
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


def compute_repetition_ratio(midi_path: str, n: int = 4) -> float:
    """
    Compute repetition ratio using n-gram analysis.
    
    Args:
        midi_path: Path to MIDI file
        n: N-gram size (default: 4)
        
    Returns:
        Repetition ratio (# repeated n-grams / # total n-grams)
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        
        # Extract pitch sequence sorted by onset time
        notes = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                notes.extend(instrument.notes)
        
        if len(notes) < n:
            return 0.0
        
        # Sort by start time
        notes.sort(key=lambda x: x.start)
        
        # Extract pitch sequence
        pitch_sequence = [note.pitch for note in notes]
        
        # Compute n-grams
        ngrams = []
        for i in range(len(pitch_sequence) - n + 1):
            ngram = tuple(pitch_sequence[i:i+n])
            ngrams.append(ngram)
        
        if len(ngrams) == 0:
            return 0.0
        
        # Count repetitions
        ngram_counts = Counter(ngrams)
        repeated = sum(1 for count in ngram_counts.values() if count > 1)
        
        return repeated / len(ngrams)
    
    except Exception as e:
        print(f"Error computing repetition ratio for {midi_path}: {e}")
        return 0.0


def evaluate_midi_file(midi_path: str, 
                      ref_midi: str = None) -> dict:
    """
    Compute all metrics for a MIDI file.
    
    Args:
        midi_path: Path to generated MIDI file
        ref_midi: Path to reference MIDI file (for pitch similarity)
        
    Returns:
        Dictionary of metric values
    """
    metrics = {}
    
    # Rhythm diversity
    metrics['rhythm_diversity'] = compute_rhythm_diversity(midi_path)
    
    # Repetition ratio
    metrics['repetition_ratio'] = compute_repetition_ratio(midi_path)
    
    # Pitch histogram similarity (if reference provided)
    if ref_midi is not None:
        metrics['pitch_similarity'] = pitch_histogram_similarity(midi_path, ref_midi)
    
    return metrics


def evaluate_midi_batch(gen_midis: List[str],
                       ref_midis: List[str] = None) -> dict:
    """
    Evaluate a batch of MIDI files and compute statistics.
    
    Args:
        gen_midis: List of generated MIDI file paths
        ref_midis: List of reference MIDI file paths (optional)
        
    Returns:
        Dictionary with mean and std for each metric
    """
    all_metrics = {
        'rhythm_diversity': [],
        'repetition_ratio': [],
        'pitch_similarity': []
    }
    
    for i, gen_midi in enumerate(gen_midis):
        ref_midi = ref_midis[i] if ref_midis and i < len(ref_midis) else None
        metrics = evaluate_midi_file(gen_midi, ref_midi)
        
        all_metrics['rhythm_diversity'].append(metrics['rhythm_diversity'])
        all_metrics['repetition_ratio'].append(metrics['repetition_ratio'])
        
        if 'pitch_similarity' in metrics:
            all_metrics['pitch_similarity'].append(metrics['pitch_similarity'])
    
    # Compute statistics
    results = {}
    for key, values in all_metrics.items():
        if values:
            results[f'{key}_mean'] = np.mean(values)
            results[f'{key}_std'] = np.std(values)
            results[f'{key}_values'] = values
    
    return results


def print_evaluation_results(results: dict, model_name: str = "Model"):
    """
    Print evaluation results in a formatted table.
    
    Args:
        results: Dictionary of evaluation results
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {model_name}")
    print(f"{'='*60}")
    
    for key in ['pitch_similarity', 'rhythm_diversity', 'repetition_ratio']:
        mean_key = f'{key}_mean'
        std_key = f'{key}_std'
        
        if mean_key in results:
            print(f"{key:25s}: {results[mean_key]:.3f} ± {results[std_key]:.3f}")
    
    print(f"{'='*60}\n")
