"""
Piano-roll representation utilities.
Converts MIDI files to binary piano-roll matrices.
"""

import numpy as np
import pretty_midi
from typing import List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import WINDOW_LEN, FS, MIN_PITCH, MAX_PITCH, NUM_PITCHES, MIN_DENSITY


def midi_to_piano_roll(midi_path: str, 
                       fs: int = FS,
                       min_pitch: int = MIN_PITCH,
                       max_pitch: int = MAX_PITCH) -> Optional[np.ndarray]:
    """
    Convert MIDI file to binary piano-roll representation.
    
    Args:
        midi_path: Path to MIDI file
        fs: Frames per second (time resolution)
        min_pitch: Minimum MIDI pitch
        max_pitch: Maximum MIDI pitch (exclusive)
        
    Returns:
        Binary piano-roll array of shape (T, 88) or None if conversion fails
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        
        # Get piano-roll (128 pitches x T time steps)
        pr = midi.get_piano_roll(fs=fs)
        
        # Extract piano range (21-108)
        pr = pr[min_pitch:max_pitch, :]
        
        # Binarize (discard velocity information)
        pr = (pr > 0).astype(np.float32)
        
        # Transpose to (T, 88)
        pr = pr.T
        
        return pr
    except Exception as e:
        print(f"Error converting {midi_path}: {e}")
        return None


def segment_piano_roll(piano_roll: np.ndarray, 
                       window_len: int = WINDOW_LEN,
                       overlap: int = 0) -> List[np.ndarray]:
    """
    Segment piano-roll into fixed-length windows.
    
    Args:
        piano_roll: Piano-roll array of shape (T, 88)
        window_len: Length of each window
        overlap: Number of overlapping frames between windows
        
    Returns:
        List of window arrays, each of shape (window_len, 88)
    """
    windows = []
    step = window_len - overlap
    
    for i in range(0, len(piano_roll) - window_len + 1, step):
        window = piano_roll[i:i + window_len]
        windows.append(window)
    
    return windows


def filter_sparse_windows(windows: List[np.ndarray], 
                         min_density: float = MIN_DENSITY) -> List[np.ndarray]:
    """
    Filter out windows with too few active notes.
    
    Args:
        windows: List of window arrays
        min_density: Minimum fraction of active cells
        
    Returns:
        Filtered list of windows
    """
    filtered = []
    for window in windows:
        density = np.mean(window)
        if density >= min_density:
            filtered.append(window)
    return filtered


def midi_to_windows(midi_path: str,
                   window_len: int = WINDOW_LEN,
                   fs: int = FS,
                   min_density: float = MIN_DENSITY) -> List[np.ndarray]:
    """
    Complete pipeline: MIDI -> piano-roll -> windows -> filtered.
    
    Args:
        midi_path: Path to MIDI file
        window_len: Length of each window
        fs: Frames per second
        min_density: Minimum density for filtering
        
    Returns:
        List of filtered window arrays
    """
    # Convert to piano-roll
    pr = midi_to_piano_roll(midi_path, fs=fs)
    if pr is None:
        return []
    
    # Segment into windows
    windows = segment_piano_roll(pr, window_len=window_len)
    
    # Filter sparse windows
    windows = filter_sparse_windows(windows, min_density=min_density)
    
    return windows


def piano_roll_to_midi(piano_roll: np.ndarray,
                      fs: int = FS,
                      min_pitch: int = MIN_PITCH,
                      program: int = 0,
                      velocity: int = 80) -> pretty_midi.PrettyMIDI:
    """
    Convert binary piano-roll back to MIDI.
    
    Args:
        piano_roll: Binary piano-roll array of shape (T, 88)
        fs: Frames per second
        min_pitch: Minimum MIDI pitch
        program: MIDI program number (0 = Acoustic Grand Piano)
        velocity: Note velocity
        
    Returns:
        PrettyMIDI object
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)
    
    frame_duration = 1.0 / fs
    
    # Process each pitch
    for pitch_idx in range(piano_roll.shape[1]):
        pitch = min_pitch + pitch_idx
        
        # Find note onsets and offsets
        active = piano_roll[:, pitch_idx] > 0
        changes = np.diff(np.concatenate([[0], active.astype(int), [0]]))
        
        onsets = np.where(changes == 1)[0]
        offsets = np.where(changes == -1)[0]
        
        # Create notes
        for onset, offset in zip(onsets, offsets):
            start_time = onset * frame_duration
            end_time = offset * frame_duration
            
            # Ensure minimum duration
            if end_time - start_time < frame_duration:
                end_time = start_time + frame_duration
            
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start_time,
                end=end_time
            )
            instrument.notes.append(note)
    
    midi.instruments.append(instrument)
    return midi


def save_piano_roll_as_midi(piano_roll: np.ndarray,
                            output_path: str,
                            fs: int = FS,
                            **kwargs):
    """
    Save piano-roll as MIDI file.
    
    Args:
        piano_roll: Binary piano-roll array
        output_path: Output MIDI file path
        fs: Frames per second
        **kwargs: Additional arguments for piano_roll_to_midi
    """
    midi = piano_roll_to_midi(piano_roll, fs=fs, **kwargs)
    midi.write(output_path)


def compute_sparsity(piano_roll: np.ndarray) -> float:
    """
    Compute sparsity (fraction of zero cells).
    
    Args:
        piano_roll: Piano-roll array
        
    Returns:
        Sparsity ratio (0 to 1)
    """
    return 1.0 - np.mean(piano_roll)


def get_active_pitch_range(piano_roll: np.ndarray) -> tuple:
    """
    Get the range of active pitches in a piano-roll.
    
    Args:
        piano_roll: Piano-roll array of shape (T, 88)
        
    Returns:
        (min_active_pitch, max_active_pitch) tuple
    """
    active_pitches = np.where(np.any(piano_roll > 0, axis=0))[0]
    if len(active_pitches) == 0:
        return (0, 0)
    return (active_pitches[0], active_pitches[-1])
