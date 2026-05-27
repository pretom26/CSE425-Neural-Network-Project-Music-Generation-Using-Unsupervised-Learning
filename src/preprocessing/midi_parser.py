"""
MIDI file parsing utilities.
Handles loading and basic processing of MIDI files.
"""

import pretty_midi
import numpy as np
from typing import List, Tuple, Optional


def load_midi(midi_path: str) -> Optional[pretty_midi.PrettyMIDI]:
    """
    Load a MIDI file.
    
    Args:
        midi_path: Path to MIDI file
        
    Returns:
        PrettyMIDI object or None if loading fails
    """
    try:
        return pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading {midi_path}: {e}")
        return None


def get_notes(midi: pretty_midi.PrettyMIDI) -> List[pretty_midi.Note]:
    """
    Extract all notes from a MIDI file.
    
    Args:
        midi: PrettyMIDI object
        
    Returns:
        List of Note objects
    """
    notes = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    return notes


def get_duration(midi: pretty_midi.PrettyMIDI) -> float:
    """
    Get the duration of a MIDI file in seconds.
    
    Args:
        midi: PrettyMIDI object
        
    Returns:
        Duration in seconds
    """
    return midi.get_end_time()


def get_tempo_changes(midi: pretty_midi.PrettyMIDI) -> List[Tuple[float, float]]:
    """
    Get tempo changes in a MIDI file.
    
    Args:
        midi: PrettyMIDI object
        
    Returns:
        List of (time, tempo) tuples
    """
    return midi.get_tempo_changes()


def filter_piano_range(notes: List[pretty_midi.Note], 
                       min_pitch: int = 21, 
                       max_pitch: int = 108) -> List[pretty_midi.Note]:
    """
    Filter notes to piano range.
    
    Args:
        notes: List of Note objects
        min_pitch: Minimum MIDI pitch (default: 21 = A0)
        max_pitch: Maximum MIDI pitch (default: 108 = C8)
        
    Returns:
        Filtered list of notes
    """
    return [n for n in notes if min_pitch <= n.pitch <= max_pitch]


def get_pitch_range(notes: List[pretty_midi.Note]) -> Tuple[int, int]:
    """
    Get the pitch range of a list of notes.
    
    Args:
        notes: List of Note objects
        
    Returns:
        (min_pitch, max_pitch) tuple
    """
    if not notes:
        return (0, 0)
    pitches = [n.pitch for n in notes]
    return (min(pitches), max(pitches))


def get_note_density(notes: List[pretty_midi.Note], duration: float) -> float:
    """
    Calculate note density (notes per second).
    
    Args:
        notes: List of Note objects
        duration: Total duration in seconds
        
    Returns:
        Notes per second
    """
    if duration == 0:
        return 0.0
    return len(notes) / duration


def quantize_duration(duration: float, resolution: float = 0.05) -> float:
    """
    Quantize note duration to nearest resolution.
    
    Args:
        duration: Duration in seconds
        resolution: Quantization resolution (default: 50ms)
        
    Returns:
        Quantized duration
    """
    return round(duration / resolution) * resolution


def get_unique_durations(notes: List[pretty_midi.Note], 
                         resolution: float = 0.05) -> int:
    """
    Count unique note durations.
    
    Args:
        notes: List of Note objects
        resolution: Quantization resolution
        
    Returns:
        Number of unique durations
    """
    durations = set()
    for note in notes:
        duration = note.end - note.start
        quantized = quantize_duration(duration, resolution)
        durations.add(quantized)
    return len(durations)
