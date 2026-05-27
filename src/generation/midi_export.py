"""
MIDI file export utilities.
Converts piano-roll and token representations to MIDI files.
"""

import pretty_midi
import numpy as np
from typing import Optional, List


def piano_roll_to_midi(piano_roll: np.ndarray,
                      fs: int = 16,
                      min_pitch: int = 21,
                      program: int = 0,
                      velocity: int = 80,
                      tempo: float = 120.0) -> pretty_midi.PrettyMIDI:
    """
    Convert binary piano-roll to MIDI file.
    
    Args:
        piano_roll: Binary array of shape (T, 88)
        fs: Frames per second (time resolution)
        min_pitch: Minimum MIDI pitch (default: 21 = A0)
        program: MIDI program number (0 = Acoustic Grand Piano)
        velocity: Note velocity (0-127)
        tempo: Tempo in BPM
        
    Returns:
        PrettyMIDI object
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=program)
    
    frame_duration = 1.0 / fs
    
    # Process each pitch
    for pitch_idx in range(piano_roll.shape[1]):
        pitch = min_pitch + pitch_idx
        
        # Find note onsets and offsets
        active = piano_roll[:, pitch_idx] > 0
        
        # Detect changes (0->1 = onset, 1->0 = offset)
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
                            fs: int = 16,
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


def tokens_to_midi(token_ids: List[int],
                  tokenizer,
                  output_path: str):
    """
    Convert token sequence to MIDI file.
    
    Args:
        token_ids: List of token IDs
        tokenizer: MIDITokenizer instance
        output_path: Output MIDI file path
    """
    try:
        tokenizer.decode(token_ids, output_path)
    except Exception as e:
        print(f"Error converting tokens to MIDI: {e}")


def add_tempo_changes(midi: pretty_midi.PrettyMIDI,
                     tempo_changes: List[tuple]):
    """
    Add tempo changes to MIDI file.
    
    Args:
        midi: PrettyMIDI object
        tempo_changes: List of (time, tempo) tuples
    """
    for time, tempo in tempo_changes:
        midi.tempo_changes.append((time, tempo))


def add_time_signature(midi: pretty_midi.PrettyMIDI,
                      numerator: int = 4,
                      denominator: int = 4,
                      time: float = 0.0):
    """
    Add time signature to MIDI file.
    
    Args:
        midi: PrettyMIDI object
        numerator: Time signature numerator
        denominator: Time signature denominator
        time: Time in seconds
    """
    midi.time_signature_changes.append(
        pretty_midi.TimeSignature(numerator, denominator, time)
    )


def quantize_midi(midi: pretty_midi.PrettyMIDI,
                 resolution: float = 0.125) -> pretty_midi.PrettyMIDI:
    """
    Quantize note timings in MIDI file.
    
    Args:
        midi: Input PrettyMIDI object
        resolution: Quantization resolution in seconds
        
    Returns:
        Quantized PrettyMIDI object
    """
    quantized_midi = pretty_midi.PrettyMIDI(initial_tempo=midi.estimate_tempo())
    
    for instrument in midi.instruments:
        new_instrument = pretty_midi.Instrument(
            program=instrument.program,
            is_drum=instrument.is_drum,
            name=instrument.name
        )
        
        for note in instrument.notes:
            # Quantize start and end times
            start = round(note.start / resolution) * resolution
            end = round(note.end / resolution) * resolution
            
            # Ensure minimum duration
            if end - start < resolution:
                end = start + resolution
            
            new_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=start,
                end=end
            )
            new_instrument.notes.append(new_note)
        
        quantized_midi.instruments.append(new_instrument)
    
    return quantized_midi


def merge_midi_files(midi_files: List[str],
                    output_path: str):
    """
    Merge multiple MIDI files into one.
    
    Args:
        midi_files: List of MIDI file paths
        output_path: Output MIDI file path
    """
    merged = pretty_midi.PrettyMIDI()
    
    for midi_path in midi_files:
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            for instrument in midi.instruments:
                merged.instruments.append(instrument)
        except Exception as e:
            print(f"Error loading {midi_path}: {e}")
    
    merged.write(output_path)


def split_midi_by_instrument(midi_path: str,
                            output_dir: str):
    """
    Split MIDI file into separate files per instrument.
    
    Args:
        midi_path: Input MIDI file path
        output_dir: Output directory
    """
    import os
    
    midi = pretty_midi.PrettyMIDI(midi_path)
    
    for i, instrument in enumerate(midi.instruments):
        new_midi = pretty_midi.PrettyMIDI()
        new_midi.instruments.append(instrument)
        
        output_path = os.path.join(
            output_dir,
            f"instrument_{i}_{instrument.name}.mid"
        )
        new_midi.write(output_path)
