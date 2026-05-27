"""
Token-based MIDI representation using miditok.
Used for Transformer model (Task 3).
"""

from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from pathlib import Path
from typing import List, Dict, Optional
import torch


class MIDITokenizer:
    """Wrapper for miditok REMI tokenizer."""
    
    def __init__(self, 
                 pitch_range: tuple = (21, 109),
                 beat_res: Dict = None,
                 num_velocities: int = 32,
                 special_tokens: List[str] = None):
        """
        Initialize REMI tokenizer.
        
        Args:
            pitch_range: (min_pitch, max_pitch) tuple
            beat_res: Beat resolution dictionary
            num_velocities: Number of velocity bins
            special_tokens: List of special token names
        """
        if beat_res is None:
            beat_res = {(0, 4): 8, (4, 12): 4}
        
        if special_tokens is None:
            special_tokens = ["PAD", "BOS", "EOS"]
        
        # Create tokenizer config
        config = TokenizerConfig(
            pitch_range=pitch_range,
            beat_res=beat_res,
            num_velocities=num_velocities,
            special_tokens=special_tokens,
            use_chords=False,
            use_rests=False,
            use_tempos=True,
            use_time_signatures=True,
            use_programs=False,
        )
        
        self.tokenizer = REMI(config)
        self.vocab_size = len(self.tokenizer)
        self.pad_token_id = self.tokenizer.special_tokens_ids.get("PAD", 0)
        self.bos_token_id = self.tokenizer.special_tokens_ids.get("BOS", 1)
        self.eos_token_id = self.tokenizer.special_tokens_ids.get("EOS", 2)
    
    def encode(self, midi_path: str) -> List[int]:
        """
        Encode MIDI file to token sequence.
        
        Args:
            midi_path: Path to MIDI file
            
        Returns:
            List of token IDs
        """
        try:
            tokens = self.tokenizer(midi_path)
            return tokens.ids
        except Exception as e:
            print(f"Error encoding {midi_path}: {e}")
            return []
    
    def decode(self, token_ids: List[int], output_path: str):
        """
        Decode token sequence to MIDI file.
        
        Args:
            token_ids: List of token IDs
            output_path: Output MIDI file path
        """
        try:
            midi = self.tokenizer.tokens_to_midi([token_ids])
            midi.dump(output_path)
        except Exception as e:
            print(f"Error decoding to {output_path}: {e}")
    
    def encode_batch(self, midi_paths: List[str]) -> List[List[int]]:
        """
        Encode multiple MIDI files.
        
        Args:
            midi_paths: List of MIDI file paths
            
        Returns:
            List of token ID sequences
        """
        return [self.encode(path) for path in midi_paths]
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size
    
    def save(self, save_path: str):
        """Save tokenizer configuration."""
        self.tokenizer.save(Path(save_path))
    
    @classmethod
    def load(cls, load_path: str):
        """Load tokenizer from saved configuration."""
        tokenizer = REMI(params=Path(load_path) / "config.txt")
        wrapper = cls()
        wrapper.tokenizer = tokenizer
        wrapper.vocab_size = len(tokenizer)
        return wrapper


def create_token_dataset(midi_dir: str,
                        tokenizer: MIDITokenizer,
                        max_seq_len: int = 512) -> DatasetMIDI:
    """
    Create PyTorch dataset from MIDI directory.
    
    Args:
        midi_dir: Directory containing MIDI files
        tokenizer: MIDITokenizer instance
        max_seq_len: Maximum sequence length
        
    Returns:
        DatasetMIDI object
    """
    dataset = DatasetMIDI(
        files_paths=list(Path(midi_dir).glob("**/*.mid*")),
        tokenizer=tokenizer.tokenizer,
        max_seq_len=max_seq_len,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return dataset


def create_data_collator(tokenizer: MIDITokenizer) -> DataCollator:
    """
    Create data collator for batching.
    
    Args:
        tokenizer: MIDITokenizer instance
        
    Returns:
        DataCollator object
    """
    return DataCollator(
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


def tokens_to_tensor(token_ids: List[int], 
                    max_len: Optional[int] = None,
                    pad_token_id: int = 0) -> torch.Tensor:
    """
    Convert token IDs to padded tensor.
    
    Args:
        token_ids: List of token IDs
        max_len: Maximum length (pads or truncates)
        pad_token_id: Padding token ID
        
    Returns:
        Tensor of shape (max_len,)
    """
    if max_len is not None:
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        elif len(token_ids) < max_len:
            token_ids = token_ids + [pad_token_id] * (max_len - len(token_ids))
    
    return torch.tensor(token_ids, dtype=torch.long)


def create_attention_mask(token_ids: torch.Tensor, 
                         pad_token_id: int = 0) -> torch.Tensor:
    """
    Create attention mask (1 for real tokens, 0 for padding).
    
    Args:
        token_ids: Token ID tensor
        pad_token_id: Padding token ID
        
    Returns:
        Attention mask tensor
    """
    return (token_ids != pad_token_id).long()
