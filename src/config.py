"""
Configuration file for music generation project.
Contains all hyperparameters and paths.
"""

import os
import torch

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MULTI_GPU = torch.cuda.device_count() > 1

# ============================================================================
# PATHS
# ============================================================================
DATA_ROOT = "all_year_files"
CSV_PATH = os.path.join(DATA_ROOT, "maestro-v3.0.0.csv")
MIDI_DIR = DATA_ROOT

# Output directories
OUTPUT_DIR = "all_outputs"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
MIDI_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "generated_midis")

# Processed data
PROCESSED_DIR = "data/processed"
TRAIN_PIANOROLL = os.path.join(PROCESSED_DIR, "train_pianoroll.npy")
VAL_PIANOROLL = os.path.join(PROCESSED_DIR, "val_pianoroll.npy")

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================
# Piano-roll settings
WINDOW_LEN = 128          # Time steps per window (8 seconds at fs=16)
FS = 16                   # Frames per second
MIN_PITCH = 21            # A0 (lowest piano key)
MAX_PITCH = 109           # C8 (highest piano key)
NUM_PITCHES = 88          # Total piano keys

# Filtering
MIN_DENSITY = 0.02        # Minimum fraction of active notes per window

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================
# Common
LATENT_DIM = 64
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3

# Autoencoder (Task 1)
AE_LEARNING_RATE = 1e-3
AE_EPOCHS = 40
AE_BATCH_SIZE = 64

# VAE (Task 2)
VAE_LEARNING_RATE = 1e-3
VAE_EPOCHS = 40
VAE_BATCH_SIZE = 64
VAE_BETA_MAX = 1.0
VAE_KL_WARMUP = 0.3       # Fraction of epochs for KL annealing

# Transformer (Task 3)
TR_D_MODEL = 256
TR_N_HEADS = 8
TR_N_LAYERS = 6
TR_D_FF = 1024
TR_DROPOUT = 0.1
TR_MAX_SEQ_LEN = 512
TR_LEARNING_RATE = 1e-4
TR_EPOCHS = 30
TR_BATCH_SIZE = 32

# RLHF (Task 4)
RLHF_LEARNING_RATE = 1e-5
RLHF_ITERATIONS = 100
RLHF_SAMPLES_PER_ITER = 10

# ============================================================================
# LOSS FUNCTION PARAMETERS
# ============================================================================
# Focal Loss
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# ============================================================================
# GENERATION PARAMETERS
# ============================================================================
GEN_THRESHOLD = 0.35      # Binarization threshold for piano-roll
GEN_TEMPERATURE = 1.0     # Sampling temperature for Transformer
GEN_TOP_K = 50            # Top-k sampling parameter

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================
NGRAM_SIZE = 4            # For repetition ratio computation

# ============================================================================
# TEST MODE (for quick debugging)
# ============================================================================
TEST_MODE = False
MAX_FILES = 30 if TEST_MODE else None

if TEST_MODE:
    AE_EPOCHS = 5
    VAE_EPOCHS = 5
    TR_EPOCHS = 5
    AE_BATCH_SIZE = 16
    VAE_BATCH_SIZE = 16
    TR_BATCH_SIZE = 16

# ============================================================================
# CREATE DIRECTORIES
# ============================================================================
def create_directories():
    """Create all necessary output directories."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MIDI_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(os.path.join(MIDI_OUTPUT_DIR, "best_rlhf"), exist_ok=True)

if __name__ == "__main__":
    create_directories()
    print(f"Device: {DEVICE}")
    print(f"Multi-GPU: {USE_MULTI_GPU}")
    print(f"Test Mode: {TEST_MODE}")
