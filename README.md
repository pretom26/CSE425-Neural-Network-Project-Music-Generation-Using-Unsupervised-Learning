# Unsupervised Music Generation

This project implements three deep learning approaches for unsupervised music generation using the MAESTRO v3.0.0 dataset.

## Project Structure

```
music-generation-unsupervised/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── data/                        # Data directory
│   ├── raw_midi/               # Raw MIDI files from MAESTRO dataset
│   ├── processed/              # Preprocessed piano-roll data
│   └── train_test_split/       # Train/validation splits
├── notebooks/                   # Jupyter notebooks
│   ├── preprocessing.ipynb     # Data preprocessing pipeline
│   └── baseline_markov.ipynb   # Markov chain baseline
├── src/                        # Source code
│   ├── config.py               # Configuration and hyperparameters
│   ├── preprocessing/          # Data preprocessing modules
│   │   ├── midi_parser.py      # MIDI file parsing
│   │   ├── tokenizer.py        # Music tokenization
│   │   └── piano_roll.py       # Piano-roll conversion
│   ├── models/                 # Model architectures
│   │   ├── autoencoder.py      # LSTM Autoencoder (Task 1)
│   │   ├── vae.py              # Variational Autoencoder (Task 2)
│   │   ├── transformer.py      # Transformer Generator (Task 3)
│   │   └── diffusion.py        # Diffusion model (optional)
│   ├── training/               # Training scripts
│   │   ├── train_ae.py         # Train autoencoder
│   │   ├── train_vae.py        # Train VAE
│   │   └── train_transformer.py # Train transformer
│   ├── evaluation/             # Evaluation metrics
│   │   ├── metrics.py          # Combined metrics
│   │   ├── pitch_histogram.py  # Pitch histogram similarity
│   │   └── rhythm_score.py     # Rhythm diversity score
│   └── generation/             # Generation utilities
│       ├── sample_latent.py    # Latent space sampling
│       ├── generate_music.py   # Music generation
│       └── midi_export.py      # MIDI export utilities
├── outputs/                    # Generated outputs
│   ├── generated_midis/        # Generated MIDI files
│   ├── plots/                  # Training curves and visualizations
│   └── survey_results/         # Human evaluation results
└── report/                     # Project report
    ├── final_report.tex        # LaTeX report
    ├── architecture_diagrams/  # Model architecture diagrams
    └── references.bib          # Bibliography
```

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd music-generation-unsupervised
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download MAESTRO dataset**:
   - Download from: https://magenta.tensorflow.org/datasets/maestro
   - Extract MIDI files to `data/raw_midi/`

## Quick Start

### 1. Data Preprocessing

Run the preprocessing notebook or use the preprocessing modules:

```bash
# Using notebook
jupyter notebook notebooks/preprocessing.ipynb

# Or using Python
python -c "from src.preprocessing.piano_roll import preprocess_dataset; preprocess_dataset()"
```

This will:
- Convert MIDI files to piano-roll representations
- Create fixed-length windows (128 timesteps)
- Split into train/validation sets
- Save processed data to `data/processed/`

### 2. Train Models

#### Task 1: LSTM Autoencoder
```bash
python src/training/train_ae.py \
    --train_data data/processed/train_pianoroll.npy \
    --val_data data/processed/val_pianoroll.npy \
    --epochs 40 \
    --batch_size 64
```

#### Task 2: Variational Autoencoder (VAE)
```bash
python src/training/train_vae.py \
    --train_data data/processed/train_pianoroll.npy \
    --val_data data/processed/val_pianoroll.npy \
    --epochs 40 \
    --batch_size 64 \
    --beta_warmup 0.3
```

#### Task 3: Transformer
```bash
python src/training/train_transformer.py \
    --train_data data/processed/train_tokens.npy \
    --val_data data/processed/val_tokens.npy \
    --vocab_size 512 \
    --epochs 30 \
    --batch_size 32
```

### 3. Generate Music

```python
from src.generation.generate_music import generate_from_model
from src.models.vae import VAE
import torch

# Load trained model
model = VAE()
model.load_state_dict(torch.load('outputs/models/vae_final.pth'))

# Generate samples
samples = generate_from_model(model, num_samples=5, seq_len=128)

# Export to MIDI
from src.generation.midi_export import piano_roll_to_midi
for i, sample in enumerate(samples):
    piano_roll_to_midi(sample, f'outputs/generated_midis/sample_{i}.midi')
```

## Model Architectures

### Task 1: LSTM Autoencoder
- **Encoder**: 2-layer bidirectional LSTM (256 hidden units)
- **Latent Space**: 64-dimensional continuous representation
- **Decoder**: 2-layer LSTM with attention
- **Loss**: Focal Loss (α=0.25, γ=2.0)

### Task 2: Variational Autoencoder (VAE)
- **Architecture**: Extends LSTM Autoencoder
- **Latent Distribution**: Gaussian with learned mean and variance
- **Training**: KL annealing (β-VAE) with 30% warmup
- **Loss**: Reconstruction + β × KL divergence

### Task 3: Transformer
- **Architecture**: Decoder-only (GPT-style)
- **Layers**: 6 transformer blocks
- **Attention**: 8 heads, 256-dimensional
- **Feedforward**: 1024-dimensional
- **Training**: Autoregressive with teacher forcing

## Evaluation Metrics

### Quantitative Metrics
1. **Pitch Histogram Similarity**: Measures pitch class distribution similarity
2. **Rhythm Diversity Score**: Evaluates inter-onset interval (IOI) diversity
3. **Reconstruction Loss**: Model-specific loss on validation set
4. **Perplexity**: For transformer model

### Qualitative Evaluation
- Human listening tests (survey)
- Musical coherence assessment
- Genre consistency evaluation

## Configuration

All hyperparameters are defined in `src/config.py`:

```python
# Model hyperparameters
LATENT_DIM = 64
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3

# Training hyperparameters
AE_EPOCHS = 40
VAE_EPOCHS = 40
TRANSFORMER_EPOCHS = 30
AE_BATCH_SIZE = 64
VAE_BATCH_SIZE = 64
TRANSFORMER_BATCH_SIZE = 32

# Data parameters
WINDOW_LEN = 128
FS = 16  # Sampling frequency (Hz)
NUM_PITCHES = 88  # Piano keys
```

## Results

Training results and generated samples are saved to:
- **Models**: `outputs/models/`
- **Training curves**: `outputs/plots/`
- **Generated MIDI**: `outputs/generated_midis/`

## Baseline Comparison

A Markov chain baseline is provided for comparison:

```bash
jupyter notebook notebooks/baseline_markov.ipynb
```

The baseline models first-order transitions between pitch classes and serves as a simple probabilistic baseline.

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
```bash
# Format code
black src/

# Check style
flake8 src/
```

## Citation

If you use this code, please cite:

```bibtex
@misc{music-generation-unsupervised,
  title={Unsupervised Music Generation with Deep Learning},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/yourusername/music-generation-unsupervised}}
}
```

## Dataset Citation

```bibtex
@inproceedings{hawthorne2019enabling,
  title={Enabling Factorized Piano Music Modeling and Generation with the {MAESTRO} Dataset},
  author={Curtis Hawthorne and Andriy Stasyuk and Adam Roberts and Ian Simon and Cheng-Zhi Anna Huang and Sander Dieleman and Erich Elsen and Jesse Engel and Douglas Eck},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=r1lYRjC9F7}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MAESTRO dataset by Google Magenta
- PyTorch framework
- pretty_midi library for MIDI processing


