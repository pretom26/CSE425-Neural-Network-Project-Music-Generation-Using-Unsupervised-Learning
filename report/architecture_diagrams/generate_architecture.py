"""
Architecture Diagram Generator for Music Generation Project
Creates hand-drawn style diagrams to appear more natural
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Set style for hand-drawn appearance
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Comic Sans MS', 'Arial']
plt.rcParams['font.size'] = 9

def add_jitter(x, y, amount=0.02):
    """Add slight jitter for hand-drawn effect"""
    return x + np.random.uniform(-amount, amount), y + np.random.uniform(-amount, amount)

def draw_box(ax, x, y, width, height, text, color='lightblue', textcolor='black'):
    """Draw a box with hand-drawn style"""
    # Add slight irregularity to corners
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.05",
        edgecolor='black',
        facecolor=color,
        linewidth=1.5,
        linestyle='-',
        alpha=0.8
    )
    ax.add_patch(box)
    
    # Add text with slight offset for natural look
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center',
            fontsize=9, weight='normal',
            color=textcolor,
            wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label=''):
    """Draw arrow with hand-drawn style"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.3,head_length=0.3',
        color='black',
        linewidth=1.5,
        linestyle='-',
        mutation_scale=20
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.1, label,
                ha='center', va='bottom',
                fontsize=7, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7, 9.5, 'Music Generation System Architecture',
        ha='center', va='top', fontsize=16, weight='bold')

# ============ DATA LAYER ============
ax.text(1, 8.8, 'Data Layer', ha='left', va='top', fontsize=11, weight='bold', style='italic')

# Raw MIDI Data
draw_box(ax, 0.5, 7.8, 2, 0.8, 'MAESTRO\nDataset\n(1,276 MIDI)', 'lightyellow')

# Preprocessing
draw_box(ax, 3, 7.8, 2, 0.8, 'MIDI Parser\n& Tokenizer', 'lightcyan')
draw_arrow(ax, 2.5, 8.2, 3, 8.2, 'parse')

# Piano Roll
draw_box(ax, 5.5, 7.8, 2, 0.8, 'Piano Roll\nRepresentation', 'lightgreen')
draw_arrow(ax, 5, 8.2, 5.5, 8.2, 'convert')

# Train/Test Split
draw_box(ax, 8, 7.8, 2, 0.8, 'Train/Test\nSplit', 'lightcoral')
draw_arrow(ax, 7.5, 8.2, 8, 8.2, 'split')

# ============ MODEL LAYER ============
ax.text(1, 7.2, 'Model Layer', ha='left', va='top', fontsize=11, weight='bold', style='italic')

# Baseline Model
draw_box(ax, 0.5, 5.8, 2.5, 1, 'Baseline:\nMarkov Chain\n(n-gram)', 'wheat')

# Autoencoder
draw_box(ax, 3.5, 5.8, 2.5, 1, 'Autoencoder\n(AE)\nLatent: 128', 'lightblue')

# VAE
draw_box(ax, 6.5, 5.8, 2.5, 1, 'Variational\nAutoencoder\n(VAE)', 'lightsteelblue')

# Transformer
draw_box(ax, 9.5, 5.8, 2, 1, 'Transformer\n(GPT-style)', 'plum')

# Diffusion
draw_box(ax, 12, 5.8, 1.8, 1, 'Diffusion\nModel', 'lavender')

# Arrows from data to models
draw_arrow(ax, 9, 7.8, 1.75, 6.8, 'train')
draw_arrow(ax, 9, 7.8, 4.75, 6.8, 'train')
draw_arrow(ax, 9, 7.8, 7.75, 6.8, 'train')
draw_arrow(ax, 9, 7.8, 10.5, 6.8, 'train')
draw_arrow(ax, 9, 7.8, 12.9, 6.8, 'train')

# ============ LATENT SPACE ============
ax.text(1, 5.2, 'Latent Space', ha='left', va='top', fontsize=11, weight='bold', style='italic')

# Latent representation
draw_box(ax, 3.5, 4.2, 5.5, 0.8, 'Latent Space Sampling & Interpolation', 'mistyrose')

# Arrows from AE/VAE to latent space
draw_arrow(ax, 4.75, 5.8, 5, 5, '')
draw_arrow(ax, 7.75, 5.8, 7.5, 5, '')

# ============ GENERATION LAYER ============
ax.text(1, 3.8, 'Generation Layer', ha='left', va='top', fontsize=11, weight='bold', style='italic')

# Decoder
draw_box(ax, 3.5, 2.8, 2.5, 0.8, 'Decoder\n(Reconstruction)', 'lightgreen')

# Sampling
draw_box(ax, 6.5, 2.8, 2.5, 0.8, 'Sampling\n(Generation)', 'lightgreen')

# Arrows from latent to generation
draw_arrow(ax, 5, 4.2, 4.75, 3.6, 'decode')
draw_arrow(ax, 7.5, 4.2, 7.75, 3.6, 'sample')

# Direct generation from Transformer/Diffusion
draw_arrow(ax, 10.5, 5.8, 7.75, 3.6, 'generate')
draw_arrow(ax, 12.9, 5.8, 7.75, 3.6, 'denoise')

# ============ OUTPUT LAYER ============
ax.text(1, 2.4, 'Output Layer', ha='left', va='top', fontsize=11, weight='bold', style='italic')

# MIDI Export
draw_box(ax, 5, 1.4, 4, 0.8, 'MIDI Export & Post-processing', 'lightyellow')

# Arrows to output
draw_arrow(ax, 4.75, 2.8, 6, 2.2, '')
draw_arrow(ax, 7.75, 2.8, 8, 2.2, '')

# Generated Music
draw_box(ax, 5, 0.4, 4, 0.8, 'Generated Music Files', 'gold')
draw_arrow(ax, 7, 1.4, 7, 1.2, 'export')

# ============ EVALUATION LAYER ============
ax.text(11, 3.8, 'Evaluation', ha='left', va='top', fontsize=11, weight='bold', style='italic')

# Metrics boxes
draw_box(ax, 10.5, 2.8, 1.8, 0.5, 'Pitch\nHistogram', 'lightcoral')
draw_box(ax, 10.5, 2.2, 1.8, 0.5, 'Rhythm\nScore', 'lightcoral')
draw_box(ax, 10.5, 1.6, 1.8, 0.5, 'Reconstruction\nLoss', 'lightcoral')
draw_box(ax, 10.5, 1.0, 1.8, 0.5, 'Perplexity', 'lightcoral')

# Arrows to evaluation
draw_arrow(ax, 9, 1.8, 10.5, 3.05, 'evaluate')
draw_arrow(ax, 9, 1.8, 10.5, 2.45, '')
draw_arrow(ax, 9, 1.8, 10.5, 1.85, '')
draw_arrow(ax, 9, 1.8, 10.5, 1.25, '')

# ============ LEGEND ============
ax.text(0.5, 0.3, 'Legend:', ha='left', va='top', fontsize=9, weight='bold')
draw_box(ax, 0.5, -0.2, 0.6, 0.3, 'Data', 'lightyellow', 'black')
draw_box(ax, 1.3, -0.2, 0.6, 0.3, 'Process', 'lightcyan', 'black')
draw_box(ax, 2.1, -0.2, 0.6, 0.3, 'Model', 'lightblue', 'black')
draw_box(ax, 2.9, -0.2, 0.6, 0.3, 'Output', 'gold', 'black')

# Add annotations
ax.text(7, -0.5, 'CSE425 Project: Unsupervised Music Generation',
        ha='center', va='top', fontsize=8, style='italic', color='gray')

plt.tight_layout()
plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Architecture diagram saved as 'architecture_diagram.png'")
plt.close()

# ============ CREATE DETAILED MODEL ARCHITECTURE ============
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(6, 9.5, 'Detailed Model Architectures',
        ha='center', va='top', fontsize=16, weight='bold')

# ============ AUTOENCODER ============
ax.text(2, 8.8, 'Autoencoder (AE)', ha='center', va='top', fontsize=12, weight='bold')

# Encoder
draw_box(ax, 0.5, 7.5, 1.2, 0.6, 'Input\n88×T', 'lightyellow')
draw_box(ax, 0.5, 6.7, 1.2, 0.6, 'Conv1D\n256', 'lightblue')
draw_box(ax, 0.5, 5.9, 1.2, 0.6, 'Conv1D\n512', 'lightblue')
draw_box(ax, 0.5, 5.1, 1.2, 0.6, 'Flatten', 'lightcyan')
draw_box(ax, 0.5, 4.3, 1.2, 0.6, 'Dense\n128', 'lightgreen')

# Decoder
draw_box(ax, 2.8, 4.3, 1.2, 0.6, 'Dense\n512', 'lightgreen')
draw_box(ax, 2.8, 5.1, 1.2, 0.6, 'Reshape', 'lightcyan')
draw_box(ax, 2.8, 5.9, 1.2, 0.6, 'ConvT1D\n256', 'lightblue')
draw_box(ax, 2.8, 6.7, 1.2, 0.6, 'ConvT1D\n128', 'lightblue')
draw_box(ax, 2.8, 7.5, 1.2, 0.6, 'Output\n88×T', 'lightyellow')

# Arrows
for i in range(4):
    draw_arrow(ax, 1.1, 7.8 - i*0.8, 1.1, 7.4 - i*0.8, '')
for i in range(4):
    draw_arrow(ax, 3.4, 4.6 + i*0.8, 3.4, 5.0 + i*0.8, '')

# Latent connection
draw_arrow(ax, 1.7, 4.6, 2.8, 4.6, 'z (128)')

# ============ VAE ============
ax.text(6, 8.8, 'Variational Autoencoder (VAE)', ha='center', va='top', fontsize=12, weight='bold')

# Encoder
draw_box(ax, 4.5, 7.5, 1.2, 0.6, 'Input\n88×T', 'lightyellow')
draw_box(ax, 4.5, 6.7, 1.2, 0.6, 'Conv1D\n256', 'lightblue')
draw_box(ax, 4.5, 5.9, 1.2, 0.6, 'Conv1D\n512', 'lightblue')

# Mean and Variance
draw_box(ax, 4.0, 5.0, 0.8, 0.5, 'μ', 'mistyrose')
draw_box(ax, 4.9, 5.0, 0.8, 0.5, 'σ²', 'mistyrose')

# Sampling
draw_box(ax, 4.4, 4.2, 1.0, 0.5, 'Sample\nz~N(μ,σ²)', 'lavender')

# Decoder
draw_box(ax, 6.8, 4.2, 1.2, 0.6, 'Dense\n512', 'lightgreen')
draw_box(ax, 6.8, 5.0, 1.2, 0.6, 'Reshape', 'lightcyan')
draw_box(ax, 6.8, 5.8, 1.2, 0.6, 'ConvT1D\n256', 'lightblue')
draw_box(ax, 6.8, 6.6, 1.2, 0.6, 'ConvT1D\n128', 'lightblue')
draw_box(ax, 6.8, 7.4, 1.2, 0.6, 'Output\n88×T', 'lightyellow')

# Arrows
draw_arrow(ax, 5.1, 7.5, 5.1, 7.1, '')
draw_arrow(ax, 5.1, 6.7, 5.1, 6.3, '')
draw_arrow(ax, 4.8, 5.9, 4.4, 5.5, '')
draw_arrow(ax, 5.4, 5.9, 5.3, 5.5, '')
draw_arrow(ax, 4.4, 5.0, 4.6, 4.7, '')
draw_arrow(ax, 5.3, 5.0, 5.2, 4.7, '')
draw_arrow(ax, 5.4, 4.45, 6.8, 4.45, 'z')

for i in range(4):
    draw_arrow(ax, 7.4, 4.5 + i*0.8, 7.4, 4.9 + i*0.8, '')

# ============ TRANSFORMER ============
ax.text(10, 8.8, 'Transformer', ha='center', va='top', fontsize=12, weight='bold')

draw_box(ax, 8.8, 7.5, 1.2, 0.6, 'Token\nEmbed', 'lightyellow')
draw_box(ax, 8.8, 6.7, 1.2, 0.6, 'Pos\nEmbed', 'lightyellow')
draw_box(ax, 8.8, 5.9, 1.2, 0.6, 'Multi-Head\nAttention', 'lightblue')
draw_box(ax, 8.8, 5.1, 1.2, 0.6, 'FFN', 'lightblue')
draw_box(ax, 8.8, 4.3, 1.2, 0.6, 'Layer\nNorm', 'lightcyan')
draw_box(ax, 8.8, 3.5, 1.2, 0.6, 'Output\nProjection', 'lightgreen')

# Arrows
for i in range(5):
    draw_arrow(ax, 9.4, 7.8 - i*0.8, 9.4, 7.4 - i*0.8, '')

# Skip connection
draw_arrow(ax, 10.2, 6.2, 10.2, 5.4, 'skip')
draw_arrow(ax, 10.2, 5.4, 9.4, 5.4, '')

# ============ DIFFUSION ============
ax.text(2, 3.2, 'Diffusion Model', ha='center', va='top', fontsize=12, weight='bold')

draw_box(ax, 0.5, 2.0, 1.2, 0.5, 'x₀\n(clean)', 'lightyellow')
draw_box(ax, 2.0, 2.0, 1.2, 0.5, 'x_t\n(noisy)', 'lightcoral')
draw_box(ax, 3.5, 2.0, 1.2, 0.5, 'U-Net\nDenoiser', 'lightblue')
draw_box(ax, 0.5, 1.0, 1.2, 0.5, 'x₀\n(recon)', 'lightgreen')

draw_arrow(ax, 1.7, 2.25, 2.0, 2.25, 'add noise')
draw_arrow(ax, 3.2, 2.25, 3.5, 2.25, 'predict')
draw_arrow(ax, 3.5, 2.0, 2.0, 1.5, 'denoise')
draw_arrow(ax, 2.0, 1.5, 1.7, 1.25, 'iterate')

# Add training info boxes
draw_box(ax, 5.5, 1.5, 2.5, 0.8, 'Training:\n• Adam optimizer\n• LR: 1e-4', 'wheat')
draw_box(ax, 8.5, 1.5, 2.5, 0.8, 'Hyperparameters:\n• Batch: 32\n• Epochs: 50', 'wheat')

plt.tight_layout()
plt.savefig('model_architectures.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Model architecture diagram saved as 'model_architectures.png'")
plt.close()

print("\nBoth diagrams generated successfully!")
print("Files created:")
print("  - architecture_diagram.png")
print("  - model_architectures.png")
