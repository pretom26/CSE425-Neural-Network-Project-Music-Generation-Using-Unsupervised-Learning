"""
Latent space sampling utilities for generative models.
"""

import torch
import numpy as np
from typing import Optional, Tuple


def sample_gaussian(batch_size: int,
                   latent_dim: int,
                   device: str = 'cpu') -> torch.Tensor:
    """
    Sample from standard Gaussian distribution N(0, I).
    
    Args:
        batch_size: Number of samples
        latent_dim: Dimension of latent space
        device: Device to create tensor on
        
    Returns:
        Tensor of shape (batch_size, latent_dim)
    """
    return torch.randn(batch_size, latent_dim, device=device)


def sample_uniform(batch_size: int,
                  latent_dim: int,
                  low: float = -1.0,
                  high: float = 1.0,
                  device: str = 'cpu') -> torch.Tensor:
    """
    Sample from uniform distribution U(low, high).
    
    Args:
        batch_size: Number of samples
        latent_dim: Dimension of latent space
        low: Lower bound
        high: Upper bound
        device: Device to create tensor on
        
    Returns:
        Tensor of shape (batch_size, latent_dim)
    """
    return torch.rand(batch_size, latent_dim, device=device) * (high - low) + low


def sample_sphere(batch_size: int,
                 latent_dim: int,
                 radius: float = 1.0,
                 device: str = 'cpu') -> torch.Tensor:
    """
    Sample uniformly from surface of hypersphere.
    
    Args:
        batch_size: Number of samples
        latent_dim: Dimension of latent space
        radius: Radius of sphere
        device: Device to create tensor on
        
    Returns:
        Tensor of shape (batch_size, latent_dim)
    """
    # Sample from Gaussian
    z = torch.randn(batch_size, latent_dim, device=device)
    
    # Normalize to unit sphere
    z = z / torch.norm(z, dim=1, keepdim=True)
    
    # Scale to desired radius
    z = z * radius
    
    return z


def interpolate_latent(z1: torch.Tensor,
                      z2: torch.Tensor,
                      num_steps: int = 10,
                      mode: str = 'linear') -> torch.Tensor:
    """
    Interpolate between two latent codes.
    
    Args:
        z1: First latent code of shape (latent_dim,) or (1, latent_dim)
        z2: Second latent code of same shape
        num_steps: Number of interpolation steps
        mode: Interpolation mode ('linear' or 'spherical')
        
    Returns:
        Interpolated codes of shape (num_steps, latent_dim)
    """
    # Ensure 2D tensors
    if z1.dim() == 1:
        z1 = z1.unsqueeze(0)
    if z2.dim() == 1:
        z2 = z2.unsqueeze(0)
    
    if mode == 'linear':
        # Linear interpolation: z_α = (1-α)z1 + αz2
        alphas = torch.linspace(0, 1, num_steps, device=z1.device)
        z_interp = []
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            z_interp.append(z)
        return torch.cat(z_interp, dim=0)
    
    elif mode == 'spherical':
        # Spherical linear interpolation (SLERP)
        # More appropriate for normalized latent spaces
        z1_norm = z1 / torch.norm(z1, dim=1, keepdim=True)
        z2_norm = z2 / torch.norm(z2, dim=1, keepdim=True)
        
        # Compute angle between vectors
        dot = torch.sum(z1_norm * z2_norm, dim=1, keepdim=True)
        dot = torch.clamp(dot, -1.0, 1.0)
        omega = torch.acos(dot)
        
        # Interpolate
        alphas = torch.linspace(0, 1, num_steps, device=z1.device)
        z_interp = []
        for alpha in alphas:
            if omega.abs() < 1e-6:
                # Vectors are parallel, use linear interpolation
                z = (1 - alpha) * z1 + alpha * z2
            else:
                # SLERP formula
                sin_omega = torch.sin(omega)
                z = (torch.sin((1-alpha)*omega) / sin_omega) * z1 + \
                    (torch.sin(alpha*omega) / sin_omega) * z2
            z_interp.append(z)
        return torch.cat(z_interp, dim=0)
    
    else:
        raise ValueError(f"Unknown interpolation mode: {mode}")


def sample_grid_2d(latent_dim: int,
                  grid_size: int = 10,
                  range_min: float = -3.0,
                  range_max: float = 3.0,
                  device: str = 'cpu') -> torch.Tensor:
    """
    Sample a 2D grid in the first two dimensions of latent space.
    Useful for visualizing latent space structure.
    
    Args:
        latent_dim: Dimension of latent space
        grid_size: Number of points per dimension
        range_min: Minimum value for grid
        range_max: Maximum value for grid
        device: Device to create tensor on
        
    Returns:
        Tensor of shape (grid_size^2, latent_dim)
    """
    # Create grid in first two dimensions
    x = torch.linspace(range_min, range_max, grid_size, device=device)
    y = torch.linspace(range_min, range_max, grid_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    # Flatten grid
    grid_2d = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    # Pad with zeros for remaining dimensions
    if latent_dim > 2:
        padding = torch.zeros(grid_size**2, latent_dim-2, device=device)
        grid = torch.cat([grid_2d, padding], dim=1)
    else:
        grid = grid_2d
    
    return grid


def sample_random_walk(start: torch.Tensor,
                      num_steps: int,
                      step_size: float = 0.1,
                      device: str = 'cpu') -> torch.Tensor:
    """
    Perform random walk in latent space.
    
    Args:
        start: Starting latent code of shape (latent_dim,)
        num_steps: Number of steps
        step_size: Size of each random step
        device: Device to create tensor on
        
    Returns:
        Tensor of shape (num_steps, latent_dim)
    """
    latent_dim = start.shape[0]
    trajectory = [start.unsqueeze(0)]
    
    current = start
    for _ in range(num_steps - 1):
        # Random step
        step = torch.randn(latent_dim, device=device) * step_size
        current = current + step
        trajectory.append(current.unsqueeze(0))
    
    return torch.cat(trajectory, dim=0)
