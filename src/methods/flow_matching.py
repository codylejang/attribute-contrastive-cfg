"""
Flow Matching for Generative Modeling

Implements the Flow Matching algorithm from Lipman et al., 2023.
The model learns a velocity field that transports noise to data.
"""

import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class FlowMatching(BaseMethod):
    """
    Flow Matching generative model.
    
    Uses optimal transport conditional flow matching with linear interpolation:
    - Forward: x_t = (1 - t) * x_0 + t * noise
    - Velocity: v_t = noise - x_0
    - Model learns to predict v_t given x_t and t
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        sigma_min: float = 1e-4,
    ):
        """
        Initialize Flow Matching.
        
        Args:
            model: Neural network that predicts velocity field v(x_t, t)
            device: Device to run computations on
            sigma_min: Minimum noise level (for numerical stability)
        """
        super().__init__(model, device)
        self.sigma_min = sigma_min
        self.to(device)
    
    def forward_process(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Interpolate between data and noise.
        
        Args:
            x_0: Clean data samples (batch_size, channels, height, width)
            t: Time values in [0, 1] (batch_size,)
            noise: Optional pre-generated noise (x_1)
            
        Returns:
            x_t: Interpolated samples at time t
            noise: The noise (x_1)
            velocity: True velocity field (noise - x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Reshape t for broadcasting: (batch_size,) -> (batch_size, 1, 1, 1)
        t_expanded = t.view(-1, 1, 1, 1)
        
        # Linear interpolation: x_t = (1 - t) * x_0 + t * noise
        x_t = (1 - t_expanded) * x_0 + t_expanded * noise
        
        # Velocity field: v = dx/dt = noise - x_0
        velocity = noise - x_0
        
        return x_t, noise, velocity
    
    def compute_loss(
        self,
        x_0: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Flow Matching training loss.
        
        Args:
            x_0: Clean data samples (batch_size, channels, height, width)
            
        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary of metrics for logging
        """
        batch_size = x_0.shape[0]
        
        # Sample time uniformly from [0, 1]
        # Use small offset to avoid t=0 and t=1 edge cases
        t = torch.rand(batch_size, device=x_0.device) * (1 - 2 * self.sigma_min) + self.sigma_min
        
        # Get interpolated samples and true velocity
        x_t, noise, velocity_true = self.forward_process(x_0, t)
        
        # Convert t to timestep format expected by the model
        # The UNet expects integer timesteps, so we scale t to [0, 999]
        t_scaled = (t * 999).long()
        
        # Predict velocity
        velocity_pred = self.model(x_t, t_scaled)
        
        # MSE loss between predicted and true velocity
        loss = F.mse_loss(velocity_pred, velocity_true)
        
        metrics = {
            "loss": loss,
            "mse": loss,
        }
        
        return loss, metrics
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: int = 100,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples using Euler integration of the velocity field.
        
        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            num_steps: Number of Euler integration steps
            
        Returns:
            samples: Generated samples (batch_size, *image_shape)
        """
        self.eval_mode()
        
        # Start from pure noise (t=1)
        x = torch.randn(batch_size, *image_shape, device=self.device)
        
        # Time step size
        dt = 1.0 / num_steps
        
        # Euler integration from t=1 to t=0
        for i in range(num_steps):
            # Current time (going from 1 to 0)
            t = 1.0 - i * dt
            t_tensor = torch.full((batch_size,), t, device=self.device)
            
            # Scale t to timestep format for the model
            t_scaled = (t_tensor * 999).long()
            
            # Predict velocity
            v = self.model(x, t_scaled)
            
            # Euler step: x_{t-dt} = x_t - v * dt
            # (negative because we're going from t=1 to t=0)
            x = x - v * dt
        
        return x
    
    def to(self, device: torch.device) -> "FlowMatching":
        """Move to device."""
        super().to(device)
        return self
    
    def state_dict(self) -> Dict:
        """Get state dict for checkpointing."""
        state = super().state_dict()
        state["sigma_min"] = self.sigma_min
        return state
    
    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "FlowMatching":
        """Create FlowMatching from config."""
        fm_config = config.get("flow_matching", {})
        return cls(
            model=model,
            device=device,
            sigma_min=fm_config.get("sigma_min", 1e-4),
        )
