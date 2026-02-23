"""
Denoising Diffusion Probabilistic Models (DDPM)
"""

import math
from typing import Dict, Tuple, Optional, Literal, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class DDPM(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
        beta_schedule: str = "linear",
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        
        # Compute beta schedule
        betas = self._get_beta_schedule(beta_schedule, num_timesteps, beta_start, beta_end)
        
        # Compute alphas and related quantities
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        
        # Forward process precompute
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        
        # Reverse process precompute
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", 
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        
        # Posterior mean coefficients
        self.register_buffer("posterior_mean_coef1",
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        
        # Move all buffers to the specified device
        self.to(device)

    def _get_beta_schedule(
        self, 
        schedule: str, 
        num_timesteps: int, 
        beta_start: float, 
        beta_end: float
    ) -> torch.Tensor:
        """Get the beta schedule."""
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            steps = num_timesteps + 1
            s = 0.008
            t = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((t / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """Extract values from a 1D tensor at indices t, reshape for broadcasting."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    # =========================================================================
    # Forward process
    # =========================================================================

    def forward_process(
        self, 
        x_0: torch.Tensor, 
        t: torch.Tensor, 
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward (noise adding) process of DDPM.
        
        Args:
            x_0: Clean data samples (batch_size, channels, height, width)
            t: Timesteps (batch_size,)
            noise: Optional pre-generated noise
            
        Returns:
            x_t: Noisy samples at time t
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
        
        return x_t, noise

    # =========================================================================
    # Training loss
    # =========================================================================

    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the DDPM training loss.

        Args:
            x_0: Clean data samples (batch_size, channels, height, width)
            **kwargs: Additional method-specific arguments
        
        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary of metrics for logging
        """
        batch_size = x_0.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device, dtype=torch.long)
        
        # Sample noise and get noisy samples
        noise = torch.randn_like(x_0)
        x_t, _ = self.forward_process(x_0, t, noise)
        
        # Predict noise
        noise_pred = self.model(x_t, t)
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        metrics = {
            "loss": loss,
            "mse": loss,
        }
        
        return loss, metrics

    # =========================================================================
    # Reverse process (sampling)
    # =========================================================================
    
    def _predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        sqrt_recip_alphas_cumprod_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * eps
    
    def _posterior_mean(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the posterior mean of p(x_{t-1} | x_t, x_0)."""
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        return coef1 * x_0 + coef2 * x_t
    
    @torch.no_grad()
    def reverse_process(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        One step of the DDPM reverse process.

        Args:
            x_t: Noisy samples at time t (batch_size, channels, height, width)
            t: Timestep tensor (batch_size,)
        
        Returns:
            x_prev: Samples at time t-1
        """
        batch_size = x_t.shape[0]
        
        # Predict noise and x_0
        eps_pred = self.model(x_t, t)
        x_0_pred = self._predict_x0_from_eps(x_t, t, eps_pred)
        x_0_pred = torch.clamp(x_0_pred, -1, 1)
        
        # Compute posterior mean and variance
        posterior_mean = self._posterior_mean(x_0_pred, x_t, t)
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        
        # Sample noise (no noise for t=0)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        
        x_prev = posterior_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise
        
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        sampler: str = "ddpm",
        **kwargs
    ) -> torch.Tensor:
        """
        Sample from the model using DDPM or DDIM.

        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            num_steps: Number of sampling steps (default: num_timesteps)
            sampler: Sampling method - "ddpm" or "ddim"
            **kwargs: Additional arguments
        
        Returns:
            samples: Generated samples (batch_size, *image_shape)
        """
        self.eval_mode()
        
        if sampler == "ddim":
            return self.sample_ddim(batch_size, image_shape, num_steps)
        else:
            return self.sample_ddpm(batch_size, image_shape, num_steps)
    
    @torch.no_grad()
    def sample_ddpm(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        DDPM sampling loop (stochastic).

        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            num_steps: Number of sampling steps (default: num_timesteps)
        
        Returns:
            samples: Generated samples (batch_size, *image_shape)
        """
        if num_steps is None:
            num_steps = self.num_timesteps
        
        # Compute timesteps
        if num_steps < self.num_timesteps:
            step_size = self.num_timesteps // num_steps
            timesteps = list(range(0, self.num_timesteps, step_size))[:num_steps]
            timesteps = list(reversed(timesteps))
        else:
            timesteps = list(reversed(range(self.num_timesteps)))
        
        # Start from noise
        x = torch.randn(batch_size, *image_shape, device=self.device)
        
        # Iterate through timesteps
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            x = self.reverse_process(x, t_batch)
        
        return x
    
    @torch.no_grad()
    def sample_ddim(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM sampling loop (deterministic when eta=0).
        
        DDIM allows for fewer sampling steps with the same trained model.
        
        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            num_steps: Number of sampling steps
            eta: Stochasticity parameter (0 = deterministic, 1 = DDPM)
        
        Returns:
            samples: Generated samples (batch_size, *image_shape)
        """
        if num_steps is None:
            num_steps = self.num_timesteps
        
        # Create subsequence of timesteps
        # e.g., for num_steps=100 and num_timesteps=1000: [999, 989, 979, ..., 9]
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))  # [999, 989, ..., 9, 0] approximately
        
        # Start from noise
        x = torch.randn(batch_size, *image_shape, device=self.device)
        
        # DDIM sampling loop
        for i in range(len(timesteps)):
            t = timesteps[i]
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            eps_pred = self.model(x, t_batch)
            
            # Get alpha values
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=self.device)
            
            # Predict x_0
            x_0_pred = (x - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
            x_0_pred = torch.clamp(x_0_pred, -1, 1)
            
            # Compute sigma for stochasticity (eta=0 means deterministic)
            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * eps_pred
            
            # DDIM update
            x = torch.sqrt(alpha_t_prev) * x_0_pred + dir_xt
            
            # Add noise if eta > 0
            if eta > 0 and t_prev > 0:
                noise = torch.randn_like(x)
                x = x + sigma_t * noise
        
        return x

    # =========================================================================
    # Device / state
    # =========================================================================

    def to(self, device: torch.device) -> "DDPM":
        # Call parent which calls nn.Module.to() - moves all buffers and parameters
        super().to(device)
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["num_timesteps"] = self.num_timesteps
        state["beta_start"] = self.beta_start
        state["beta_end"] = self.beta_end
        state["beta_schedule"] = self.beta_schedule
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "DDPM":
        ddpm_config = config.get("ddpm", config)
        return cls(
            model=model,
            device=device,
            num_timesteps=ddpm_config["num_timesteps"],
            beta_start=ddpm_config["beta_start"],
            beta_end=ddpm_config["beta_end"],
            beta_schedule=ddpm_config.get("beta_schedule", "linear"),
        )
