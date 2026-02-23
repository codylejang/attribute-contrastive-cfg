"""
Classifier-Free Guidance DDPM

Extends DDPM with classifier-free guidance for conditional generation.
During training, the condition is randomly dropped with probability p_uncond
to jointly learn conditional and unconditional distributions.
At sampling time, the model output is interpolated:
    eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

Reference: Ho & Salimans, "Classifier-Free Diffusion Guidance", 2022.
"""

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ddpm import DDPM


class CfgDDPM(DDPM):
    """
    DDPM with Classifier-Free Guidance.

    Adds conditional generation capability on top of the base DDPM.
    The model learns both p(x|c) and p(x) by randomly dropping the
    condition during training.

    Args:
        model: UNet with conditioning support (num_classes > 0)
        device: Torch device
        num_timesteps: Number of diffusion timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
        beta_schedule: Type of beta schedule
        p_uncond: Probability of dropping condition during training
        guidance_scale: Default guidance scale for sampling (w in the paper)
        num_classes: Number of condition dimensions (e.g. 40 for CelebA attributes)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
        beta_schedule: str = "linear",
        p_uncond: float = 0.1,
        guidance_scale: float = 2.0,
        num_classes: int = 40,
    ):
        super().__init__(
            model=model,
            device=device,
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        )
        self.p_uncond = p_uncond
        self.guidance_scale = guidance_scale
        self.num_classes = num_classes

    def compute_loss(
        self,
        x_0: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute CFG-DDPM training loss.

        During training, with probability p_uncond we replace the condition
        with zeros (unconditional). Otherwise we use the real condition.

        Args:
            x_0: Clean images (B, C, H, W)
            cond: Condition tensor (B, num_classes), e.g. binary CelebA attributes
        """
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device, dtype=torch.long)

        # Sample noise and create noisy images
        noise = torch.randn_like(x_0)
        x_t, _ = self.forward_process(x_0, t, noise)

        # Classifier-free guidance: randomly drop condition
        if cond is not None:
            # Create mask: 1 = drop condition (unconditional), 0 = keep condition
            drop_mask = (torch.rand(batch_size, device=x_0.device) < self.p_uncond).float()
            # Zero out condition for dropped samples
            cond = cond * (1 - drop_mask.unsqueeze(1))
        else:
            cond = torch.zeros(batch_size, self.num_classes, device=x_0.device)

        # Predict noise with conditioning
        noise_pred = self.model(x_t, t, cond=cond)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)

        metrics = {
            "loss": loss,
            "mse": loss,
        }

        return loss, metrics

    @torch.no_grad()
    def reverse_process_cfg(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """
        One step of CFG reverse process.

        Computes both conditional and unconditional predictions, then
        combines them with the guidance scale.
        """
        batch_size = x_t.shape[0]

        # Unconditional prediction (condition = zeros)
        cond_uncond = torch.zeros_like(cond)
        eps_uncond = self.model(x_t, t, cond=cond_uncond)

        # Conditional prediction
        eps_cond = self.model(x_t, t, cond=cond)

        # CFG: eps = eps_uncond + w * (eps_cond - eps_uncond)
        eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # Predict x_0
        x_0_pred = self._predict_x0_from_eps(x_t, t, eps_pred)
        x_0_pred = torch.clamp(x_0_pred, -1, 1)

        # Posterior mean and variance
        posterior_mean = self._posterior_mean(x_0_pred, x_t, t)
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)

        # Sample (no noise at t=0)
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
        cond: Optional[torch.Tensor] = None,
        cond_anchor: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = None,
        sampler: str = "ddpm",
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate conditional samples using CFG or Attribute-Contrastive CFG.

        Args:
            batch_size: Number of samples
            image_shape: (C, H, W)
            num_steps: Sampling steps (default: num_timesteps)
            cond: Target condition tensor (batch_size, num_classes)
            cond_anchor: Anchor condition for contrastive CFG (batch_size, num_classes).
                         When provided, uses attribute-contrastive guidance instead of
                         standard CFG. The anchor should be identical to cond except
                         that the focal attribute(s) are flipped.
                         When None, falls back to standard CFG (null baseline).
            guidance_scale: CFG scale (default: self.guidance_scale)
            sampler: "ddpm" or "ddim"
        """
        self.eval_mode()

        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        if cond is None:
            cond = torch.zeros(batch_size, self.num_classes, device=self.device)
        else:
            cond = cond.to(self.device)

        if cond_anchor is not None:
            cond_anchor = cond_anchor.to(self.device)
            # Attribute-contrastive CFG
            if sampler == "ddim":
                return self.sample_ddim_contrastive(batch_size, image_shape, num_steps, cond, cond_anchor, guidance_scale)
            else:
                return self.sample_ddpm_contrastive(batch_size, image_shape, num_steps, cond, cond_anchor, guidance_scale)
        else:
            # Standard CFG
            if sampler == "ddim":
                return self.sample_ddim_cfg(batch_size, image_shape, num_steps, cond, guidance_scale)
            else:
                return self.sample_ddpm_cfg(batch_size, image_shape, num_steps, cond, guidance_scale)

    @torch.no_grad()
    def sample_ddpm_cfg(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        cond: Optional[torch.Tensor] = None,
        guidance_scale: float = 2.0,
    ) -> torch.Tensor:
        """DDPM sampling with classifier-free guidance."""
        if num_steps is None:
            num_steps = self.num_timesteps

        if num_steps < self.num_timesteps:
            step_size = self.num_timesteps // num_steps
            timesteps = list(range(0, self.num_timesteps, step_size))[:num_steps]
            timesteps = list(reversed(timesteps))
        else:
            timesteps = list(reversed(range(self.num_timesteps)))

        x = torch.randn(batch_size, *image_shape, device=self.device)

        for t_val in timesteps:
            t_batch = torch.full((batch_size,), t_val, device=self.device, dtype=torch.long)
            x = self.reverse_process_cfg(x, t_batch, cond, guidance_scale)

        return x

    @torch.no_grad()
    def sample_ddim_cfg(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        cond: Optional[torch.Tensor] = None,
        guidance_scale: float = 2.0,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM sampling with classifier-free guidance."""
        if num_steps is None:
            num_steps = self.num_timesteps

        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        cond_uncond = torch.zeros_like(cond)

        x = torch.randn(batch_size, *image_shape, device=self.device)

        for i in range(len(timesteps)):
            t = timesteps[i]
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0

            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # CFG: combine conditional and unconditional predictions
            eps_uncond = self.model(x, t_batch, cond=cond_uncond)
            eps_cond = self.model(x, t_batch, cond=cond)
            eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=self.device)

            x_0_pred = (x - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
            x_0_pred = torch.clamp(x_0_pred, -1, 1)

            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
            dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * eps_pred

            x = torch.sqrt(alpha_t_prev) * x_0_pred + dir_xt

            if eta > 0 and t_prev > 0:
                noise = torch.randn_like(x)
                x = x + sigma_t * noise

        return x

    # =========================================================================
    # Attribute-Contrastive CFG sampling
    # =========================================================================

    @torch.no_grad()
    def reverse_process_contrastive(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        cond_anchor: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """
        One step of attribute-contrastive CFG reverse process.

        Instead of using the null condition as the guidance baseline, we use a
        targeted anchor condition that differs from cond only in the focal
        attribute(s). The guidance direction then captures only the causal effect
        of those attributes, isolated from spurious correlations.

            eps_guided = eps_anchor + w * (eps_target - eps_anchor)

        When cond_anchor is the null condition (all zeros), this reduces exactly
        to standard CFG.

        Args:
            x_t: Noisy samples at time t
            t: Timestep tensor
            cond: Target condition (what we want to generate)
            cond_anchor: Anchor condition (same as cond but with focal attrs flipped)
            guidance_scale: CFG scale w
        """
        # Anchor prediction (semantic baseline — all attrs except the focal ones)
        eps_anchor = self.model(x_t, t, cond=cond_anchor)

        # Target prediction (includes the focal attribute)
        eps_target = self.model(x_t, t, cond=cond)

        # Contrastive guidance: steer from anchor toward target
        eps_pred = eps_anchor + guidance_scale * (eps_target - eps_anchor)

        # Predict x_0 and compute posterior (same as standard CFG reverse step)
        x_0_pred = self._predict_x0_from_eps(x_t, t, eps_pred)
        x_0_pred = torch.clamp(x_0_pred, -1, 1)

        posterior_mean = self._posterior_mean(x_0_pred, x_t, t)
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)

        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))

        x_prev = posterior_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise
        return x_prev

    @torch.no_grad()
    def sample_ddpm_contrastive(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        cond: Optional[torch.Tensor] = None,
        cond_anchor: Optional[torch.Tensor] = None,
        guidance_scale: float = 2.0,
    ) -> torch.Tensor:
        """
        DDPM sampling with attribute-contrastive CFG.

        Args:
            batch_size: Number of samples
            image_shape: (C, H, W)
            num_steps: Sampling steps
            cond: Target condition (B, num_classes)
            cond_anchor: Anchor condition (B, num_classes) — focal attrs flipped
            guidance_scale: CFG scale
        """
        if num_steps is None:
            num_steps = self.num_timesteps

        if cond is None:
            cond = torch.zeros(batch_size, self.num_classes, device=self.device)
        if cond_anchor is None:
            # Fall back to standard CFG (null anchor)
            cond_anchor = torch.zeros_like(cond)

        if num_steps < self.num_timesteps:
            step_size = self.num_timesteps // num_steps
            timesteps = list(range(0, self.num_timesteps, step_size))[:num_steps]
            timesteps = list(reversed(timesteps))
        else:
            timesteps = list(reversed(range(self.num_timesteps)))

        x = torch.randn(batch_size, *image_shape, device=self.device)

        for t_val in timesteps:
            t_batch = torch.full((batch_size,), t_val, device=self.device, dtype=torch.long)
            x = self.reverse_process_contrastive(x, t_batch, cond, cond_anchor, guidance_scale)

        return x

    @torch.no_grad()
    def sample_ddim_contrastive(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        cond: Optional[torch.Tensor] = None,
        cond_anchor: Optional[torch.Tensor] = None,
        guidance_scale: float = 2.0,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM sampling with attribute-contrastive CFG.

        Args:
            batch_size: Number of samples
            image_shape: (C, H, W)
            num_steps: Sampling steps
            cond: Target condition (B, num_classes)
            cond_anchor: Anchor condition (B, num_classes) — focal attrs flipped
            guidance_scale: CFG scale
            eta: Stochasticity (0 = deterministic, 1 = DDPM)
        """
        if num_steps is None:
            num_steps = self.num_timesteps

        if cond is None:
            cond = torch.zeros(batch_size, self.num_classes, device=self.device)
        if cond_anchor is None:
            cond_anchor = torch.zeros_like(cond)

        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        x = torch.randn(batch_size, *image_shape, device=self.device)

        for i in range(len(timesteps)):
            t = timesteps[i]
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0

            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # Contrastive guidance: anchor baseline → target
            eps_anchor = self.model(x, t_batch, cond=cond_anchor)
            eps_target = self.model(x, t_batch, cond=cond)
            eps_pred = eps_anchor + guidance_scale * (eps_target - eps_anchor)

            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=self.device)

            x_0_pred = (x - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
            x_0_pred = torch.clamp(x_0_pred, -1, 1)

            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
            dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * eps_pred

            x = torch.sqrt(alpha_t_prev) * x_0_pred + dir_xt

            if eta > 0 and t_prev > 0:
                noise = torch.randn_like(x)
                x = x + sigma_t * noise

        return x

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "CfgDDPM":
        ddpm_config = config.get("ddpm", config)
        cfg_config = config.get("cfg", {})
        model_config = config.get("model", {})
        return cls(
            model=model,
            device=device,
            num_timesteps=ddpm_config["num_timesteps"],
            beta_start=ddpm_config["beta_start"],
            beta_end=ddpm_config["beta_end"],
            beta_schedule=ddpm_config.get("beta_schedule", "linear"),
            p_uncond=cfg_config.get("p_uncond", 0.1),
            guidance_scale=cfg_config.get("guidance_scale", 2.0),
            num_classes=model_config.get("num_classes", 40),
        )
