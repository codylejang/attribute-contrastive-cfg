"""
Sampling Script for DDPM (Denoising Diffusion Probabilistic Models)

Generate samples from a trained model. By default, saves individual images to avoid
memory issues with large sample counts. Use --grid to generate a single grid image.

Usage:
    # Sample from DDPM (saves individual images to ./samples/)
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --num_samples 64

    # With custom number of sampling steps
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --num_steps 500

    # Generate a grid image instead of individual images
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --num_samples 64 --grid

    # Save individual images to custom directory
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --output_dir my_samples

What you need to implement:
- Incorporate your sampling scheme to this pipeline
- Save generated samples as images for logging
"""

import os
import sys
import argparse
from datetime import datetime

import yaml
import torch
from tqdm import tqdm

from src.models import create_model_from_config
from src.data import save_image, unnormalize
from src.methods import DDPM, FlowMatching, CfgDDPM
from src.data import CELEBA_ATTRIBUTES
from src.utils import EMA


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load checkpoint and return model, config, and EMA."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = create_model_from_config(config).to(device)
    model.load_state_dict(checkpoint['model'])
    
    # Create EMA and load
    ema = EMA(model, decay=config['training']['ema_decay'])
    ema.load_state_dict(checkpoint['ema'])
    
    return model, config, ema


def save_samples(
    samples: torch.Tensor,
    save_path: str,
    nrow: int = 8,
) -> None:
    """
    Save generated samples as an image grid.

    Args:
        samples: Generated samples tensor (num_samples, C, H, W) in [-1, 1].
        save_path: File path to save the image grid.
        nrow: Number of images per row in the grid.
    """
    # Convert from [-1, 1] to [0, 1]
    samples = unnormalize(samples)
    samples = torch.clamp(samples, 0, 1)
    save_image(samples, save_path, nrow=nrow)


def main():
    parser = argparse.ArgumentParser(description='Generate samples from trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--method', type=str, required=True,
                       choices=['ddpm', 'flow_matching', 'cfg_ddpm'],
                       help='Method used for training (ddpm, flow_matching, or cfg_ddpm)')
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='samples',
                       help='Directory to save individual images (default: samples)')
    parser.add_argument('--grid', action='store_true',
                       help='Save as grid image instead of individual images')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for grid (only used with --grid, default: samples_<timestamp>.png)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for generation')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    # Sampling arguments
    parser.add_argument('--num_steps', type=int, default=None,
                       help='Number of sampling steps (default: from config)')
    parser.add_argument('--sampler', type=str, default='ddpm',
                       choices=['ddpm', 'ddim'],
                       help='Sampler to use for DDPM models (ddpm or ddim)')
    
    # CFG-specific arguments
    parser.add_argument('--guidance_scale', type=float, default=None,
                       help='Classifier-free guidance scale (default: from config)')
    parser.add_argument('--attributes', type=str, default=None,
                       help='Comma-separated CelebA attributes to condition on (e.g., "Smiling,Young,Male")')

    # Attribute-Contrastive CFG arguments
    parser.add_argument('--contrastive', action='store_true',
                       help='Use attribute-contrastive CFG instead of standard CFG. '
                            'Requires --focal_attributes to specify which attribute(s) to isolate.')
    parser.add_argument('--focal_attributes', type=str, default=None,
                       help='Comma-separated attributes to isolate via contrastive guidance. '
                            'The anchor condition is constructed by flipping these attributes '
                            'relative to --attributes. Positive focal attrs (set to 1 in --attributes) '
                            'guide toward the attribute; negative focal attrs (set to 0) suppress it. '
                            'Example: --attributes "Eyeglasses,Male,Young" --focal_attributes "Eyeglasses"')
    
    # Other options
    parser.add_argument('--no_ema', action='store_true',
                       help='Use training weights instead of EMA weights')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, config, ema = load_checkpoint(args.checkpoint, device)
    
    # Create method
    if args.method == 'ddpm':
        method = DDPM.from_config(model, config, device)
    elif args.method == 'flow_matching':
        method = FlowMatching.from_config(model, config, device)
    elif args.method == 'cfg_ddpm':
        method = CfgDDPM.from_config(model, config, device)
    else:
        raise ValueError(f"Unknown method: {args.method}. Supported: 'ddpm', 'flow_matching', 'cfg_ddpm'")
    
    # Build condition tensor for CFG models
    cfg_cond = None
    cfg_cond_anchor = None
    if args.method == 'cfg_ddpm':
        num_classes = config.get('model', {}).get('num_classes', 40)
        cfg_cond = torch.zeros(args.num_samples, num_classes, device=device)
        if args.attributes:
            attr_list = [a.strip() for a in args.attributes.split(',')]
            for attr in attr_list:
                if attr in CELEBA_ATTRIBUTES:
                    idx = CELEBA_ATTRIBUTES.index(attr)
                    cfg_cond[:, idx] = 1.0
                else:
                    print(f"Warning: Unknown attribute '{attr}'. Available: {CELEBA_ATTRIBUTES}")
            print(f"Conditioning on: {attr_list}")
        else:
            print("No attributes specified, generating unconditionally")

        # Build anchor condition for contrastive CFG
        if args.contrastive:
            if not args.focal_attributes:
                raise ValueError("--contrastive requires --focal_attributes to specify which attribute(s) to isolate.")
            focal_list = [a.strip() for a in args.focal_attributes.split(',')]
            # Start from the target condition and flip each focal attribute
            cfg_cond_anchor = cfg_cond.clone()
            for attr in focal_list:
                if attr in CELEBA_ATTRIBUTES:
                    idx = CELEBA_ATTRIBUTES.index(attr)
                    # Flip: 1 → 0 (positive isolation) or 0 → 1 (negative suppression)
                    cfg_cond_anchor[:, idx] = 1.0 - cfg_cond_anchor[:, idx]
                else:
                    print(f"Warning: Unknown focal attribute '{attr}'. Available: {CELEBA_ATTRIBUTES}")
            print(f"Contrastive CFG enabled. Focal attribute(s): {focal_list}")
            print(f"  Target:  {dict(zip(attr_list, cfg_cond[0, [CELEBA_ATTRIBUTES.index(a) for a in attr_list if a in CELEBA_ATTRIBUTES]].tolist()))}")
            print(f"  Anchor:  focal attrs flipped")
    
    # Apply EMA weights
    if not args.no_ema:
        print("Using EMA weights")
        ema.apply_shadow()
    else:
        print("Using training weights (no EMA)")
    
    method.eval_mode()
    
    # Image shape
    data_config = config['data']
    image_shape = (data_config['channels'], data_config['image_size'], data_config['image_size'])
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")

    all_samples = []
    remaining = args.num_samples
    sample_idx = 0

    # Create output directory if saving individual images
    if not args.grid:
        os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        pbar = tqdm(total=args.num_samples, desc="Generating samples")
        while remaining > 0:
            batch_size = min(args.batch_size, remaining)

            num_steps = args.num_steps or config.get('sampling', {}).get('num_steps', 100)

            # Build sampling kwargs
            sample_kwargs = {
                'batch_size': batch_size,
                'image_shape': image_shape,
                'num_steps': num_steps,
            }
            
            # Add sampler argument (applies to both ddpm and cfg_ddpm)
            if args.method in ('ddpm', 'cfg_ddpm'):
                sample_kwargs['sampler'] = args.sampler

            # Add CFG-specific arguments
            if args.method == 'cfg_ddpm':
                if cfg_cond is not None:
                    sample_kwargs['cond'] = cfg_cond[sample_idx:sample_idx + batch_size]
                if cfg_cond_anchor is not None:
                    sample_kwargs['cond_anchor'] = cfg_cond_anchor[sample_idx:sample_idx + batch_size]
                if args.guidance_scale is not None:
                    sample_kwargs['guidance_scale'] = args.guidance_scale

            samples = method.sample(**sample_kwargs)

            # Save individual images immediately or collect for grid
            if args.grid:
                all_samples.append(samples)
            else:
                for i in range(samples.shape[0]):
                    img_path = os.path.join(args.output_dir, f"{sample_idx:06d}.png")
                    save_samples(samples[i:i+1], img_path, 1)  # Save individual sample
                    sample_idx += 1

            remaining -= batch_size
            pbar.update(batch_size)

        pbar.close()

    # Save samples
    if args.grid:
        # Concatenate all samples for grid
        all_samples = torch.cat(all_samples, dim=0)[:args.num_samples]

        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"samples_{timestamp}.png"

        save_samples(all_samples, args.output, nrow=8)
        print(f"Saved grid to {args.output}")
    else:
        print(f"Saved {args.num_samples} individual images to {args.output_dir}")

    # Restore EMA if applied
    if not args.no_ema:
        ema.restore()


if __name__ == '__main__':
    main()
