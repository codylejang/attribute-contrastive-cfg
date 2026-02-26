"""
LPIPS Diversity Evaluation

Computes mean pairwise LPIPS distance within sets of generated samples.
Higher LPIPS = more diverse samples. Lower LPIPS = mode collapse.

Usage:
    python evaluate_diversity.py \
        --output_json outputs/hw4_results/diversity_results.json
"""

import os
import json
import argparse
import random
from pathlib import Path

import torch
import lpips
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


def load_images(image_dir: str, n_samples: int = 100, seed: int = 42) -> torch.Tensor:
    """Load n_samples random PNG images from a directory as a tensor."""
    image_paths = sorted(Path(image_dir).glob("*.png"))
    if len(image_paths) == 0:
        raise ValueError(f"No PNG images found in {image_dir}")

    random.seed(seed)
    if len(image_paths) > n_samples:
        image_paths = random.sample(image_paths, n_samples)

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        # LPIPS expects images in [-1, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        images.append(transform(img))

    return torch.stack(images)


def compute_pairwise_lpips(images: torch.Tensor, lpips_fn, device: torch.device,
                           batch_size: int = 64) -> dict:
    """Compute mean pairwise LPIPS distance for a set of images.

    Args:
        images: (N, 3, H, W) tensor in [-1, 1]
        lpips_fn: LPIPS distance function
        batch_size: Number of pairs to evaluate at once

    Returns:
        dict with lpips_mean, lpips_std, n_pairs
    """
    n = images.shape[0]
    all_distances = []

    # Generate all pairs
    pairs_i = []
    pairs_j = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs_i.append(i)
            pairs_j.append(j)

    n_pairs = len(pairs_i)

    # Process in batches
    for start in range(0, n_pairs, batch_size):
        end = min(start + batch_size, n_pairs)
        idx_i = pairs_i[start:end]
        idx_j = pairs_j[start:end]

        imgs_a = images[idx_i].to(device)
        imgs_b = images[idx_j].to(device)

        with torch.no_grad():
            d = lpips_fn(imgs_a, imgs_b)
        all_distances.append(d.cpu().squeeze())

    all_distances = torch.cat(all_distances)
    return {
        "lpips_mean": float(all_distances.mean()),
        "lpips_std": float(all_distances.std()),
        "n_pairs": n_pairs,
        "n_images": n,
    }


def main():
    parser = argparse.ArgumentParser(description="LPIPS diversity evaluation")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of images to sample per directory (default: 100)")
    parser.add_argument("--output_json", type=str,
                        default="outputs/hw4_results/diversity_results.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load LPIPS (AlexNet backbone — lightweight)
    print("Loading LPIPS model...")
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    base = "outputs/hw4_results/kid_samples"
    results = {}

    # Find all sample directories
    sample_dirs = []
    for dirname in sorted(os.listdir(base)):
        d = os.path.join(base, dirname)
        if os.path.isdir(d) and dirname != "celeba_reference":
            png_count = len(list(Path(d).glob("*.png")))
            if png_count > 0:
                sample_dirs.append((dirname, d, png_count))

    print(f"\nFound {len(sample_dirs)} sample directories")

    for dirname, dirpath, n_images in sample_dirs:
        n_to_load = min(args.n_samples, n_images)
        print(f"\n{'='*60}")
        print(f"  {dirname}: {n_images} images, sampling {n_to_load}")

        images = load_images(dirpath, n_samples=n_to_load, seed=args.seed)
        result = compute_pairwise_lpips(images, lpips_fn, device)
        results[dirname] = result

        print(f"  LPIPS: {result['lpips_mean']:.4f} +/- {result['lpips_std']:.4f} "
              f"({result['n_pairs']} pairs)")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: LPIPS Diversity (higher = more diverse)")
    print(f"{'='*60}")
    print(f"{'Directory':<40} {'LPIPS Mean':>12} {'LPIPS Std':>12}")
    print("-" * 65)
    for dirname, r in results.items():
        print(f"{dirname:<40} {r['lpips_mean']:>12.4f} {r['lpips_std']:>12.4f}")

    # Save results
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.output_json}")


if __name__ == "__main__":
    main()
