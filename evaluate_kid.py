"""
KID (Kernel Inception Distance) Evaluation Script

Computes KID between generated samples and CelebA reference images.
Uses torch-fidelity for computation.

Usage:
    python evaluate_kid.py \
        --generated outputs/hw4_results/kid_samples/standard_eyeglasses \
        --reference_dir data/celeba-subset \
        --name "Standard CFG"
"""

import os
import argparse
import torch
import torch_fidelity
from src.data import CELEBA_ATTRIBUTES


def compute_kid(generated_dir: str, reference_dir: str, name: str = "") -> dict:
    """Compute KID between generated samples and CelebA reference."""
    print(f"\nComputing KID for: {name or generated_dir}")
    print(f"  Generated: {generated_dir} ({len(os.listdir(generated_dir))} images)")

    metrics = torch_fidelity.calculate_metrics(
        input1=generated_dir,
        input2=reference_dir,
        kid=True,
        fid=False,
        isc=False,
        kid_subset_size=min(1000, len(os.listdir(generated_dir))),
        verbose=False,
        cuda=torch.cuda.is_available(),
    )

    kid_mean = metrics["kernel_inception_distance_mean"]
    kid_std = metrics["kernel_inception_distance_std"]
    print(f"  KID: {kid_mean:.4f} ± {kid_std:.4f}")
    return {"name": name, "kid_mean": kid_mean, "kid_std": kid_std}


def main():
    parser = argparse.ArgumentParser(description="Compute KID for generated samples")
    parser.add_argument("--generated", type=str, required=False, default="",
                        help="Directory of generated images")
    parser.add_argument("--reference_dir", type=str, required=True,
                        help="Directory of reference (CelebA) images")
    parser.add_argument("--name", type=str, default="",
                        help="Label for this run")
    parser.add_argument("--compare_all", action="store_true",
                        help="Compare all three hw4 KID sample directories at once")
    args = parser.parse_args()

    if args.compare_all:
        base = "outputs/hw4_results/kid_samples"
        results = []
        configs = [
            ("unconditional",           "Unconditional CFG"),
            ("standard_eyeglasses",     "Standard CFG (Eyeglasses,Male,Young)"),
            ("contrastive_eyeglasses",  "Contrastive CFG (focal=Eyeglasses)"),
        ]
        for dirname, label in configs:
            d = os.path.join(base, dirname)
            if os.path.isdir(d) and len(os.listdir(d)) > 0:
                r = compute_kid(d, args.reference_dir, label)
                results.append(r)

        print("\n" + "="*60)
        print("SUMMARY: KID Results")
        print("="*60)
        print(f"{'Method':<45} {'KID Mean':>10} {'KID Std':>10}")
        print("-"*65)
        for r in results:
            print(f"{r['name']:<45} {r['kid_mean']:>10.4f} {r['kid_std']:>10.4f}")
    else:
        compute_kid(args.generated, args.reference_dir, args.name)


if __name__ == "__main__":
    main()
