"""
KID (Kernel Inception Distance) Evaluation Script

Computes KID between generated samples and CelebA reference images.
Uses torch-fidelity for computation with proper bootstrap subsampling
for reliable error estimates.

Usage:
    # Single evaluation
    python evaluate_kid.py \
        --generated outputs/hw4_results/kid_samples/standard_eyeglasses \
        --reference_dir outputs/hw4_results/kid_samples/celeba_reference \
        --name "Standard CFG"

    # Compare all HW4 conditions and save JSON
    python evaluate_kid.py \
        --compare_all \
        --reference_dir outputs/hw4_results/kid_samples/celeba_reference \
        --output_json outputs/hw4_results/kid_results.json
"""

import os
import json
import argparse
import torch
import torch_fidelity


def compute_kid(generated_dir: str, reference_dir: str, name: str = "",
                kid_subset_size: int = 100) -> dict:
    """Compute KID between generated samples and CelebA reference.

    Args:
        generated_dir: Directory of generated images
        reference_dir: Directory of reference images
        name: Label for this evaluation
        kid_subset_size: Size of random subsets for KID bootstrap.
            Smaller values (e.g. 100) produce more subsets and more
            reliable std estimates. Default torch-fidelity is 1000,
            but with 1000 generated and 1000 reference images that
            gives only 1 subset (no variance estimate).
    """
    n_gen = len([f for f in os.listdir(generated_dir) if f.endswith('.png')])
    print(f"\nComputing KID for: {name or generated_dir}")
    print(f"  Generated: {generated_dir} ({n_gen} images)")
    print(f"  kid_subset_size: {kid_subset_size}")

    metrics = torch_fidelity.calculate_metrics(
        input1=generated_dir,
        input2=reference_dir,
        kid=True,
        fid=False,
        isc=False,
        kid_subset_size=kid_subset_size,
        verbose=False,
        cuda=torch.cuda.is_available(),
    )

    kid_mean = metrics["kernel_inception_distance_mean"]
    kid_std = metrics["kernel_inception_distance_std"]
    print(f"  KID: {kid_mean:.6f} +/- {kid_std:.6f}")
    return {"name": name, "kid_mean": kid_mean, "kid_std": kid_std, "n_samples": n_gen}


def main():
    parser = argparse.ArgumentParser(description="Compute KID for generated samples")
    parser.add_argument("--generated", type=str, default="",
                        help="Directory of generated images")
    parser.add_argument("--reference_dir", type=str, required=True,
                        help="Directory of reference (CelebA) images")
    parser.add_argument("--name", type=str, default="",
                        help="Label for this run")
    parser.add_argument("--kid_subset_size", type=int, default=100,
                        help="KID bootstrap subset size (default: 100 for reliable std)")
    parser.add_argument("--compare_all", action="store_true",
                        help="Compare all HW4 KID sample directories")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    if args.compare_all:
        base = "outputs/hw4_results/kid_samples"
        results = {}
        configs = [
            ("unconditional",           "Unconditional CFG"),
            ("standard_eyeglasses",     "Standard CFG (Eyeglasses,Male,Young)"),
            ("contrastive_eyeglasses",  "Contrastive CFG (focal=Eyeglasses)"),
        ]
        for dirname, label in configs:
            d = os.path.join(base, dirname)
            if os.path.isdir(d) and len(os.listdir(d)) > 0:
                r = compute_kid(d, args.reference_dir, label,
                                kid_subset_size=args.kid_subset_size)
                results[dirname] = r

        print("\n" + "=" * 70)
        print("SUMMARY: KID Results")
        print("=" * 70)
        print(f"{'Method':<45} {'KID Mean':>12} {'KID Std':>12}")
        print("-" * 70)
        for key, r in results.items():
            print(f"{r['name']:<45} {r['kid_mean']:>12.6f} {r['kid_std']:>12.6f}")

        if args.output_json:
            os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved results to {args.output_json}")

    else:
        r = compute_kid(args.generated, args.reference_dir, args.name,
                        kid_subset_size=args.kid_subset_size)
        if args.output_json:
            os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
            with open(args.output_json, 'w') as f:
                json.dump(r, f, indent=2)
            print(f"\nSaved results to {args.output_json}")


if __name__ == "__main__":
    main()
