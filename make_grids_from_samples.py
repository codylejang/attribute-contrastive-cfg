"""
Create grid images from individual 64x64 sample PNGs.

For each smiling sample directory (standard and contrastive, w=1..7),
pick 16 random images and tile them into a 4x4 grid.

Outputs:
  - outputs/hw4_results/ablations/guidance_scale/standard_smiling_w{X}.png
  - outputs/hw4_results/ablations/guidance_scale/contrastive_smiling_w{X}.png
  - outputs/hw4_results/qualitative/standard_smiling_w2.png   (for fig2)
  - outputs/hw4_results/qualitative/contrastive_smiling_w2.png (for fig2)
"""

import os
import random
import glob
from PIL import Image

SAMPLES_DIR = "outputs/hw4_results/kid_samples"
ABL_DIR = "outputs/hw4_results/ablations/guidance_scale"
QUAL_DIR = "outputs/hw4_results/qualitative"

os.makedirs(ABL_DIR, exist_ok=True)
os.makedirs(QUAL_DIR, exist_ok=True)

GRID_SIZE = 4  # 4x4 grid
N_IMAGES = GRID_SIZE * GRID_SIZE
SEED = 42

random.seed(SEED)

methods = ["standard", "contrastive"]
ws = ["1.0", "2.0", "3.0", "5.0", "7.0"]


def make_grid(image_paths, grid_size=4):
    """Tile images into a grid_size x grid_size grid."""
    imgs = [Image.open(p).convert("RGB") for p in image_paths[:grid_size * grid_size]]
    if not imgs:
        raise ValueError("No images found")
    w, h = imgs[0].size
    grid = Image.new("RGB", (w * grid_size, h * grid_size))
    for idx, img in enumerate(imgs):
        row, col = divmod(idx, grid_size)
        grid.paste(img, (col * w, row * h))
    return grid


for method in methods:
    for w in ws:
        dir_name = f"{method}_smiling_w{w}"
        sample_dir = os.path.join(SAMPLES_DIR, dir_name)
        if not os.path.isdir(sample_dir):
            print(f"SKIP {dir_name} (directory not found)")
            continue

        pngs = sorted(glob.glob(os.path.join(sample_dir, "*.png")))
        if len(pngs) < N_IMAGES:
            print(f"SKIP {dir_name} (only {len(pngs)} images, need {N_IMAGES})")
            continue

        selected = random.sample(pngs, N_IMAGES)
        grid = make_grid(selected, GRID_SIZE)

        # Save to ablation directory
        abl_path = os.path.join(ABL_DIR, f"{dir_name}.png")
        grid.save(abl_path)
        print(f"Saved {abl_path}")

        # For w=2.0, also save to qualitative directory (for fig2 side-by-side)
        if w == "2.0":
            qual_path = os.path.join(QUAL_DIR, f"{method}_smiling_w2.png")
            grid.save(qual_path)
            print(f"Saved {qual_path}")

print("\nDone creating grids.")
