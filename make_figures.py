"""
Generate HW4 visualizations for the writeup.

Figures produced:
  fig1_kid_bar.png              — KID comparison bar chart
  fig2_eyeglasses_side_by_side.png  — Standard vs Contrastive at w=2
  fig3_guidance_scale_ablation.png  — w={1,2,3,5,7} for both methods
  fig4_attribute_comparison.png     — Blond Hair + Smiling (control conditions)
  fig5_negative_suppression.png     — Negative suppression of Eyeglasses
  fig6_composite_summary.png        — All key findings in one figure
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from PIL import Image

OUT = "outputs/hw4_results/figures"
QUAL = "outputs/hw4_results/qualitative"
ABL  = "outputs/hw4_results/ablations/guidance_scale"
os.makedirs(OUT, exist_ok=True)

# ─── Color palette ────────────────────────────────────────────────────────────
C_STD  = "#4878CF"   # blue  — Standard CFG
C_CTR  = "#D65F5F"   # red   — Contrastive CFG
C_UNC  = "#6ACC65"   # green — Unconditional
GRAY   = "#888888"
BG     = "#F8F8F8"

def load(path):
    return Image.open(path).convert("RGB")

def add_label(ax, text, color="white", fontsize=11, bg=None):
    """Add a bold label box in the top-left of an axes."""
    ax.text(0.02, 0.97, text,
            transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", color=color,
            va="top", ha="left",
            bbox=dict(facecolor=bg or "black", alpha=0.55, pad=3, edgecolor="none"))

def strip_axes(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 — KID bar chart
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
ax.set_facecolor(BG)

labels  = ["Unconditional\nCFG", "Standard CFG\n(Eyeglasses,Male,Young)", "Contrastive CFG\n(focal=Eyeglasses) ★"]
values  = [0.0235, 0.0316, 0.0263]
colors  = [C_UNC, C_STD, C_CTR]

bars = ax.bar(labels, values, color=colors, width=0.5, zorder=3,
              edgecolor="white", linewidth=1.2)

# Value labels on top
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.0005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

# Improvement annotation
ax.annotate("", xy=(2, 0.0263), xytext=(1, 0.0316),
            arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.5))
ax.text(1.5, 0.032, "−16.8%", ha="center", va="bottom",
        fontsize=11, fontweight="bold", color=C_CTR)

ax.set_ylabel("KID (lower = better)", fontsize=12)
ax.set_title("Kernel Inception Distance vs. CelebA Reference\n"
             "(1000 generated samples, DDIM 100 steps, w=2.0)", fontsize=12)
ax.set_ylim(0, 0.038)
ax.grid(axis="y", alpha=0.4, zorder=0)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/fig1_kid_bar.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved fig1_kid_bar.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Side-by-side Eyeglasses (Standard vs Contrastive, w=2)
# ═══════════════════════════════════════════════════════════════════════════════
std_img = load(f"{QUAL}/standard_eyeglasses_w2.png")
ctr_img = load(f"{QUAL}/contrastive_eyeglasses_w2.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), facecolor="white")
fig.suptitle('Condition: Eyeglasses=1, Male=1, Young=1   |   Guidance scale w=2.0',
             fontsize=13, fontweight="bold", y=1.01)

axes[0].imshow(std_img)
strip_axes(axes[0])
axes[0].set_title("Standard CFG\n(null baseline  ε̃ = ε_uncond + w·(ε_cond − ε_uncond))",
                  fontsize=11, color=C_STD, fontweight="bold")
add_label(axes[0], "STANDARD", bg=C_STD)

axes[1].imshow(ctr_img)
strip_axes(axes[1])
axes[1].set_title("Attribute-Contrastive CFG  ★\n(anchor baseline  ε̃ = ε_anchor + w·(ε_target − ε_anchor))",
                  fontsize=11, color=C_CTR, fontweight="bold")
add_label(axes[1], "CONTRASTIVE", bg=C_CTR)

plt.tight_layout()
plt.savefig(f"{OUT}/fig2_eyeglasses_side_by_side.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2_eyeglasses_side_by_side.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Guidance scale ablation (both methods, 5 w values)
# ═══════════════════════════════════════════════════════════════════════════════
ws = [1.0, 2.0, 3.0, 5.0, 7.0]

fig = plt.figure(figsize=(18, 5.5), facecolor="white")
fig.suptitle("Guidance Scale Ablation — Eyeglasses,Male,Young  (DDIM 100 steps)",
             fontsize=14, fontweight="bold", y=1.01)

gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.08, wspace=0.05)

for col, w in enumerate(ws):
    # Row 0: Standard
    ax_s = fig.add_subplot(gs[0, col])
    ax_s.imshow(load(f"{ABL}/standard_eyeglasses_w{w}.png"))
    strip_axes(ax_s)
    if col == 0:
        ax_s.set_ylabel("Standard CFG", fontsize=11, color=C_STD, fontweight="bold",
                        labelpad=6)
    ax_s.set_title(f"w = {w}", fontsize=11, fontweight="bold")
    if w == 7.0:
        for spine in ax_s.spines.values():
            spine.set_visible(True); spine.set_edgecolor("red"); spine.set_linewidth(3)

    # Row 1: Contrastive
    ax_c = fig.add_subplot(gs[1, col])
    ax_c.imshow(load(f"{ABL}/contrastive_eyeglasses_w{w}.png"))
    strip_axes(ax_c)
    if col == 0:
        ax_c.set_ylabel("Contrastive CFG ★", fontsize=11, color=C_CTR, fontweight="bold",
                        labelpad=6)
    if w == 7.0:
        for spine in ax_c.spines.values():
            spine.set_visible(True); spine.set_edgecolor(C_CTR); spine.set_linewidth(3)

# Legend annotations
fig.text(0.895, 0.52, "← Mode collapse\n   (standard CFG)", fontsize=9.5,
         color="red", ha="left", va="center", fontweight="bold")
fig.text(0.895, 0.15, "← Diverse, natural\n   (contrastive CFG)", fontsize=9.5,
         color=C_CTR, ha="left", va="center", fontweight="bold")

plt.savefig(f"{OUT}/fig3_guidance_scale_ablation.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved fig3_guidance_scale_ablation.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Attribute comparison: Blond Hair + Smiling (control conditions)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), facecolor="white")
fig.suptitle("Control Conditions — Attributes that Standard CFG Already Handles Well",
             fontsize=13, fontweight="bold", y=1.01)

imgs = [
    (f"{QUAL}/standard_blond_hair_w2.png",   "Standard CFG",   "Blond_Hair, Young"),
    (f"{QUAL}/contrastive_blond_hair_w2.png", "Contrastive CFG ★", "Blond_Hair, Young"),
    (f"{QUAL}/standard_smiling_w2.png",       "Standard CFG",   "Smiling"),
    (f"{QUAL}/contrastive_smiling_w2.png",    "Contrastive CFG ★", "Smiling"),
]

row_labels = ["Blond_Hair, Young", "Smiling"]
for i, (path, method, cond) in enumerate(imgs):
    r, c = divmod(i, 2)
    ax = axes[r][c]
    ax.imshow(load(path))
    strip_axes(ax)
    color = C_CTR if "Contrastive" in method else C_STD
    label = "CONTRASTIVE" if "Contrastive" in method else "STANDARD"
    ax.set_title(f"{method}\n(w=2.0)", fontsize=11, color=color, fontweight="bold")
    add_label(ax, label, bg=color)

# Row labels on the left
fig.text(0.01, 0.75, "Blond_Hair\n+ Young", fontsize=12, fontweight="bold",
         va="center", rotation=90)
fig.text(0.01, 0.28, "Smiling", fontsize=12, fontweight="bold",
         va="center", rotation=90)

# Annotation: contrastive = identical
fig.text(0.5, -0.01,
         "Contrastive CFG produces identical results on easy attributes — it is a strict generalization of standard CFG.",
         fontsize=10.5, ha="center", style="italic", color=GRAY)

plt.tight_layout(rect=[0.03, 0, 1, 1])
plt.savefig(f"{OUT}/fig4_attribute_comparison.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved fig4_attribute_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Negative suppression
# ═══════════════════════════════════════════════════════════════════════════════
std_img   = load(f"{QUAL}/standard_eyeglasses_w2.png")
suppr_img = load(f"{QUAL}/suppress_eyeglasses_w2.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), facecolor="white")
fig.suptitle("Negative Suppression via Contrastive CFG\n"
             "Target: Male=1, Young=1  |  Focal: Eyeglasses (0→1 flip in anchor = suppress glasses)",
             fontsize=12, fontweight="bold", y=1.03)

axes[0].imshow(std_img)
strip_axes(axes[0])
axes[0].set_title("Standard CFG\nMale=1, Young=1, Eyeglasses=1", fontsize=11,
                  color=C_STD, fontweight="bold")
add_label(axes[0], "STANDARD\n(attract Eyeglasses)", bg=C_STD)

axes[1].imshow(suppr_img)
strip_axes(axes[1])
axes[1].set_title("Contrastive CFG — Negative Suppression  ★\nMale=1, Young=1  +  suppress Eyeglasses",
                  fontsize=11, color=C_CTR, fontweight="bold")
add_label(axes[1], "CONTRASTIVE\n(suppress Eyeglasses)", bg=C_CTR)

plt.tight_layout()
plt.savefig(f"{OUT}/fig5_negative_suppression.png", dpi=150,
            bbox_inches="tight")
plt.close()
print("Saved fig5_negative_suppression.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6 — Composite summary figure (all key findings)
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 12), facecolor="white")
fig.suptitle("HW4: Attribute-Contrastive CFG — Key Results Summary",
             fontsize=16, fontweight="bold", y=1.005)

gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.35, wspace=0.05)

# ── Row 0: KID bar (left 2 cols) + Eyeglasses side-by-side (right 4 cols) ──
ax_kid = fig.add_subplot(gs[0, :2])
ax_kid.set_facecolor(BG)

bars = ax_kid.bar(["Unconditional", "Standard CFG", "Contrastive CFG ★"],
                  [0.0235, 0.0316, 0.0263],
                  color=[C_UNC, C_STD, C_CTR], width=0.5, zorder=3,
                  edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, [0.0235, 0.0316, 0.0263]):
    ax_kid.text(bar.get_x() + bar.get_width()/2, val + 0.0003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
ax_kid.annotate("", xy=(2, 0.0263), xytext=(1, 0.0316),
                arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.2))
ax_kid.text(1.5, 0.031, "−16.8%", ha="center", fontsize=9, fontweight="bold", color=C_CTR)
ax_kid.set_ylabel("KID ↓", fontsize=10)
ax_kid.set_title("(a) KID vs. CelebA (w=2.0)", fontsize=10, fontweight="bold")
ax_kid.set_ylim(0, 0.037); ax_kid.grid(axis="y", alpha=0.35, zorder=0)
ax_kid.spines[["top","right"]].set_visible(False)
ax_kid.tick_params(axis="x", labelsize=8)

ax_std_e = fig.add_subplot(gs[0, 2:4])
ax_std_e.imshow(load(f"{QUAL}/standard_eyeglasses_w2.png"))
strip_axes(ax_std_e)
ax_std_e.set_title("(b) Standard CFG — Eyeglasses w=2", fontsize=9.5,
                   color=C_STD, fontweight="bold")
add_label(ax_std_e, "STANDARD", bg=C_STD, fontsize=9)

ax_ctr_e = fig.add_subplot(gs[0, 4:6])
ax_ctr_e.imshow(load(f"{QUAL}/contrastive_eyeglasses_w2.png"))
strip_axes(ax_ctr_e)
ax_ctr_e.set_title("(c) Contrastive CFG ★ — Eyeglasses w=2", fontsize=9.5,
                   color=C_CTR, fontweight="bold")
add_label(ax_ctr_e, "CONTRASTIVE", bg=C_CTR, fontsize=9)

# ── Row 1: Guidance scale ablation (w=2 and w=7, both methods) ──
titles_r1 = [
    ("Standard w=2",     f"{ABL}/standard_eyeglasses_w2.0.png",     C_STD),
    ("Standard w=3",     f"{ABL}/standard_eyeglasses_w3.0.png",     C_STD),
    ("Standard w=5",     f"{ABL}/standard_eyeglasses_w5.0.png",     C_STD),
    ("Standard w=7 ✗",   f"{ABL}/standard_eyeglasses_w7.0.png",     "red"),
    ("Contrastive w=5",  f"{ABL}/contrastive_eyeglasses_w5.0.png",  C_CTR),
    ("Contrastive w=7 ✓",f"{ABL}/contrastive_eyeglasses_w7.0.png",  C_CTR),
]
for col, (title, path, color) in enumerate(titles_r1):
    ax = fig.add_subplot(gs[1, col])
    ax.imshow(load(path))
    strip_axes(ax)
    ax.set_title(title, fontsize=9, color=color, fontweight="bold")
    if "7 ✗" in title:
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor("red"); sp.set_linewidth(2.5)
    if "7 ✓" in title:
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(C_CTR); sp.set_linewidth(2.5)

fig.text(0.5, 0.37, "(d) Guidance scale ablation — standard CFG collapses at w=7; contrastive CFG degrades gracefully",
         ha="center", fontsize=10, fontweight="bold")

# ── Row 2: Control conditions (Blond Hair) + Suppression ──
row2 = [
    ("(e) Standard — Blond_Hair,Young",   f"{QUAL}/standard_blond_hair_w2.png",   C_STD),
    ("(e) Contrastive ★ — Blond_Hair",    f"{QUAL}/contrastive_blond_hair_w2.png", C_CTR),
    ("(f) Standard — Smiling",            f"{QUAL}/standard_smiling_w2.png",       C_STD),
    ("(f) Contrastive ★ — Smiling",       f"{QUAL}/contrastive_smiling_w2.png",    C_CTR),
    ("(g) Suppress Eyeglasses",           f"{QUAL}/suppress_eyeglasses_w2.png",    C_CTR),
    ("(h) Std w=7 collapse",              f"{ABL}/standard_eyeglasses_w7.0.png",   "red"),
]
for col, (title, path, color) in enumerate(row2):
    ax = fig.add_subplot(gs[2, col])
    ax.imshow(load(path))
    strip_axes(ax)
    ax.set_title(title, fontsize=8.5, color=color, fontweight="bold")

# Overall caption
fig.text(0.5, -0.01,
         "Attribute-Contrastive CFG: replaces the null CFG baseline with a targeted anchor condition (same attrs, focal attr flipped).\n"
         "Result: −16.8% KID, identical performance on easy attributes, and greatly improved stability at high guidance scales.",
         ha="center", fontsize=10, style="italic", color="#444444")

plt.savefig(f"{OUT}/fig6_composite_summary.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved fig6_composite_summary.png")

print(f"\nAll figures saved to {OUT}/")
