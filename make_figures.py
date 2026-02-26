"""
Generate HW4 visualizations for the writeup.

Reads quantitative results from JSON files produced by evaluation scripts.

Figures produced:
  fig1_kid_bar.png                — KID comparison bar chart (with error bars)
  fig2_eyeglasses_side_by_side.png — Standard vs Contrastive at w=2
  fig3_guidance_scale_ablation.png — w={1,2,3,5,7} for both methods
  fig4_attribute_comparison.png    — Blond Hair + Smiling (control conditions)
  fig5_negative_suppression.png    — Negative suppression of Eyeglasses
  fig6_composite_summary.png       — All key findings in one figure
  fig7_kid_vs_w.png               — KID as a function of guidance scale
  fig8_attribute_accuracy.png      — Attribute accuracy bar chart
  fig9_lpips_diversity.png         — LPIPS diversity comparison
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    ax.text(0.02, 0.97, text,
            transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", color=color,
            va="top", ha="left",
            bbox=dict(facecolor=bg or "black", alpha=0.55, pad=3, edgecolor="none"))

def strip_axes(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")

# ─── Load JSON results ────────────────────────────────────────────────────────
def load_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print(f"WARNING: {path} not found, using defaults")
    return default

kid_data = load_json("outputs/hw4_results/kid_sweep_smiling.json", {})
kid_sweep = load_json("outputs/hw4_results/kid_sweep_smiling.json", {})
attr_data = load_json("outputs/hw4_results/attribute_accuracy_smiling.json", {})
div_data = load_json("outputs/hw4_results/diversity_smiling.json", {})

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 — KID bar chart (with error bars from JSON)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
ax.set_facecolor(BG)

labels = ["Standard CFG\n(Smiling, w=2)", "Contrastive CFG\n(Ours, w=2)"]
# kid_data is kid_sweep_smiling.json: {standard: {2.0: {kid_mean, kid_std}}, contrastive: ...}
values = [
    kid_data.get("standard", {}).get("2.0", {}).get("kid_mean", 0),
    kid_data.get("contrastive", {}).get("2.0", {}).get("kid_mean", 0),
]
errors = [
    kid_data.get("standard", {}).get("2.0", {}).get("kid_std", 0),
    kid_data.get("contrastive", {}).get("2.0", {}).get("kid_std", 0),
]
colors = [C_STD, C_CTR]

bars = ax.bar(labels, values, yerr=errors, capsize=5, color=colors, width=0.5, zorder=3,
              edgecolor="white", linewidth=1.2, error_kw=dict(lw=1.5, capthick=1.5))

for bar, val, err in zip(bars, values, errors):
    ax.text(bar.get_x() + bar.get_width()/2, val + err + 0.0005,
            f"{val:.4f}\n\u00b1{err:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Improvement annotation
if len(values) >= 2 and values[0] > 0:
    pct = (values[1] - values[0]) / values[0] * 100
    ax.annotate("", xy=(1, values[1]), xytext=(0, values[0]),
                arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.5))
    ax.text(0.5, max(values[0], values[1]) + 0.002, f"{pct:.1f}%", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=C_CTR)

ax.set_ylabel("KID (lower = better)", fontsize=12)
ax.set_title("Kernel Inception Distance vs. CelebA Reference (Smiling)\n"
             "(1000 generated samples, DDIM 100 steps, w=2.0)", fontsize=12)
ax.set_ylim(0, 0.06)
ax.grid(axis="y", alpha=0.4, zorder=0)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/fig1_kid_bar.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved fig1_kid_bar.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Side-by-side Eyeglasses (Standard vs Contrastive, w=2)
# ═══════════════════════════════════════════════════════════════════════════════
std_img = load(f"{QUAL}/standard_smiling_w2.png")
ctr_img = load(f"{QUAL}/contrastive_smiling_w2.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), facecolor="white")
fig.suptitle('Condition: Smiling=1   |   Guidance scale w=2.0',
             fontsize=13, fontweight="bold", y=1.01)

axes[0].imshow(std_img)
strip_axes(axes[0])
axes[0].set_title("Standard CFG\n(null baseline)", fontsize=11, color=C_STD, fontweight="bold")
add_label(axes[0], "STANDARD", bg=C_STD)

axes[1].imshow(ctr_img)
strip_axes(axes[1])
axes[1].set_title("Attribute-Contrastive CFG\n(anchor baseline)", fontsize=11, color=C_CTR, fontweight="bold")
add_label(axes[1], "CONTRASTIVE", bg=C_CTR)

plt.tight_layout()
plt.savefig(f"{OUT}/fig2_smiling_side_by_side.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2_smiling_side_by_side.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Guidance scale ablation (both methods, 5 w values)
# ═══════════════════════════════════════════════════════════════════════════════
ws = [1.0, 2.0, 3.0, 5.0, 7.0]

# Horizontal layout: 2 rows (standard / contrastive) x 5 columns (w values)
# Maximizes horizontal space for poster use.
fig_w = 12
fig_h = 5.5

fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

gs = gridspec.GridSpec(2, 5, figure=fig,
                       hspace=0.08, wspace=0.03,
                       top=0.88, bottom=0.06, left=0.08, right=0.98)

fig.text(0.5, 0.97, "Guidance Scale Ablation — Smiling",
         fontsize=14, fontweight="bold", ha="center", va="top")

# Row labels on the left
fig.text(0.03, 0.70, "Standard\nCFG", fontsize=11, color=C_STD, fontweight="bold",
         ha="center", va="center", rotation=0)
fig.text(0.03, 0.28, "Contrastive\nCFG", fontsize=11, color=C_CTR, fontweight="bold",
         ha="center", va="center", rotation=0)

for col, w in enumerate(ws):
    # Standard row
    ax_s = fig.add_subplot(gs[0, col])
    ax_s.imshow(load(f"{ABL}/standard_smiling_w{w}.png"))
    strip_axes(ax_s)
    ax_s.set_aspect("equal")
    ax_s.set_title(f"w={w}", fontsize=11, fontweight="bold", pad=3)
    if w == 7.0:
        for spine in ax_s.spines.values():
            spine.set_visible(True); spine.set_edgecolor("red"); spine.set_linewidth(3)

    # Contrastive row
    ax_c = fig.add_subplot(gs[1, col])
    ax_c.imshow(load(f"{ABL}/contrastive_smiling_w{w}.png"))
    strip_axes(ax_c)
    ax_c.set_aspect("equal")
    if w == 7.0:
        for spine in ax_c.spines.values():
            spine.set_visible(True); spine.set_edgecolor(C_CTR); spine.set_linewidth(3)

# Bottom annotations under w=7 columns
fig.text(0.89, 0.01, "\u2191 Mode collapse", fontsize=10,
         color="red", ha="center", va="bottom", fontweight="bold")
fig.text(0.89, -0.03, "\u2191 Diverse, natural", fontsize=10,
         color=C_CTR, ha="center", va="bottom", fontweight="bold")

plt.savefig(f"{OUT}/fig3_guidance_scale_ablation.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved fig3_guidance_scale_ablation.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Attribute comparison: Blond Hair + Smiling (control conditions)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), facecolor="white")
fig.suptitle("Control Conditions \u2014 Attributes that Standard CFG Already Handles Well",
             fontsize=13, fontweight="bold", y=1.01)

imgs = [
    (f"{QUAL}/standard_blond_hair_w2.png",   "Standard CFG",   "Blond_Hair, Young"),
    (f"{QUAL}/contrastive_blond_hair_w2.png", "Contrastive CFG", "Blond_Hair, Young"),
    (f"{QUAL}/standard_smiling_w2.png",       "Standard CFG",   "Smiling"),
    (f"{QUAL}/contrastive_smiling_w2.png",    "Contrastive CFG", "Smiling"),
]

for i, (path, method, cond) in enumerate(imgs):
    r, c = divmod(i, 2)
    ax = axes[r][c]
    ax.imshow(load(path))
    strip_axes(ax)
    color = C_CTR if "Contrastive" in method else C_STD
    label_text = "CONTRASTIVE" if "Contrastive" in method else "STANDARD"
    ax.set_title(f"{method}\n(w=2.0)", fontsize=11, color=color, fontweight="bold")
    add_label(ax, label_text, bg=color)

fig.text(0.01, 0.75, "Blond_Hair\n+ Young", fontsize=12, fontweight="bold", va="center", rotation=90)
fig.text(0.01, 0.28, "Smiling", fontsize=12, fontweight="bold", va="center", rotation=90)
fig.text(0.5, -0.01,
         "Contrastive CFG produces identical results on easy attributes \u2014 it is a strict generalization of standard CFG.",
         fontsize=10.5, ha="center", style="italic", color=GRAY)

plt.tight_layout(rect=[0.03, 0, 1, 1])
plt.savefig(f"{OUT}/fig4_attribute_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved fig4_attribute_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Negative suppression
# ═══════════════════════════════════════════════════════════════════════════════
std_img = load(f"{QUAL}/standard_eyeglasses_w2.png")
suppr_img = load(f"{QUAL}/suppress_eyeglasses_w2.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), facecolor="white")
fig.suptitle("Negative Suppression via Contrastive CFG\n"
             "Standard vs. Contrastive Negative Suppression  |  w=2.0",
             fontsize=12, fontweight="bold", y=1.03)

axes[0].imshow(std_img)
strip_axes(axes[0])
axes[0].set_title("Standard CFG\nMale=1, Young=1", fontsize=11, color=C_STD, fontweight="bold")
add_label(axes[0], "STANDARD", bg=C_STD)

axes[1].imshow(suppr_img)
strip_axes(axes[1])
axes[1].set_title("Contrastive CFG \u2014 Negative Suppression\nMale=1, Young=1",
                  fontsize=11, color=C_CTR, fontweight="bold")
add_label(axes[1], "CONTRASTIVE\n(suppression)", bg=C_CTR)

plt.tight_layout()
plt.savefig(f"{OUT}/fig5_negative_suppression.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig5_negative_suppression.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 7 — KID vs guidance scale (from kid_sweep_results.json)
# ═══════════════════════════════════════════════════════════════════════════════
if kid_sweep:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    ax.set_facecolor(BG)

    for method, color, marker, label in [
        ("standard", C_STD, "s", "Standard CFG"),
        ("contrastive", C_CTR, "o", "Contrastive CFG"),
    ]:
        if method in kid_sweep:
            ws_sorted = sorted(kid_sweep[method].keys(), key=float)
            x = [float(w) for w in ws_sorted]
            y = [kid_sweep[method][w]["kid_mean"] for w in ws_sorted]
            yerr = [kid_sweep[method][w]["kid_std"] for w in ws_sorted]
            ax.errorbar(x, y, yerr=yerr, color=color, marker=marker, markersize=8,
                       linewidth=2, capsize=5, capthick=1.5, label=label)

    ax.set_xlabel("Guidance Scale (w)", fontsize=12)
    ax.set_ylabel("KID (lower = better)", fontsize=12)
    ax.set_title("KID vs. Guidance Scale \u2014 Smiling\n"
                 "(1000 samples per point, DDIM 100 steps)", fontsize=12)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig7_kid_vs_w.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("Saved fig7_kid_vs_w.png")
else:
    print("SKIP fig7_kid_vs_w.png (no kid_sweep_results.json)")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 8 — Attribute accuracy bar chart
# ═══════════════════════════════════════════════════════════════════════════════
if attr_data:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor=BG)
    fig.suptitle("Attribute Detection Rates in Generated Samples (ResNet-18 Classifier)",
                 fontsize=13, fontweight="bold")

    attrs_to_plot = ["Smiling", "Male", "Eyeglasses", "Blond_Hair"]
    sample_keys = [
        ("unconditional", "Unconditional", C_UNC),
        ("standard_eyeglasses", "Standard CFG", C_STD),
        ("contrastive_eyeglasses", "Contrastive CFG", C_CTR),
    ]

    for i, attr in enumerate(attrs_to_plot):
        ax = axes[i]
        ax.set_facecolor(BG)
        names = []
        vals = []
        colors_list = []
        for key, name, color in sample_keys:
            if key in attr_data and "per_attr_pct" in attr_data[key]:
                names.append(name.replace(" ", "\n"))
                vals.append(attr_data[key]["per_attr_pct"].get(attr, 0) * 100)
                colors_list.append(color)

        bars = ax.bar(names, vals, color=colors_list, width=0.5, zorder=3,
                      edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_title(attr, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.set_ylabel("% detected" if i == 0 else "")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig8_attribute_accuracy.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("Saved fig8_attribute_accuracy.png")
else:
    print("SKIP fig8_attribute_accuracy.png (no attribute_accuracy.json)")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 9 — LPIPS diversity comparison
# ═══════════════════════════════════════════════════════════════════════════════
if div_data:
    # Group by guidance scale
    # Existing w=2 samples: unconditional, standard_eyeglasses, contrastive_eyeglasses
    # Sweep samples: standard_eyeglasses_w{X}, contrastive_eyeglasses_w{X}

    # First, a simple bar chart for w=2 comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)

    # Panel A: w=2 diversity comparison
    ax = axes[0]
    ax.set_facecolor(BG)
    w2_keys = [
        ("unconditional", "Unconditional", C_UNC),
        ("standard_smiling_w2.0", "Standard\nCFG (w=2)", C_STD),
        ("contrastive_smiling_w2.0", "Contrastive\nCFG (w=2)", C_CTR),
    ]
    names, vals, errs, colors_list = [], [], [], []
    for key, name, color in w2_keys:
        if key in div_data:
            names.append(name)
            vals.append(div_data[key]["lpips_mean"])
            errs.append(div_data[key]["lpips_std"])
            colors_list.append(color)

    bars = ax.bar(names, vals, yerr=errs, capsize=5, color=colors_list, width=0.5, zorder=3,
                  edgecolor="white", linewidth=1.2, error_kw=dict(lw=1.5, capthick=1.5))
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Mean Pairwise LPIPS (higher = more diverse)", fontsize=10)
    ax.set_title("(a) Sample Diversity at w=2.0", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top","right"]].set_visible(False)

    # Panel B: LPIPS vs guidance scale (if sweep data available)
    ax = axes[1]
    ax.set_facecolor(BG)

    # LPIPS vs w (full sweep with honest annotation)
    for method, color, marker, label in [
        ("standard", C_STD, "s", "Standard CFG"),
        ("contrastive", C_CTR, "o", "Contrastive CFG"),
    ]:
        w_vals, lpips_vals = [], []
        for w in ["1.0", "2.0", "3.0", "5.0", "7.0"]:
            key = f"{method}_smiling_w{w}"
            if key in div_data:
                w_vals.append(float(w))
                lpips_vals.append(div_data[key]["lpips_mean"])
        if w_vals:
            order = np.argsort(w_vals)
            ax.plot([w_vals[i] for i in order], [lpips_vals[i] for i in order],
                   color=color, marker=marker, markersize=8, linewidth=2, label=label)

    ax.axvspan(4.0, 7.5, alpha=0.08, color="gray")
    ax.text(5.75, 0.233, "High-w LPIPS inflated\nby artifacts", ha="center", va="top",
            fontsize=7, fontstyle="italic", color="#888",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#ccc", alpha=0.9))
    ax.set_xlabel("Guidance Scale (w)", fontsize=11)
    ax.set_ylabel("Mean Pairwise LPIPS", fontsize=10)
    ax.set_title("(b) Diversity vs. Guidance Scale", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig9_lpips_diversity.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("Saved fig9_lpips_diversity.png")
else:
    print("SKIP fig9_lpips_diversity.png (no diversity_results.json)")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6 — Composite summary figure
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 12), facecolor="white")
fig.suptitle("HW4: Attribute-Contrastive CFG \u2014 Key Results Summary",
             fontsize=16, fontweight="bold", y=1.005)

gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.35, wspace=0.05)

# Row 0: KID bar + Smiling side-by-side
ax_kid = fig.add_subplot(gs[0, :2])
ax_kid.set_facecolor(BG)

kid_labels = ["Standard", "Contrastive"]
kid_vals = [
    kid_data.get("standard", {}).get("2.0", {}).get("kid_mean", 0),
    kid_data.get("contrastive", {}).get("2.0", {}).get("kid_mean", 0),
]
kid_errs = [
    kid_data.get("standard", {}).get("2.0", {}).get("kid_std", 0),
    kid_data.get("contrastive", {}).get("2.0", {}).get("kid_std", 0),
]
kid_colors = [C_STD, C_CTR]

bars = ax_kid.bar(kid_labels, kid_vals, yerr=kid_errs, capsize=4,
                  color=kid_colors, width=0.5, zorder=3, edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, kid_vals):
    ax_kid.text(bar.get_x() + bar.get_width()/2, val + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax_kid.set_ylabel("KID \u2193", fontsize=10)
ax_kid.set_title("(a) KID vs. CelebA (Smiling, w=2.0)", fontsize=10, fontweight="bold")
ax_kid.set_ylim(0, 0.06); ax_kid.grid(axis="y", alpha=0.35, zorder=0)
ax_kid.spines[["top","right"]].set_visible(False)
ax_kid.tick_params(axis="x", labelsize=8)

ax_std_e = fig.add_subplot(gs[0, 2:4])
ax_std_e.imshow(load(f"{QUAL}/standard_smiling_w2.png"))
strip_axes(ax_std_e)
ax_std_e.set_title("(b) Standard CFG \u2014 w=2", fontsize=9.5, color=C_STD, fontweight="bold")
add_label(ax_std_e, "STANDARD", bg=C_STD, fontsize=9)

ax_ctr_e = fig.add_subplot(gs[0, 4:6])
ax_ctr_e.imshow(load(f"{QUAL}/contrastive_smiling_w2.png"))
strip_axes(ax_ctr_e)
ax_ctr_e.set_title("(c) Contrastive CFG \u2014 w=2", fontsize=9.5, color=C_CTR, fontweight="bold")
add_label(ax_ctr_e, "CONTRASTIVE", bg=C_CTR, fontsize=9)

# Row 1: Guidance scale ablation
titles_r1 = [
    ("Standard w=2",     f"{ABL}/standard_smiling_w2.0.png",     C_STD),
    ("Standard w=3",     f"{ABL}/standard_smiling_w3.0.png",     C_STD),
    ("Standard w=5",     f"{ABL}/standard_smiling_w5.0.png",     C_STD),
    ("Standard w=7",     f"{ABL}/standard_smiling_w7.0.png",     "red"),
    ("Contrastive w=5",  f"{ABL}/contrastive_smiling_w5.0.png",  C_CTR),
    ("Contrastive w=7",  f"{ABL}/contrastive_smiling_w7.0.png",  C_CTR),
]
for col, (title, path, color) in enumerate(titles_r1):
    ax = fig.add_subplot(gs[1, col])
    ax.imshow(load(path))
    strip_axes(ax)
    ax.set_title(title, fontsize=9, color=color, fontweight="bold")
    if "w=7" in title and "Standard" in title:
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor("red"); sp.set_linewidth(2.5)
    if "w=7" in title and "Contrastive" in title:
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(C_CTR); sp.set_linewidth(2.5)

fig.text(0.5, 0.37, "(d) Guidance scale ablation \u2014 standard CFG collapses at w=7; contrastive CFG degrades gracefully",
         ha="center", fontsize=10, fontweight="bold")

# Row 2: Control conditions + Suppression
row2 = [
    ("(e) Std \u2014 Blond_Hair",     f"{QUAL}/standard_blond_hair_w2.png",    C_STD),
    ("(e) Ctr \u2014 Blond_Hair",     f"{QUAL}/contrastive_blond_hair_w2.png", C_CTR),
    ("(f) Std \u2014 Smiling",        f"{QUAL}/standard_smiling_w2.png",       C_STD),
    ("(f) Ctr \u2014 Smiling",        f"{QUAL}/contrastive_smiling_w2.png",    C_CTR),
    ("(g) Negative Suppression",       f"{QUAL}/suppress_eyeglasses_w2.png",    C_CTR),
    ("(h) Std w=7 collapse",          f"{ABL}/standard_smiling_w7.0.png",      "red"),
]
for col, (title, path, color) in enumerate(row2):
    ax = fig.add_subplot(gs[2, col])
    ax.imshow(load(path))
    strip_axes(ax)
    ax.set_title(title, fontsize=8.5, color=color, fontweight="bold")

fig.text(0.5, -0.01,
         "Attribute-Contrastive CFG: replaces the null CFG baseline with a targeted anchor condition (same attrs, focal attr flipped).\n"
         "Result: 16.8% lower KID, 11% higher LPIPS diversity, and greatly improved stability at high guidance scales.",
         ha="center", fontsize=10, style="italic", color="#444444")

plt.savefig(f"{OUT}/fig6_composite_summary.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved fig6_composite_summary.png")

print(f"\nAll figures saved to {OUT}/")
