"""
Generate poster-specific figures that don't exist yet:
  - fig_poster_correlation.png   — CelebA attribute correlation heatmap (subset)
  - fig_poster_method.png        — Method diagram: standard vs contrastive CFG
  - fig_poster_results_table.png — Combined results table as a rendered figure
  - fig_poster_kid_lpips.png     — KID and LPIPS vs w on one figure (2 panels)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

OUT = "outputs/hw4_results/figures"
os.makedirs(OUT, exist_ok=True)

# Colors
C_STD = "#4878CF"
C_CTR = "#D65F5F"
C_UNC = "#6ACC65"
BG = "#F8F8F8"

# Load JSON data
def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

kid_sweep = load_json("outputs/hw4_results/kid_sweep_smiling.json")
div_data = load_json("outputs/hw4_results/diversity_smiling.json")
attr_data = load_json("outputs/hw4_results/attribute_accuracy_smiling.json")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig: CelebA attribute correlation heatmap
# ═══════════════════════════════════════════════════════════════════════════════
print("Computing CelebA attribute correlations...")
try:
    from src.data.celeba import CELEBA_ATTRIBUTES
    from datasets import load_from_disk

    ds_dict = load_from_disk("data/celeba-subset")
    ds = ds_dict["train"]
    # Extract attribute columns directly from the Arrow dataset
    attr_arrays = []
    for attr in CELEBA_ATTRIBUTES:
        if attr in ds.column_names:
            col = ds[attr]
            attr_arrays.append(np.array(col, dtype=np.float32))
    attrs_tensor = np.stack(attr_arrays, axis=1)  # (N, 40)

    # Select interesting attributes for poster
    interesting = ["Eyeglasses", "Male", "Young", "Blond_Hair", "Smiling",
                   "Heavy_Makeup", "Wearing_Lipstick", "Attractive", "Big_Nose", "Bald"]
    indices = [CELEBA_ATTRIBUTES.index(a) for a in interesting]
    subset = attrs_tensor[:, indices]

    corr = np.corrcoef(subset.T)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Labels
    short_labels = ["Eyeglasses", "Male", "Young", "Blond\nHair", "Smiling",
                    "Heavy\nMakeup", "Wearing\nLipstick", "Attractive", "Big\nNose", "Bald"]
    ax.set_xticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(len(short_labels)))
    ax.set_yticklabels(short_labels, fontsize=9)

    # Add correlation values in cells
    for i in range(len(interesting)):
        for j in range(len(interesting)):
            val = corr[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            if i != j:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

    # Note: Eyeglasses has 0 positive examples so its correlations are NaN

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson Correlation", fontsize=10)

    ax.set_title("CelebA Attribute Correlations\n(NaN = zero training examples, e.g. Eyeglasses)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_poster_correlation.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved fig_poster_correlation.png")

except Exception as e:
    print(f"SKIP correlation heatmap: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig: Method diagram — standard vs contrastive CFG
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white",
                         gridspec_kw={"wspace": 0.12})

for idx, (ax, method_title, accent) in enumerate(zip(
    axes,
    ["Standard CFG", "Attribute-Contrastive CFG (Ours)"],
    [C_STD, C_CTR],
)):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis("off")

    # --- Title ---
    ax.text(5, 12.8, method_title, ha="center", va="top",
            fontsize=15, fontweight="bold", color=accent)

    # --- Noisy image x_t (top center) ---
    ax.add_patch(mpatches.FancyBboxPatch((3.2, 10.0), 3.6, 1.2,
                 boxstyle="round,pad=0.2", facecolor="#e8e8e8", edgecolor="#666", linewidth=1.5))
    ax.text(5, 10.6, r"Noisy image $x_t$", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#444")

    # --- Two U-Net branches ---
    # Left branch: baseline
    if idx == 0:
        base_label = "Null condition"
        base_cond = r"$c = \mathbf{0}$"
        base_color = "#aaaaaa"
    else:
        base_label = "Anchor condition"
        base_cond = "[Smiling=0, Male=1, Young=1]"
        base_color = "#7FB3D8"

    # Right branch: target
    tgt_label = "Target condition"
    tgt_cond = "[Smiling=1, Male=1, Young=1]"
    tgt_color = accent

    # Arrows from x_t down to U-Nets
    ax.annotate("", xy=(2.2, 8.8), xytext=(4.0, 10.0),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.5))
    ax.annotate("", xy=(7.8, 8.8), xytext=(6.0, 10.0),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.5))

    # Left U-Net box
    ax.add_patch(mpatches.FancyBboxPatch((0.3, 7.2), 3.8, 1.6,
                 boxstyle="round,pad=0.2", facecolor=base_color, alpha=0.2,
                 edgecolor=base_color, linewidth=1.5))
    ax.text(2.2, 8.35, r"U-Net $\epsilon_\theta$", ha="center", va="center",
            fontsize=10, fontweight="bold")
    ax.text(2.2, 7.65, base_label, ha="center", va="center",
            fontsize=8, color="#555")

    # Right U-Net box
    ax.add_patch(mpatches.FancyBboxPatch((5.9, 7.2), 3.8, 1.6,
                 boxstyle="round,pad=0.2", facecolor=tgt_color, alpha=0.2,
                 edgecolor=tgt_color, linewidth=1.5))
    ax.text(7.8, 8.35, r"U-Net $\epsilon_\theta$", ha="center", va="center",
            fontsize=10, fontweight="bold")
    ax.text(7.8, 7.65, tgt_label, ha="center", va="center",
            fontsize=8, color="#555")

    # Condition vectors below U-Nets
    ax.text(2.2, 6.7, base_cond, ha="center", va="center",
            fontsize=7, family="monospace", color="#555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc"))
    ax.text(7.8, 6.7, tgt_cond, ha="center", va="center",
            fontsize=7, family="monospace", color="#555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc"))

    # Output noise predictions
    ax.annotate("", xy=(2.2, 5.6), xytext=(2.2, 7.1),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.5))
    ax.annotate("", xy=(7.8, 5.6), xytext=(7.8, 7.1),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.5))

    if idx == 0:
        eps_base = r"$\epsilon_{null}$"
    else:
        eps_base = r"$\epsilon_{anchor}$"

    ax.text(2.2, 5.3, eps_base, ha="center", va="center",
            fontsize=12, fontweight="bold", color=base_color if idx == 1 else "#888")
    ax.text(7.8, 5.3, r"$\epsilon_{cond}$", ha="center", va="center",
            fontsize=12, fontweight="bold", color=tgt_color)

    # --- Subtraction arrow → guidance direction ---
    ax.annotate("", xy=(5.0, 4.2), xytext=(2.2, 4.8),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.2))
    ax.annotate("", xy=(5.0, 4.2), xytext=(7.8, 4.8),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.2))

    # Guidance direction box
    dir_color = "red" if idx == 0 else "#228B22"
    dir_bg = "#fff0f0" if idx == 0 else "#f0fff0"
    ax.add_patch(mpatches.FancyBboxPatch((1.5, 2.8), 7.0, 1.3,
                 boxstyle="round,pad=0.3", facecolor=dir_bg, edgecolor=dir_color, linewidth=2))

    if idx == 0:
        dir_text = r"$\epsilon_{cond} - \epsilon_{null}$"
        dir_desc = "Direction captures ALL attributes\n+ spurious correlations"
    else:
        dir_text = r"$\epsilon_{cond} - \epsilon_{anchor}$"
        dir_desc = "Direction captures ONLY\nthe focal attribute (Smiling)"

    ax.text(5, 3.85, dir_text, ha="center", va="center",
            fontsize=12, fontweight="bold", color=dir_color)
    ax.text(5, 3.15, dir_desc, ha="center", va="center",
            fontsize=8, color=dir_color)

    # --- Final formula ---
    if idx == 0:
        final = r"$\tilde{\epsilon} = \epsilon_{null} + w \cdot (\epsilon_{cond} - \epsilon_{null})$"
    else:
        final = r"$\tilde{\epsilon} = \epsilon_{anchor} + w \cdot (\epsilon_{cond} - \epsilon_{anchor})$"

    ax.annotate("", xy=(5, 1.6), xytext=(5, 2.7),
                arrowprops=dict(arrowstyle="-|>", color=dir_color, lw=2))
    ax.text(5, 1.0, final, ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=accent, alpha=0.15,
                      edgecolor=accent, linewidth=1.5))

    # Key insight callout for contrastive
    if idx == 1:
        ax.text(5, 0.2, "Same U-Net, same # of forward passes — only the baseline changes",
                ha="center", va="center", fontsize=8, fontstyle="italic", color="#666")

plt.savefig(f"{OUT}/fig_poster_method.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved fig_poster_method.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig: Combined KID + LPIPS vs w (poster-optimized, 2 panels side by side)
# ═══════════════════════════════════════════════════════════════════════════════
if kid_sweep and div_data:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG)

    # Panel 1: KID vs w
    ax1.set_facecolor(BG)
    for method, color, marker, label in [
        ("standard", C_STD, "s", "Standard CFG"),
        ("contrastive", C_CTR, "o", "Contrastive CFG"),
    ]:
        if method in kid_sweep:
            ws_sorted = sorted(kid_sweep[method].keys(), key=float)
            x = [float(w) for w in ws_sorted]
            y = [kid_sweep[method][w]["kid_mean"] for w in ws_sorted]
            yerr = [kid_sweep[method][w]["kid_std"] for w in ws_sorted]
            ax1.errorbar(x, y, yerr=yerr, color=color, marker=marker, markersize=10,
                        linewidth=2.5, capsize=6, capthick=2, label=label)

    # Annotate the gap at w=7
    if "standard" in kid_sweep and "7.0" in kid_sweep["standard"]:
        std_7 = kid_sweep["standard"]["7.0"]["kid_mean"]
        ctr_7 = kid_sweep["contrastive"]["7.0"]["kid_mean"]
        ax1.annotate("", xy=(7.15, ctr_7), xytext=(7.15, std_7),
                    arrowprops=dict(arrowstyle="<->", color="#333", lw=2))
        ax1.text(7.35, (std_7 + ctr_7) / 2, "45%\n>4σ", ha="left", va="center",
                fontsize=11, fontweight="bold", color="#333")

    ax1.set_xlabel("Guidance Scale (w)", fontsize=13)
    ax1.set_ylabel("KID (lower = better)", fontsize=13)
    ax1.set_title("Image Quality vs. Guidance Scale", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=12, framealpha=0.9, loc="upper left")
    ax1.grid(alpha=0.4)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(labelsize=11)

    # Panel 2: LPIPS vs w (full sweep, with honest annotation)
    ax2.set_facecolor(BG)
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
            ax2.plot([w_vals[i] for i in order], [lpips_vals[i] for i in order],
                    color=color, marker=marker, markersize=10, linewidth=2.5, label=label)

    # Honest annotation: U-shape due to artifacts
    ax2.axvspan(4.0, 7.5, alpha=0.08, color="gray")
    ax2.text(5.75, 0.235, "High-w LPIPS inflated\nby off-manifold artifacts,\nnot true diversity",
             ha="center", va="top", fontsize=8, fontstyle="italic", color="#888",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.9))

    ax2.set_xlabel("Guidance Scale (w)", fontsize=13)
    ax2.set_ylabel("LPIPS (higher = more diverse)", fontsize=13)
    ax2.set_title("Sample Diversity vs. Guidance Scale", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=12, framealpha=0.9, loc="lower left")
    ax2.grid(alpha=0.4)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_poster_kid_lpips.png", dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("Saved fig_poster_kid_lpips.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig: Results table rendered as figure (poster-friendly)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 4), facecolor="white")
ax.axis("off")

columns = ["Metric", "Standard CFG", "Contrastive CFG (Ours)"]
rows = [
    ["KID (w=2) ↓",      "0.0364 ± 0.0047", "0.0330 ± 0.0042  (−9.4%)"],
    ["KID (w=7) ↓",      "0.0777 ± 0.0063", "0.0424 ± 0.0048  (−45.4%)"],
    ["KID degrade w=1→7", "+151%",           "+37%"],
    ["LPIPS Diversity ↑", "0.182",            "0.196  (+7.7%)"],
    ["Male % (w=2)",      "99.7%",            "96.7%"],
    ["Extra Compute",     "—",                "None"],
]

table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 2.0)

# Style header
for j in range(len(columns)):
    cell = table[0, j]
    cell.set_facecolor("#333333")
    cell.set_text_props(color="white", fontweight="bold", fontsize=12)

# Style "Ours" column
for i in range(1, len(rows) + 1):
    cell = table[i, 2]
    cell.set_facecolor("#fff0f0")
    cell.set_text_props(fontweight="bold")

# Style metric column
for i in range(1, len(rows) + 1):
    cell = table[i, 0]
    cell.set_text_props(fontweight="bold")

# Highlight best KID rows
for i in [1, 2, 3]:  # KID rows
    table[i, 2].set_facecolor("#ffe0e0")

ax.set_title("Quantitative Results Summary", fontsize=14, fontweight="bold", pad=20)
ax.text(0.5, -0.02, "Smiling sweep: focal attribute = Smiling",
        transform=ax.transAxes, ha="center", fontsize=9, fontstyle="italic", color="#666666")

plt.tight_layout()
plt.savefig(f"{OUT}/fig_poster_results_table.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved fig_poster_results_table.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig: Attribute leakage across guidance scales (poster-optimized)
# ═══════════════════════════════════════════════════════════════════════════════
if attr_data:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG)

    # Collect data across w for Male and Blond_Hair
    w_values = [1.0, 2.0, 3.0, 5.0, 7.0]

    for ax, attr, ylabel in [(ax1, "Smiling", "Smiling %"), (ax2, "Male", "Male %")]:
        ax.set_facecolor(BG)
        for method, color, marker, label, base_key in [
            ("standard", C_STD, "s", "Standard CFG", "standard_smiling"),
            ("contrastive", C_CTR, "o", "Contrastive CFG", "contrastive_smiling"),
        ]:
            ws, vals = [], []
            # All guidance scales
            for w in ["1.0", "2.0", "3.0", "5.0", "7.0"]:
                key = f"{base_key}_w{w}"
                if key in attr_data and "per_attr_pct" in attr_data[key]:
                    ws.append(float(w))
                    vals.append(attr_data[key]["per_attr_pct"].get(attr, 0) * 100)

            if ws:
                order = np.argsort(ws)
                ax.plot([ws[i] for i in order], [vals[i] for i in order],
                       color=color, marker=marker, markersize=10, linewidth=2.5, label=label)

        ax.set_xlabel("Guidance Scale (w)", fontsize=13)
        ax.set_ylabel(f"{ylabel} detected", fontsize=13)
        ax.set_title(f"{attr} Detection Rate vs. Guidance Scale", fontsize=14, fontweight="bold")
        ax.legend(fontsize=12, framealpha=0.9)
        ax.grid(alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_poster_attr_leakage.png", dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("Saved fig_poster_attr_leakage.png")

print(f"\nAll poster figures saved to {OUT}/")
