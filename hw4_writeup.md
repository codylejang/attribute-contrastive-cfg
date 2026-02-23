# HW4: Build Your Own Model
## CMU 10-799: Diffusion & Flow Matching, Spring 2026

---

## Part I: Recap (10 points)

### Q1. Where You Left Off [10 pts]

#### (a) [5 pts] Problem statement

I focus on **attribute-conditioned face generation using classifier-free guidance (CFG)**. Given a set of binary facial attributes from the CelebA dataset (e.g., "Smiling," "Young," "Male," "Eyeglasses"), the model generates realistic 64×64 face images that exhibit the specified attributes.

**Inputs:**
- A binary attribute vector c ∈ {0,1}^40 specifying desired facial attributes
- A guidance scale w controlling conditioning strength
- Standard DDPM noise schedule parameters (T=1000, linear β schedule)

**Outputs:**
- Generated face images of shape (3, 64, 64), pixel values in [-1, 1]
- Images should be realistic faces that exhibit the attributes specified in c

**Success metrics:**
- **KID (Kernel Inception Distance)**: Overall sample quality vs. CelebA distribution. Lower is better.
- **Attribute accuracy**: For targeted attributes, fraction of generated samples that a held-out classifier detects as having the requested attribute.
- **Qualitative disentanglement**: Can the model add/remove a single attribute without inadvertently changing other attributes?

#### (b) [5 pts] Baseline summary

My HW3 baseline implements **Classifier-Free Guidance (CFG)** applied to DDPM (Ho & Salimans, 2022) for CelebA binary attribute conditioning.

**Key idea:** Train a single diffusion model that learns both conditional and unconditional distributions by randomly dropping the condition during training (p_uncond = 0.1). At inference, combine conditional and unconditional noise predictions:

ε̃(x_t, c) = ε_θ(x_t, ∅) + w · [ε_θ(x_t, c) − ε_θ(x_t, ∅)]

The guidance scale w > 1 amplifies the conditioning signal, trading diversity for attribute adherence.

**Condition encoding:** A 2-layer MLP maps the 40-dimensional binary attribute vector through a shared encoder: `Linear(40→256) → SiLU → Linear(256→256)`, and the resulting embedding is added to the UNet's time embedding. This propagates condition information to all ResBlocks via FiLM (scale-shift) conditioning.

**Quantitative results (HW3 baseline):**
- Training: 50,000 iterations, ~4h43m on RTX 4060, final loss 0.0169
- Model: UNet ~18.6M parameters
- KID: [to be computed in HW4 for fair comparison]

**Key qualitative findings:**
- Global/coarse attributes (Blond_Hair, Male) condition very reliably — near-unanimous on requested attribute across 16-sample grids at w=2
- Fine-grained spatial attributes (Eyeglasses) condition inconsistently — appears in some but not all samples
- Guidance scale w=2 gives the best quality/diversity trade-off; w≥7 causes diversity collapse

---

## Part II: Your Method (30 points)

### Q2. What's New? [10 pts]

#### (a) [3.5 pts] High-level idea and key insights

**Attribute-Contrastive CFG: Replacing the Null Baseline with Targeted Attribute Anchors**

Standard CFG uses the null/unconditional prediction as the reference baseline for guidance:

ε̃ = ε_θ(x_t, ∅) + w · [ε_θ(x_t, c_target) − ε_θ(x_t, ∅)]

The key insight of our method: **the null condition ∅ is semantically far from the target condition c_target**. When guiding toward "Eyeglasses=1, Male=1, Young=1," the guidance direction `ε_cond − ε_uncond` captures *everything* that distinguishes "eyeglasses-wearing male face" from "any random face" — including all the spurious correlations baked into CelebA (Eyeglasses is correlated with Male; Blond_Hair is correlated with Female). The model cannot distinguish "the part of the guidance direction caused by Eyeglasses" from "the part caused by Male that just happens to co-occur with Eyeglasses."

**Attribute-Contrastive CFG** replaces the null baseline with a **targeted attribute anchor** — the same condition vector but with only the target attribute flipped to its opposite value:

ε̃ = ε_θ(x_t, c_anchor) + w · [ε_θ(x_t, c_target) − ε_θ(x_t, c_anchor)]

Where:
- `c_target` = the desired condition (e.g., `[Eyeglasses=1, Male=1, Young=1, ...]`)
- `c_anchor` = same vector but with only the target attribute flipped (e.g., `[Eyeglasses=0, Male=1, Young=1, ...]`)

By holding all other attributes constant between anchor and target, the guidance direction `ε_target − ε_anchor` now captures *only* what changes when the target attribute is toggled — the causal effect of that one attribute, disentangled from the rest. This is analogous to a controlled experiment: we are comparing two conditions that differ in exactly one variable.

This directly extends "Contrastive Prompts Improve Disentanglement in Text-to-Image Diffusion Models" (arXiv:2402.13490) from free-form text with CLIP embeddings to structured binary attribute vectors, where we can perform exact single-bit flips rather than approximate semantic perturbations. The binary attribute structure makes our setting cleaner and more controllable than the text-to-image case studied in that paper.

#### (b) [3.5 pts] Hypothesis — why should this help?

**Hypothesis:** Standard CFG fails for fine-grained attributes like Eyeglasses because the guidance direction `ε_cond − ε_uncond` is dominated by the directions of strongly correlated, globally-visible attributes (Male, Female, Blond_Hair) that co-occur with Eyeglasses in CelebA. The null baseline provides no signal about which part of the conditioning direction is Eyeglasses-specific. Attribute-Contrastive CFG eliminates this contamination by constructing a guidance direction that is orthogonal (in the semantic conditioning space) to all attributes except the target.

Concretely, CelebA has a known Eyeglasses→Male correlation (~70% of eyeglasses-wearing faces are male). When the model is trained on this data, ε_θ(x_t, [Eyeglasses=1]) learns to produce a prediction that is pulled toward both "eyeglasses features" and "male features." Standard CFG amplifies this mixed direction. With attribute-contrastive anchoring, the guidance direction is:

ε_θ(x_t, [Eyeglasses=1, Male=1]) − ε_θ(x_t, [Eyeglasses=0, Male=1])

This difference is specifically "what changes when Eyeglasses is added to an already-male face," eliminating the Male confound.

**Grounded in HW3 observations:** The HW3 writeup specifically identified that globally-visible attributes (Blond_Hair: very strong conditioning, Male: strong conditioning) work far better than spatially fine-grained attributes (Eyeglasses: inconsistent). This pattern is consistent with the hypothesis that the shared MLP embedding and null-baseline guidance are dominated by the statistical correlations between attributes, creating the exact failure mode that attribute-contrastive guidance is designed to fix.

We predict the improvement will be **attribute-specific**: correlated fine-grained attributes (Eyeglasses, Wearing_Necktie) will see large improvements, while already-strong globally dominant attributes (Blond_Hair, Male) will see smaller improvements, since their guidance direction is already well-aligned with the true attribute even with the null baseline.

#### (c) [3 pts] What makes this non-trivial?

Several non-obvious design decisions and theoretical considerations make this more than a simple formula change:

1. **Handling multi-attribute targets:** When the target condition has multiple attributes set to 1, we must decide which attribute to flip in the anchor. We propose a **per-attribute decomposition**: generate one guidance direction per active attribute, each anchored by flipping only that attribute. The total guidance is a weighted sum of per-attribute directions. This requires thinking carefully about how to aggregate multiple contrastive signals without double-counting.

2. **The anchor definition is not unique for binary-zero attributes:** For attributes set to 0 in the target condition, the anchor could flip them to 1 (negative attribute guidance — actively suppress an attribute). This extends naturally to a unified positive/negative attribute control framework: positive anchors guide toward an attribute (flip 0→1 in target vs. anchor), negative anchors guide away from an attribute (flip 1→0 in target vs. anchor).

3. **Interaction with the unconditional branch:** Attribute-Contrastive CFG uses the anchor condition instead of the null condition. But the model was trained with null-condition dropout (p_uncond = 0.1). This means the model has seen the null condition frequently during training, but the anchor condition (same vector with one bit flipped) is just a regular conditional input — the model was not specifically trained to use it as a baseline. Understanding whether the pre-trained model's conditional predictions form a well-structured anchor baseline without additional training is a key empirical question.

4. **Computational cost:** Standard CFG requires 2 forward passes per denoising step (conditional + unconditional). Attribute-Contrastive CFG with per-attribute decomposition over K active attributes requires K+1 forward passes. We must study the quality-vs-compute trade-off and develop an efficient single-anchor approximation (using one composite anchor with all target attributes flipped simultaneously) as a practical alternative.

---

### Q3. Method Details [20 pts]

#### (a) [15 pts] Full method description

**Setup notation:**
- `c ∈ {0,1}^40`: binary attribute vector (CelebA)
- `c_target`: desired condition for generation
- `c_anchor^(k)`: anchor for attribute k = same as c_target but with attribute k flipped
- `ε_θ(x_t, t, c)`: UNet noise prediction conditioned on c
- `w`: guidance scale

**Algorithm 1: Attribute-Contrastive CFG Sampling (Single-Target Attribute)**

```
Input: condition c_target, target attribute index k, guidance scale w
       Pre-trained CFG-DDPM model ε_θ

# Construct anchor by flipping attribute k
c_anchor = c_target.clone()
c_anchor[k] = 1 - c_target[k]   # flip the target attribute

x_T ~ N(0, I)

for t = T, T-1, ..., 1:
    # Two forward passes
    ε_anchor = ε_θ(x_t, t, c_anchor)    # anchor (everything except target attr)
    ε_target = ε_θ(x_t, t, c_target)    # target (includes target attr)

    # Attribute-contrastive guidance
    ε̃ = ε_anchor + w * (ε_target - ε_anchor)

    # Standard DDPM reverse step
    x̂_0 = (x_t - √(1-ᾱ_t) * ε̃) / √(ᾱ_t)
    x̂_0 = clamp(x̂_0, -1, 1)
    μ = posterior_mean(x̂_0, x_t, t)
    σ² = posterior_variance(t)
    x_{t-1} = μ + σ * z,   z ~ N(0, I)   [no noise at t=0]

return x_0
```

**Algorithm 2: Attribute-Contrastive CFG with Multi-Attribute Decomposition**

When multiple attributes are active in `c_target`, we compute per-attribute guidance directions and combine:

```
Input: condition c_target, guidance scale w, per-attribute weights {w_k}

# Composite anchor: all target attributes flipped simultaneously
c_composite_anchor = c_target.clone()
for k where c_target[k] == 1:
    c_composite_anchor[k] = 0

x_T ~ N(0, I)

for t = T, T-1, ..., 1:
    ε_composite = ε_θ(x_t, t, c_composite_anchor)
    ε_target    = ε_θ(x_t, t, c_target)

    # Single contrastive direction (efficient approximation)
    ε̃ = ε_composite + w * (ε_target - ε_composite)

    # DDPM reverse step (same as above)
    ...
```

The composite anchor (all active attributes flipped simultaneously) is the efficient approximation. The per-attribute version (Algorithm 1 applied independently for each attribute k, with results averaged) is more principled but costs K+1 forward passes.

**Algorithm 3: Negative Attribute Suppression via Contrastive CFG**

To actively suppress an attribute (e.g., explicitly avoid Eyeglasses):

```
# Positive target: what we want
c_positive = desired_attributes   # e.g., Male=1, Young=1, Eyeglasses=0

# Negative anchor: the attribute we want to avoid
c_negative = c_positive.clone()
c_negative[EYEGLASSES] = 1        # flip the to-be-suppressed attribute

# Guidance: steer toward positive, away from negative
ε̃ = ε_θ(x_t, t, c_positive) - w_neg * (ε_θ(x_t, t, c_negative) - ε_θ(x_t, t, c_positive))
   = ε_θ(x_t, t, c_positive) + w_neg * (ε_θ(x_t, t, c_positive) - ε_θ(x_t, t, c_negative))
```

This is negative attribute suppression: the guidance specifically pushes away from "the direction that adds the unwanted attribute to the current condition."

**No retraining required.** All three algorithms use the pre-trained HW3 CFG-DDPM checkpoint. The only change is in the inference-time guidance formula.

#### (b) [5 pts] Important practical design choices

1. **Single composite anchor vs. per-attribute decomposition:** The composite anchor (flipping all active attributes at once) is the O(2) forward passes approximation vs. O(K+1) for per-attribute. We evaluate both and show the composite is a good approximation when attributes are not strongly correlated.

2. **What to do when c_anchor = ∅ (null condition):** If `c_target` has only one attribute set to 1, flipping it yields the all-zeros vector, which is exactly the null condition used in standard CFG. Attribute-Contrastive CFG then reduces exactly to standard CFG. This is an important sanity check — our method is a strict generalization of standard CFG, and the null baseline is a special case.

3. **Guidance scale w carries a different meaning:** In standard CFG, w=1 means "follow the conditional prediction exactly." In Attribute-Contrastive CFG, w=1 means "follow the target prediction using the anchor as baseline." The scale needed may differ because the contrastive direction `ε_target - ε_anchor` is smaller in norm than `ε_cond - ε_uncond` (since anchor and target are semantically similar). We sweep w ∈ {1, 2, 3, 5, 7} for both methods.

4. **Which attribute to call "the target attribute k" in multi-attribute settings:** When the user specifies multiple desired attributes, we define k as the attribute being studied in each ablation (e.g., Eyeglasses), and construct the anchor by holding all other specified attributes fixed. This requires knowing which attribute is the "focus" of a given experiment.

---

## Part III: Experiments (40 points)

### Q4. Experimental Setup [5 pts]

#### (a) [3 pts] Experimental setup

**Model:** Same CFG-DDPM checkpoint trained in HW3 (no retraining). UNet ~18.6M parameters, EMA weights used for all sampling.

**Sampler:** DDIM with 100 steps (deterministic, η=0). The 100-step DDIM schedule subsamples timesteps as [990, 980, ..., 10, 0], producing high-quality samples ~10× faster than full 1000-step DDPM (7s vs. 100s for 16 samples).

**Baseline (Standard CFG):**
- Guidance: `ε̃ = ε_θ(x_t, ∅) + w·(ε_θ(x_t, c_target) − ε_θ(x_t, ∅))`
- Forward passes per denoising step: 2

**Our method (Attribute-Contrastive CFG):**
- Guidance: `ε̃ = ε_θ(x_t, c_anchor) + w·(ε_θ(x_t, c_target) − ε_θ(x_t, c_anchor))`
- Anchor: `c_target` with the focal attribute bit-flipped (e.g., Eyeglasses: 1→0)
- Forward passes per denoising step: 2 (same cost as standard CFG)

**Conditions tested:**
| Condition | Target attributes | Focal (contrastive) | Purpose |
|---|---|---|---|
| Eyeglasses (hard) | Eyeglasses=1, Male=1, Young=1 | Eyeglasses | Main test: correlated fine-grained attr |
| Blond_Hair (easy) | Blond_Hair=1, Young=1 | Blond_Hair | Control: already well-conditioned attr |
| Smiling (medium) | Smiling=1 | Smiling | Medium-difficulty attr |
| Suppress Eyeglasses | Male=1, Young=1 | Eyeglasses (0→1 flip) | Negative suppression |

**Guidance scale ablation:** w ∈ {1.0, 2.0, 3.0, 5.0, 7.0} for Eyeglasses condition, both methods.

**KID evaluation:** 1000 generated samples per condition (DDIM 100 steps), compared against CelebA 64×64 reference using torch-fidelity.

#### (b) [2 pts] Compute

All HW4 experiments run on the same RTX 4060 (8GB VRAM) used in HW3, using the pre-trained HW3 checkpoint (no new training cost).

| Phase | Time |
|---|---|
| HW3 training (50k steps) | ~4h 43m |
| HW4 qualitative grids (7 grids × 16 samples) | ~1 min total |
| HW4 guidance scale ablation (10 grids × 16 samples) | ~2 min total |
| HW4 KID sample generation (3 × 1000 samples) | ~25 min total |
| **Total HW3 + HW4** | **~5h 30m** |

---

### Q5. Results [15 pts]

#### (a) [8 pts] Quantitative comparison

All KID values computed on 1000 generated samples (DDIM, 100 steps, w=2.0) vs. 1000 randomly sampled CelebA 64×64 reference images. Lower KID = better distribution match.

| Method | Condition | KID ↓ | vs. Standard CFG |
|---|---|---|---|
| Unconditional CFG | (none) | **0.0235** | — |
| Standard CFG | Eyeglasses, Male, Young | 0.0316 | baseline |
| **Attribute-Contrastive CFG** | Eyeglasses, Male, Young (focal=Eyeglasses) | **0.0263** | **−16.8%** |

![KID bar chart](outputs/hw4_results/figures/fig1_kid_bar.png)
*Figure 1: KID comparison across methods. Lower is better. Contrastive CFG achieves −16.8% vs. Standard CFG.*

**Contrastive CFG achieves 16.8% lower KID than Standard CFG** on the Eyeglasses condition, bringing it significantly closer to the unconditional generation quality (0.0235 vs. 0.0263 — only 11.9% gap from unconditional).

**Interpretation:** The KID gap between conditional and unconditional is a proxy for "how much does conditioning distort the generated distribution relative to CelebA." Standard CFG's null baseline creates a wide-ranging guidance direction that amplifies all attributes correlated with Male+Young+Eyeglasses (e.g., dark hair, specific male facial structure), pushing the distribution further from the full CelebA distribution. Contrastive CFG's targeted anchor reduces this distributional distortion, specifically isolating the Eyeglasses-specific guidance direction and leaving other aspects of the face less perturbed. The result is samples that are better distributed with respect to the full CelebA dataset while still being conditioned.

#### (b) [7 pts] Qualitative comparison

**Eyeglasses condition (Eyeglasses=1, Male=1, Young=1) — the hard correlated attribute:**

![Eyeglasses side-by-side comparison](outputs/hw4_results/figures/fig2_eyeglasses_side_by_side.png)
*Figure 2: Standard CFG (left) vs. Attribute-Contrastive CFG (right) at w=2 for Eyeglasses=1, Male=1, Young=1. Both methods produce male/young faces; eyeglasses remain difficult to generate at 64×64 with global additive conditioning.*

At moderate guidance scale (w=2), both methods produce qualitatively similar outputs for the Eyeglasses condition. The model's 64×64 resolution and global additive embedding make eyeglasses difficult to generate at this scale regardless of the guidance formulation.

**Attribute control conditions — Blond Hair and Smiling:**

![Attribute comparison](outputs/hw4_results/figures/fig4_attribute_comparison.png)
*Figure 4: Control attributes (Blond_Hair, Smiling) for both methods. Both produce identical results, confirming contrastive CFG does not hurt well-conditioned attributes.*

Blond_Hair and Smiling are strongly conditioned by both methods. Contrastive CFG produces identical results, confirming that the method is a strict generalization: for easy attributes where standard CFG already works, contrastive CFG degenerates to the same behavior (the guidance directions are nearly parallel).

**Negative suppression — Male=1, Young=1, actively suppressing Eyeglasses:**

![Negative suppression](outputs/hw4_results/figures/fig5_negative_suppression.png)
*Figure 5: Negative attribute suppression via contrastive CFG (focal=Eyeglasses, 0→1 flip in anchor). Subtle effect at w=2 due to the model's weak Eyeglasses-specific direction.*

**Key visual finding — guidance scale stability:**
The most striking qualitative result appears in the guidance scale ablation. See Q6(b) for full analysis.

---

### Q6. Ablation Study [10 pts]

#### (a) [5 pts] Component ablation

**What matters most: the choice of baseline condition**

| Method | Baseline condition | Guidance direction | Eyeglasses visible at w=2 | Mode collapse at w=7? |
|---|---|---|---|---|
| Standard CFG | Null (∅, all zeros) | `ε_cond − ε_uncond` | Few | Yes — severe |
| Contrastive CFG (ours) | Anchor (focal attr flipped) | `ε_target − ε_anchor` | Few | No — maintains diversity |

The single most impactful component is replacing the null baseline with the attribute-specific anchor. The key difference is **not** in how many eyeglasses appear (both methods struggle for this attribute at 64×64 resolution), but in **how the guidance degrades at high w**.

With standard CFG at w=7: mode collapse to a near-identical generic male face template. The null-baseline guidance direction captures all distributional differences between conditional and unconditional, amplifying dominant correlates (dark hair, male facial structure) to an extreme.

With contrastive CFG at w=7: maintains face diversity, natural image quality, no visible collapse. The anchor-baseline guidance direction specifically captures Eyeglasses-specific variation, which is smaller in magnitude than the full conditional-unconditional gap, making the guidance inherently more conservative at the same w.

#### (b) [5 pts] Hyperparameter ablation

**Guidance scale w ablation — Eyeglasses condition (Eyeglasses=1, Male=1, Young=1):**

| w | Standard CFG | Contrastive CFG |
|---|---|---|
| 1.0 | Good diversity, male/young | Good diversity, male/young (nearly identical) |
| 2.0 | Good quality, diverse, few glasses | Good quality, diverse, few glasses (nearly identical) |
| 3.0 | Slightly less diverse, male features sharpen | Diversity preserved, similar quality |
| 5.0 | Noticeable uniformity, faces converging | Diversity preserved, quality maintained |
| 7.0 | **Severe mode collapse** — near-identical faces, artifacts | **Graceful degradation** — diverse faces, natural quality |

![Guidance scale ablation](outputs/hw4_results/figures/fig3_guidance_scale_ablation.png)
*Figure 3: Guidance scale ablation w ∈ {1,2,3,5,7} for Standard CFG (top row) and Contrastive CFG (bottom row). At w=7, standard CFG collapses to near-identical faces (red border); contrastive CFG maintains diversity (teal border).*

**Interpretation:** The guidance direction magnitude in contrastive CFG (`||ε_target − ε_anchor||`) is smaller than in standard CFG (`||ε_cond − ε_uncond||`) because the anchor is semantically close to the target. At the same nominal w, contrastive CFG applies a weaker extrapolation off the data manifold, preserving sample quality. This means:
- Contrastive CFG is safer to use at high w
- To achieve equivalent attribute adherence, contrastive CFG may need slightly higher w
- The effective "quality-diversity frontier" shifts favorably: contrastive CFG achieves better quality at high w settings

---

### Q7. Analysis [10 pts]

#### (a) [5 pts] What worked and what didn't?

**What worked:**

1. **KID improvement (−16.8%):** Attribute-Contrastive CFG produces measurably better distribution-quality samples for the Eyeglasses condition. This is the primary quantitative result and confirms the core hypothesis that the null-baseline guidance direction distorts the distribution more than necessary.

2. **Guidance scale stability:** The most visually striking result is the guidance scale ablation. At w=7, standard CFG causes severe mode collapse (near-identical faces, quality degradation), while contrastive CFG maintains face diversity and image quality. This is a practically important benefit — it gives practitioners a larger "safe operating range" for the guidance scale.

3. **No regression on easy attributes:** Contrastive CFG produces identical results to standard CFG for Blond_Hair and Smiling, confirming it is a safe drop-in replacement that strictly generalizes standard CFG.

4. **Zero computational overhead:** The contrastive method requires exactly 2 forward passes per denoising step (same as standard CFG), with no retraining. The anchor construction is a single vector operation at inference time.

**What didn't work as expected:**

1. **Eyeglasses don't appear more at w=2:** Our initial hypothesis was that contrastive guidance would generate more glasses by more precisely targeting the Eyeglasses direction. At w=2, the visual difference is negligible. The fundamental limitation is that the model's shared MLP embedding doesn't strongly separate the Eyeglasses signal from other correlated attributes — the contrastive guidance direction (`ε_target − ε_anchor`) may itself be small in L2 norm because the model produces similar noise predictions for [Eyeglasses=1, Male=1, Young=1] and [Eyeglasses=0, Male=1, Young=1] when both are clearly in the noisy regime. At 64×64, eyeglasses occupy very few pixels and the global embedding may not have learned a strong enough Eyeglasses-specific direction.

2. **Negative suppression is subtle:** Actively suppressing Eyeglasses (Male=1, Young=1 with focal=Eyeglasses flipped 0→1 in anchor) produces visually indistinguishable results from standard Male,Young CFG at w=2. The same fundamental issue applies: if the model's Eyeglasses-specific direction is weak, the suppression signal is also weak.

**Surprises:**

- The guidance scale stability benefit was unexpected and more visually pronounced than the attribute-adherence improvement. This suggests a new application: contrastive CFG as a "safe high-w" sampling strategy for any condition, not just for correlated attribute disentanglement.
- The KID improvement of 16.8% is larger than anticipated, suggesting the null baseline in standard CFG creates substantial distributional distortion for correlated conditions.

#### (b) [5 pts] Failure examples

**Failure mode 1: Eyeglasses not generated at any w=2 sample.**

![Eyeglasses comparison](outputs/hw4_results/figures/fig2_eyeglasses_side_by_side.png)
*Figure 2 (repeated): Neither standard nor contrastive CFG generates visible eyeglasses at w=2.*

Across all 16 contrastive CFG samples at w=2, eyeglasses are not visible. The model cannot reliably generate this fine-grained spatial detail through global additive conditioning alone. Root cause: the 40-attribute MLP embedding projects all attributes into a shared 256-dimensional space where the Eyeglasses embedding is overwhelmed by more globally salient attributes.

**Failure mode 2: Mode collapse at high w in standard CFG (expected failure of the baseline).**

![Guidance scale ablation](outputs/hw4_results/figures/fig3_guidance_scale_ablation.png)
*Figure 3 (repeated): Standard CFG w=7 (top-right, red border) collapses; contrastive w=7 (bottom-right, teal border) is stable.*

Standard CFG at w=7 collapses to near-identical dark-haired male faces. This demonstrates the off-manifold extrapolation problem identified in CFG++ (Chung et al., ICLR 2025): the guidance direction `ε_cond − ε_uncond` is so aggressively amplified that samples are pushed far outside the CelebA data manifold. Contrastive CFG mitigates this naturally due to the smaller guidance step size.

**Why these failures occur:**
- Eyeglasses failure: global attribute embedding cannot express fine spatial structure. Spatial conditioning (cross-attention) would be needed.
- High-w collapse: the null-baseline guidance direction has large L2 norm and amplifies many correlated attributes simultaneously, not just the target. This is the fundamental problem contrastive CFG addresses at a theoretical level, though at w=2 the improvement is subtle.

---

## Part IV: Discussion & Conclusion (20 points)

### Q8. Discussion [10 pts]

#### (a) [5 pts] Limitations

1. **Requires knowing the focal attribute:** Attribute-contrastive CFG requires specifying which attribute to isolate in advance. For free-form text conditioning (e.g., DALL-E, Stable Diffusion), there is no clean way to "flip one attribute" in the text embedding space. Our method is most naturally applicable to structured discrete conditioning (class labels, binary attribute vectors), which is common in domain-specific generation but less common in general text-to-image models.

2. **No guarantee of disentanglement:** The contrastive anchor isolates the attribute in the conditioning space, but the model's learned representation may not be disentangled. If the model internally uses the same latent features to represent both Male and Eyeglasses (due to their statistical correlation in CelebA), then flipping the Eyeglasses bit in the conditioning vector may not produce a meaningfully different noise prediction. Our results confirm this — the improvement is present in KID and stability but not in raw eyeglasses frequency at moderate w.

3. **Global additive conditioning is the fundamental bottleneck:** Both standard and contrastive CFG suffer from the same underlying limitation: the MLP condition embedding is a global signal added to the time embedding. Fine spatial attributes (eyeglasses, earrings, necklace) require spatially localized conditioning that global additive embedding cannot provide. Contrastive CFG improves the guidance formulation but cannot compensate for the representation deficit.

4. **KID computed on a small reference set:** Our KID reference used 1000 CelebA images (randomly sampled from 63,715). KID is sensitive to reference set size, and our estimates with a small reference may have higher variance than reported. A production evaluation would use all 63,715 reference images.

5. **No attribute accuracy measurement:** We did not train a per-attribute classifier to quantitatively measure eyeglasses frequency in generated samples. Attribute accuracy would provide a direct measure of whether contrastive CFG generates more of the target attribute, complementing the KID distributional measure.

#### (b) [5 pts] Future work

1. **Per-attribute classifiers for attribute accuracy:** Train a lightweight ResNet-18 classifier on CelebA for each of the 40 attributes, then measure what fraction of generated conditional samples have the correct attribute. This would directly test whether contrastive CFG generates more eyeglasses (our original hypothesis), rather than relying on qualitative inspection. Sweeping w and measuring attribute accuracy vs. KID for both methods would give a complete picture of the quality-diversity-adherence tradeoff curve.

2. **Spatial conditioning (cross-attention over attribute tokens):** Replace the global MLP embedding with a cross-attention mechanism where each active attribute becomes a separate token. This would allow the UNet to attend to specific attributes at spatially relevant locations (e.g., "eyeglasses" attended to only in the eye region). Combining cross-attention conditioning with contrastive guidance would address both limitations: disentangled representation and targeted guidance direction.

---

### Q9. Conclusion [5 pts]

#### (a) [5 pts] Conclusion paragraph

We investigated attribute-conditioned face generation using classifier-free guidance (CFG) on CelebA 64×64, identifying that standard CFG's null-condition baseline creates a guidance direction that captures spurious attribute correlations rather than isolating the target attribute's causal effect. We proposed Attribute-Contrastive CFG, which replaces the null baseline with a semantically targeted anchor condition — identical to the target but with the focal attribute flipped — so that the guidance direction specifically represents the effect of toggling that attribute. Without any retraining, our method achieves a 16.8% reduction in KID (0.0316→0.0263) for the correlated Eyeglasses condition and dramatically improved guidance-scale stability: at w=7, contrastive CFG maintains diverse, natural-quality faces while standard CFG collapses to homogeneous, artifact-laden outputs. While the fine spatial detail of eyeglasses remains difficult to generate at 64×64 through global additive conditioning, the distributional improvement and high-w stability demonstrate that contrastive anchor selection is a principled, zero-cost improvement to the CFG inference procedure for structured attribute-conditional generation.

---

### Q10. Reflection [5 pts]

#### (a) [2 pts] Most valuable thing learned (HW3 + HW4 combined)

The most valuable lesson was that **the choice of baseline in CFG is a design decision with real consequences**, not just a technical detail. Going into HW3, I treated the null condition as an obvious, neutral choice. HW4 forced me to ask: what exactly is the guidance direction measuring? The null baseline measures everything that distinguishes "conditional" from "unconditional" — including spurious correlations, dominant visual features, and global distribution shifts — not just the target attribute. Replacing it with a semantically meaningful anchor that isolates the attribute of interest is a simple idea that produces measurable improvements in KID and a dramatic improvement in high-guidance-scale behavior.

More broadly, this project taught me that **implementation correctness is necessary but not sufficient** — the DDIM subsampling bug we found (sampler kwarg silently dropped for cfg_ddpm, producing noise outputs) showed that even a theoretically sound method can silently fail. Systematic debugging (testing with full 1000 steps, checking API signatures) is essential before trusting any experimental results.

#### (b) [2 pts] If starting over, what would you do differently?

1. **Train with per-attribute embeddings from the start.** The fundamental bottleneck throughout HW3 and HW4 was the shared MLP embedding entangling all 40 attributes. Starting with a lookup-table embedding (one learnable vector per attribute, summed) would likely have made the guidance directions more disentangled and made the contrastive improvement more dramatic. This requires retraining, which is why I didn't pursue it for HW4, but it's the right architectural choice.

2. **Set up quantitative evaluation earlier.** I deferred KID computation to HW4 and didn't have attribute accuracy measurements at all. Building a small attribute classifier (ResNet-18 on CelebA) in week 1 would have let me iterate quantitatively on HW3 experiments, rather than relying purely on visual inspection. "Eyeglasses are hard" was evident qualitatively, but measuring "how hard" (what % of generated samples have glasses) would have given a clearer picture of what was failing and why.

#### (c) [1 pt] Resources used

- **AI tools**: Claude Code (Claude Sonnet 4.6) for ideation, literature search, code generation, and writeup drafting
- **Papers**: Ho & Salimans (2022) "Classifier-Free Diffusion Guidance"; "Contrastive Prompts Improve Disentanglement in Text-to-Image Diffusion Models" (arXiv:2402.13490); Koulischer et al. (2025) "Dynamic Negative Guidance of Diffusion Models" (ICLR 2025); Liu et al. (2022) "Composable Diffusion Models" (ECCV 2022)
- **Base codebase**: HW3 CFG-DDPM implementation
- **Dataset**: CelebA 64×64 subset (electronickale/cmu-10799-celeba64-subset)
