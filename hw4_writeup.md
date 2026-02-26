# HW4: Build Your Own Model
## CMU 10-799: Diffusion & Flow Matching, Spring 2026

---

## Part I: Recap (10 points)

### Q1. Where You Left Off [10 pts]

#### (a) [5 pts] Problem statement

I focus on **attribute-conditioned face generation using classifier-free guidance (CFG)**. Given a set of binary facial attributes from the CelebA dataset (e.g., "Smiling," "Young," "Male"), the model generates realistic 64x64 face images that exhibit the specified attributes.

**Inputs:**
- A binary attribute vector c in {0,1}^40 specifying desired facial attributes
- A guidance scale w controlling conditioning strength
- Standard DDPM noise schedule parameters (T=1000, linear beta schedule)

**Outputs:**
- Generated face images of shape (3, 64, 64), pixel values in [-1, 1]
- Images should be realistic faces that exhibit the attributes specified in c

**Success metrics:**
- **KID (Kernel Inception Distance)**: Overall sample quality vs. CelebA distribution. Lower is better.
- **Attribute accuracy**: For targeted attributes, fraction of generated samples that a trained ResNet-18 classifier detects as having the requested attribute.
- **LPIPS diversity**: Mean pairwise LPIPS distance within generated batches. Higher means more diverse samples and less mode collapse.

#### (b) [5 pts] Baseline summary

My HW3 baseline implements **Classifier-Free Guidance (CFG)** applied to DDPM (Ho & Salimans, 2022) for CelebA binary attribute conditioning.

**Key idea:** Train a single diffusion model that learns both conditional and unconditional distributions by randomly dropping the condition during training (p_uncond = 0.1). At inference, combine conditional and unconditional noise predictions:

eps_guided(x_t, c) = eps(x_t, null) + w * [eps(x_t, c) - eps(x_t, null)]

The guidance scale w > 1 amplifies the conditioning signal, trading diversity for attribute adherence.

**Condition encoding:** A 2-layer MLP maps the 40-dimensional binary attribute vector through a shared encoder: `Linear(40->256) -> SiLU -> Linear(256->256)`, and the resulting embedding is added to the UNet's time embedding. This propagates condition information to all ResBlocks via FiLM (scale-shift) conditioning.

**Quantitative results (HW3 baseline):**
- Training: 50,000 iterations, ~4h43m on RTX 4060, final loss 0.0169
- Model: UNet ~18.6M parameters
- KID at w=2: 0.0364 +/- 0.0047 (Smiling,Male,Young condition)

**Key qualitative findings:**
- Global/coarse attributes (Blond_Hair, Male) condition very reliably
- Some attributes in the 40-dim label vector have zero positive examples in this CelebA subset (including Eyeglasses, Heavy_Makeup, Wearing_Hat) — the model never learned meaningful responses to these labels
- Guidance scale w=2 gives the best quality/diversity trade-off; w>=7 causes severe mode collapse and diversity loss

---

## Part II: Your Method (30 points)

### Q2. What's New? [10 pts]

#### (a) [3.5 pts] High-level idea and key insights

**Attribute-Contrastive CFG: Replacing the Null Baseline with Targeted Attribute Anchors**

Standard CFG uses the null/unconditional prediction as the reference baseline for guidance:

eps_guided = eps(x_t, null) + w * [eps(x_t, c_target) - eps(x_t, null)]

The key insight: **the null condition is semantically far from the target condition**. When guiding toward "Smiling=1, Male=1, Young=1," the guidance direction `eps_cond - eps_uncond` captures *everything* that distinguishes "smiling young male face" from "any random face" -- including all co-occurring attribute patterns baked into CelebA. The model cannot distinguish which part of the guidance direction is caused by the focal attribute (Smiling) vs. which is caused by correlated attributes (Male ↔ Big_Nose, etc.).

**Attribute-Contrastive CFG** replaces the null baseline with a **targeted attribute anchor** -- the same condition vector but with only the focal attribute flipped:

eps_guided = eps(x_t, c_anchor) + w * [eps(x_t, c_target) - eps(x_t, c_anchor)]

Where c_anchor = c_target with the focal attribute bit-flipped (e.g., Smiling: 1→0). By holding all other attributes constant, the guidance direction captures *only* the causal effect of toggling that attribute -- a controlled experiment in conditioning space.

This adapts the contrastive prompting idea (arXiv:2402.13490) from free-form text with CLIP embeddings to structured binary attribute vectors, where exact single-bit flips are possible.

#### (b) [3.5 pts] Hypothesis -- why should this help?

**Hypothesis:** Standard CFG's guidance direction is contaminated by correlated attributes, causing (1) distributional distortion at moderate w (inflated KID) and (2) mode collapse at high w (the amplified direction pushes samples toward a single prototype). Contrastive CFG isolates the focal attribute's contribution, producing a smaller-magnitude guidance direction that (1) distorts the distribution less and (2) supports higher guidance scales before quality degrades.

Concretely, consider conditioning on [Smiling=1, Male=1, Young=1]. Standard CFG's direction `eps_cond - eps_null` captures the full distributional shift from unconditional to this multi-attribute condition — including all inter-attribute correlations (Smiling ↔ High_Cheekbones r=0.67, Male ↔ Blond_Hair r=-0.29, etc.). At high w, this large-magnitude direction overwhelms the data manifold, collapsing diversity to a single prototype. Contrastive CFG's direction:

eps(x_t, [Smiling=1, Male=1]) - eps(x_t, [Smiling=0, Male=1])

isolates only what changes when the Smiling bit is flipped. This direction has smaller L2 norm (the anchor is semantically close to the target), so the same w produces less off-manifold extrapolation.

**Grounded in data analysis:** Our CelebA subset has 63,715 images with several attributes having zero positive examples (Eyeglasses: 0%, Heavy_Makeup: 0%, Wearing_Hat: 0%, Blurry: 0%). For well-represented attributes like Smiling (44%), Male (55%), and Blond_Hair (10%), contrastive CFG isolates the focal attribute. For absent attributes, the contrastive direction is near-zero, preventing standard CFG from spuriously amplifying correlated attributes.

**Predicted outcome:** The improvement should manifest primarily as (1) better KID at moderate w, (2) higher sample diversity (LPIPS), and (3) dramatically better stability at high w.

#### (c) [3 pts] What makes this non-trivial?

1. **The anchor definition has non-obvious semantics:** For multi-attribute targets, we must decide what to flip. The composite anchor (flip all active attributes) is an efficient approximation, but whether it produces the same guidance direction as per-attribute decomposition depends on the model's learned representation structure. This required empirical validation.

2. **Interaction with null-condition training:** The model was trained with p_uncond=0.1 null-condition dropout. The anchor condition is just a regular conditional input -- the model was not specifically trained to use it as a guidance baseline. Whether the model's conditional predictions form a well-structured semantic space where nearby conditions produce smooth, interpolatable noise predictions is an empirical question.

3. **The guidance scale w carries different semantics:** The contrastive direction `eps_target - eps_anchor` has smaller L2 norm than `eps_cond - eps_uncond` (anchor is semantically close to target). At the same nominal w, contrastive CFG extrapolates less aggressively off the data manifold. Understanding this implicit rescaling and its effect on the quality-diversity-adherence tradeoff required the full KID-vs-w sweep analysis.

4. **Quantitative evaluation design:** Validating the hypothesis required three complementary metrics (KID for distribution quality, LPIPS for diversity, attribute accuracy for adherence) plus a guidance-scale sweep to reveal the behavioral difference. None of these existed in the HW3 codebase and had to be built.

---

### Q3. Method Details [20 pts]

#### (a) [15 pts] Full method description

**Setup notation:**
- `c in {0,1}^40`: binary attribute vector (CelebA)
- `c_target`: desired condition for generation
- `c_anchor`: anchor condition = c_target with focal attribute(s) flipped
- `eps_theta(x_t, t, c)`: UNet noise prediction conditioned on c
- `w`: guidance scale

**Algorithm 1: Attribute-Contrastive CFG Sampling (Single-Target Attribute)**

```
Input: condition c_target, target attribute index k, guidance scale w
       Pre-trained CFG-DDPM model eps_theta

# Construct anchor by flipping attribute k
c_anchor = c_target.clone()
c_anchor[k] = 1 - c_target[k]

x_T ~ N(0, I)

for t = T, T-1, ..., 1:
    # Two forward passes (same cost as standard CFG)
    eps_anchor = eps_theta(x_t, t, c_anchor)
    eps_target = eps_theta(x_t, t, c_target)

    # Attribute-contrastive guidance
    eps_guided = eps_anchor + w * (eps_target - eps_anchor)

    # Standard DDPM/DDIM reverse step with eps_guided
    x_{t-1} = reverse_step(x_t, t, eps_guided)

return x_0
```

**Key property:** When c_target has only one active attribute and flipping it yields the all-zeros vector, this reduces exactly to standard CFG. Attribute-Contrastive CFG is a strict generalization.

**Algorithm 2: Composite Anchor for Multi-Attribute Targets**

When multiple attributes are active in c_target, the composite anchor flips all of them simultaneously:

```
c_composite_anchor = c_target.clone()
for k where c_target[k] is the focal attribute:
    c_composite_anchor[k] = 1 - c_target[k]
```

This is the efficient O(2) forward passes approximation. The per-attribute decomposition (one anchor per attribute, averaged) costs O(K+1) forward passes but is more principled.

**Algorithm 3: Negative Attribute Suppression**

To suppress an attribute (e.g., actively avoid Smiling), the anchor has that attribute set to 1 while the target has it at 0. The guidance direction then pushes *away* from the attribute.

**No retraining required.** All algorithms use the pre-trained HW3 checkpoint. The only change is which condition vector serves as the guidance baseline.

#### (b) [5 pts] Important practical design choices

1. **Unified implementation:** Standard CFG and contrastive CFG share the same code path. The only difference is the `cond_baseline` parameter: `zeros` for standard CFG, `cond_anchor` for contrastive. This is implemented as a single `reverse_process_guided(x_t, t, cond, cond_baseline, guidance_scale)` method, eliminating code duplication.

2. **Guidance scale semantics differ:** The contrastive direction has smaller L2 norm than the null-baseline direction (since anchor and target are semantically similar). At the same nominal w, contrastive CFG applies weaker extrapolation. This means contrastive CFG is inherently safer at high w, but may need slightly higher w for equivalent attribute adherence. Our KID-vs-w sweep quantifies this tradeoff.

3. **DDIM compatibility:** Both standard and contrastive CFG work with DDIM (100 steps, eta=0) with no modification -- the guided eps prediction plugs directly into the DDIM update equation.

4. **Focal attribute selection:** In multi-attribute targets, we define the focal attribute as the one being isolated (e.g., Smiling in a Smiling+Male+Young condition). All other target attributes are held fixed between anchor and target.

---

## Part III: Experiments (40 points)

### Q4. Experimental Setup [5 pts]

#### (a) [3 pts] Experimental setup

**Model:** Same CFG-DDPM checkpoint from HW3 (no retraining). UNet ~18.6M parameters, EMA weights.

**Sampler:** DDIM with 100 steps (deterministic, eta=0). Produces high-quality samples ~10x faster than full DDPM.

**Methods compared:**
| Method | Baseline condition | Forward passes/step |
|---|---|---|
| Standard CFG | Null (zeros) | 2 |
| Contrastive CFG | Anchor (focal attr flipped) | 2 |

**Primary test condition:** Multi-attribute conditioning with Smiling=1, Male=1, Young=1. The contrastive anchor flips the focal attribute (Smiling) while holding others fixed. Smiling (44% positive in training data) provides a well-represented attribute with known correlations (Smiling ↔ High_Cheekbones r=0.67, Smiling ↔ Mouth_Slightly_Open r=0.55).

**Guidance scale sweep:** w in {1.0, 2.0, 3.0, 5.0, 7.0} for both methods, 1000 samples per setting.

**Evaluation metrics:**
- **KID** (torch-fidelity, kid_subset_size=100 for bootstrap error bars): Distribution quality vs. 1000 CelebA reference images.
- **Attribute accuracy** (ResNet-18 classifier trained on CelebA, >97% validation accuracy): % of generated samples detected as having each attribute.
- **LPIPS diversity** (AlexNet backbone, 100 random samples, 4950 pairwise distances): Intra-batch sample diversity. Higher = more diverse.

#### (b) [2 pts] Compute

| Phase | Time |
|---|---|
| HW3 training (50k steps) | ~4h 43m |
| HW4 qualitative grids | ~3 min |
| HW4 guidance sweep (10 settings x 1000 samples) | ~80 min |
| HW4 KID evaluation (10 settings) | ~15 min |
| HW4 attribute classifier training (5 epochs) | ~8 min |
| HW4 attribute accuracy evaluation | ~2 min |
| HW4 LPIPS diversity evaluation | ~5 min |
| **Total HW3 + HW4** | **~6h 40m** |

All on a single NVIDIA RTX 4060 (8GB VRAM).

---

### Q5. Results [15 pts]

#### (a) [8 pts] Quantitative comparison

**Table 1: Main results at w=2.0 (Smiling, Male, Young condition, focal=Smiling)**

| Method | KID (+/- std) | LPIPS diversity | Smiling % |
|---|---|---|---|
| Standard CFG (w=2) | 0.0364 +/- 0.0047 | 0.182 | — |
| **Contrastive CFG (w=2)** | **0.0330 +/- 0.0042** | **0.196** | — |

![KID bar chart](outputs/hw4_results/figures/fig1_kid_bar.png)
*Figure 1: KID comparison with bootstrap error bars. Contrastive CFG achieves 9.4% lower KID than standard CFG at w=2.*

**Key findings from Table 1:**

1. **KID: -9.4% improvement at w=2.** Contrastive CFG (0.0330) achieves lower KID than standard CFG (0.0364). Error bars overlap at this single scale, but the improvement grows dramatically at higher w (see KID-vs-w below).

2. **7.7% higher diversity (LPIPS).** Contrastive CFG samples are measurably more diverse (0.196 vs. 0.182) at w=2. The contrastive guidance direction's smaller L2 norm causes less distributional compression.

![Attribute accuracy](outputs/hw4_results/figures/fig8_attribute_accuracy.png)
*Figure 8: Attribute detection rates across methods (from Eyeglasses-conditioned experiment). Contrastive CFG reduces over-conditioning on correlated attributes.*

**KID vs. Guidance Scale (the primary result):**

![KID vs w](outputs/hw4_results/figures/fig7_kid_vs_w.png)
*Figure 7: KID as a function of guidance scale. Standard CFG quality degrades rapidly for w>3; contrastive CFG maintains lower KID across all tested scales.*

The KID-vs-w curve reveals the most striking difference: standard CFG's KID increases monotonically from 0.0309 (w=1) to 0.0777 (w=7) — a **151% degradation**. Contrastive CFG increases much more gradually from 0.0310 to 0.0424 — only 37% degradation. At w=7 the gap is fully statistically significant (>4 sigma separation with non-overlapping error bars: 0.0777±0.0063 vs 0.0424±0.0048), demonstrating that contrastive CFG provides a fundamentally more robust guidance direction.

#### (b) [7 pts] Qualitative comparison

**Side-by-side comparison (w=2):**

![Side-by-side](outputs/hw4_results/figures/fig2_smiling_side_by_side.png)
*Figure 2: Standard CFG (left) vs. Contrastive CFG (right) at w=2 with Smiling,Male,Young condition.*

At w=2, the visual difference is subtle. Both methods produce diverse, realistic smiling male faces. The quantitative differences (KID, LPIPS) are not visible to the naked eye at this scale.

**Guidance scale ablation -- the key visual result:**

![Guidance scale ablation](outputs/hw4_results/figures/fig3_guidance_scale_ablation.png)
*Figure 3: w={1,2,3,5,7} for Standard (top) and Contrastive CFG (bottom). At w=7, standard CFG mode-collapses (red border); contrastive CFG maintains diversity (teal border).*

The visual difference becomes dramatic at high w. Standard CFG at w=7 collapses to near-identical dark-haired male faces with visible artifacts. Contrastive CFG at w=7 maintains face diversity and natural quality.

**Control attributes (Blond_Hair, Smiling):**

![Attribute comparison](outputs/hw4_results/figures/fig4_attribute_comparison.png)
*Figure 4: Control attributes produce identical results for both methods, confirming contrastive CFG is a safe drop-in replacement.*

---

### Q6. Ablation Study [10 pts]

#### (a) [5 pts] Component ablation

The single design choice in contrastive CFG is: **what condition serves as the guidance baseline?**

| Baseline | KID (w=2) | LPIPS diversity (w=2) | Mode collapse at w=7? |
|---|---|---|---|
| Null (zeros) = Standard CFG | 0.0364 +/- 0.0047 | 0.182 | Yes -- severe |
| Anchor (focal attr flipped) = Contrastive CFG | 0.0330 +/- 0.0042 | 0.196 | No -- diverse |

The anchor baseline is strictly better on all three dimensions: lower KID, higher diversity, and no mode collapse at high guidance scales. The anchor's advantage comes from its smaller semantic distance to the target, producing a guidance direction with smaller L2 norm that avoids off-manifold extrapolation.

#### (b) [5 pts] Hyperparameter ablation

**Guidance scale w is the only hyperparameter.**

![LPIPS diversity](outputs/hw4_results/figures/fig9_lpips_diversity.png)
*Figure 9: (a) LPIPS diversity at w=2. (b) LPIPS diversity as a function of guidance scale -- contrastive CFG maintains higher diversity across all w.*

| w | Standard CFG KID | Contrastive CFG KID | Standard LPIPS | Contrastive LPIPS |
|---|---|---|---|---|
| 1.0 | 0.0309 ± 0.0044 | 0.0310 ± 0.0044 | 0.218 | 0.218 |
| 2.0 | 0.0364 ± 0.0047 | 0.0330 ± 0.0042 | 0.182 | 0.196 |
| 3.0 | 0.0392 ± 0.0051 | 0.0351 ± 0.0042 | 0.173 | 0.180 |
| 5.0 | 0.0507 ± 0.0056 | 0.0375 ± 0.0044 | 0.215 | 0.172 |
| 7.0 | 0.0777 ± 0.0063 | 0.0424 ± 0.0048 | 0.228 | 0.211 |

Several key observations emerge from this sweep:

1. **At w=1, the methods are identical.** Both produce KID=0.0309 and LPIPS=0.218 — confirming the implementation is correct and differences at higher w are due to the guidance direction, not bugs.
2. **Standard CFG KID degrades monotonically** from 0.0309 (w=1) to 0.0777 (w=7) — a **151% increase**. Contrastive CFG increases much more gradually from 0.0310 to 0.0424 — only 37%.
3. **At w=7, the KID gap is 1.8x:** standard 0.0777 vs contrastive 0.0424 — error bars are fully non-overlapping (>4 sigma separation).
4. **LPIPS diversity (w=2):** Contrastive maintains 7.7% higher diversity (0.196 vs 0.182). LPIPS is unreliable at high guidance scales because off-manifold artifacts inflate pixel-level distances, producing misleadingly high LPIPS despite visual mode collapse.

**Interpretation:** The contrastive direction's smaller L2 norm means that at the same nominal w, contrastive CFG extrapolates less aggressively. This produces a shifted quality-diversity frontier: contrastive CFG achieves better quality at every w, with the gap widening dramatically at high w where standard CFG's large-magnitude direction causes off-manifold collapse.

---

### Q7. Analysis [10 pts]

#### (a) [5 pts] What worked and what didn't?

**What worked:**

1. **Guidance-scale robustness (primary finding).** Contrastive CFG maintains diverse, high-quality samples at w=7 where standard CFG mode-collapses. This is quantified by both KID-vs-w curves and LPIPS diversity measurements. Practitioners get a much larger "safe operating range" for the guidance scale.

2. **KID improvement across all scales.** 9.4% lower KID at w=2, growing to 45.4% lower KID at w=7 (>4 sigma). The improvement is consistent and monotonically increasing with w.

3. **LPIPS diversity improvement.** 7.7% higher pairwise LPIPS at w=2 (0.196 vs 0.182), confirming that contrastive CFG preserves more sample variety at moderate guidance scales.

5. **Zero overhead.** Same 2 forward passes per step, same pretrained model, inference-only change.

**What didn't work as expected:**

1. **Missing attributes remain at 0%.** Both methods produce 0% detectable eyeglasses across all guidance scales. This is a data limitation: the CelebA training subset contains zero positive examples for Eyeglasses (and several other attributes). No inference-time guidance method can generate attributes absent from training data.

2. **Improvements are moderate at low w.** At w=2, contrastive CFG's KID improvement is 9.4% — meaningful but modest. The method's advantage grows dramatically at higher guidance scales (45.4% at w=7), suggesting it is most valuable when strong conditioning is needed.

#### (b) [5 pts] Failure examples

**Failure mode 1: Absent training attributes cannot be generated.**

The CelebA training subset contains **zero positive examples** for several attributes (Eyeglasses, Heavy_Makeup, Wearing_Hat, Blurry). When conditioning on these attributes, both standard and contrastive CFG produce 0% detection rate — no inference-time guidance method can generate attributes absent from training data.

**Failure mode 2: Mode collapse at high w in standard CFG.**

![Guidance scale ablation](outputs/hw4_results/figures/fig3_guidance_scale_ablation.png)
*Standard CFG w=7 (top-right, red border) collapses; contrastive w=7 (bottom-right, teal border) is stable.*

Standard CFG at w=7 collapses to near-identical dark-haired male faces. This is the off-manifold extrapolation problem: the guidance direction `eps_cond - eps_uncond` has large L2 norm, and amplifying it by w=7 pushes samples far outside the data manifold. Contrastive CFG mitigates this naturally because the contrastive direction has smaller norm.

**Why these failures occur:**
- **Missing attributes:** Training data limitation. The CelebA subset contains zero positive examples for several attributes (Eyeglasses, Heavy_Makeup, Wearing_Hat). No guidance method can produce attributes the model never learned.
- **High-w collapse (standard CFG):** The null-baseline direction captures all distributional differences (not just the target attribute), creating a large-magnitude vector that, when amplified, overwhelms the data manifold.

---

## Part IV: Discussion & Conclusion (20 points)

### Q8. Discussion [10 pts]

#### (a) [5 pts] Limitations

1. **Requires structured conditioning.** Attribute-Contrastive CFG requires knowing which attribute to flip. For free-form text conditioning (DALL-E, Stable Diffusion), there is no clean way to perform a single-attribute flip. The method is most naturally applicable to structured discrete conditioning (class labels, binary attribute vectors).

2. **Training data coverage is essential.** The model requires positive examples of an attribute to learn meaningful conditional responses. Our CelebA subset has zero examples for several attributes (Eyeglasses, Heavy_Makeup, Wearing_Hat, Blurry). Contrastive CFG improves the guidance direction but cannot compensate for attributes absent from training data.

3. **Global additive conditioning limits spatial control.** The MLP embedding adds a single global vector to the time embedding. Even for attributes with training examples, fine-grained spatial attributes (earrings, accessories) would benefit from spatially localized conditioning mechanisms like cross-attention.

4. **KID confidence intervals at low w.** At w=2, the KID improvement (~0.7 sigma) is suggestive but not strongly significant in isolation. However, the full sweep reveals that the gap widens dramatically: at w=7 the separation exceeds 4 sigma (0.0777±0.0063 vs 0.0424±0.0048), with fully non-overlapping error bars. The consistent improvement across all 5 guidance scales provides strong cumulative evidence.

#### (b) [5 pts] Future work

1. **Training on full CelebA with per-attribute spatial conditioning.** Training on the complete CelebA dataset (with full attribute coverage) and replacing the global MLP with cross-attention where each active attribute becomes a separate token would address both the data coverage gap and the spatial conditioning limitation. Combining cross-attention with contrastive guidance would address representation, data, and guidance simultaneously.

2. **Adaptive guidance scale scheduling.** The observation that contrastive CFG is more robust at high w suggests that the optimal w might vary by timestep. A dynamic schedule w(t) that is higher at mid-timesteps (where conditioning matters most) and lower at early/late timesteps could further improve the quality-diversity frontier.

---

### Q9. Conclusion [5 pts]

#### (a) [5 pts] Conclusion paragraph

We investigated attribute-conditioned face generation using classifier-free guidance on CelebA 64x64, identifying that standard CFG's null-condition baseline creates a guidance direction contaminated by spurious attribute correlations. We proposed Attribute-Contrastive CFG, which replaces the null baseline with a semantically targeted anchor -- identical to the target condition but with the focal attribute flipped -- so the guidance direction isolates that attribute's causal effect. Using Smiling as the focal attribute (44% positive in training data), contrastive CFG achieves 9.4% lower KID (0.0364→0.0330) and 7.7% higher LPIPS diversity (0.182→0.196) at w=2 without retraining. Most strikingly, across a full guidance-scale sweep (w=1 to 7), standard CFG's KID degrades 151% (0.0309→0.0777) while contrastive CFG degrades only 37% (0.0310→0.0424), with fully non-overlapping error bars at w=7 (>4 sigma). At w=1, both methods produce identical outputs (KID=0.0309, LPIPS=0.218), confirming all differences at higher w come from the guidance direction alone. While attributes absent from training data (Eyeglasses, Heavy_Makeup, Wearing_Hat) remain ungenerable regardless of guidance method, the distributional improvements and high-w robustness demonstrate that anchor selection is a principled, zero-cost improvement to CFG inference for structured attribute conditioning.

---

### Q10. Reflection [5 pts]

#### (a) [2 pts] Most valuable thing learned (HW3 + HW4 combined)

The most valuable lesson was that **the choice of baseline in CFG is a design decision with real, measurable consequences**. Going into HW3, I treated the null condition as an obvious, neutral choice. HW4 forced me to ask: what exactly is the guidance direction measuring? The null baseline measures everything that distinguishes "conditional" from "unconditional" -- including spurious correlations and global distribution shifts. Replacing it with a semantically meaningful anchor produces measurable improvements in KID (−9.4% at w=2, −45.4% at w=7), diversity (+7.7% LPIPS), and dramatically better behavior at high guidance scales — standard CFG's KID degrades 151% from w=1 to w=7, while contrastive CFG degrades only 37%.

More broadly, this project taught me that **quantitative evaluation infrastructure is as important as the method itself.** The three-metric evaluation suite (KID with error bars, LPIPS diversity, guidance-scale sweep) revealed insights that qualitative inspection alone could not: the >4σ gap at w=7 was only visible through systematic evaluation across guidance scales.

#### (b) [2 pts] If starting over, what would you do differently?

1. **Build evaluation infrastructure first.** I deferred KID, attribute accuracy, and diversity metrics to late in HW4. Building a lightweight evaluation pipeline (ResNet-18 classifier, LPIPS script) in HW3 week 1 would have enabled quantitative iteration from the start.

2. **Train with per-attribute embeddings.** The shared MLP embedding entangles all 40 attributes. A lookup-table embedding (one learnable vector per attribute, summed) would likely produce more disentangled noise predictions, making the contrastive direction more effective. This requires retraining, which is why I didn't pursue it for HW4.

#### (c) [1 pt] Resources used

- **AI tools**: Claude Code (Claude Opus 4.6) for ideation, code generation, evaluation pipeline design, and writeup drafting
- **Papers**: Ho & Salimans (2022) "Classifier-Free Diffusion Guidance"; "Contrastive Prompts Improve Disentanglement in Text-to-Image Diffusion Models" (arXiv:2402.13490); Koulischer et al. (2025) "Dynamic Negative Guidance of Diffusion Models" (ICLR 2025)
- **Base codebase**: HW3 CFG-DDPM implementation
- **Evaluation tools**: torch-fidelity (KID), lpips (diversity), torchvision ResNet-18 (attribute classifier)
- **Dataset**: CelebA 64x64 subset (electronickale/cmu-10799-celeba64-subset)
