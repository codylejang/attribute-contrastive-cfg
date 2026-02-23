# HW3: Baseline Implementation & Research Foundation
## CMU 10-799: Diffusion & Flow Matching, Spring 2026

---

## Part I: Motivation & Problem Definition (20 points)

### Q1. Track Selection [1 pt]

**Controllability**

---

### Q2. Motivation [9 pts]

#### (a) [3 pts] Why did you choose this track? What personally interests you about this challenge?

I chose the Controllability track because the ability to steer generative models is arguably the most practically impactful research direction in diffusion modeling today. While unconditional generation produces beautiful random samples, real-world applications almost always require user control: a designer needs a face with specific attributes, a content creator needs images matching a text description, and medical imaging requires generation conditioned on patient metadata. The gap between "generate something random" and "generate what I want" is the core challenge of controllability, and solving it unlocks the vast majority of practical diffusion model applications.

What personally interests me is the elegance of classifier-free guidance (CFG) as a training paradigm. The idea that a single model can learn both the conditional and unconditional distribution simultaneously, and that interpolating between them at inference time produces dramatically better results, is a beautiful insight with deep connections to Bayesian inference and energy-based modeling. I find this conceptual elegance compelling and want to understand it deeply through implementation.

#### (b) [3 pts] What is the core challenge?

The fundamental challenge in controllable diffusion generation is learning a conditional distribution p(x|c) that is both faithful to the condition c and produces high-quality, diverse samples. There are two competing objectives: **adherence** (generated images should match the specified condition) and **fidelity** (generated images should look realistic and diverse). Naively conditioning a diffusion model on additional information often leads to the model ignoring the condition or producing low-quality samples that overly fixate on it.

A second core challenge is **training efficiency**: conditional generation traditionally required training separate models or classifiers for guidance. Classifier guidance requires a pre-trained classifier and can introduce artifacts from adversarial gradients. The question is: can we build a single model that handles conditional generation without auxiliary networks, while providing a tunable knob between sample quality and condition adherence?

#### (c) [3 pts] Why does this matter?

1. **Personalized content creation**: Text-to-image and attribute-conditioned generation powers tools like DALL-E, Midjourney, and Stable Diffusion. Improving controllability directly translates to better creative tools used by millions of designers, artists, and content creators.

2. **Medical image synthesis**: Generating synthetic medical images conditioned on diagnoses, demographics, or anatomical features is critical for data augmentation in healthcare. For example, generating CXR images conditioned on "pneumonia + elderly" helps train diagnostic models when labeled data is scarce and privacy concerns limit real data sharing.

3. **Face generation for privacy-preserving datasets**: Generating synthetic faces with specific attributes (age, expression, ethnicity) enables creating balanced, privacy-compliant training datasets for face recognition research. Rather than using real people's photos, controllable generation can produce synthetic datasets with desired demographic distributions, addressing both ethical and data scarcity challenges.

---

### Q3. Problem Definition [10 pts]

#### (a) [4 pts] What specific problem are you solving?

I focus on **attribute-conditioned face generation using classifier-free guidance**. Specifically, given a set of binary facial attributes (e.g., "Smiling," "Young," "Male," "Eyeglasses"), the model should generate realistic 64x64 face images that exhibit the specified attributes. This is a well-defined instance of text-to-image generation where the "text" is a structured attribute vector.

This subproblem is interesting because:
- It provides a clean, quantitatively evaluable testbed for controllability (we can measure whether generated faces actually have the requested attributes)
- CelebA's 40 binary attributes enable rich compositional conditioning (e.g., "smiling young woman with glasses")
- It directly maps to the broader text-to-image paradigm but at a scale tractable for academic compute
- The baseline (classifier-free guidance) is the same technique used in state-of-the-art text-to-image models like DALL-E 2, Imagen, and Stable Diffusion

#### (b) [4 pts] What are your inputs, outputs, and assumptions?

**Input:**
- A binary attribute vector c of dimension 40 (one per CelebA attribute), where 1 indicates the desired attribute and 0 indicates absence/don't care
- A guidance scale w (scalar) controlling the strength of conditioning
- Standard DDPM noise schedule parameters

**Output:**
- Generated face images of shape (3, 64, 64) in RGB, with pixel values in [-1, 1]
- The generated images should be realistic faces that exhibit the attributes specified in c

**Assumptions:**
- Access to the CelebA 64x64 subset (~63,715 training images) with ground-truth attribute labels
- No pretrained models or external data beyond CelebA (the model is trained from scratch)
- The same UNet architecture from HW1/HW2 is used as the backbone, extended with condition embedding
- Binary attribute labels are a sufficient proxy for "text" conditioning at this resolution
- We assume the attribute annotations in CelebA are reasonably accurate

#### (c) [2 pts] How will you measure success?

**Primary metrics:**
- **KID (Kernel Inception Distance)**: Measures overall sample quality and distribution match against the CelebA dataset. Lower is better. Computed on the standard CelebA 64x64 subset.
- **Conditional accuracy**: For specific attributes (e.g., Smiling, Male, Eyeglasses), we can train a simple attribute classifier on real CelebA images and measure whether generated conditional samples actually exhibit the requested attributes.

**Qualitative evaluation:**
- A "successful sample" should show a clearly recognizable face at 64x64 resolution. When conditioned on "Smiling + Young + Female," the face should visibly be smiling, youthful, and feminine. When conditioned on "Male + Eyeglasses + No_Beard=0," the face should be male with glasses and a beard.
- Side-by-side comparisons of unconditional samples (w=0) vs. conditional samples (w=2, w=5) should show a clear progression from diverse-but-uncontrolled to focused-on-attributes.
- Grid visualizations showing the same condition with different random seeds to verify diversity.

---

## Part II: Related Works (20 points)

### Q4. Literature Survey [20 pts]

#### (a) [12 pts] Survey of 4-6 relevant methods

**1. Classifier Guidance (Dhariwal & Nichol, 2021)**
- *"Diffusion Models Beat GANs on Image Synthesis." NeurIPS 2021.*
- **Key idea:** Train a separate classifier on noisy images at each diffusion timestep, then use its gradient to guide the reverse process toward a desired class. The score function becomes: ∇ log p(x_t|y) = ∇ log p(x_t) + s · ∇ log p(y|x_t), where s is the guidance scale.
- **Pros:** Simple concept; dramatically improves class-conditional FID on ImageNet; works with any pretrained diffusion model by just adding a classifier.
- **Cons:** Requires training a separate noisy-image classifier, which is expensive and must match the noise schedule. The classifier gradients can introduce adversarial artifacts. Limited to class labels the classifier was trained on.

**2. Classifier-Free Guidance (Ho & Salimans, 2022)**
- *"Classifier-Free Diffusion Guidance." NeurIPS 2021 Workshop on DGMs.*
- **Key idea:** Train a single diffusion model that can operate both conditionally and unconditionally by randomly dropping the condition during training (with probability p_uncond). At inference, combine: ε̃ = ε_uncond + w · (ε_cond - ε_uncond), where w > 1 amplifies the conditioning signal.
- **Pros:** No separate classifier needed; single model for both modes; simple to implement; produces state-of-the-art results. The guidance scale w provides a smooth trade-off between diversity and condition adherence. This is the de facto standard in modern text-to-image models.
- **Cons:** Requires two forward passes per denoising step during inference (doubling cost). Very high guidance scales can cause oversaturation and reduced diversity. The unconditional drop rate p_uncond is a hyperparameter that needs tuning.

**3. DALL-E 2 (Ramesh et al., 2022)**
- *"Hierarchical Text-Conditional Image Generation with CLIP Latents." arXiv:2204.06125.*
- **Key idea:** Uses a two-stage approach: (1) a CLIP text encoder maps text to a CLIP image embedding via a diffusion prior, then (2) a diffusion decoder generates images conditioned on the CLIP embedding. Uses classifier-free guidance in both stages.
- **Pros:** Leverages powerful CLIP embeddings for rich text understanding; produces high-quality 1024x1024 images; supports image variations via CLIP latent manipulation.
- **Cons:** Requires pre-trained CLIP (large external model); two-stage pipeline is complex; sometimes struggles with compositional text prompts (e.g., "a red cube on top of a blue sphere").

**4. Imagen (Saharia et al., 2022)**
- *"Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding." NeurIPS 2022.*
- **Key idea:** Uses a frozen T5-XXL text encoder (not CLIP) to encode text, then applies classifier-free guidance at multiple resolutions in a cascaded diffusion pipeline (64x64 → 256x256 → 1024x1024).
- **Pros:** Demonstrates that larger language models (T5-XXL) are more important than larger diffusion models for text-image alignment. Achieves state-of-the-art FID on COCO. Shows classifier-free guidance scales of 5-10 are optimal.
- **Cons:** Requires a very large frozen language model; cascaded pipeline adds complexity; high compute cost for training and inference.

**5. Conditional Flow Matching (Lipman et al., 2023; Tong et al., 2024)**
- *"Flow Matching for Generative Modeling." ICLR 2023. "Improving and Generalizing Flow-Based Generative Models." ICLR 2024.*
- **Key idea:** Extend flow matching to conditional generation by incorporating conditioning information directly into the learned velocity field. The optimal transport formulation naturally supports conditioning without the need for noise schedule-dependent guidance.
- **Pros:** Deterministic sampling; theoretically elegant OT formulation; can be more efficient than diffusion at few steps; naturally extends to conditional settings.
- **Cons:** Less mature than diffusion-based guidance methods; fewer pretrained models available; classifier-free guidance is less well-studied in the flow matching framework.

**6. ControlNet (Zhang et al., 2023)**
- *"Adding Conditional Control to Text-to-Image Diffusion Models." ICCV 2023.*
- **Key idea:** Adds a trainable copy of the encoder blocks of a pretrained diffusion model (e.g., Stable Diffusion), connected via zero-initialized convolutions. This "control net" learns to incorporate spatial conditioning signals (edges, depth maps, poses) without destroying the pretrained model's generative capability.
- **Pros:** Preserves pretrained model quality; works with diverse spatial conditions (Canny edges, depth, segmentation, pose); zero-initialization ensures stable fine-tuning.
- **Cons:** Requires a pretrained base model; limited to spatial conditioning (not semantic/attribute conditioning); adds significant parameter count (duplicating encoder).

#### (b) [8 pts] Comparison table

| Method | No Separate Classifier | Attribute-Level Control | Single Model | Tunable Quality-Diversity | Training Cost |
|--------|:---------------------:|:----------------------:|:------------:|:------------------------:|:-------------:|
| Classifier Guidance | No | Yes (class labels) | No | Yes (guidance scale) | High (classifier + diffusion) |
| **Classifier-Free Guidance** | **Yes** | **Yes** | **Yes** | **Yes (guidance scale)** | **Low (single model)** |
| DALL-E 2 | Yes | Yes (text) | No (two-stage) | Yes | Very High |
| Imagen | Yes | Yes (text) | No (cascaded) | Yes | Very High |
| Conditional Flow Matching | Yes | Yes | Yes | Partial | Low |
| ControlNet | Yes | No (spatial only) | No (needs pretrained) | Limited | Medium |

**Dimensions chosen:**
1. **No Separate Classifier**: Whether the method avoids training an auxiliary classifier (simpler pipeline)
2. **Attribute-Level Control**: Whether the method supports fine-grained attribute/text control
3. **Single Model**: Whether a single model handles both conditional and unconditional generation
4. **Tunable Quality-Diversity**: Whether there's an explicit knob to trade off quality vs. diversity
5. **Training Cost**: Relative computational cost to train from scratch on CelebA 64x64

---

## Part III: Baseline Selection & Preliminaries (25 points)

### Q5. Choose Your Baseline [7 pts]

#### (a) [3 pts] Which method will you implement?

I will implement **Classifier-Free Diffusion Guidance (CFG)** applied to DDPM with CelebA attribute conditioning. Specifically, I implement the method from Ho & Salimans (2022), "Classifier-Free Diffusion Guidance," adapting it from class-conditional ImageNet generation to attribute-conditional CelebA face generation.

#### (b) [3 pts] Why this method?

I chose classifier-free guidance over the alternatives for several reasons:

1. **Simplicity**: CFG requires only minor modifications to the existing DDPM codebase from HW1/HW2. We add a condition embedding to the UNet, randomly drop the condition during training, and combine conditional/unconditional predictions at inference. No external models (classifiers, CLIP, T5) are needed.

2. **Foundational importance**: CFG is the conditioning method used in virtually all modern text-to-image models (DALL-E 2, Imagen, Stable Diffusion). Understanding it deeply by implementing from scratch provides insight into the entire field.

3. **Tractable compute**: Unlike DALL-E 2 or Imagen, CFG can be implemented and trained from scratch on a single GPU within hours on CelebA 64x64. Methods like ControlNet require pretrained models we don't have.

4. **Clean evaluation**: CelebA's binary attributes provide ground truth for measuring whether conditioning actually works, unlike free-form text where evaluation is subjective.

5. **Clear HW4 path**: CFG provides a natural baseline to improve upon in HW4 (e.g., better condition encoding, compositional control, guidance schedule tuning, classifier-free guidance for flow matching).

#### (c) [1 pt] Resources

- **Reference code**: The lucidrains/classifier-free-guidance-pytorch repository for architecture patterns
- **Paper**: Ho & Salimans, "Classifier-Free Diffusion Guidance" (2022)
- **Base code**: HW1/HW2 codebase (DDPM + Flow Matching implementations)
- **AI tools**: Cursor AI assistant for code generation and debugging
- **Dataset**: CelebA 64x64 subset from HuggingFace (electronickale/cmu-10799-celeba64-subset)

---

### Q6. Technical Deep Dive [18 pts]

#### (a) [5 pts] Key insight

The key insight of classifier-free guidance is that **you don't need a separate classifier to guide diffusion sampling**. Instead, you can train a single model that implicitly learns both the conditional score ∇ log p(x_t|c) and the unconditional score ∇ log p(x_t) simultaneously. By randomly dropping the condition during training (replacing c with a null token ∅ with probability p_uncond), the model learns to operate in both modes.

At inference time, the guided prediction extrapolates away from the unconditional prediction toward the conditional prediction:

ε̃(x_t, c) = ε(x_t, ∅) + w · [ε(x_t, c) - ε(x_t, ∅)]

When w = 0, we get unconditional generation. When w = 1, we get standard conditional generation. When w > 1, we get "extra" conditioning that produces samples more strongly aligned with c at the cost of reduced diversity. This creates a smooth, tunable trade-off between sample diversity and condition adherence.

The deep insight is that this guidance formula is equivalent to sampling from the distribution p(x|c) ∝ p(c|x)^w · p(x), i.e., Bayesian inference with a sharpened likelihood. The guidance scale w controls how much we "sharpen" the posterior.

#### (b) [5 pts] Algorithm pseudocode

**Training Algorithm:**
```
Input: Dataset D = {(x, c)} of images with attribute labels
       Model ε_θ(x_t, t, c) with condition embedding
       Unconditional probability p_uncond

for each training step:
    Sample (x_0, c) from D
    Sample t ~ Uniform({1, ..., T})
    Sample ε ~ N(0, I)
    
    # Classifier-free: randomly drop condition
    if random() < p_uncond:
        c_input = 0  (zero vector = unconditional)
    else:
        c_input = c  (real condition)
    
    # Forward process
    x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
    
    # Predict noise with condition
    ε_pred = ε_θ(x_t, t, c_input)
    
    # MSE loss
    L = ||ε_pred - ε||²
    
    Update θ to minimize L
```

**Sampling Algorithm (CFG):**
```
Input: Condition c, guidance scale w, number of steps T

x_T ~ N(0, I)

for t = T, T-1, ..., 1:
    # Two forward passes
    ε_uncond = ε_θ(x_t, t, 0)      # unconditional
    ε_cond   = ε_θ(x_t, t, c)      # conditional
    
    # Classifier-free guidance
    ε̃ = ε_uncond + w * (ε_cond - ε_uncond)
    
    # DDPM reverse step using guided noise prediction
    x̂_0 = (x_t - sqrt(1 - ᾱ_t) * ε̃) / sqrt(ᾱ_t)
    μ = posterior_mean(x̂_0, x_t, t)
    σ² = posterior_variance(t)
    
    if t > 1:
        x_{t-1} = μ + σ * z,  z ~ N(0, I)
    else:
        x_{t-1} = μ

return x_0
```

#### (c) [5 pts] Key equation

The central equation of classifier-free guidance is:

**ε̃(x_t, t, c) = (1 - w) · ε_θ(x_t, t, ∅) + w · ε_θ(x_t, t, c)**

Or equivalently:

**ε̃(x_t, t, c) = ε_θ(x_t, t, ∅) + w · [ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅)]**

Where:
- **ε_θ(x_t, t, c)**: The model's noise prediction conditioned on attributes c at timestep t for noisy image x_t. This represents the conditional score function, telling the model "what does a face with these attributes look like?"
- **ε_θ(x_t, t, ∅)**: The model's unconditional noise prediction (condition zeroed out). This represents the unconditional score, telling the model "what does a generic face look like?"
- **w**: The guidance scale. Controls the strength of conditioning. w=1 gives standard conditional generation. w>1 amplifies the difference between conditional and unconditional predictions, producing samples more strongly aligned with c but with less diversity.
- **ε̃**: The guided noise prediction used in the reverse process. This is a linear extrapolation that moves the prediction further in the direction indicated by the condition.

This equation implicitly samples from a tilted distribution proportional to p(x) · p(c|x)^(w-1), which sharpens the conditional likelihood.

#### (d) [3 pts] Important hyperparameters

| Hyperparameter | Original Paper Value | Our Value | Rationale |
|---|---|---|---|
| **p_uncond** (unconditional dropout rate) | 0.1-0.2 | 0.1 | 10% unconditional training as in the original paper. Ensures the model learns good unconditional generation while primarily being conditional. |
| **Guidance scale w** | 1.0-5.0 (varies by task) | 2.0 (default, tunable) | Moderate guidance. Original paper shows w=2-4 works well for class-conditional ImageNet. CelebA attributes are "softer" than class labels, so moderate guidance is appropriate. |
| **Condition embedding** | Class embedding | MLP(40 → 256 → 256) | Maps 40 binary attributes through a 2-layer MLP to match the time embedding dimension. Added to time embedding for joint conditioning. |
| **Beta schedule** | Linear, 1000 steps | Linear, β₁=0.0001, β_T=0.02, T=1000 | Standard DDPM schedule from Ho et al. (2020) |
| **Learning rate** | 2e-4 | 2e-4 | Standard AdamW with (β₁, β₂) = (0.9, 0.999) |
| **EMA decay** | 0.9999 | 0.9999 | Standard EMA for stable sampling |

---

## Part IV: Implementation & Preliminary Results (25 points)

### Q7. Experimental Setup [10 pts]

#### (a) [8 pts] Experimental setup

**Model architecture:**
- UNet with base_channels=64, channel_mult=[1,2,2,4], num_res_blocks=2
- Attention at resolution 16x16
- 4 attention heads, dropout=0.1, FiLM (scale-shift) conditioning
- Condition embedding: Linear(40→256) → SiLU → Linear(256→256), added to time embedding
- Total parameters: ~18.6M

**Training setup:**
- **Batch size**: 64
- **Learning rate**: 2e-4 (AdamW, β₁=0.9, β₂=0.999)
- **Weight decay**: 0.0
- **EMA decay**: 0.9999 (starts at step 2000)
- **Gradient clipping**: norm 1.0
- **Mixed precision**: FP16 (AMP)
- **Total iterations**: 50,000
- **Dataset**: CelebA 64x64 subset (63,715 training images with 40 binary attribute labels)

**Inference setup:**
- **Sampling steps**: 1000 (full DDPM schedule)
- **Guidance scale**: 2.0 (tunable at inference)
- **Sampler**: DDPM reverse process with CFG (also supports DDIM with CFG)
- **Condition**: Binary attribute vector (40 dims), zeros for unconditional

**Modifications from original paper:**
- The original CFG paper uses ImageNet class labels (single integer → class embedding). We instead use a 40-dimensional binary attribute vector processed through an MLP, which is a richer conditioning signal.
- The condition embedding is added to the time embedding (additive), following the standard practice for DDPM conditioning.
- We use a smaller UNet (~18.6M params vs. the original paper's ~400M) due to compute constraints.

#### (b) [2 pts] Compute

- **GPU**: NVIDIA RTX 4060 (8GB VRAM), 6.2GB VRAM used during training
- **Training time**: 4 hours 43 minutes for 50,000 iterations at ~2.94 it/s (with AMP)
- **Inference time**: ~5 seconds per image at 1000 DDPM steps with CFG (2 forward passes per step); ~80 seconds per 16-sample grid
- **Total compute**: ~5 hours training + ~30 minutes sampling for all results

---

### Q8. Preliminary Results [15 pts]

#### (a) [10 pts] Qualitative results

Training completed in 4 hours 43 minutes (50,000 iterations). Final training loss: 0.0169.

**Training progression** (conditioned on "Smiling + Young" during training-time sampling):
- Step 5,000: Pure noise, no structure
- Step 10,000: Blurry face-like shapes emerging with rough color patterns
- Step 25,000: Recognizable but blurry faces, face structure is clear
- Step 50,000: Sharp, realistic 64x64 faces with good detail

![Training Progression](outputs/hw3_results/composite_training_progression.png)

**Conditional generation results** (all at 1000 DDPM steps, guidance scale w=2.0):
- **"Smiling + Young" (w=2)**: Grid of 16 faces showing mostly smiling, young-looking faces. The conditioning is clearly effective -- faces are predominantly smiling with youthful features.
- **"Male + Eyeglasses" (w=2)**: Grid of 16 clearly male faces. The "Male" attribute is strongly conditioned, though "Eyeglasses" is a harder attribute and appears in some but not all samples.
- **"Blond_Hair + Young" (w=2)**: Striking results -- nearly all 16 faces have blonde hair and look young, demonstrating very strong hair color conditioning.
- **Unconditional (w=0)**: Diverse mix of male/female, various ages and hair colors, demonstrating the model's unconditional diversity.

![Conditional Generation Results](outputs/hw3_results/composite_conditional_generation.png)

**Guidance scale ablation** (condition: "Smiling + Young", varying w):
- **w=0.0**: Diverse faces (mix of male/female, various ages, not all smiling) -- pure unconditional
- **w=1.0**: Slight trend toward smiling/young, still diverse
- **w=2.0**: Mostly smiling and youthful faces, good diversity maintained
- **w=4.0**: More uniformly smiling and young, slightly reduced diversity
- **w=7.0**: Nearly all smiling + young, noticeably reduced diversity, more "sharpened" appearance

This guidance scale progression demonstrates the expected CFG trade-off between condition adherence and sample diversity.

![Guidance Scale Ablation](outputs/hw3_results/composite_guidance_ablation.png)

All generated sample grids are saved in `outputs/hw3_results/`.

#### (b) [0 pts, optional] Quantitative results

Quantitative evaluation is planned for HW4 with:
- **KID**: Will compute on 1000+ generated samples vs. CelebA training set
- **Attribute accuracy**: Will train a ResNet-18 attribute classifier on CelebA and measure accuracy of conditional samples
- **Guidance scale ablation**: KID and attribute accuracy as a function of w ∈ {0, 1, 2, 3, 5, 7, 10}

#### (c) [5 pts] Analysis of preliminary results

**What's working well:**
- The training loss converges rapidly and smoothly. Starting from ~0.1, the loss drops to ~0.025 within the first 1000 iterations and reaches 0.0169 by iteration 50,000. This convergence is comparable to our unconditional DDPM from HW1/HW2, indicating the conditioning mechanism does not hinder optimization.
- **Conditioning is clearly effective**: The "Blond_Hair" condition produces near-unanimous blonde faces, "Male" produces uniformly male faces, and "Smiling" produces smiling faces. This demonstrates that the simple additive embedding approach successfully encodes attribute information.
- **Guidance scale works as expected**: Increasing w from 0 to 7 shows a smooth transition from diverse unconditional samples to strongly conditioned samples, confirming the CFG mechanism is functioning correctly.
- Mixed precision (AMP) training works without numerical issues. Training ran at ~2.94 it/s on an RTX 4060 with 6.2GB/8.2GB VRAM used.

**What needs improvement:**
- **Rare/subtle attributes**: "Eyeglasses" is less consistently generated than "Male" or "Blond_Hair." This is likely because eyeglasses are a fine spatial detail that's harder to condition via a global embedding. Cross-attention conditioning or spatially-adaptive conditioning could help.
- **High guidance scales**: At w=7, diversity drops significantly and some faces show slight artifacts. A dynamic guidance schedule (varying w by timestep) could improve this.
- **Sampling speed**: Full 1000-step DDPM with CFG requires 2000 forward passes per sample (~5 seconds per image). DDIM support with CFG would speed this up significantly. (The DDIM-CFG sampler is implemented but not yet fully validated.)

**Surprises:**
- The 10% unconditional dropout rate works excellently. The model learns both modes without any visible degradation in either conditional or unconditional quality.
- Adding the condition to the time embedding (rather than cross-attention) works remarkably well for attribute conditioning. The MLP embedding (40→256→256, ~0.1M extra params) is sufficient to encode all 40 binary attributes.
- **"Blond_Hair" is the most visually striking condition** -- it consistently produces near-perfect blonde hair across all 16 samples at w=2. This may be because hair color is a highly salient, globally-visible feature that's easy to learn from a dataset-level embedding.

---

## Part V: Brainstorming for HW4 (5 points)

### Q9. Ideas for Improvement [5 pts]

#### (a) [5 pts] 2-3 ideas for HW4 improvement

**1. Compositional Guidance with Attribute-Specific Embeddings**
Instead of mapping all 40 attributes through a single MLP, assign each attribute its own learned embedding vector and combine them via attention or summation. This would allow the model to better disentangle individual attributes and enable more precise compositional control (e.g., "add glasses" without changing other features). The hypothesis is that the current shared MLP creates entangled representations where changing one attribute inadvertently affects others.

**2. Dynamic Guidance Schedule**
Instead of using a fixed guidance scale w throughout sampling, vary w as a function of timestep. Recent work (e.g., Karras et al., 2024) suggests that guidance is most beneficial at intermediate noise levels and can be harmful at very high or very low noise. Implementing a cosine or step-function guidance schedule w(t) that starts low, peaks at mid-timesteps, and decays could improve both sample quality and diversity simultaneously.

**3. Negative Prompting / Contrastive Guidance**
Extend CFG to support "negative attributes" -- attributes the user explicitly doesn't want. For example, conditioning on "Smiling=1, Eyeglasses=0" where the 0 means "actively avoid eyeglasses." This can be implemented by modifying the guidance formula to: ε̃ = ε_uncond + w_pos · (ε_pos - ε_uncond) - w_neg · (ε_neg - ε_uncond), where ε_neg is conditioned on the negative attributes. This is the technique behind "negative prompts" in Stable Diffusion and could significantly improve control precision.

---

## Part VI: Reflection (5 points)

### Q10. Reflection [5 pts]

#### (a) [2 pts] Most interesting/surprising finding from literature survey

The most surprising finding was how dominant classifier-free guidance has become despite its simplicity. When I started the survey, I expected that more sophisticated conditioning mechanisms (cross-attention with CLIP embeddings, ControlNet-style architectures) would be fundamentally different from CFG. But in reality, nearly all modern text-to-image models use CFG as their core guidance mechanism, with the differences being primarily in how the condition is encoded (CLIP vs. T5 vs. attribute vectors) rather than how guidance is applied. The formula ε̃ = ε_uncond + w · (ε_cond - ε_uncond) is remarkably universal.

Also surprising was the finding from Imagen that scaling the language model (T5-XXL) matters more than scaling the diffusion model for text-image alignment. This suggests that the bottleneck in controllability is often in understanding the condition, not in the generation capacity.

#### (b) [2 pts] Did implementing the baseline change your understanding?

Yes, implementing CFG from scratch clarified several things:

1. **The condition dropout is doing more than regularization.** I initially thought of p_uncond as just a regularization technique. But implementing it made clear that it's actually teaching the model two distinct modes of operation, and the quality of unconditional generation directly affects how well guidance works (since guidance extrapolates from the unconditional direction). Our unconditional samples (w=0) are just as high-quality as our conditional ones, confirming both modes are well-learned.

2. **The additive conditioning is surprisingly effective.** I expected that simply adding the attribute embedding to the time embedding would be too crude -- the model would need cross-attention or FiLM conditioning at every layer. But adding to the time embedding works because the time embedding already modulates every ResBlock via scale-and-shift, so the condition information propagates through the entire network. The "Blond_Hair" results are particularly striking evidence of this.

3. **Two forward passes per step is a real computational cost.** Reading about CFG, I underestimated the 2x inference cost. Implementing it and seeing sampling take ~5 seconds per image (vs ~2.5 for unconditional) made me appreciate why methods that avoid this (e.g., distillation, amortized guidance) are an active research area.

4. **Not all attributes are equal.** Globally visible attributes (hair color, gender) condition much more reliably than fine-grained spatial attributes (eyeglasses, goatee). This suggests that for HW4, attribute-specific embeddings or spatial conditioning would be valuable improvements.

#### (c) [1 pt] Resources used

- **AI tools**: Cursor AI assistant (Claude) for code generation, debugging, and writeup drafting
- **Papers**: Ho & Salimans (2022) "Classifier-Free Diffusion Guidance"; Dhariwal & Nichol (2021) "Diffusion Models Beat GANs"; Ramesh et al. (2022) "DALL-E 2"; Saharia et al. (2022) "Imagen"; Lipman et al. (2023) "Flow Matching"; Zhang et al. (2023) "ControlNet"
- **Open source code**: lucidrains/classifier-free-guidance-pytorch (reference patterns)
- **Base codebase**: HW1/HW2 DDPM and Flow Matching implementations
- **Dataset**: electronickale/cmu-10799-celeba64-subset on HuggingFace
