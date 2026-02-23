# attribute-contrastive-cfg

Attribute-Contrastive Classifier-Free Guidance (CFG) for diffusion models. Replaces the null-condition baseline with semantically targeted attribute anchors to isolate attribute-specific guidance directions, reducing spurious correlations. Improves guidance-scale stability for attribute-conditioned face generation on CelebA.

## About

This repository implements **Attribute-Contrastive CFG**, an improved classifier-free guidance method for diffusion models. The key innovation is replacing the standard null-condition baseline with targeted attribute anchors that isolate the causal effect of specific attributes, reducing spurious correlations in the guidance direction.

### Key Results

- **16.8% KID improvement** (0.0316 → 0.0263) on CelebA attribute-conditioned generation
- **Better guidance-scale stability**: No mode collapse at high guidance scales (w=7)
- **Zero computational overhead**: Same 2 forward passes as standard CFG, no retraining required
- **Unified framework**: Supports both positive attribute guidance and negative attribute suppression

## Method

Standard CFG uses the null/unconditional prediction as the baseline:
```
ε̃ = ε_θ(x_t, ∅) + w · [ε_θ(x_t, c_target) − ε_θ(x_t, ∅)]
```

Attribute-Contrastive CFG replaces the null baseline with a targeted anchor:
```
ε̃ = ε_θ(x_t, c_anchor) + w · [ε_θ(x_t, c_target) − ε_θ(x_t, c_anchor)]
```

Where `c_anchor` is identical to `c_target` except the focal attribute is flipped. This isolates the attribute-specific guidance direction, eliminating spurious correlations.

## Installation

```bash
git clone https://github.com/codylejang/attribute-contrastive-cfg.git
cd attribute-contrastive-cfg

# Setup environment (see setup instructions below)
./setup-uv.sh  # or ./setup.sh

# Download CelebA dataset
python download_dataset.py
```

## Usage

### Training

Train a CFG-DDPM model (baseline from HW3):

```bash
python train.py --method cfg_ddpm --config configs/ddpm_cfg.yaml
```

### Sampling with Attribute-Contrastive CFG

```bash
# Standard CFG (baseline)
python sample.py \
    --checkpoint checkpoints/cfg_ddpm_final.pt \
    --method cfg_ddpm \
    --attributes "Eyeglasses,Male,Young" \
    --guidance_scale 2.0

# Attribute-Contrastive CFG
python sample.py \
    --checkpoint checkpoints/cfg_ddpm_final.pt \
    --method cfg_ddpm \
    --attributes "Eyeglasses,Male,Young" \
    --focal_attributes "Eyeglasses" \
    --contrastive \
    --guidance_scale 2.0
```

### Evaluation

```bash
# Compute KID scores
python evaluate_kid.py \
    --checkpoint checkpoints/cfg_ddpm_final.pt \
    --method cfg_ddpm \
    --num_samples 1000
```

## Results

See `hw4_writeup.md` for detailed experimental results, ablations, and analysis.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{attribute-contrastive-cfg,
  title={Attribute-Contrastive Classifier-Free Guidance for Diffusion Models},
  author={Cody Lejang},
  year={2026},
  howpublished={\url{https://github.com/codylejang/attribute-contrastive-cfg}}
}
```

## References

- Ho & Salimans (2022) "Classifier-Free Diffusion Guidance"
- "Contrastive Prompts Improve Disentanglement in Text-to-Image Diffusion Models" (arXiv:2402.13490)
- Koulischer et al. (2025) "Dynamic Negative Guidance of Diffusion Models" (ICLR 2025)

## License

This project is part of CMU 10-799: Diffusion & Flow Matching (Spring 2026).
