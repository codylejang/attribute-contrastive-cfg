#!/bin/bash
# Generate 1000 samples at each guidance scale for both standard and contrastive CFG
# Then run KID and LPIPS evaluation on all generated samples
#
# Usage: bash scripts/run_guidance_sweep.sh

set -e

CHECKPOINT="logs/cfg_ddpm_20260217_023852/checkpoints/cfg_ddpm_final.pt"
BASE_DIR="outputs/hw4_results/kid_samples"
REF_DIR="${BASE_DIR}/celeba_reference"
SEED=42

# Guidance scales to sweep (w=2 already exists as standard_eyeglasses / contrastive_eyeglasses)
SCALES="1.0 3.0 5.0 7.0"

echo "============================================================"
echo "Guidance Scale Sweep: Sample Generation"
echo "============================================================"

for W in $SCALES; do
    # Standard CFG
    OUT_DIR="${BASE_DIR}/standard_eyeglasses_w${W}"
    if [ -d "$OUT_DIR" ] && [ "$(ls -1 "$OUT_DIR"/*.png 2>/dev/null | wc -l)" -ge 1000 ]; then
        echo "SKIP: ${OUT_DIR} already has 1000+ samples"
    else
        echo ""
        echo ">>> Standard CFG w=${W} ..."
        python sample.py \
            --checkpoint "$CHECKPOINT" \
            --method cfg_ddpm --sampler ddim --num_steps 100 \
            --num_samples 1000 --batch_size 64 \
            --attributes "Eyeglasses,Male,Young" \
            --guidance_scale "$W" \
            --output_dir "$OUT_DIR" \
            --seed $SEED
    fi

    # Contrastive CFG
    OUT_DIR="${BASE_DIR}/contrastive_eyeglasses_w${W}"
    if [ -d "$OUT_DIR" ] && [ "$(ls -1 "$OUT_DIR"/*.png 2>/dev/null | wc -l)" -ge 1000 ]; then
        echo "SKIP: ${OUT_DIR} already has 1000+ samples"
    else
        echo ""
        echo ">>> Contrastive CFG w=${W} ..."
        python sample.py \
            --checkpoint "$CHECKPOINT" \
            --method cfg_ddpm --sampler ddim --num_steps 100 \
            --num_samples 1000 --batch_size 64 \
            --attributes "Eyeglasses,Male,Young" \
            --contrastive --focal_attributes "Eyeglasses" \
            --guidance_scale "$W" \
            --output_dir "$OUT_DIR" \
            --seed $SEED
    fi
done

echo ""
echo "============================================================"
echo "Running KID evaluation on all guidance scales"
echo "============================================================"

# Build KID sweep results
python -c "
import os, json, torch, torch_fidelity

base = '${BASE_DIR}'
ref = '${REF_DIR}'
results = {'standard': {}, 'contrastive': {}}

# w=2 (existing directories)
for method, dirname in [('standard', 'standard_eyeglasses'), ('contrastive', 'contrastive_eyeglasses')]:
    d = os.path.join(base, dirname)
    if os.path.isdir(d):
        print(f'Computing KID: {dirname}')
        m = torch_fidelity.calculate_metrics(input1=d, input2=ref, kid=True, fid=False, isc=False,
                                              kid_subset_size=100, verbose=False, cuda=torch.cuda.is_available())
        results[method]['2.0'] = {'kid_mean': m['kernel_inception_distance_mean'],
                                   'kid_std': m['kernel_inception_distance_std']}
        print(f'  KID: {m[\"kernel_inception_distance_mean\"]:.6f} +/- {m[\"kernel_inception_distance_std\"]:.6f}')

# Other w values
for w in ['1.0', '3.0', '5.0', '7.0']:
    for method in ['standard', 'contrastive']:
        dirname = f'{method}_eyeglasses_w{w}'
        d = os.path.join(base, dirname)
        if os.path.isdir(d) and len(os.listdir(d)) > 0:
            print(f'Computing KID: {dirname}')
            m = torch_fidelity.calculate_metrics(input1=d, input2=ref, kid=True, fid=False, isc=False,
                                                  kid_subset_size=100, verbose=False, cuda=torch.cuda.is_available())
            results[method][w] = {'kid_mean': m['kernel_inception_distance_mean'],
                                    'kid_std': m['kernel_inception_distance_std']}
            print(f'  KID: {m[\"kernel_inception_distance_mean\"]:.6f} +/- {m[\"kernel_inception_distance_std\"]:.6f}')

with open('outputs/hw4_results/kid_sweep_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved kid_sweep_results.json')
"

echo ""
echo "============================================================"
echo "Running LPIPS diversity on all guidance scales"
echo "============================================================"
python evaluate_diversity.py --output_json outputs/hw4_results/diversity_results.json

echo ""
echo "============================================================"
echo "Running attribute accuracy on all guidance scales"
echo "============================================================"
python evaluate_attributes.py \
    --data_root data/celeba-subset \
    --classifier_path outputs/hw4_results/eyeglasses_classifier.pt \
    --eval_only \
    --output_json outputs/hw4_results/attribute_accuracy.json

echo ""
echo "============================================================"
echo "All done! Results saved to outputs/hw4_results/"
echo "============================================================"
