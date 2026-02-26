#!/bin/bash
# Generate 1000 samples at each guidance scale for both standard and contrastive CFG
# Using Smiling as the focal attribute (44% positive in training data)
#
# Usage: bash scripts/run_smiling_sweep.sh

set -e

CHECKPOINT="logs/cfg_ddpm_20260217_023852/checkpoints/cfg_ddpm_final.pt"
BASE_DIR="outputs/hw4_results/kid_samples"
REF_DIR="${BASE_DIR}/celeba_reference"
SEED=42

SCALES="1.0 2.0 3.0 5.0 7.0"

echo "============================================================"
echo "Guidance Scale Sweep: Smiling (focal), Male, Young"
echo "============================================================"

for W in $SCALES; do
    # Standard CFG
    OUT_DIR="${BASE_DIR}/standard_smiling_w${W}"
    if [ -d "$OUT_DIR" ] && [ "$(ls -1 "$OUT_DIR"/*.png 2>/dev/null | wc -l)" -ge 1000 ]; then
        echo "SKIP: ${OUT_DIR} already has 1000+ samples"
    else
        echo ""
        echo ">>> Standard CFG w=${W} ..."
        python sample.py \
            --checkpoint "$CHECKPOINT" \
            --method cfg_ddpm --sampler ddim --num_steps 100 \
            --num_samples 1000 --batch_size 64 \
            --attributes "Smiling,Male,Young" \
            --guidance_scale "$W" \
            --output_dir "$OUT_DIR" \
            --seed $SEED
    fi

    # Contrastive CFG
    OUT_DIR="${BASE_DIR}/contrastive_smiling_w${W}"
    if [ -d "$OUT_DIR" ] && [ "$(ls -1 "$OUT_DIR"/*.png 2>/dev/null | wc -l)" -ge 1000 ]; then
        echo "SKIP: ${OUT_DIR} already has 1000+ samples"
    else
        echo ""
        echo ">>> Contrastive CFG w=${W} ..."
        python sample.py \
            --checkpoint "$CHECKPOINT" \
            --method cfg_ddpm --sampler ddim --num_steps 100 \
            --num_samples 1000 --batch_size 64 \
            --attributes "Smiling,Male,Young" \
            --contrastive --focal_attributes "Smiling" \
            --guidance_scale "$W" \
            --output_dir "$OUT_DIR" \
            --seed $SEED
    fi
done

echo ""
echo "============================================================"
echo "Running KID evaluation on Smiling sweep"
echo "============================================================"

python -c "
import os, json, torch, torch_fidelity

base = '${BASE_DIR}'
ref = '${REF_DIR}'
results = {'standard': {}, 'contrastive': {}}

for w in ['1.0', '2.0', '3.0', '5.0', '7.0']:
    for method in ['standard', 'contrastive']:
        dirname = f'{method}_smiling_w{w}'
        d = os.path.join(base, dirname)
        if os.path.isdir(d) and len(os.listdir(d)) > 0:
            print(f'Computing KID: {dirname}')
            m = torch_fidelity.calculate_metrics(input1=d, input2=ref, kid=True, fid=False, isc=False,
                                                  kid_subset_size=100, verbose=False, cuda=torch.cuda.is_available())
            results[method][w] = {'kid_mean': m['kernel_inception_distance_mean'],
                                    'kid_std': m['kernel_inception_distance_std']}
            print(f'  KID: {m[\"kernel_inception_distance_mean\"]:.6f} +/- {m[\"kernel_inception_distance_std\"]:.6f}')

with open('outputs/hw4_results/kid_sweep_smiling.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved kid_sweep_smiling.json')
"

echo ""
echo "============================================================"
echo "Running LPIPS diversity on Smiling sweep"
echo "============================================================"
python evaluate_diversity.py --output_json outputs/hw4_results/diversity_smiling.json

echo ""
echo "============================================================"
echo "Running attribute accuracy on Smiling sweep"
echo "============================================================"
python evaluate_attributes.py \
    --data_root data/celeba-subset \
    --classifier_path outputs/hw4_results/eyeglasses_classifier.pt \
    --eval_only \
    --output_json outputs/hw4_results/attribute_accuracy_smiling.json

echo ""
echo "============================================================"
echo "Smiling sweep complete! Results saved to outputs/hw4_results/"
echo "============================================================"
