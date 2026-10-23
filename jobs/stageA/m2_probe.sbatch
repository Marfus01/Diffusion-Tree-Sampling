#!/bin/bash
#SBATCH -J A-M2-probe
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# A-R006 preparation: image-path unit checks, guiding ResNet-18 and its per-class test accuracy,
# throughput of DTS (tree batch 1 / 16 / 64) and of SMC/FK, TDS, plus one offline evaluation.
set -euo pipefail
export DTS_CODE=$HOME/run/dts/wt-m2
source $DTS_CODE/jobs/stageA/env.sh
B=$DTS_BASE_DIR/cifar10
RES=$B/img/results_tree_viz/cifar10
COMMON="--dataset cifar10 --sample_classes 0 --population_size 5000 --seed 0 --no_progress_bar --base_dir $B"

echo "[$(date)] unit checks"
python -m pytest -q tests/test_sampling_img.py

echo "[$(date)] guiding classifier"
python -m jobs.stageA.m2_classifier --base_dir $B --out $B/m2_classifier/per_class_accuracy.json

for bs in 1 16 64; do
  echo "[$(date)] DTS tree batch $bs"
  python -m scripts.run_img $COMMON --do_mcts_rollout --compute_budget 20000 --mcts_batch_size $bs \
    --exp_name m2_probe_dts_bs$bs --no_save_samples
done

echo "[$(date)] SMC/FK and TDS"
python -m scripts.run_img $COMMON --do_smc --do_smc_guidance --compute_budget 10000 100000 --exp_name m2_probe_smc

echo "[$(date)] offline evaluation"
python -m eval.eval_cifar --run_dir $RES/0_m2_probe_smc --data_path $DTS_DATA_DIR
echo "[$(date)] done"
