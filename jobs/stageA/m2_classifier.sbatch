#!/bin/bash
#SBATCH -J A-M2-classifier
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# Guiding ResNet-18 (trained when the checkpoint is missing) with its per-class test accuracy,
# then the offline metrics of the probe's SMC/FK and TDS samples.
set -euo pipefail
export DTS_CODE=$HOME/run/dts/wt-m2
source $DTS_CODE/jobs/stageA/env.sh
B=$DTS_BASE_DIR/cifar10
python -m jobs.stageA.m2_classifier --base_dir $B --out $B/m2_classifier/per_class_accuracy.json
echo "[$(date)] offline evaluation"
python -m eval.eval_cifar --run_dir $B/img/results_tree_viz/cifar10/0_m2_probe_smc --data_path $DTS_DATA_DIR
echo "[$(date)] done"
