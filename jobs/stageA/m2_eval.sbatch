#!/bin/bash
#SBATCH -J A-M2-eval
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --exclude=d1n41a14g01
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# Offline metrics (eval/eval_cifar.py) for the run directories named in RUNS (relative to the
# CIFAR-10 results directory), e.g. sbatch --export=ALL,RUNS="5_m2c_DTS__bs1_s0 5_m2c_TDS__s1_s0" ...
set -euo pipefail
export DTS_CODE=$HOME/run/dts/wt-m2
source $DTS_CODE/jobs/stageA/env.sh
RES=$DTS_BASE_DIR/cifar10/img/results_tree_viz/cifar10
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
dirs=()
for r in $RUNS; do dirs+=("$RES/$r"); done
python -m eval.eval_cifar --run_dir "${dirs[@]}" --data_path $DTS_DATA_DIR
echo "[$(date)] done"
