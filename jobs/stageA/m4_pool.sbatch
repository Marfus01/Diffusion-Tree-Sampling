#!/bin/bash
#SBATCH -J A-M4-pool
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=02:30:00
#SBATCH --exclude=d1n41a14g01
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# A-R012: prior pool (2e5 samples, 1e7 NFE), pi*_0 reference and prior importance resampling
# (jobs/stageA/m4_prior_pool.py), then the offline metrics of the draws with log p(c|x) under both
# classifiers. STAGE=draw skips generation (finished chunks are skipped anyway).
set -euo pipefail
export DTS_CODE=$HOME/run/dts/wt-m4img
source $DTS_CODE/jobs/stageA/env.sh
POOL=$DTS_BASE_DIR/cifar10/m4_prior_pool
RES=$DTS_BASE_DIR/cifar10/img/results_tree_viz/cifar10
HF=$DTS_ROOT/scratch/classifiers/edadaltocg_resnet18_cifar10/pytorch_model.bin
OWN=$DTS_BASE_DIR/cifar10/img/checkpoints/cifar10/cifar10_resnet_best.pt
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
[ "${STAGE:-all}" = draw ] || python jobs/stageA/m4_prior_pool.py generate --pool_dir $POOL \
  --classifier hf resnet18_cifar10_edadaltocg $HF --classifier own resnet $OWN
python jobs/stageA/m4_prior_pool.py draw --pool_dir $POOL --results_dir $RES
dirs=("$RES/m4pi_PiStar_hf" "$RES/m4pi_PiStar_own" "$RES/m4ir_IR_s0" "$RES/m4ir_IR_s1" "$RES/m4ir_IR_s2")
python -m eval.eval_cifar --run_dir "${dirs[@]}" --data_path $DTS_DATA_DIR \
  --reward_classifier hf resnet18_cifar10_edadaltocg $HF --reward_classifier own resnet $OWN
echo "[$(date)] done"
