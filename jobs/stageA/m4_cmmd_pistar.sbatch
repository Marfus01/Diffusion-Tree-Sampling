#!/bin/bash
#SBATCH -J A-M4-cmmdpi
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=02:30:00
#SBATCH --exclude=d1n41a14g01
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# CMMD of the M2 grid, IR and whole-pool pi*_0 sample sets to the pi*_0 reference from pool
# chunks 3-9 (jobs/stageA/m4_cmmd_pistar.py).
set -euo pipefail
export DTS_CODE=$HOME/run/dts/wt-m4img
source $DTS_CODE/jobs/stageA/env.sh
RES=$DTS_BASE_DIR/cifar10/img/results_tree_viz/cifar10
OUT=$DTS_BASE_DIR/cifar10/m4_cmmd_pistar
mkdir -p $OUT
python -m jobs.stageA.m4_cmmd_pistar --pool_dir $DTS_BASE_DIR/cifar10/m4_prior_pool \
    --cache_dir $OUT/clip_cache --out $OUT/cmmd_to_pistar.csv \
    $RES/*_m2g_* $RES/m4ir_* $RES/m4pi_*
echo "[$(date)] done"
