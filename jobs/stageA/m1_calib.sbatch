#!/bin/bash
#SBATCH -J A-M1-calib
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# A-R004: per-method calibration on the analytic GMM, d_x in {8, 80}, instance 0, 1e5 NFE,
# 5 seeds; best and runner-up rechecked at 1e6 NFE over 3 seeds, the lower recheck SW is chosen.
# CPU only: batch-1 tree search is faster on the job's 8 CPU cores than on the GPU, one
# single-threaded run_2d process per core.
set -euo pipefail
source $HOME/run/dts/Diffusion-Tree-Sampling/jobs/stageA/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""

echo "[$(date)] unit checks"
python -m pytest -q tests/

echo "[$(date)] A-R004 calibration"
python jobs/stageA/m1_calibrate.py --dims 8 80 --instance 0 --nfe 100000 --num_seeds 5 \
  --recheck_nfe 1000000 --recheck_seeds 3 --workers ${SLURM_CPUS_ON_NODE:-8} --base_dir $DTS_BASE_DIR
echo "[$(date)] done"
