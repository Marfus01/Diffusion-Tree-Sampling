#!/bin/bash
#SBATCH -J A-M1-calib-dtse1
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# A-R004, DTS variant with epsilon = 1 (uniform child selection during tree construction): alpha
# calibrated on the analytic GMM, d_x in {8, 80}, instance 0, same protocol as m1_calib.sbatch (CPU only).
set -euo pipefail
source $HOME/run/dts/Diffusion-Tree-Sampling/jobs/stageA/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""

echo "[$(date)] A-R004 DTS epsilon = 1, alpha"
python jobs/stageA/m1_calibrate.py --dims 8 80 --instance 0 --nfe 100000 --num_seeds 5 \
  --recheck_nfe 1000000 --recheck_seeds 3 --workers ${SLURM_CPUS_ON_NODE:-8} --base_dir $DTS_BASE_DIR \
  --chains DTS --dts_fixed_eps 1.0 --name m1_calibration_dts_eps1 --run_prefix calibdtse1
echo "[$(date)] done"
