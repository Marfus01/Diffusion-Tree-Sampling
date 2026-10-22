#!/bin/bash
#SBATCH -J A-M0-2d
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# A-R001 unit checks, A-R002 2D gaussian_mixture check (code defaults, 1 seed),
# and a GMM d_x=80 throughput probe for the M1 budget.
set -euo pipefail
source jobs/stageA/env.sh

echo "[$(date)] A-R001 unit checks"
python -m pytest -q tests/

echo "[$(date)] A-R002 prior training"
CKPT=$DTS_BASE_DIR/toy/checkpoints/gaussian_mixture/diffusion_model.pt
python -m src.models.diffusion_2d --dataset gaussian_mixture --epochs 5000 --base_dir $DTS_BASE_DIR --no-progress-bar \
  || test -f $CKPT
python -m scripts.gen_2d_targets --dataset gaussian_mixture --base_dir $DTS_BASE_DIR

echo "[$(date)] A-R002 run_2d"
python -m scripts.run_2d --dataset gaussian_mixture --exp_name m0_check --seed 0 --num_seeds 1 \
  --do_mcts_rollout --do_smc --do_guidance --do_smc_guidance --do_smc_guidance_tempering \
  --compute_budget 1000 10000 100000 1000000 --no_progress_bar

echo "[$(date)] GMM d_x=80 throughput probe"
python -m scripts.run_2d --dataset gmm --gmm_dim 80 --gmm_instance 0 --exp_name m0_throughput --seed 0 --num_seeds 1 \
  --do_mcts_rollout --do_smc --do_smc_guidance --compute_budget 10000 100000 --population_size 2000 --no_progress_bar
echo "[$(date)] done"
