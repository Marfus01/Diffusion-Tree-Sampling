#!/bin/bash
#SBATCH -J A-M0-sd
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# A-R003: SD-v1.5 + ImageReward, one prompt, 10^3 NFE: Best-of-N, FK, DTS, DTS*, then offline metrics.
set -euo pipefail
source jobs/stageA/env.sh

PROMPT="a photo of a black teddy bear"
COMMON="--prompts \"$PROMPT\" --sd_model_id stable-diffusion-v1-5/stable-diffusion-v1-5 --guiding_reward_fns ImageReward \
  --compute_budget 1000 --population_size 16 --seed 0 --no_progress_bar ${SD_EXTRA:-}"

echo "[$(date)] BoN, FK, DTS"
eval python -m scripts.run_sd $COMMON --do_bon --do_fkd --do_mcts_rollout --exp_name m0
echo "[$(date)] DTS*"
eval python -m scripts.run_sd $COMMON --do_mcts_rollout --maximize --exp_name m0_star

echo "[$(date)] offline metrics"
for RUN in user_prompts_m0 user_prompts_m0_star; do
  python -m eval.eval_sd --image_dir $DTS_BASE_DIR/sd/$RUN/images/ImageReward --prompts "$PROMPT" \
    --out $DTS_BASE_DIR/sd/$RUN/metrics/offline_eval.csv
done
echo "[$(date)] done"
