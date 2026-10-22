#!/bin/bash
#SBATCH -J A-M1-grid
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# A-R005: analytic GMM, d_x in {8, 80}, problem instances 1-3 (instance 0 is the calibration
# unit), sampler seeds $SEEDS (default 0), 1e4-1e6 NFE, with the configurations frozen in
# calibration.md. One single-threaded run_2d process per (d_x, instance, method, seed) on the
# job's CPU cores; the per-seed sample_metrics_*.csv are the results (all_runs_raw_summary.csv is
# rewritten by whichever seed of a directory finishes last).
set -euo pipefail
source $HOME/run/dts/Diffusion-Tree-Sampling/jobs/stageA/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
SEEDS=${SEEDS:-0}

declare -A CFG
CFG[8,DTS]="--do_mcts_rollout --pw_alpha 0.8 --pw_alpha_root 0.8"
CFG[8,FK]="--do_smc --resampling_steps 20 80 20 --adaptive_resampling false"
CFG[8,DPS-RG]="--do_guidance --guidance_scale 2.0"
CFG[8,TDS]="--do_smc_guidance --guidance_scale 0.25 --resampling_steps 20 80 20 --adaptive_resampling false"
CFG[8,DAS]="--do_smc_guidance_tempering --guidance_scale 0.25 --das_gamma 0.00703 --resampling_steps 20 80 20 --adaptive_resampling true"
CFG[80,DTS]="--do_mcts_rollout"
CFG[80,FK]="--do_smc --resampling_steps 20 80 20 --adaptive_resampling true"
CFG[80,DPS-RG]="--do_guidance --guidance_scale 0.25"
CFG[80,TDS]="--do_smc_guidance --guidance_scale 0.25 --resampling_steps 20 80 20 --adaptive_resampling true"
CFG[80,DAS]="--do_smc_guidance_tempering --guidance_scale 0.25 --das_gamma 0.0081 --resampling_steps 20 80 20 --adaptive_resampling true"

mkdir -p logs/A-M1-grid_${SLURM_JOB_ID}
CMDS=logs/A-M1-grid_${SLURM_JOB_ID}/commands.txt
: > $CMDS
for m in DTS FK DPS-RG TDS DAS; do          # longest (DTS) first for load balance
  for d in 8 80; do
    for i in 1 2 3; do
      for s in $SEEDS; do
        echo "python -m scripts.run_2d --dataset gmm --gmm_dim $d --gmm_instance $i --seed $s --num_seeds 1 \
--compute_budget 10000 30000 100000 300000 1000000 --device cpu --no_progress_bar --base_dir $DTS_BASE_DIR \
--exp_name m1_grid_${m} ${CFG[$d,$m]} > logs/A-M1-grid_${SLURM_JOB_ID}/d${d}_i${i}_${m}_s${s}.log 2>&1 \
|| echo FAILED d${d}_i${i}_${m}_s${s}" >> $CMDS
      done
    done
  done
done

echo "[$(date)] unit checks"
python -m pytest -q tests/

echo "[$(date)] A-R005 $(wc -l < $CMDS) runs on ${SLURM_CPUS_ON_NODE:-8} cores"
xargs -P ${SLURM_CPUS_ON_NODE:-8} -I{} bash -c "{}" < $CMDS
echo "[$(date)] done"
