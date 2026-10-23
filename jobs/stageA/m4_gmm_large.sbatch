#!/bin/bash
#SBATCH -J A-M4-gmmlarge
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=02:20:00
#SBATCH --exclude=d1n41a14g01
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# A-R013: analytic GMM d_x = 80, problem instances 1-3, sampler seeds $SEEDS (default 0), on the job's
# CPU cores, one single-threaded run_2d process per (instance, method, seed), all started together.
#  - DTS with alpha 0.3 (the frozen code value: 0.3 for nodes, 0.4 for the meta-root, i.e. the M1 DTS),
#    0.5 and 0.8 (nodes and meta-root), epsilon at the code value, snapshots 1e4 ... 1e7 NFE of one tree
#    with --tree_stats and --compact_tree; the 1e4-1e6 snapshots of alpha 0.3 repeat the M1 runs.
#  - SMC/FK and DAS with the frozen d_x = 80 configurations of M1 at 3e6 and 1e7 NFE (1e4-1e6 are in M1).
# Metrics of every snapshot are also in the logs, so a run stopped by the time limit keeps its smaller budgets.
set -uo pipefail
export DTS_CODE=${DTS_CODE:-$HOME/run/dts/wt-m4sd}
source $DTS_CODE/jobs/stageA/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
SEEDS=${SEEDS:-0}
L=logs/A-M4-gmmlarge_${SLURM_JOB_ID}; mkdir -p $L

declare -A CFG BUDGET
DTS_BUDGET="10000 30000 100000 300000 1000000 3000000 10000000"
CFG[DTS-a0.3]="--do_mcts_rollout --tree_stats --compact_tree"
CFG[DTS-a0.5]="--do_mcts_rollout --tree_stats --compact_tree --pw_alpha 0.5 --pw_alpha_root 0.5"
CFG[DTS-a0.8]="--do_mcts_rollout --tree_stats --compact_tree --pw_alpha 0.8 --pw_alpha_root 0.8"
CFG[FK]="--do_smc --resampling_steps 20 80 20 --adaptive_resampling true"
CFG[DAS]="--do_smc_guidance_tempering --guidance_scale 0.25 --das_gamma 0.0081 --resampling_steps 20 80 20 --adaptive_resampling true"
for m in DTS-a0.3 DTS-a0.5 DTS-a0.8; do BUDGET[$m]=$DTS_BUDGET; done
BUDGET[FK]="3000000 10000000"; BUDGET[DAS]="3000000 10000000"

CMDS=$L/commands.txt; : > $CMDS
for m in DTS-a0.3 DTS-a0.5 DTS-a0.8 FK DAS; do
  for i in 1 2 3; do
    for s in $SEEDS; do
      echo "/usr/bin/time -v python -m scripts.run_2d --dataset gmm --gmm_dim 80 --gmm_instance $i --seed $s --num_seeds 1 \
--compute_budget ${BUDGET[$m]} --device cpu --no_progress_bar --base_dir $DTS_BASE_DIR --exp_name m4_large_${m} ${CFG[$m]} \
> $L/i${i}_${m}_s${s}.log 2>&1 || echo FAILED i${i}_${m}_s${s}" >> $CMDS
    done
  done
done

echo "[$(date)] unit checks"
python -m pytest -q tests/test_sampling_2d.py 2>&1 | tail -2
echo "[$(date)] A-R013 $(wc -l < $CMDS) runs on ${SLURM_CPUS_ON_NODE:-8} cores"
xargs -P $(wc -l < $CMDS) -I{} bash -c "{}" < $CMDS
for f in $L/i*.log; do
  echo "== $f $(grep -E 'Maximum resident|Elapsed \(wall' $f | sed 's/^\s*//' | tr '\n' ' ')"
done
echo "[$(date)] done"
