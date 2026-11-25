#!/bin/bash
#SBATCH -J A-M2-par
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --exclude=d1n41a14g01
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
# Several run_img processes side by side on one GPU (tree batch 1 DTS is bound by kernel launches
# and Python, so one process leaves the GPU mostly idle), then the offline metrics of all of them.
# FROZEN is a file of lines "method|run_img flags"; RUNS lists "method:class:seed" items, one process
# each; with an array, task a takes the a-th group of PER_TASK items. Run directories are
# <class>_<PREFIX>_<method>_s<seed> (PREFIX default m2g, as in m2_grid.sbatch).
#   sbatch --array=0-2 --export=ALL,FROZEN=jobs/stageA/m2_frozen_dts.txt,RUNS="DTS:1:1 DTS:1:2 DTS:2:1 DTS:2:2 DTS:3:1 DTS:3:2",PER_TASK=2 jobs/stageA/m2_par.sbatch
set -euo pipefail
export DTS_CODE=$HOME/run/dts/wt-m2
source $DTS_CODE/jobs/stageA/env.sh
B=$DTS_BASE_DIR/cifar10
RES=$B/img/results_tree_viz/cifar10
CLF="--classifier_model resnet18_cifar10_edadaltocg --classifier_path $DTS_ROOT/scratch/classifiers/edadaltocg_resnet18_cifar10/pytorch_model.bin"
BUDGET=${BUDGET:-10000 30000 100000 300000 1000000}
PREFIX=${PREFIX:-m2g}
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

read -r -a items <<< "$RUNS"
per=${PER_TASK:-${#items[@]}}
a=${SLURM_ARRAY_TASK_ID:-0}
mine=("${items[@]:$((a * per)):$per}")
dirs=()
pids=()
for it in "${mine[@]}"; do
  IFS=':' read -r method class seed <<< "$it"
  flags=$(grep -v '^#' "$FROZEN" | grep "^${method}|" | head -1 | cut -d'|' -f2)
  name=${PREFIX}_${method}_s${seed}
  echo "[$(date)] class $class $name: $flags"
  python -m scripts.run_img --dataset cifar10 --sample_classes $class --compute_budget $BUDGET \
    --population_size 5000 --seed $seed --no_progress_bar --base_dir $B --exp_name $name $CLF $flags \
    < /dev/null > logs/${SLURM_JOB_NAME:-par}_${SLURM_JOB_ID:-0}_${class}_${name}.log 2>&1 &
  pids+=($!)
  dirs+=("$RES/${class}_$name")
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "[$(date)] run_img finished, status $status"
[ "${NO_EVAL:-0}" = 1 ] || python -m eval.eval_cifar --run_dir "${dirs[@]}" --data_path $DTS_DATA_DIR
echo "[$(date)] done"
exit $status
