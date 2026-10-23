#!/bin/bash
#SBATCH -J A-M4-abl
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --exclude=d1n41a14g01
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%j.err
# A-R010: several run_img processes side by side on one GPU, then the offline metrics of all of
# them with log p(c|x) under both classifiers. RUNS lists "cell:impl:clf:seed" items (cell from
# m4_cells.txt, impl fixed|upstream, clf own|hf), one process each running CLASSES at BUDGET; with
# an array, task a takes the a-th group of PER_TASK items. Run directories are
# subset_<PREFIX>_<cell>_<impl>_<clf>_s<seed>.
set -euo pipefail
export DTS_CODE=$HOME/run/dts/wt-m4img
source $DTS_CODE/jobs/stageA/env.sh
B=$DTS_BASE_DIR/cifar10
RES=$B/img/results_tree_viz/cifar10
HF=$DTS_ROOT/scratch/classifiers/edadaltocg_resnet18_cifar10/pytorch_model.bin
OWN=$B/img/checkpoints/cifar10/cifar10_resnet_best.pt
CELLS=${CELLS:-jobs/stageA/m4_cells.txt}
BUDGET=${BUDGET:-100000}
CLASSES=${CLASSES:-1 2 3}
PREFIX=${PREFIX:-m4a}
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

read -r -a items <<< "$RUNS"
per=${PER_TASK:-${#items[@]}}
a=${SLURM_ARRAY_TASK_ID:-0}
mine=("${items[@]:$((a * per)):$per}")
dirs=()
pids=()
for it in "${mine[@]}"; do
  IFS=':' read -r cell impl clf seed <<< "$it"
  flags=$(grep -v '^#' "$CELLS" | grep "^${cell}|" | head -1 | cut -d'|' -f2)
  [ -n "$flags" ] || { echo "unknown cell $cell"; exit 1; }
  case $clf in
    hf) cf="--classifier_model resnet18_cifar10_edadaltocg --classifier_path $HF" ;;
    own) cf="--classifier_model resnet --classifier_path $OWN" ;;
    *) echo "unknown classifier $clf"; exit 1 ;;
  esac
  name=${PREFIX}_${cell}_${impl}_${clf}_s${seed}
  echo "[$(date)] $name: $flags --smc_impl $impl $cf"
  python -m scripts.run_img --dataset cifar10 --sample_classes $CLASSES --compute_budget $BUDGET \
    --population_size 5000 --seed $seed --no_progress_bar --base_dir $B --exp_name $name $cf $flags --smc_impl $impl \
    < /dev/null > logs/${SLURM_JOB_NAME:-par}_${SLURM_JOB_ID:-0}_${name}.log 2>&1 &
  pids+=($!)
  if [ "$(echo $CLASSES | wc -w)" -gt 1 ]; then dirs+=("$RES/subset_$name"); else dirs+=("$RES/${CLASSES}_$name"); fi
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "[$(date)] run_img finished, status $status"
[ "${NO_EVAL:-0}" = 1 ] || python -m eval.eval_cifar --run_dir "${dirs[@]}" --data_path $DTS_DATA_DIR \
  --reward_classifier hf resnet18_cifar10_edadaltocg $HF --reward_classifier own resnet $OWN
echo "[$(date)] done"
exit $status
