#!/bin/bash
#SBATCH -J A-M2-grid
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --exclude=d1n41a14g01
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
# A-R007: evaluation classes x methods at NFE 1e4, 3e4, 1e5, 3e5, 1e6 with the frozen configurations.
# FROZEN is a file of lines "method|run_img flags"; array task a runs line a // (number of CLASSES)
# on class CLASSES[a mod number of CLASSES], then the offline metrics of eval/eval_cifar.py.
#   sbatch --array=0-2 --export=ALL,FROZEN=jobs/stageA/m2_frozen_dts.txt,CLASSES="1 2 3",SEED=0 jobs/stageA/m2_grid.sbatch
set -euo pipefail
export DTS_CODE=$HOME/run/dts/wt-m2
source $DTS_CODE/jobs/stageA/env.sh
B=$DTS_BASE_DIR/cifar10
RES=$B/img/results_tree_viz/cifar10
CLF="--classifier_model resnet18_cifar10_edadaltocg --classifier_path $DTS_ROOT/scratch/classifiers/edadaltocg_resnet18_cifar10/pytorch_model.bin"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

read -r -a classes <<< "$CLASSES"
a=${SLURM_ARRAY_TASK_ID:-0}
line=$((a / ${#classes[@]}))
class=${classes[$((a % ${#classes[@]}))]}
IFS='|' read -r method flags <<< "$(grep -v '^#' "$FROZEN" | grep . | sed -n "$((line + 1))p")"
name=m2g_${method}_s${SEED}
echo "[$(date)] class $class $name: $flags"
python -m scripts.run_img --dataset cifar10 --sample_classes $class \
  --compute_budget 10000 30000 100000 300000 1000000 --population_size 5000 --seed $SEED \
  --no_progress_bar --base_dir $B --exp_name $name $CLF $flags
python -m eval.eval_cifar --run_dir $RES/${class}_$name --data_path $DTS_DATA_DIR
echo "[$(date)] done"
