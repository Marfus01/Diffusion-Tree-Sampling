#!/bin/bash
#SBATCH -J A-M2-calib
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
# A-R006 calibration on the calibration class. CONFIGS is a file of lines "label|run_img flags[|seeds]";
# array task a runs the lines with (line index mod array size) == a, for every seed of the line
# (default SEEDS), each followed by the offline metrics of eval/eval_cifar.py.
#   sbatch --array=0-3 --export=ALL,CONFIGS=jobs/stageA/m2_calib_l1.txt,CLASS=3,NFE="100000",SEEDS="0 1" jobs/stageA/m2_calib.sbatch
set -euo pipefail
export DTS_CODE=$HOME/run/dts/wt-m2
source $DTS_CODE/jobs/stageA/env.sh
B=$DTS_BASE_DIR/cifar10
RES=$B/img/results_tree_viz/cifar10
CLF="--classifier_model resnet18_cifar10_edadaltocg --classifier_path $DTS_ROOT/scratch/classifiers/edadaltocg_resnet18_cifar10/pytorch_model.bin"
N_TASKS=${SLURM_ARRAY_TASK_COUNT:-1}
TASK=${SLURM_ARRAY_TASK_ID:-0}

i=0
while IFS='|' read -r label flags line_seeds; do
  [ -z "$label" ] && continue
  case "$label" in \#*) continue ;; esac
  if [ $((i % N_TASKS)) -eq "$TASK" ]; then
    for s in ${line_seeds:-$SEEDS}; do
      name=m2c_${label}_s${s}
      echo "[$(date)] $name: $flags"
      python -m scripts.run_img --dataset cifar10 --sample_classes $CLASS --compute_budget $NFE \
        --population_size 5000 --seed $s --no_progress_bar --base_dir $B --exp_name $name $CLF $flags < /dev/null \
        || echo "[$(date)] $name FAILED"
      if ls $RES/${CLASS}_$name/metrics/samples_*.npz >/dev/null 2>&1; then
        python -m eval.eval_cifar --run_dir $RES/${CLASS}_$name --data_path $DTS_DATA_DIR < /dev/null || echo "[$(date)] $name eval FAILED"
      fi
    done
  fi
  i=$((i + 1))
done < "$CONFIGS"
echo "[$(date)] task $TASK done"
