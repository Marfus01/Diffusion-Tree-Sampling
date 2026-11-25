#!/bin/bash
#SBATCH -J A-M2-classifier
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# Per-class test accuracy and mean log p of the guiding classifier, the pretrained
# edadaltocg/resnet18_cifar10.
set -euo pipefail
export DTS_CODE=$HOME/run/dts/wt-m2
source $DTS_CODE/jobs/stageA/env.sh
B=$DTS_BASE_DIR/cifar10
python -m jobs.stageA.m2_classifier --base_dir $B --out $B/m2_classifier/edadaltocg_resnet18_cifar10.json \
  --classifier_model resnet18_cifar10_edadaltocg \
  --classifier_path $DTS_ROOT/scratch/classifiers/edadaltocg_resnet18_cifar10/pytorch_model.bin
echo "[$(date)] done"
