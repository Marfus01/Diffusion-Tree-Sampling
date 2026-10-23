#!/bin/bash
#SBATCH -J A-M4-cfgprobe
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# A-R011 unit check: UNet inputs of the first denoising step of FK-max (40 particles, batch 20) under
# upstream DTS (UPSTREAM_CODE), repro-m3 (M3_CODE) and this checkout with cfg_pairing fixed / upstream.
set -euo pipefail
export DTS_CODE=${DTS_CODE:-$HOME/run/dts/wt-m4sd}
source $DTS_CODE/jobs/stageA/env.sh
OUT=$DTS_BASE_DIR/sd/m4_cfg_probe_$SLURM_JOB_ID; mkdir -p $OUT
PROBE=$DTS_CODE/jobs/stageA/m4_cfg_probe.py
python $PROBE --code_dir ${UPSTREAM_CODE:?} --out $OUT/upstream_code.pt
python $PROBE --code_dir ${M3_CODE:?} --out $OUT/m3_code.pt
python $PROBE --code_dir $DTS_CODE --cfg_pairing upstream --out $OUT/ours_upstream.pt
python $PROBE --code_dir $DTS_CODE --cfg_pairing fixed --out $OUT/ours_fixed.pt
python $PROBE --code_dir $DTS_CODE --out $OUT/ours_default.pt
python $PROBE --compare $OUT/ours_upstream.pt $OUT/upstream_code.pt
python $PROBE --compare $OUT/ours_fixed.pt $OUT/m3_code.pt
python $PROBE --compare $OUT/ours_default.pt $OUT/m3_code.pt
python $PROBE --compare $OUT/ours_upstream.pt $OUT/ours_fixed.pt
echo "[$(date)] done"
