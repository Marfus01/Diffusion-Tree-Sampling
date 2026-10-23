#!/bin/bash
#SBATCH -J A-M4-gmmprobe
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=00:12:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# A-R013 probe on the job's 8 CPU cores (d_x = 80, instance 1, seed 0):
#  - unit checks of tree_statistics and the SD CFG pairing;
#  - DTS 1e4 / 3e4 NFE samples of this checkout with and without --tree_stats, and of OLD_CODE
#    (the commit before tree_statistics), compared bitwise;
#  - wall-clock and peak RSS of DTS (alpha 0.3, 0.5, 0.8, --tree_stats) to 1e6 NFE and of FK / DAS
#    at 1e6 NFE, stopped at 9 minutes, for the extrapolation to 1e7.
set -uo pipefail
export DTS_CODE=${DTS_CODE:-$HOME/run/dts/wt-m4sd}
source $DTS_CODE/jobs/stageA/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
OUT=$DTS_BASE_DIR/toy/m4_gmm_probe_$SLURM_JOB_ID; L=logs/A-M4-gmmprobe_$SLURM_JOB_ID; mkdir -p $OUT $L
G="--dataset gmm --gmm_dim 80 --gmm_instance 1 --seed 0 --num_seeds 1 --device cpu --no_progress_bar"
echo "[$(date)] unit checks"
python -m pytest -q tests/test_sampling_2d.py -k tree_statistics tests/test_sampling_sd.py 2>&1 | tail -3

echo "[$(date)] bitwise check and timing"
SMALL="$G --compute_budget 10000 30000 --do_mcts_rollout"
(cd ${OLD_CODE:?} && python -m scripts.run_2d $SMALL --base_dir $OUT/old --exp_name bit > $DTS_CODE/$L/bit_old.log 2>&1) &
python -m scripts.run_2d $SMALL --base_dir $OUT/new --exp_name bit > $L/bit_new.log 2>&1 &
python -m scripts.run_2d $SMALL --base_dir $OUT/stats --exp_name bit --tree_stats > $L/bit_stats.log 2>&1 &
for a in 0.3 0.5 0.8; do
  timeout 540 /usr/bin/time -v python -m scripts.run_2d $G --compute_budget 10000 100000 300000 1000000 --do_mcts_rollout \
    --pw_alpha $a --pw_alpha_root $a --tree_stats --no_save_samples --base_dir $OUT/time --exp_name dts_a$a > $L/time_dts_a$a.log 2>&1 &
done
timeout 540 /usr/bin/time -v python -m scripts.run_2d $G --compute_budget 1000000 --do_smc --resampling_steps 20 80 20 \
  --adaptive_resampling true --no_save_samples --base_dir $OUT/time --exp_name fk > $L/time_fk.log 2>&1 &
timeout 540 /usr/bin/time -v python -m scripts.run_2d $G --compute_budget 1000000 --do_smc_guidance_tempering --guidance_scale 0.25 \
  --das_gamma 0.0081 --resampling_steps 20 80 20 --adaptive_resampling true --no_save_samples --base_dir $OUT/time --exp_name das > $L/time_das.log 2>&1 &
wait
python - $OUT <<'PY'
import sys, glob, numpy as np
out = sys.argv[1]
f = {k: glob.glob(f"{out}/{k}/**/samples_*.npz", recursive=True)[0] for k in ("old", "new", "stats")}
z = {k: np.load(v) for k, v in f.items()}
for k in ("new", "stats"):
    same = sorted(z[k].files) == sorted(z["old"].files) and all(np.array_equal(z[k][a], z["old"][a]) for a in z["old"].files)
    print(f"{k} vs old: {'IDENTICAL' if same else 'DIFFERENT'} ({len(z['old'].files)} arrays)")
PY
for f in $L/time_*.log; do
  echo "== $f"; grep -E "NFE [0-9]+ \(actual|tree: roots|Maximum resident|Elapsed \(wall" $f | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-260
  grep "Tree building progress" $f | tail -1
done
echo "[$(date)] done"
