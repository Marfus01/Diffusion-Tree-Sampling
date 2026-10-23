#!/bin/bash
#SBATCH -J A-M4-gmmprobe2
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=00:08:00
#SBATCH --exclude=d1n41a14g01
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# A-R013 second probe on the job's CPU cores (d_x = 80, instance 1, seed 0, DTS with the frozen
# alpha 0.3 / 0.4 of M1):
#  - tree_statistics and compact-tree unit checks with full tracebacks, alone and together with
#    tests/test_sampling_sd.py, and the CPU model of the node;
#  - DTS to 3e5 NFE with --tree_stats, with and without --compact_tree: samples, metrics and tree
#    statistics compared exactly, wall-clock and peak RSS of each.
set -uo pipefail
export DTS_CODE=${DTS_CODE:-$HOME/run/dts/wt-m4sd}
source $DTS_CODE/jobs/stageA/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
OUT=$DTS_BASE_DIR/toy/m4_gmm_probe2_$SLURM_JOB_ID; L=logs/A-M4-gmmprobe2_$SLURM_JOB_ID; mkdir -p $OUT $L
lscpu | grep -E "Model name"
G="--dataset gmm --gmm_dim 80 --gmm_instance 1 --seed 0 --num_seeds 1 --device cpu --no_progress_bar --do_mcts_rollout \
--tree_stats --compute_budget 10000 30000 100000 300000"
/usr/bin/time -v python -m scripts.run_2d $G --base_dir $OUT/full --exp_name cmp > $L/full.log 2>&1 &
/usr/bin/time -v python -m scripts.run_2d $G --compact_tree --base_dir $OUT/compact --exp_name cmp > $L/compact.log 2>&1 &
python -m pytest -vv --tb=long tests/test_sampling_2d.py -k "tree_statistics or compact" > $L/pytest_2d.log 2>&1
tail -4 $L/pytest_2d.log
python -m pytest -vv --tb=long tests/test_sampling_2d.py -k tree_statistics tests/test_sampling_sd.py > $L/pytest_with_sd.log 2>&1
tail -4 $L/pytest_with_sd.log
wait
python - $OUT <<'PY'
import sys, glob, json, numpy as np, pandas as pd
out = sys.argv[1]
def one(v, pat):
    return glob.glob(f"{out}/{v}/**/{pat}", recursive=True)[0]
a, b = np.load(one("full", "samples_*.npz")), np.load(one("compact", "samples_*.npz"))
same = sorted(a.files) == sorted(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
print(f"samples compact vs full: {'IDENTICAL' if same else 'DIFFERENT'} ({len(a.files)} arrays)")
drop = lambda r: {k: v for k, v in r.items() if k not in ("BuildTime", "StatsTime")}
sa, sb = (json.load(open(one(v, "tree_stats_*.json"))) for v in ("full", "compact"))
print("tree stats compact vs full:", "IDENTICAL" if [drop(r) for r in sa] == [drop(r) for r in sb] else "DIFFERENT")
ca, cb = (pd.read_csv(one(v, "sample_metrics_*.csv")) for v in ("full", "compact"))
cols = [c for c in ca.columns if c not in ("BuildTime", "SampleTime", "Time", "PeakMemMB")]
print("metrics compact vs full:", "IDENTICAL" if ca[cols].equals(cb[cols]) else "DIFFERENT")
PY
for f in $L/full.log $L/compact.log; do
  echo "== $f"; grep -E "NFE [0-9]+ \(actual|Maximum resident|Elapsed \(wall" $f | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-200
done
echo "[$(date)] done"
