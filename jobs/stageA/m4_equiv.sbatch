#!/bin/bash
#SBATCH -J A-M4-equiv
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --exclude=d1n41a14g01
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# A-R010 unit checks of --smc_impl (jobs/stageA/m4_smc_equiv.py, deterministic CUDA algorithms):
# the upstream implementation against the upstream checkout UP (origin/main), and the default
# (fixed) implementation against the repro-m2 worktree M2; then run_img end to end in both
# worktrees (default implementation, small budget), comparing the saved evaluation samples.
set -euo pipefail
export DTS_CODE=$HOME/run/dts/wt-m4img
source $DTS_CODE/jobs/stageA/env.sh
UP=${UP:-$DTS_ROOT/wt-upstream}
M2=${M2:-$DTS_ROOT/wt-m2}
CLF=$DTS_BASE_DIR/cifar10/img/checkpoints/cifar10/cifar10_resnet_best.pt
S=$DTS_ROOT/scratch/m4_equiv/${SLURM_JOB_ID:-0}
mkdir -p $S
echo "upstream $(git -C $UP rev-parse HEAD)  repro-m2 $(git -C $M2 rev-parse HEAD)"
T="python jobs/stageA/m4_smc_equiv.py"
C="--classifier_path $CLF --device cuda --deterministic"
$T run --code_root $UP --side upstream --out $S/up.pt --base_dir $S/b_up $C > $S/up.log 2>&1 &
$T run --code_root $DTS_CODE --side fork --smc_impl upstream --out $S/fu.pt --base_dir $S/b_fu $C > $S/fu.log 2>&1 &
$T run --code_root $M2 --side fork --frozen_das_gamma 0.0142 --out $S/m2.pt --base_dir $S/b_m2 $C > $S/m2.log 2>&1 &
$T run --code_root $DTS_CODE --side fork --smc_impl fixed --frozen_das_gamma 0.0142 --out $S/ff.pt --base_dir $S/b_ff $C > $S/ff.log 2>&1 &
wait
tail -n 3 $S/*.log
echo "== upstream checkout vs --smc_impl upstream"
$T compare $S/up.pt $S/fu.pt || true
echo "== repro-m2 vs --smc_impl fixed"
$T compare $S/m2.pt $S/ff.pt || true

# run_img end to end, default implementation, in both worktrees
F="--dataset cifar10 --sample_classes 1 --compute_budget 5000 --population_size 200 --seed 0 --no_progress_bar --do_smc --do_smc_guidance --do_smc_guidance_tempering --adaptive_resampling true"
for w in m2:$M2 m4:$DTS_CODE; do
  n=${w%%:*}; d=${w#*:}
  (cd $d && python -m scripts.run_img $F --base_dir $S/cli_$n --exp_name e2e > $S/cli_$n.log 2>&1) &
done
wait
python - "$S" <<'PY'
import sys, glob, os, numpy as np, pandas as pd
S = sys.argv[1]
a = sorted(glob.glob(f'{S}/cli_m2/img/results_tree_viz/cifar10/*/metrics/samples_*.npz'))
ok = bool(a)
for f in a:
    g = f.replace('/cli_m2/', '/cli_m4/')
    same = os.path.exists(g) and np.array_equal(np.load(f)['images'], np.load(g)['images'])
    print('IDENTICAL' if same else 'DIFFERENT', os.path.basename(f)); ok &= same
drop = ['Time', 'PeakMemMB', 'BuildTime', 'SampleTime']
for f in sorted(glob.glob(f'{S}/cli_m2/img/results_tree_viz/cifar10/*/metrics/*_class_1_metrics.csv')):
    x = pd.read_csv(f).drop(columns=drop, errors='ignore'); y = pd.read_csv(f.replace('/cli_m2/', '/cli_m4/')).drop(columns=drop, errors='ignore')
    same = x.equals(y); print('IDENTICAL' if same else 'DIFFERENT', os.path.basename(f)); ok &= same
print('E2E ALL IDENTICAL' if ok else 'E2E MISMATCH')
PY
echo "[$(date)] done"
