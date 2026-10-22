"""A-R004: per-method hyperparameter calibration on the analytic GMM (CPU).

Each method's parameters are fixed one at a time (the others at the code defaults) by the mean
SW over `--num_seeds` seeds at `--nfe`; a scale that wins at the edge of its grid is extended by
one step. The best and runner-up configuration of each method's last stage are re-run at
`--recheck_nfe` (seed 0) to confirm the ranking. Every run_2d call is one method and one
configuration; results stay in the run_2d output tree, and summary.json / summary.md are written
to <base_dir>/toy/m1_calibration/.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

SCHEDULES = {'branch': (20, 80, 20), 'every': (0, 99, 1)}
SCALES = [0.5, 1.0, 2.0, 4.0]
SCALE_EXTENSION = {0.5: 0.25, 4.0: 8.0}
DAS_GAMMAS = [0.00703, 0.0081, 0.03]   # tempering factor reaches 1 at the last step / at 87% of 100 steps / early
DTS_ALPHAS = [None, 0.5, 0.8]          # None: hard-coded 0.3 (nodes) / 0.4 (meta-root)
DTS_EPS = [None, 0.0]                  # None: linear 0.1 -> 0.05
METHOD_FLAG = {'FK': '--do_smc', 'DPS-RG': '--do_guidance', 'TDS': '--do_smc_guidance',
               'DAS': '--do_smc_guidance_tempering', 'DTS': '--do_mcts_rollout'}


def cli_args(params):
    out = []
    if 'schedule' in params:
        out += ['--resampling_steps', *map(str, SCHEDULES[params['schedule']])]
    if 'adaptive' in params:
        out += ['--adaptive_resampling', str(params['adaptive']).lower()]
    if 'scale' in params:
        out += ['--guidance_scale', str(params['scale'])]
    if 'gamma' in params:
        out += ['--das_gamma', str(params['gamma'])]
    if params.get('alpha') is not None:
        out += ['--pw_alpha', str(params['alpha']), '--pw_alpha_root', str(params['alpha'])]
    if params.get('eps') is not None:
        out += ['--eps_explore', str(params['eps'])]
    return out


def tag(method, params):
    return method + '_' + '_'.join(f"{k}{v}" for k, v in sorted(params.items()))


class Runner:
    def __init__(self, args):
        self.args = args
        self.pool = ThreadPoolExecutor(max_workers=args.workers)
        self.log_dir = Path(args.base_dir) / 'toy' / 'm1_calibration' / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def submit(self, dim, method, params, nfe, num_seeds):
        name = f"calib_n{nfe}_{tag(method, params)}"
        cmd = [sys.executable, '-m', 'scripts.run_2d', '--dataset', 'gmm', '--gmm_dim', str(dim),
               '--gmm_instance', str(self.args.instance), '--seed', '0', '--num_seeds', str(num_seeds),
               '--compute_budget', str(nfe), '--device', 'cpu', '--no_progress_bar', '--no_save_samples',
               '--base_dir', self.args.base_dir, '--exp_name', name, METHOD_FLAG[method], *cli_args(params)]
        return self.pool.submit(self._run, dim, method, params, name, cmd)

    def _run(self, dim, method, params, name, cmd):
        log = self.log_dir / f"d{dim}_{name}.log"
        with open(log, 'w') as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        csv = (Path(self.args.base_dir) / 'toy' / 'results_tree_viz' / f"gmm_d{dim}_i{self.args.instance}"
               / name / 'metrics' / 'all_runs_raw_summary.csv')
        row = {'dim': dim, 'method': method, 'params': params, 'run': name, 'returncode': rc}
        if rc != 0 or not csv.exists():
            row.update(SW=float('inf'), SW_std=None, ModeTV=None, diverged_seeds=None)
            return row
        df = pd.read_csv(csv)
        diverged = int(df['Diverged'].fillna(0).sum()) if 'Diverged' in df else 0
        row.update(SW=float('inf') if diverged else float(df['SW'].mean()),
                   SW_std=float(df['SW'].std()) if len(df) > 1 else None,
                   ModeTV=float(df['ModeTV'].mean()) if 'ModeTV' in df else None,
                   diverged_seeds=diverged)
        return row

    def stage(self, dim, method, base, key, values, results):
        futures = [self.submit(dim, method, {**base, key: v}, self.args.nfe, self.args.num_seeds) for v in values]
        rows = [f.result() for f in futures]
        if key == 'scale':
            best = min(rows, key=lambda r: r['SW'])
            ext = SCALE_EXTENSION.get(best['params']['scale'])
            if ext is not None and all(r['params']['scale'] != ext for r in rows):
                rows.append(self.submit(dim, method, {**base, key: ext}, self.args.nfe, self.args.num_seeds).result())
        results.extend(rows)
        return sorted(rows, key=lambda r: r['SW'])


def calibrate_dim(runner, dim, results):
    """Returns {method: ranked rows of its last stage}."""
    final = {}

    def schedule_stage(method, base):
        """Resampling steps x adaptive resampling, jointly (4 configurations)."""
        rows = [f.result() for f in [runner.submit(dim, method, {**base, 'schedule': s, 'adaptive': a},
                                                   runner.args.nfe, runner.args.num_seeds)
                                     for s in SCHEDULES for a in (False, True)]]
        results.extend(rows)
        return sorted(rows, key=lambda r: r['SW'])

    def fk():
        final['FK'] = schedule_stage('FK', {})

    def dps():
        final['DPS-RG'] = runner.stage(dim, 'DPS-RG', {}, 'scale', SCALES, results)

    def tds_das():
        tds_scale = runner.stage(dim, 'TDS', {}, 'scale', SCALES, results)[0]['params']['scale']
        final['TDS'] = schedule_stage('TDS', {'scale': tds_scale})
        gamma = runner.stage(dim, 'DAS', {'scale': tds_scale}, 'gamma', DAS_GAMMAS, results)[0]['params']['gamma']
        final['DAS'] = schedule_stage('DAS', {'scale': tds_scale, 'gamma': gamma})

    def dts():
        alpha = runner.stage(dim, 'DTS', {}, 'alpha', DTS_ALPHAS, results)[0]['params']['alpha']
        final['DTS'] = runner.stage(dim, 'DTS', {'alpha': alpha}, 'eps', DTS_EPS, results)

    with ThreadPoolExecutor(max_workers=4) as chains:
        for f in [chains.submit(fn) for fn in (fk, dps, tds_das, dts)]:
            f.result()
    return final


def write_summary(out_dir, results, rechecks, chosen):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump({'calibration': results, 'recheck': rechecks, 'chosen': chosen}, f, indent=2, default=str)
    lines = ['| d_x | method | params | SW (mean) | SW std | ModeTV | diverged seeds |', '|---|---|---|---|---|---|---|']
    for r in sorted(results, key=lambda r: (r['dim'], r['method'], r['SW'])):
        lines.append(f"| {r['dim']} | {r['method']} | {r['params']} | {r['SW']:.4g} | {r['SW_std']} | {r['ModeTV']} | {r['diverged_seeds']} |")
    lines += ['', '| d_x | method | rank at calibration NFE | params | SW at recheck NFE |', '|---|---|---|---|---|']
    for r in rechecks:
        lines.append(f"| {r['dim']} | {r['method']} | {r['rank']} | {r['params']} | {r['SW']:.4g} |")
    (out_dir / 'summary.md').write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dims', nargs='+', type=int, default=[8, 80])
    ap.add_argument('--instance', type=int, default=0)
    ap.add_argument('--nfe', type=int, default=10_000)
    ap.add_argument('--num_seeds', type=int, default=3)
    ap.add_argument('--recheck_nfe', type=int, default=1_000_000)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--base_dir', default=os.environ.get('DTS_BASE_DIR'))
    args = ap.parse_args()
    runner = Runner(args)
    out_dir = Path(args.base_dir) / 'toy' / 'm1_calibration'

    results, finals = [], {}
    with ThreadPoolExecutor(max_workers=len(args.dims)) as dims:
        futures = {d: dims.submit(calibrate_dim, runner, d, results) for d in args.dims}
        finals = {d: f.result() for d, f in futures.items()}
    chosen = {d: {m: rows[0]['params'] for m, rows in fin.items()} for d, fin in finals.items()}
    write_summary(out_dir, results, [], chosen)

    recheck_futures = []
    for d, fin in finals.items():
        for m, rows in fin.items():
            for rank, r in enumerate(rows[:2], start=1):
                recheck_futures.append((d, m, rank, r['params'], runner.submit(d, m, r['params'], args.recheck_nfe, 1)))
    rechecks = []
    for d, m, rank, params, f in recheck_futures:
        row = f.result()
        rechecks.append({'dim': d, 'method': m, 'rank': rank, 'params': params, 'SW': row['SW'], 'run': row['run']})
    write_summary(out_dir, results, rechecks, chosen)
    print(f"summary: {out_dir / 'summary.md'}")


if __name__ == '__main__':
    main()
