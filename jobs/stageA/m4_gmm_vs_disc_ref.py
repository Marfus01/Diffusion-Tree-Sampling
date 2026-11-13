"""M1 GMM metrics of every saved run against both references: the analytic pi*_0 and the SNIS
reference of the discrete chain's tilted target (m4_gmm_disc_ref.py).

For each samples_<dataset>_seed<s>.npz under <results>/gmm_d<d>_i<i>/<run>/images and each NFE:
SW to the run's own exact pi*_0 draw ('target', 2000 samples) and to block s of the resampled
SNIS reference (2000 samples); mode-mass TV to w~_k and to the SNIS masses; reward gap to
E_{pi*_0}[r] and to the SNIS E[r]. Reference rows give the same quantities for an exact pi*_0
draw of the same size, so each column has its sampling floor.

Usage: python -m jobs.stageA.m4_gmm_vs_disc_ref --results <dir> --instances <dir> --disc_ref <dir>
           --out <csv> --dims 8 80 --instance_ids 1 2 3
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.gmm_analytic import GMMInverseProblem, sliced_wasserstein

KEY = re.compile(r'(?P<method>.+)__N(?P<nfe>\d+)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True, help='results_tree_viz directory of the toy runs')
    ap.add_argument('--instances', required=True, help='directory of d<d>_i<i>.json instance files')
    ap.add_argument('--disc_ref', required=True, help='output directory of m4_gmm_disc_ref.py')
    ap.add_argument('--dims', type=int, nargs='+', default=[8, 80])
    ap.add_argument('--instance_ids', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--sw_projections', type=int, default=10_000)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    rows = []
    for d in a.dims:
        for i in a.instance_ids:
            name = f'd{d}_i{i}'
            problem = GMMInverseProblem(**json.loads((Path(a.instances) / f'{name}.json').read_text()))
            disc = json.loads((Path(a.disc_ref) / f'{name}.json').read_text())
            ref = np.load(Path(a.disc_ref) / f'{name}_ref.npz')['ref']
            w_tilde, m_disc = problem.posterior_weights, np.asarray(disc['mass'])
            er_exact, er_disc = problem.target_mean_reward(), disc['mean_reward']

            def metrics(x, target, block):
                comp = np.bincount(problem.map_component(x), minlength=len(w_tilde)) / len(x)
                r = float(problem.log_likelihood(torch.from_numpy(np.asarray(x, dtype=np.float32))).double().mean())
                return {'SW_exact': sliced_wasserstein(x, target, n_projections=a.sw_projections),
                        'SW_disc': sliced_wasserstein(x, block, n_projections=a.sw_projections),
                        'ModeTV_exact': 0.5 * float(np.abs(comp - w_tilde).sum()),
                        'ModeTV_disc': 0.5 * float(np.abs(comp - m_disc).sum()),
                        'RewardGap_exact': r - er_exact, 'RewardGap_disc': r - er_disc}

            for f in sorted(Path(a.results).glob(f'gmm_{name}/*/images/samples_gmm_{name}_seed*.npz')):
                seed = int(f.stem.rsplit('seed', 1)[1])
                z = np.load(f)
                n = len(z['target'])
                block = ref[seed * n:(seed + 1) * n]
                assert len(block) == n, f'reference has {len(ref)} samples, seed {seed} needs block of {n}'
                base = {'dim': d, 'instance': i, 'run': f.parent.parent.name, 'seed': seed}
                if not any(r['run'] == 'reference' and r['dim'] == d and r['instance'] == i and r['seed'] == seed
                           for r in rows):
                    exact = problem.sample_target(n, seed=50_000 + seed)
                    rows.append({**base, 'run': 'reference', 'method': 'exact-pi*', 'N': None,
                                 **metrics(exact, z['target'], block)})
                for k in z.files:
                    m = KEY.fullmatch(k)
                    if m:
                        rows.append({**base, 'method': m['method'], 'N': int(m['nfe']),
                                     **metrics(z[k], z['target'], block)})
            print(f'{name}: {sum(r["dim"] == d and r["instance"] == i for r in rows)} rows', flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == '__main__':
    main()
