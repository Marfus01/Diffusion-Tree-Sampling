"""Tilted target of the discrete reverse chain on the analytic GMM, by self-normalized IS.

The methods sample from the 100-step chain p_theta,0 of `AnalyticGMMDiffusion` (eta = 1, the
chain of `run_2d.sample_unguided`), so their target is p_theta,0 e^{lambda r} / Z_theta, which
differs from the analytic pi*_0 by the discretization of the reverse kernel. Draws from p_theta,0
weighted by e^{lambda r} give, for each instance,

- E[r], the component masses under the pi*_0 MAP partition (`map_component`) and their mode-mass
  TV to the analytic weights w~_k, with delta-method standard errors and the TV expected from
  SNIS noise alone;
- the same quantities minus the analytic values (E_{pi*_0}[r] closed form, w~_k);
- SW between a systematic resample of the first `--n_keep` draws and exact pi*_0 draws, next to
  SW between two exact pi*_0 draws of the same size;
- the resampled reference set (npz) for later sample-based comparisons.

The ESS fraction under the analytic prior, (E_q[e^{lambda r}])^2 / E_q[e^{2 lambda r}], is computed
in closed form from `log_soft_value` at abar = 0 (where q_0|t = q_0).

Usage: python jobs/stageA/m4_gmm_disc_ref.py --params <d8_i1.json> ... --n 10000000 --out <dir>
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.models.gmm_analytic import GMMInverseProblem, AnalyticGMMDiffusion, sliced_wasserstein


def closed_form_ess_fraction(params, lmbda):
    d = len(params['A'][0])
    log_z = [float(GMMInverseProblem(**params, lmbda=l).log_soft_value(torch.zeros(1, d), torch.zeros(1)))
             for l in (lmbda, 2 * lmbda)]
    return math.exp(2 * log_z[0] - log_z[1])


def systematic_resample(w, n, g):
    """w [N] normalized weights -> n indices."""
    u = (torch.rand(1, generator=g, dtype=torch.float64) + torch.arange(n, dtype=torch.float64)) / n
    return torch.searchsorted(torch.cumsum(w, 0), u).clamp(max=len(w) - 1)


def run_instance(path, args):
    params = json.loads(Path(path).read_text())
    problem = GMMInverseProblem(**params, lmbda=args.lmbda)
    model = AnalyticGMMDiffusion(problem, device='cpu')
    K = len(problem.weights)
    torch.manual_seed(args.seed)

    logw, r_all, comp_all, keep = [], [], [], []
    t0 = time.time()
    done = 0
    while done < args.n:
        b = min(args.batch, args.n - done)
        x = model.sample_initial_noise(b)
        for t in range(model.n_timesteps - 1, -1, -1):
            x, _ = model.p_sample(x, t)
        r = problem.log_likelihood(x).double()
        logw.append(args.lmbda * r)
        r_all.append(r)
        comp_all.append(torch.from_numpy(problem.map_component(x.numpy())).to(torch.int16))
        if done < args.n_keep:
            keep.append(x[:max(0, min(b, args.n_keep - done))].clone())
        done += b
        if done == b or done % (10 * args.batch) == 0:
            print(f"{Path(path).stem}: {done}/{args.n} draws, {done / (time.time() - t0):.0f} draws/s", flush=True)

    logw, r, comp = torch.cat(logw), torch.cat(r_all), torch.cat(comp_all).long()
    w = torch.softmax(logw, 0)                                                  # SNIS weights
    ess = float(1.0 / (w ** 2).sum())

    def snis(f):
        """Self-normalized estimate of E[f] and its delta-method standard error."""
        m = float((w * f).sum())
        return m, float(torch.sqrt((w ** 2 * (f - m) ** 2).sum()))

    er, er_se = snis(r)
    # E[1{comp = k}] and its standard error, without the [N, K] indicator matrix:
    # sum_i w_i^2 (1{c_i = k} - m_k)^2 = S2_k (1 - 2 m_k) + m_k^2 S2, S2_k = sum_{c_i = k} w_i^2.
    mass = torch.zeros(K, dtype=torch.float64).index_add_(0, comp, w)
    s2_k = torch.zeros(K, dtype=torch.float64).index_add_(0, comp, w ** 2)
    mass_se = torch.sqrt((s2_k * (1 - 2 * mass) + mass ** 2 * (w ** 2).sum()).clamp(min=0))
    w_tilde = torch.as_tensor(problem.posterior_weights)

    x_keep = torch.cat(keep)
    w_keep = torch.softmax(logw[:len(x_keep)], 0)
    g = torch.Generator().manual_seed(args.seed + 1)
    ref = x_keep[systematic_resample(w_keep, args.n_ref, g)].numpy()
    exact_a = problem.sample_target(args.n_ref, seed=args.seed + 2)
    exact_b = problem.sample_target(args.n_ref, seed=args.seed + 3)

    out = {
        'params_file': str(path), 'dim': problem.dim, 'lmbda': args.lmbda, 'n': args.n, 'seed': args.seed,
        'seconds': time.time() - t0,
        'ess': ess, 'ess_fraction': ess / args.n,
        'ess_fraction_closed_form_q0': closed_form_ess_fraction(params, args.lmbda),
        'ess_keep': float(1.0 / (w_keep ** 2).sum()),
        'mean_reward': er, 'mean_reward_se': er_se,
        'mean_reward_analytic': problem.target_mean_reward(),
        'mean_reward_minus_analytic': er - problem.target_mean_reward(),
        'mass': mass.tolist(), 'mass_se': mass_se.tolist(), 'analytic_weights': w_tilde.tolist(),
        'mode_tv_to_analytic': 0.5 * float((mass - w_tilde).abs().sum()),
        # expected mode-mass TV from SNIS noise alone: 0.5 sum_k E|N(0, se_k^2)|
        'mode_tv_noise_floor': 0.5 * math.sqrt(2 / math.pi) * float(mass_se.sum()),
        'n_ref': args.n_ref,
        'sw_ref_to_exact': sliced_wasserstein(ref, exact_a, n_projections=args.sw_projections),
        'sw_exact_to_exact': sliced_wasserstein(exact_b, exact_a, n_projections=args.sw_projections),
    }
    np.savez_compressed(Path(args.out) / f"{Path(path).stem}_ref.npz", ref=ref)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', nargs='+', required=True, help='instance files (GMMInverseProblem.instance_params())')
    ap.add_argument('--n', type=int, default=10_000_000, help='draws from p_theta,0 per instance')
    ap.add_argument('--batch', type=int, default=20_000)
    ap.add_argument('--n_keep', type=int, default=1_000_000, help='leading draws whose states are kept for resampling')
    ap.add_argument('--n_ref', type=int, default=10_000, help='size of the resampled reference set')
    ap.add_argument('--lmbda', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sw_projections', type=int, default=10_000)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for path in args.params:
        res = run_instance(path, args)
        (Path(args.out) / f"{Path(path).stem}.json").write_text(json.dumps(res, indent=2))
        print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}), flush=True)


if __name__ == '__main__':
    main()
