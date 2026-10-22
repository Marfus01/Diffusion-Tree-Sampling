"""SMC weights, NFE accounting and asymptotic exactness on the analytic GMM (CPU)."""
import math

import numpy as np
import pytest
import torch

from src.core.smc_weights import SMCWeights, gaussian_step_log_ratio
from src.core.sampling.smc_2d import SMCDiffusion2DSampler
from src.core.sampling.mcts_parallel import MCTSDiffusionSampler
from src.models.gmm_analytic import GMMInverseProblem, AnalyticGMMDiffusion, sliced_wasserstein

FKD = {'potential_type': 'diff', 'lmbda': 1.0, 'adaptive_resampling': False, 'resample_frequency': 20,
       'resampling_t_start': 20, 'resampling_t_end': 80, 'reward_min_value': 0.0}


def test_gaussian_step_log_ratio():
    torch.manual_seed(0)
    x, m, d = torch.randn(5, 3), torch.randn(5, 3), torch.randn(5, 3)
    v = torch.rand(5, 1) + 0.1
    ref = (torch.distributions.Normal(m, v.sqrt()).log_prob(x) - torch.distributions.Normal(m + d, v.sqrt()).log_prob(x)).sum(1)
    torch.testing.assert_close(gaussian_step_log_ratio(x, m, d, v), ref)
    assert torch.all(gaussian_step_log_ratio(x, m, torch.zeros_like(d), torch.zeros(5, 1)) == 0)


def test_weights_persist_across_skipped_resampling():
    w = SMCWeights(4, lmbda=2.0, adaptive_resampling=True, resampling_schedule=[], device='cpu', ess_threshold=0.0)
    r1, r2, rf = torch.tensor([0., 1., 2., 3.]), torch.tensor([1., 1., 0., 0.]), torch.tensor([5., 0., 1., 2.])
    lr = torch.tensor([0.1, -0.2, 0.3, 0.0])
    assert w.potential_step(20, r1) is None
    w.add_log_ratio(lr)
    assert w.potential_step(40, r2) is None
    w.add_log_ratio(lr)
    w.final_step(rf)
    torch.testing.assert_close(w.log_w, 2.0 * rf + 2 * lr)


def test_resampling_resets_weights_and_baseline():
    torch.manual_seed(0)
    w = SMCWeights(1000, lmbda=1.0, adaptive_resampling=False, resampling_schedule=[20], device='cpu')
    r = torch.linspace(-3, 3, 1000)
    idx = w.potential_step(20, r)
    assert idx is not None and torch.all(w.log_w == 0)
    torch.testing.assert_close(w.prev_reward, r[idx])


@pytest.fixture(scope="module")
def gmm2():
    # instance with a multimodal posterior that is far from the prior
    problem = GMMInverseProblem(dim=2, instance_seed=8)
    return problem, AnalyticGMMDiffusion(problem, device='cpu')


def run_smc(model, problem, K, use_smc, use_guidance, use_tempering=False, adaptive=False, guidance_scale=1.0, seed=0):
    torch.manual_seed(seed)
    s = SMCDiffusion2DSampler(model, use_smc=use_smc, use_guidance=use_guidance, use_tempering=use_tempering,
                              guidance_scale=guidance_scale, reward_fn=problem.log_likelihood,
                              fkd_config={**FKD, 'adaptive_resampling': adaptive}, use_progress_bar=False)
    out = s.sample(K, eta=1.0)
    return out, s.nfes


def weighted_mode_tv(problem, out):
    w = torch.softmax(out['log_weights'].double(), 0)
    comp = torch.from_numpy(problem.map_component(out['samples'].numpy()))
    mass = torch.zeros(len(problem.weights), dtype=torch.float64).index_add_(0, comp, w).numpy()
    return 0.5 * float(np.abs(mass - problem.posterior_weights).sum())


@pytest.mark.parametrize("use_guidance", [False, True])
def test_smc_nfe_count(gmm2, use_guidance):
    problem, model = gmm2
    _, nfes = run_smc(model, problem, 37, use_smc=True, use_guidance=use_guidance)
    assert nfes == 37 * model.n_timesteps * (2 if use_guidance else 1)


def test_smc_fixed_schedule_is_close_to_target(gmm2):
    problem, model = gmm2
    out, _ = run_smc(model, problem, 20_000, use_smc=True, use_guidance=False)
    idx = torch.multinomial(torch.softmax(out['log_weights'].double(), 0), 2000, replacement=True)
    target = problem.sample_target(2000, seed=1)
    floor = sliced_wasserstein(problem.sample_target(2000, seed=2), target, n_projections=2000)
    prior_gap = sliced_wasserstein(problem.sample_prior(2000, seed=3), target, n_projections=2000)
    sw = sliced_wasserstein(out['samples'][idx].numpy(), target, n_projections=2000)
    assert prior_gap > 10 * floor
    assert sw < 3 * floor + 0.1, (sw, floor, prior_gap)
    assert weighted_mode_tv(problem, out) < 0.08


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_das_adaptive_keeps_weights(gmm2, seed):
    """DAS with ESS-adaptive resampling skips resampling at most potential steps. Dropping the
    accumulated weight on a skip (the original behaviour) gives mode-mass TV of 0.06-0.84 on
    this instance across these seeds; the persistent weights give about 0.02."""
    problem, model = gmm2
    out, _ = run_smc(model, problem, 20_000, use_smc=True, use_guidance=True, use_tempering=True,
                     adaptive=True, guidance_scale=0.1, seed=seed)
    assert weighted_mode_tv(problem, out) < 0.05


def tree_nodes(root):
    stack, out = [root], []
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(n.children)
    return out


def test_dts_nfe_matches_cache_rule(gmm2):
    """Each non-terminal node with children costs one noise prediction (siblings reuse it),
    each terminal node one final denoising step; nothing else is charged."""
    problem, model = gmm2
    torch.manual_seed(0)
    s = MCTSDiffusionSampler(model, reward_fn=lambda x: problem.log_likelihood(x).numpy(),
                             expand_timesteps=[100, 80, 60, 40, 20], num_nfes=20_000, device='cpu',
                             use_progress_bar=False, use_rollout=True, batch_size=1)
    s.initialize_root_nodes()
    s.build_search_tree(num_nfes=100)
    assert s.nfes == model.n_timesteps                       # one full trajectory, T steps
    s.build_search_tree(num_nfes=20_000)
    nodes = [n for n in tree_nodes(s.meta_root) if not s._is_meta_root(n)]
    expected = sum(1 for n in nodes if n.t > 0 and n.children) + sum(1 for n in nodes if n.t == 0)
    assert s.nfes == expected
    assert any(len(n.children) > 1 for n in nodes if n.t in (80, 60, 40, 20))


def test_dts_approaches_target(gmm2):
    problem, model = gmm2
    torch.manual_seed(0)
    s = MCTSDiffusionSampler(model, reward_fn=lambda x: problem.log_likelihood(x).numpy(),
                             expand_timesteps=[100, 80, 60, 40, 20], num_nfes=200_000, device='cpu',
                             use_progress_bar=False, use_rollout=True, batch_size=1)
    s.initialize_root_nodes()
    target = problem.sample_target(2000, seed=1)
    sws = []
    for n in (10_000, 200_000):
        s.build_search_tree(num_nfes=n)
        x, _, _ = s.sample_from_tree(2000)
        sws.append(sliced_wasserstein(x.numpy(), target, n_projections=2000))
    floor = sliced_wasserstein(problem.sample_target(2000, seed=2), target, n_projections=2000)
    assert sws[1] < sws[0], sws
    assert sws[1] < 3 * floor + 0.2, (sws, floor)
