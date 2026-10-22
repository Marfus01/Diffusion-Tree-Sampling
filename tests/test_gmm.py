"""Unit checks of the analytic GMM inverse problem (closed forms against Monte Carlo)."""
import math

import numpy as np
import pytest
import torch

from src.models.gmm_analytic import GMMInverseProblem, AnalyticGMMDiffusion, sliced_wasserstein


@pytest.fixture(scope="module", params=[8, 80])
def problem(request):
    return GMMInverseProblem(dim=request.param, instance_seed=0)


def q_t_log_prob(problem, x, abar):
    means = math.sqrt(abar) * problem.means
    comps = torch.distributions.MultivariateNormal(means, torch.eye(problem.dim, dtype=torch.float64).expand(len(means), -1, -1))
    mix = torch.distributions.MixtureSameFamily(torch.distributions.Categorical(problem.weights), comps)
    return mix.log_prob(x)


@pytest.mark.parametrize("abar", [0.999, 0.5, 0.05])
def test_eps_star_matches_score(problem, abar):
    g = torch.Generator().manual_seed(1)
    x = problem.sample_q_t(64, abar, generator=g).double().requires_grad_(True)
    score = torch.autograd.grad(q_t_log_prob(problem, x, abar).sum(), x)[0]
    eps = problem.eps_star(x.detach().float(), torch.full((64,), abar)).double()
    np.testing.assert_allclose(eps.numpy(), (-math.sqrt(1 - abar) * score).numpy(), atol=2e-3, rtol=1e-3)


def test_posterior_masses_match_importance_resampling(problem):
    """Component masses of pi*_0 against self-normalized importance sampling from the prior."""
    g = torch.Generator().manual_seed(2)
    n = 400_000
    comp = torch.multinomial(problem.weights, n, replacement=True, generator=g)
    x = problem.means[comp] + torch.randn(n, problem.dim, generator=g, dtype=torch.float64)
    logw = problem.log_likelihood(x.float()).double()
    w = torch.softmax(logw, 0)
    mass = torch.zeros(len(problem.weights), dtype=torch.float64).index_add_(0, comp, w)
    ess = 1.0 / w.pow(2).sum().item()
    assert ess > 1000, "too few effective samples for the check"
    np.testing.assert_allclose(mass.numpy(), problem.posterior_weights, atol=4 / math.sqrt(ess))


def test_posterior_component_moments(problem):
    """Mean of the heaviest pi*_0 component against importance resampling within that prior component."""
    k = int(np.argmax(problem.posterior_weights))
    g = torch.Generator().manual_seed(3)
    x = problem.means[k] + torch.randn(400_000, problem.dim, generator=g, dtype=torch.float64)
    w = torch.softmax(problem.log_likelihood(x.float()).double(), 0)
    mean_is = (w[:, None] * x).sum(0).numpy()
    np.testing.assert_allclose(mean_is, problem.posterior_means[k], atol=0.05)


def test_target_samples_component_counts(problem):
    x = problem.sample_target(20_000, seed=4)
    assert problem.mode_mass_tv(x) < 0.02


def test_sliced_wasserstein_basic():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 4))
    assert sliced_wasserstein(x, x, n_projections=200) == pytest.approx(0.0, abs=1e-12)
    shift = np.zeros(4); shift[0] = 3.0
    # SW_1 of a translation by c = |c| * E|u_1| for u uniform on the sphere; E|u_1| = Gamma(2)/(sqrt(pi) Gamma(2.5)) in d=4
    expected = 3.0 * math.gamma(2) / (math.sqrt(math.pi) * math.gamma(2.5))
    assert sliced_wasserstein(x, x + shift, n_projections=20_000) == pytest.approx(expected, rel=0.02)


def test_prior_chain_close_to_q0():
    """Unguided reverse chain with eps* starts at q_{T-1} and ends near q_0 (discretization only)."""
    problem = GMMInverseProblem(dim=8, instance_seed=0)
    model = AnalyticGMMDiffusion(problem, device='cpu')
    torch.manual_seed(5)
    x = model.sample_initial_noise(2000)
    for t in range(model.n_timesteps - 1, -1, -1):
        x, _ = model.p_sample(x, t)
    q0 = problem.sample_prior(2000, seed=6)
    q0b = problem.sample_prior(2000, seed=7)
    sw = sliced_wasserstein(x.numpy(), q0, n_projections=2000)
    floor = sliced_wasserstein(q0b, q0, n_projections=2000)
    assert sw < 3 * floor + 0.3, (sw, floor)
