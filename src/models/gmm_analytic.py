"""Gaussian-mixture inverse problem with an analytic diffusion prior (MCGdiff benchmark).

Prior: 25 unit-covariance components with means (-8i, -8j, ..., -8i, -8j) in R^{d_x},
(i, j) in {-2,...,2}^2, and weights w = chi^2_1 draws normalized to one. Observation
y = A x* + sigma_y eps with A in R^{d_y x d_x}, as in `generate_measurement_equations` of
gabrielvc/benchopts_inverse_problem_diffusion_prior: A = U diag(s) V_{:d_y} with the
singular values s ~ U(0, 1) sorted, sigma_y ~ U(0, max s), x* drawn from the prior.
Each problem instance (seed) redraws the prior weights, A, sigma_y and y.

Reward r(x_0) = log N(y; A x_0, sigma_y^2 I). The target is
pi*_0(x_0) ∝ p_0(x_0) e^{lambda r(x_0)}, a Gaussian mixture with component covariance
Sigma = (I + lambda A^T A / sigma_y^2)^{-1}, means Sigma (mu_k + lambda A^T y / sigma_y^2),
and weights ∝ w_k N(y; A mu_k, sigma_y^2 / lambda I + A A^T).

Under the VP forward process x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps the marginal is
q_t = sum_k w_k N(sqrt(abar_t) mu_k, I), so the exact noise prediction is
eps*(x_t, t) = sqrt(1 - abar_t) (x_t - sum_k gamma_k(x_t) sqrt(abar_t) mu_k), with gamma_k
the posterior responsibility of component k under q_t.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from src.models.diffusion_2d import DiffusionModel


def mixture_means(dim: int) -> torch.Tensor:
    means = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            means.append(torch.tensor([-8.0 * i, -8.0 * j] * (dim // 2)))
    return torch.stack(means)  # [25, dim]


class GMMInverseProblem:
    def __init__(self, dim: int, instance_seed: int, dim_y: int = 1, lmbda: float = 1.0, device='cpu'):
        if dim % 2 != 0:
            raise ValueError("dim must be even (means repeat a 2D pattern)")
        self.dim = dim
        self.dim_y = dim_y
        self.lmbda = lmbda
        self.device = torch.device(device)
        g = torch.Generator().manual_seed(instance_seed)

        self.means = mixture_means(dim).double()                     # [K, d]
        weights = torch.randn(len(self.means), generator=g, dtype=torch.float64) ** 2
        self.weights = weights / weights.sum()                       # [K]

        A = torch.randn(dim_y, dim, generator=g, dtype=torch.float64)
        U, s, Vh = torch.linalg.svd(A, full_matrices=True)
        s = torch.sort(torch.rand(s.shape, generator=g, dtype=torch.float64), descending=True).values
        self.A = U @ torch.diag(s) @ Vh[:dim_y, :]                   # [d_y, d]
        self.sigma_y = float(torch.rand(1, generator=g, dtype=torch.float64)[0] * s.max())
        comp = torch.multinomial(self.weights, 1, generator=g).item()
        x_star = self.means[comp] + torch.randn(dim, generator=g, dtype=torch.float64)
        self.y = self.A @ x_star + self.sigma_y * torch.randn(dim_y, generator=g, dtype=torch.float64)

        self._posterior = self._compute_posterior(lmbda)
        self._means_f = self.means.float().to(self.device)
        self._log_w_f = torch.log(self.weights).float().to(self.device)
        self._A_f = self.A.float().to(self.device)
        self._y_f = self.y.float().to(self.device)

    # ---- target ----
    def _compute_posterior(self, lmbda):
        d = self.dim
        prec = torch.eye(d, dtype=torch.float64) + lmbda * self.A.T @ self.A / self.sigma_y ** 2
        cov = torch.linalg.inv(prec)
        cov = (cov + cov.T) / 2
        means = (self.means + lmbda * (self.A.T @ self.y) / self.sigma_y ** 2) @ cov  # cov symmetric
        obs_cov = (self.sigma_y ** 2 / lmbda) * torch.eye(self.dim_y, dtype=torch.float64) + self.A @ self.A.T
        resid = self.y[None, :] - self.means @ self.A.T                               # [K, d_y]
        obs_prec = torch.linalg.inv(obs_cov)
        log_lik = -0.5 * torch.einsum('ki,ij,kj->k', resid, obs_prec, resid) \
            - 0.5 * torch.logdet(2 * math.pi * obs_cov)
        log_w = torch.log(self.weights) + log_lik
        weights = torch.softmax(log_w, dim=0)
        return {'weights': weights, 'means': means, 'cov': cov, 'chol': torch.linalg.cholesky(cov)}

    @property
    def posterior_weights(self) -> np.ndarray:
        return self._posterior['weights'].numpy()

    @property
    def posterior_means(self) -> np.ndarray:
        return self._posterior['means'].numpy()

    def sample_target(self, n: int, seed: int) -> np.ndarray:
        g = torch.Generator().manual_seed(seed)
        post = self._posterior
        comp = torch.multinomial(post['weights'], n, replacement=True, generator=g)
        z = torch.randn(n, self.dim, generator=g, dtype=torch.float64)
        return (post['means'][comp] + z @ post['chol'].T).float().numpy()

    def sample_prior(self, n: int, seed: int) -> np.ndarray:
        g = torch.Generator().manual_seed(seed)
        comp = torch.multinomial(self.weights, n, replacement=True, generator=g)
        z = torch.randn(n, self.dim, generator=g, dtype=torch.float64)
        return (self.means[comp] + z).float().numpy()

    # ---- reward ----
    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        """r(x) = log N(y; A x, sigma_y^2 I); x [B, d] -> [B]. Differentiable."""
        A = self._A_f.to(x.device)
        y = self._y_f.to(x.device)
        resid = y[None, :] - x @ A.T
        return -0.5 * resid.pow(2).sum(-1) / self.sigma_y ** 2 \
            - 0.5 * self.dim_y * math.log(2 * math.pi * self.sigma_y ** 2)

    # ---- diffusion prior ----
    def eps_star(self, x: torch.Tensor, abar: torch.Tensor) -> torch.Tensor:
        """Exact noise prediction; x [B, d], abar [B] (alphas_cumprod at each sample's index)."""
        means = self._means_f.to(x.device)
        log_w = self._log_w_f.to(x.device)
        s = abar.sqrt()[:, None]                                       # [B, 1]
        centers = s[:, :, None] * means.T[None, :, :]                  # [B, d, K]
        sq = (x[:, :, None] - centers).pow(2).sum(1)                   # [B, K]
        gamma = torch.softmax(log_w[None, :] - 0.5 * sq, dim=1)        # [B, K]
        mean_t = torch.einsum('bk,kd->bd', gamma, means) * s           # E[sqrt(abar) mu | x_t]
        return (1 - abar).sqrt()[:, None] * (x - mean_t)

    def sample_q_t(self, n: int, abar: float, device=None, generator=None) -> torch.Tensor:
        device = device or self.device
        comp = torch.multinomial(self.weights.float(), n, replacement=True, generator=generator)
        z = torch.randn(n, self.dim, generator=generator)
        return (math.sqrt(abar) * self.means.float()[comp] + z).to(device)

    def map_component(self, x: np.ndarray) -> np.ndarray:
        """Most probable pi*_0 component of each sample: argmax_k log w~_k + log N(x; m_k, Sigma).
        (Nearest-mean assignment is unreliable here: components with negligible posterior mass
        can have means within one posterior standard deviation of a heavy one.)"""
        post = self._posterior
        prec = torch.linalg.inv(post['cov'])
        diff = torch.as_tensor(x, dtype=torch.float64)[:, None, :] - post['means'][None, :, :]   # [N, K, d]
        maha = torch.einsum('nki,ij,nkj->nk', diff, prec, diff)
        return (torch.log(post['weights'] + 1e-300)[None, :] - 0.5 * maha).argmax(1).numpy()

    def mode_mass_tv(self, x: np.ndarray) -> float:
        """Total variation between the component masses of pi*_0 and those of the samples, with
        each sample assigned to its most probable pi*_0 component."""
        counts = np.bincount(self.map_component(x), minlength=len(self.weights)) / len(x)
        return 0.5 * float(np.abs(counts - self.posterior_weights).sum())


class AnalyticNoisePredictor(nn.Module):
    """Drop-in replacement for NoisePredictor: forward(x, t / n_timesteps) -> eps*(x, t)."""

    def __init__(self, problem: GMMInverseProblem, alphas_cumprod: torch.Tensor, n_timesteps: int):
        super().__init__()
        self.problem = problem
        self.n_timesteps = n_timesteps
        self.register_buffer('alphas_cumprod', alphas_cumprod.clone().float())

    def forward(self, x, t_normalized):
        t_idx = torch.round(t_normalized * self.n_timesteps).long().clamp(0, self.n_timesteps - 1)
        abar = self.alphas_cumprod.to(x.device)[t_idx]
        return self.problem.eps_star(x, abar)


class AnalyticGMMDiffusion(DiffusionModel):
    """DiffusionModel whose noise predictor is the exact eps* of the GMM prior.

    The chain starts from the exact marginal q at the first sampling index (T - 1); under the
    2D schedule (100 steps, beta 1e-3 -> 0.07) q_{T-1} is far from N(0, I) for these means,
    so p_theta,0 differs from q_0 only by the discretization of the reverse kernel.
    """

    def __init__(self, problem: GMMInverseProblem, n_timesteps=100, beta_start=1e-3, beta_end=0.07, device='cpu'):
        super().__init__(input_dim=problem.dim, hidden_dim=8, n_timesteps=n_timesteps,
                         beta_start=beta_start, beta_end=beta_end, device=device)
        self.problem = problem
        self.noise_predictor = AnalyticNoisePredictor(problem, self.alphas_cumprod, n_timesteps).to(device)

    def sample_initial_noise(self, batch_size=1):
        abar = float(self.alphas_cumprod[self.n_timesteps - 1])
        return self.problem.sample_q_t(batch_size, abar, device=self.device)


class GMMTask:
    """Dataset-like wrapper used by run_2d.py (same interface as the 2D toy datasets)."""

    def __init__(self, problem: GMMInverseProblem, n_target: int, target_seed: int, device='cpu'):
        self.problem = problem
        self.name = "gmm"
        self.device = device
        self.gt_samples = problem.sample_target(n_target, seed=target_seed)
        self.gt_rewards = problem.log_likelihood(torch.from_numpy(self.gt_samples)).numpy()

    def calculate_reward_diff(self, samples: torch.Tensor) -> torch.Tensor:
        return self.problem.log_likelihood(samples)


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, n_projections: int = 10_000, seed: int = 0,
                       batch: int = 1000) -> float:
    """SW_1 with random unit directions (as in MCGdiff / DCPS): mean over directions of the 1D
    Wasserstein-1 distance. Equal sample sizes use sorted differences; otherwise quantiles."""
    xt = torch.as_tensor(x, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    total = 0.0
    done = 0
    while done < n_projections:
        m = min(batch, n_projections - done)
        proj = torch.randn(m, xt.shape[1], generator=g, dtype=torch.float64)
        proj = proj / proj.norm(dim=1, keepdim=True)
        px = torch.sort(xt @ proj.T, dim=0).values
        py = torch.sort(yt @ proj.T, dim=0).values
        if px.shape[0] != py.shape[0]:
            q = torch.linspace(0, 1, max(px.shape[0], py.shape[0]), dtype=torch.float64)
            px = torch.quantile(px, q, dim=0)
            py = torch.quantile(py, q, dim=0)
        total += (px - py).abs().mean(0).sum().item()
        done += m
    return total / n_projections
