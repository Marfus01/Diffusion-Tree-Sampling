"""Importance weights of an SMC population with the 'diff' (difference) twist potential.

Shared by the 2D, pixel-space and Stable Diffusion SMC samplers. A particle's log-weight
accumulates, from its last resampling event onward,

    lambda * (r~_k - r~_{k'})                                   at every potential step k,
    log p_theta(x_{s-1} | x_s) - log q(x_{s-1} | x_s)           at every denoising step s,

where r~_k is the (possibly tempered) reward of the particle's x_0 estimate at potential
step k and k' is its previous potential step (r~ = reward_min_value before the first one).
The potentials telescope, so with r~ at the final step equal to r(x_0) the total weight is
lambda * r(x_0) + sum_s log(p/q), i.e. the target is p_theta,0(x_0) e^{lambda r(x_0)}.

The log-weight is reset to zero only when the population is resampled. A potential step
that skips resampling (adaptive mode with ESS above the threshold) keeps the accumulated
weight, and the final weights are returned so that evaluation samples are drawn by weight.

A particle whose state, reward or log-ratio becomes non-finite (diverging gradient guidance)
gets log-weight -inf: it is never selected again and is counted in `num_invalid`. Drawing
from a population with no valid particle raises.
"""
from typing import Iterable, Optional

import torch


class SMCDivergedError(RuntimeError):
    """No particle of the population has a finite importance weight."""


class SMCWeights:
    def __init__(
        self,
        num_particles: int,
        lmbda: float,
        adaptive_resampling: bool,
        resampling_schedule: Iterable[int],
        device,
        reward_min_value: float = 0.0,
        ess_threshold: float = 0.5,
    ):
        self.num_particles = num_particles
        self.lmbda = lmbda
        self.adaptive_resampling = adaptive_resampling
        self.resampling_schedule = set(resampling_schedule)
        self.ess_threshold = ess_threshold
        self.device = device
        self.log_w = torch.zeros(num_particles, device=device)
        self.prev_reward = torch.full((num_particles,), float(reward_min_value), device=device)
        self.num_resampling_events = 0

    def _drop_nonfinite(self, *values):
        bad = ~torch.isfinite(self.log_w)
        for v in values:
            bad = bad | ~torch.isfinite(v)
        self.log_w = torch.where(bad, torch.full_like(self.log_w, -float('inf')), self.log_w)

    @property
    def num_invalid(self) -> int:
        return int((~torch.isfinite(self.log_w)).sum())

    def invalidate(self, mask: torch.Tensor) -> None:
        """Gives log-weight -inf to particles where mask [K] is True (e.g. non-finite states)."""
        self.log_w = torch.where(mask.to(self.log_w.device), torch.full_like(self.log_w, -float('inf')), self.log_w)

    def add_log_ratio(self, log_ratio: torch.Tensor) -> None:
        """Adds log p_theta(x_{s-1}|x_s) - log q(x_{s-1}|x_s) of one denoising step, shape [K]."""
        log_ratio = log_ratio.to(self.log_w)
        self.log_w = self.log_w + torch.where(torch.isfinite(log_ratio), log_ratio, torch.zeros_like(log_ratio))
        self._drop_nonfinite(log_ratio)

    def ess_ratio(self) -> float:
        if not torch.isfinite(self.log_w).any():
            return 0.0
        w = torch.softmax(self.log_w.double(), dim=0)
        return float(1.0 / (w.pow(2).sum() * self.num_particles))

    def potential_step(self, sampling_idx: int, reward: torch.Tensor) -> Optional[torch.Tensor]:
        """Applies the potential at step `sampling_idx` and resamples if required.

        Returns the ancestor indices [K] when the population is resampled, otherwise None.
        """
        reward = reward.to(self.log_w)
        incr = self.lmbda * (reward - self.prev_reward)
        self.log_w = self.log_w + torch.where(torch.isfinite(incr), incr, torch.zeros_like(incr))
        self._drop_nonfinite(incr)
        self.prev_reward = reward
        if self.adaptive_resampling:
            do_resample = self.ess_ratio() < self.ess_threshold
        else:
            do_resample = sampling_idx in self.resampling_schedule
        if not do_resample:
            return None
        indices = self.draw(self.num_particles)
        self.log_w = torch.zeros_like(self.log_w)
        self.prev_reward = self.prev_reward[indices]
        self.num_resampling_events += 1
        return indices

    def final_step(self, reward: torch.Tensor) -> None:
        """Applies the terminal potential lambda * (r(x_0) - r~_{last}); no resampling."""
        reward = reward.to(self.log_w)
        incr = self.lmbda * (reward - self.prev_reward)
        self.log_w = self.log_w + torch.where(torch.isfinite(incr), incr, torch.zeros_like(incr))
        self._drop_nonfinite(incr)
        self.prev_reward = reward

    def draw(self, n: int) -> torch.Tensor:
        """Multinomial draw of n indices with replacement, proportional to exp(log_w)."""
        if not torch.isfinite(self.log_w).any():
            raise SMCDivergedError("SMC population has no particle with a finite weight (diverged).")
        probs = torch.softmax(self.log_w.double(), dim=0)
        return torch.multinomial(probs, n, replacement=True)


def gaussian_step_log_ratio(x_prev, base_mean, shift, variance):
    """log N(x_prev; m, v) - log N(x_prev; m + shift, v), summed over non-batch dims.

    `variance` broadcasts against x_prev. Steps with zero variance (the deterministic last
    DDIM step, where the guidance shift is also zero) contribute 0.
    """
    reduce_dims = tuple(range(1, x_prev.ndim))
    variance = torch.broadcast_to(variance, x_prev.shape)
    safe_var = torch.where(variance > 0, variance, torch.ones_like(variance))
    diff = x_prev - base_mean
    log_ratio = (-(diff.pow(2)) + (diff - shift).pow(2)) / (2 * safe_var)
    log_ratio = torch.where(variance > 0, log_ratio, torch.zeros_like(log_ratio))
    return log_ratio.sum(dim=reduce_dims)


def resampling_schedule(t_start: int, t_end: int, frequency: int):
    """Sampling indices of the fixed (non-adaptive) potential/resampling steps."""
    return list(range(t_start, t_end + 1, frequency))
