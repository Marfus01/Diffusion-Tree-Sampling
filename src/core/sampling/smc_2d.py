import torch
import numpy as np
import time
import tqdm
from enum import Enum
import logging
from typing import Callable, Dict, List, Optional

from src.models.diffusion_2d import DiffusionModel
from src.core.smc_weights import SMCWeights, resampling_schedule


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PotentialType(Enum):
    """Potential functions for resampling."""
    MAX = "max"
    ADD = "add"
    DIFF = "diff"


def tempering_factor(sampling_idx: int, gamma: float) -> float:
    """DAS tempering schedule min((1 + gamma)^i - 1, 1) at sampling index i."""
    return min((1 + gamma) ** sampling_idx - 1, 1.0)


class SMCDiffusion2DSampler:
    """SMC sampler for 2D / low-dimensional diffusion models.

    Covers SMC/FK (use_smc), DPS-RG (use_guidance), TDS (use_smc + use_guidance) and
    DAS (TDS + use_tempering). Each denoising step calls the noise predictor once; with
    guidance the same forward pass is differentiated for the reward gradient, so a step
    costs 1 NFE without and 2 NFE with guidance. Importance weights follow
    `src.core.smc_weights`.
    """
    def __init__(
        self,
        diffusion_model: DiffusionModel,
        *,
        use_smc: bool = False,
        use_guidance: bool = False,
        use_tempering: bool = False,
        guidance_scale: float = 1.0,
        reward_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        fkd_config: Optional[Dict] = None,
        device: str = 'cpu',
        visualization_timesteps: Optional[List[int]] = None,
        dataset_name: str = 'gaussian_mixture',
        use_progress_bar: bool = True,
        use_rollout: bool = False,
        das_gamma: float = 0.008,
    ):
        self.model = diffusion_model
        self.device = torch.device(device)
        self.model.noise_predictor.to(self.device)
        self.visualization_timesteps = visualization_timesteps if visualization_timesteps else []
        self.dataset_name = dataset_name
        self.use_progress_bar = use_progress_bar
        self.use_rollout = use_rollout

        self.use_smc = use_smc
        self.use_guidance = use_guidance
        self.use_tempering = use_tempering
        self.guidance_scale = guidance_scale
        self.das_gamma = das_gamma
        self.reward_fn = reward_fn
        self.fkd_config = fkd_config or {}

        if self.reward_fn is None:
            self.reward_fn = self.fkd_config.get('reward_fn')
        if (self.use_guidance or self.use_smc) and self.reward_fn is None:
            raise ValueError("reward_fn must be provided for guidance or SMC")
        if self.use_smc and PotentialType(self.fkd_config.get('potential_type', 'diff')) != PotentialType.DIFF:
            raise NotImplementedError("Only the 'diff' potential is implemented with persistent importance weights.")

        self.nfes = 0

    @torch.no_grad()
    def _perform_rollout(self, start_particles: torch.Tensor, start_timestep: int) -> torch.Tensor:
        """Deterministic (eta=0) rollout from index start_timestep to the clean sample."""
        particles = start_particles.clone()
        for t in range(start_timestep, -1, -1):
            particles, _ = self.model.p_sample(particles, t, eta=0.0)
            self.nfes += particles.shape[0]
        return particles

    def sample(self, num_particles, eta=0.0, return_all_particles=False):
        """Runs the sampler with `num_particles` particles.

        Returns a dict with
          'samples':     final particles [K, D] on self.device,
          'log_weights': final log importance weights [K] (zeros without SMC),
          'history':     {t+1: {'samples', 'rewards'}} at potential steps and {0: ...} at the
                         end, only when return_all_particles is True ('rewards' are e^{r} of the
                         x_0 estimate used for weighting, i.e. the Tweedie value estimate).
        """
        start_time = time.time()
        self.nfes = 0

        num_steps = self.model.n_timesteps
        schedule = resampling_schedule(
            self.fkd_config.get('resampling_t_start', 20),
            self.fkd_config.get('resampling_t_end', 80),
            self.fkd_config.get('resample_frequency', 20),
        )
        weights = None
        if self.use_smc:
            weights = SMCWeights(
                num_particles=num_particles,
                lmbda=self.fkd_config.get('lmbda', 1.0),
                adaptive_resampling=self.fkd_config.get('adaptive_resampling', True),
                resampling_schedule=schedule,
                device=self.device,
                reward_min_value=self.fkd_config.get('reward_min_value', 0.0),
            )
        potential_steps = set(schedule)

        logger.info(f"Starting sampling: {num_particles} particles, smc={self.use_smc}, guidance={self.use_guidance}, "
                    f"tempering={self.use_tempering}, rollout={self.use_rollout}")

        particles = self.model.sample_initial_noise(batch_size=num_particles).to(self.device)
        history = {}
        if return_all_particles:
            history[num_steps] = {'samples': particles.clone()}

        timesteps = list(range(num_steps - 1, -1, -1))
        iterator = enumerate(timesteps)
        if self.use_progress_bar:
            iterator = tqdm.tqdm(iterator, total=len(timesteps), desc="Sampling")

        for i, t in iterator:
            factor = tempering_factor(i, self.das_gamma) if self.use_tempering else 1.0
            is_potential_step = self.use_smc and i in potential_steps

            guidance_grad = None
            reward = None
            if self.use_guidance:
                with torch.enable_grad():
                    x_in = particles.detach().requires_grad_(True)
                    x0_pred, noise_pred = self.model.ddim_denoise(x_in, t)
                    reward_g = self.reward_fn(x0_pred)
                    guidance_grad = torch.autograd.grad(reward_g.sum(), x_in)[0]
                # gradient guidance can produce NaNs/overflow; zero/clip them as in the original code
                guidance_grad = torch.nan_to_num(guidance_grad, nan=0.0, posinf=1e4, neginf=-1e4) * factor
                noise_pred = noise_pred.detach()
                x0_pred = x0_pred.detach()
                reward = reward_g.detach()
                self.nfes += 2 * particles.shape[0]
            else:
                with torch.no_grad():
                    x0_pred, noise_pred = self.model.ddim_denoise(particles, t)
                self.nfes += particles.shape[0]

            if is_potential_step:
                if self.use_rollout and not self.use_guidance:
                    x0_for_reward = self._perform_rollout(particles, t)
                    reward = self.reward_fn(x0_for_reward)
                elif reward is None:
                    reward = self.reward_fn(x0_pred)
                reward = reward.detach()
                if return_all_particles:
                    history[t + 1] = {'samples': particles.clone(), 'rewards': np.exp(reward.cpu().numpy())}
                indices = weights.potential_step(i, factor * reward)
                if indices is not None:
                    particles = particles[indices]
                    noise_pred = noise_pred[indices]
                    if guidance_grad is not None:
                        guidance_grad = guidance_grad[indices]

            particles, log_prob_diffusion, log_prob_proposal, _ = self.model.p_sample(
                particles, t, eta=eta,
                guidance_grad=guidance_grad, guidance_scale=self.guidance_scale,
                return_log_prob=True, noise_pred=noise_pred,
            )
            if weights is not None:
                weights.add_log_ratio(log_prob_diffusion - log_prob_proposal)
                weights.invalidate(~torch.isfinite(particles).all(dim=-1))

        log_weights = torch.zeros(num_particles, device=self.device)
        if weights is not None:
            final_reward = self.reward_fn(particles).detach()
            weights.final_step(final_reward)
            log_weights = weights.log_w
            logger.info(f"Final ESS ratio {weights.ess_ratio():.3f}; resampling events {weights.num_resampling_events}; "
                        f"invalid particles {weights.num_invalid}")
        if return_all_particles:
            final = {'samples': particles.clone()}
            if weights is not None:
                final['rewards'] = np.exp(final_reward.cpu().numpy())
            history[0] = final

        logger.info(f"Sampling finished in {time.time() - start_time:.2f} seconds, total NFEs: {self.nfes}")
        return {'samples': particles.detach(), 'log_weights': log_weights.detach(),
                'history': history if return_all_particles else None}
