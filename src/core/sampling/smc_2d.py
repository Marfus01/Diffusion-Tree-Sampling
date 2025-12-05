import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import tqdm
from collections import defaultdict
import argparse
from enum import Enum, auto
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

from src.models.diffusion_2d import DiffusionModel
from src.utils.datasets import get_2d_dataset
from src.utils.plotting_utils import visualize_dataset_comparison, plot_density_evolution
from src.evaluation.distribution_metrics import compute_mmd


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PotentialType(Enum):
    """Potential functions for resampling."""
    MAX = "max"
    ADD = "add"
    DIFF = "diff"


class FKD:
    """Implements the FKD steering mechanism for 2D diffusion."""
    def __init__(
        self,
        *,
        potential_type: PotentialType,
        lmbda: float,
        num_particles: int,
        adaptive_resampling: bool,
        resample_frequency: int,
        resampling_t_start: int,
        resampling_t_end: int,
        time_steps: int,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
        reward_min_value: float = 0.0,
        device: torch.device = torch.device('cuda'),
        **kwargs,
    ) -> None:
        if kwargs:
            logger.warning(f"FKD Steering - Unused arguments: {kwargs}")

        self.potential_type = PotentialType(potential_type)
        self.lmbda = lmbda
        self.num_particles = num_particles
        self.adaptive_resampling = adaptive_resampling
        self.resample_frequency = resample_frequency
        self.resampling_t_start = resampling_t_start
        self.resampling_t_end = resampling_t_end
        self.time_steps = time_steps
        self.reward_fn = reward_fn
        self.device = device
        self.population_rs = torch.ones(self.num_particles, device=self.device) * reward_min_value
        self.product_of_potentials = torch.ones(self.num_particles, device=self.device)

    def get_resampling_schedule(self) -> List[int]:
        """Get timesteps for non-adaptive resampling."""
        schedule = list(range(self.resampling_t_start, self.resampling_t_end + 1, self.resample_frequency))
        return sorted(list(set(schedule)), reverse=True)

    def is_potential_resampling_step(self, sampling_idx: int) -> bool:
        """Check if current timestep is within the resampling interval."""
        is_last_step = sampling_idx == self.time_steps - 1
        is_within_interval = self.resampling_t_start <= sampling_idx <= self.resampling_t_end
        is_on_frequency = (sampling_idx - self.resampling_t_start) % self.resample_frequency == 0
        return is_last_step or (is_within_interval and is_on_frequency)

    def resample(
        self, *, sampling_idx: int, x0_preds_for_reward: torch.Tensor,
        log_prob_diffusion: torch.Tensor, log_prob_proposal: torch.Tensor, tempering_factor: float,
    ) -> torch.Tensor:
        """Perform resampling based on rewards and weights."""
        rs_candidates = tempering_factor * self.reward_fn(x0_preds_for_reward)

        if self.potential_type == PotentialType.MAX:
            max_rewards = torch.max(rs_candidates, self.population_rs)
            if sampling_idx > 0:
                max_rewards += log_prob_diffusion - log_prob_proposal
            w = torch.exp(self.lmbda * max_rewards)
            if sampling_idx == self.time_steps - 1 and self.product_of_potentials.min() > 1e-9:
                w = torch.exp(self.lmbda * rs_candidates) / self.product_of_potentials
                logger.debug(f"Final weights adjusted by product: min={w.min():.2e}, max={w.max():.2e}")
        elif self.potential_type == PotentialType.ADD:
            current_total_reward = rs_candidates + self.population_rs
            if sampling_idx > 0:
                current_total_reward += log_prob_diffusion - log_prob_proposal
            w = torch.exp(self.lmbda * current_total_reward)
            if sampling_idx == self.time_steps - 1 and self.product_of_potentials.min() > 1e-9:
                w = torch.exp(self.lmbda * rs_candidates) / self.product_of_potentials
                logger.debug(f"Final weights adjusted by product: min={w.min():.2e}, max={w.max():.2e}")
        elif self.potential_type == PotentialType.DIFF:
            diffs = rs_candidates - self.population_rs
            if sampling_idx > 0:
                diffs += log_prob_diffusion - log_prob_proposal
            w = torch.exp(self.lmbda * diffs)
        else:
            raise ValueError(f"potential_type {self.potential_type} not recognized")

        w = torch.clamp(w, min=0, max=1e10)
        w[torch.isnan(w)] = 0.0
        logger.debug(f"  Weight stats at t={sampling_idx}: min={w.min().item():.2e}, max={w.max().item():.2e}, mean={w.mean().item():.2e}")

        perform_resampling = False
        fixed_schedule_timesteps = self.get_resampling_schedule()
        if self.adaptive_resampling or sampling_idx == self.time_steps - 1:
            if w.sum() > 1e-9:
                normalized_w = w / w.sum()
                ess = 1.0 / (normalized_w.pow(2).sum() + 1e-9)
                ess_ratio = ess / self.num_particles
                timestep = self.time_steps - 1 - sampling_idx
                if ess_ratio < 0.5:
                    logger.info(f"Adaptive resampling triggered at t={timestep} (sampling idx {sampling_idx}, ESS Ratio: {ess_ratio:.3f} < {0.5})")
                    perform_resampling = True
                else:
                    logger.debug(f"Skipping adaptive resampling at t={timestep} (sampling idx {sampling_idx}, ESS Ratio: {ess_ratio:.3f} >= {0.5})")
            else:
                logger.warning(f"Weights sum to zero at sampling idx {sampling_idx}, cannot calculate ESS. Forcing resampling.")
                perform_resampling = True
        elif sampling_idx in fixed_schedule_timesteps:
             perform_resampling = True
             timestep = self.time_steps - 1 - sampling_idx
             logger.info(f"Fixed resampling triggered at t={timestep} (sampling idx {sampling_idx})")

        if perform_resampling:
            if w.sum() > 1e-9:
                indices = torch.multinomial(w, num_samples=self.num_particles, replacement=True)
                new_rewards = rs_candidates[indices]

                unique_indices = torch.unique(indices)
                logger.info(f"  Resampled {len(unique_indices)} unique particles out of {self.num_particles} at t={timestep} (sampling idx {sampling_idx})")
                logger.info(f"  Reward mean before: {self.population_rs.mean():.4f}, after: {new_rewards.mean():.4f}")
                self.population_rs = new_rewards

                if self.potential_type in [PotentialType.MAX, PotentialType.ADD]:
                    self.product_of_potentials = self.product_of_potentials[indices] * w[indices]
            else:
                 timestep = self.time_steps - 1 - sampling_idx
                 logger.warning(f"All weights zero at t={timestep} (sampling idx {sampling_idx}). Sampling uniformly.")
                 indices = torch.randint(0, self.num_particles, (self.num_particles,), device=self.device)

                 self.population_rs = rs_candidates[indices]
                 self.product_of_potentials.fill_(1.0)
        else:
            self.population_rs = rs_candidates
            indices = torch.arange(x0_preds_for_reward.shape[0], device=self.device)

        return indices

# --- Main Sampler Class ---

class SMCDiffusion2DSampler:
    """SMC sampler for 2D diffusion models."""
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
        self.reward_fn = reward_fn
        self.fkd_config = fkd_config
        self.fkd: Optional[FKD] = None

        if self.use_guidance and self.reward_fn is None:
            raise ValueError("reward_fn must be provided if use_guidance is True")
        if self.use_smc and self.fkd_config is None:
             raise ValueError("fkd_config must be provided if use_smc is True")
        if self.use_smc and self.reward_fn is None:
            if 'reward_fn' not in self.fkd_config:
                 raise ValueError("reward_fn must be provided (either directly or in fkd_config) if use_smc is True")
            if self.reward_fn:
                 self.fkd_config.setdefault('reward_fn', self.reward_fn)

        self.nfes = 0

        plot_suffix = []
        if self.use_smc:
            potential = self.fkd_config.get('potential_type', 'unknown').lower()
            adaptive = "adaptive" if self.fkd_config.get('adaptive_resampling', False) else "fixed"
            plot_suffix.append(f"smc_{potential}_{adaptive}")
        if self.use_guidance:
            plot_suffix.append(f"guidance_{self.guidance_scale}")
        if not plot_suffix:
             plot_suffix.append("standard")

        self.plots_dir = os.path.join('plots', dataset_name, "_".join(plot_suffix))
        os.makedirs(self.plots_dir, exist_ok=True)

    @torch.no_grad()
    def _perform_rollout(self, start_particles: torch.Tensor, start_timestep: int) -> torch.Tensor:
        """Performs MC rollout from start_particles."""
        particles = start_particles.clone()
        for t in range(start_timestep, 0, -1):
            particles, _ = self.model.p_sample(particles, t, eta=0.0)
            self.nfes += particles.shape[0]
        return particles

    @torch.no_grad()
    def _get_x0_for_reward(self, particles: torch.Tensor, current_timestep: int) -> torch.Tensor:
        """Gets x0 prediction for reward calculation (rollout or DDIM)."""
        if current_timestep == 0:
            x0_for_reward = particles
        elif self.use_rollout:
            x0_for_reward = self._perform_rollout(particles, current_timestep)
        else:
            x0_for_reward, _ = self.model.ddim_denoise(particles, current_timestep)
            self.nfes += particles.shape[0]
        return x0_for_reward

    def sample(self, num_particles, eta=0.0, return_all_particles=False):
        """Run the sampling process (standard, guided, FKD-SMC, or guided FKD-SMC)."""

        if self.use_smc:
            fkd_constructor_args = self.fkd_config.copy()
            fkd_constructor_args['num_particles'] = num_particles
            fkd_constructor_args['device'] = self.device
            fkd_constructor_args['time_steps'] = self.model.n_timesteps
            fkd_constructor_args['reward_fn'] = self.fkd_config['reward_fn']
            self.fkd = FKD(**fkd_constructor_args)
            self.potential_name = self.fkd.potential_type.name.lower()
            adaptive_str = "adaptive" if self.fkd.adaptive_resampling else "fixed"
            smc_info = f"FKD-SMC ({self.potential_name}, λ={self.fkd.lmbda}, {adaptive_str})"
        else:
            self.fkd = None
            smc_info = "Standard Sampling"

        if self.use_guidance:
             guidance_info = f"CFG (scale={self.guidance_scale})"
        else:
             guidance_info = "No Guidance"

        rollout_str = "MC-rollout reward" if self.use_rollout and self.use_smc else "DDIM reward" if self.use_smc else ""

        logger.info(
             f"Starting Sampling: {num_particles} particles, {smc_info}, {guidance_info} {rollout_str}"
        )
        start_time = time.time()
        self.nfes = 0

        max_timestep = self.model.n_timesteps - 1
        particles = self.model.sample_initial_noise(batch_size=num_particles).to(self.device)

        visualization_samples = {}
        all_particles_history = {}
        resample_stats = defaultdict(lambda: {'rewards_mean': np.nan, 'rewards_std': np.nan,
                                              'rewards_min': np.nan, 'rewards_max': np.nan})

        initial_index = max_timestep + 1
        if initial_index in self.visualization_timesteps:
            visualization_samples[initial_index] = particles.clone().cpu()
        if return_all_particles:
             all_particles_history[initial_index] = {'samples': particles.clone()}

        timesteps = list(range(max_timestep, -1, -1))
        if self.use_progress_bar:
            iterator = tqdm.tqdm(enumerate(timesteps), total=len(timesteps), desc="Sampling")
        else:
            iterator = enumerate(timesteps)

        log_prob_diffusion_prev = torch.zeros(num_particles, device=self.device)
        log_prob_proposal_prev = torch.zeros(num_particles, device=self.device)

        tempering_gamma = 0.008
        for i, t in iterator:
            if self.use_tempering:
                tempering_factor = min((1 + tempering_gamma)**i - 1, 1.)
            else:
                tempering_factor = 1.0

            current_timestep_idx = t + 1
            if current_timestep_idx in self.visualization_timesteps:
                 visualization_samples[current_timestep_idx] = particles.clone().cpu()

            guidance_grad = None
            x0_pred_for_guidance = None
            if self.use_guidance:
                with torch.enable_grad():
                    particles.requires_grad_(True)
                    x0_pred, _ = self.model.ddim_denoise(particles, t)
                    rewards = self.reward_fn(x0_pred)
                    guidance_grad = torch.autograd.grad(rewards.sum(), particles)[0]
                    # gradient guidance can cause NaNs, so we set them to 0
                    guidance_grad = torch.nan_to_num(guidance_grad, nan=0.0, posinf=1e4, neginf=-1e4)
                    guidance_grad = guidance_grad * tempering_factor
                    particles.requires_grad_(False)
                    x0_pred_for_guidance = x0_pred.detach()
                self.nfes += particles.shape[0]

            resampled_indices = torch.arange(particles.shape[0], device=self.device)
            if self.use_smc:
                should_resample = self.fkd.is_potential_resampling_step(i)
                if should_resample:
                    if self.use_guidance and x0_pred_for_guidance is not None:
                        x0_for_reward = x0_pred_for_guidance
                    else:
                        x0_for_reward = self._get_x0_for_reward(particles, t)
                    resampled_indices = self.fkd.resample(
                        sampling_idx=i,
                        x0_preds_for_reward=x0_for_reward,
                        log_prob_diffusion=log_prob_diffusion_prev,
                        log_prob_proposal=log_prob_proposal_prev,
                        tempering_factor=tempering_factor,
                    )

                    # reset log_prob_diffusion_prev and log_prob_proposal_prev after resampling
                    log_prob_diffusion_prev = torch.zeros(num_particles, device=self.device)
                    log_prob_proposal_prev = torch.zeros(num_particles, device=self.device)

                    if return_all_particles:
                        rewards_for_history = self.fkd.reward_fn(x0_for_reward)
                        all_particles_history[current_timestep_idx] = {
                            'samples': particles.clone(),
                            'rewards': np.exp(rewards_for_history.cpu().numpy())
                        }                

            current_particles, log_prob_diffusion, log_prob_proposal, _ = self.model.p_sample(
                particles,
                t,
                eta=eta,
                guidance_grad=guidance_grad,
                guidance_scale=self.guidance_scale,
                return_log_prob=True,
                use_smc=self.use_smc,
            )
            self.nfes += particles.shape[0]

            # cumulative until next resampling
            particles = current_particles[resampled_indices]
            log_prob_diffusion_prev += log_prob_diffusion[resampled_indices]
            log_prob_proposal_prev += log_prob_proposal[resampled_indices]

        if return_all_particles:
            final_rewards = None
            if self.use_smc:
                 final_rewards = self.fkd.reward_fn(particles)

            final_history = {'samples': particles.clone()}
            if final_rewards is not None:
                final_history['rewards'] = np.exp(final_rewards.cpu().numpy())
            all_particles_history[0] = final_history

        end_time = time.time()
        logger.info(f"Sampling finished in {end_time - start_time:.2f} seconds.")
        logger.info(f"Total NFEs: {self.nfes}")

        if self.use_smc:
            self._plot_resampling_stats(resample_stats)

        if self.visualization_timesteps:
            if 0 in self.visualization_timesteps and 0 not in visualization_samples:
                 visualization_samples[0] = particles.clone().cpu()
            if max_timestep + 1 in self.visualization_timesteps and max_timestep + 1 not in visualization_samples:
                 initial_noise = all_particles_history.get(max_timestep + 1, {}).get('samples')
                 if initial_noise is not None:
                      visualization_samples[max_timestep + 1] = initial_noise.cpu()

            # plot_density_evolution(
            #     visualization_samples,
            #     target_samples=get_2d_dataset(self.dataset_name, split='test', num_samples=1000)[0],
            #     save_path=os.path.join(self.plots_dir, f"density_evolution.png")
            # )

        if return_all_particles:
            return all_particles_history
        else:
            return particles.cpu()

    def _plot_resampling_stats(self, stats):
        """Plots reward statistics at resampling timesteps."""
        recorded_timesteps = sorted([t for t in stats if not np.isnan(stats[t]['rewards_mean'])], reverse=True)
        if not recorded_timesteps:
            logger.warning("No resampling stats to plot.")
            return

        means = [stats[t]['rewards_mean'] for t in recorded_timesteps]
        stds = [stats[t]['rewards_std'] for t in recorded_timesteps]
        mins = [stats[t]['rewards_min'] for t in recorded_timesteps]
        maxs = [stats[t]['rewards_max'] for t in recorded_timesteps]

        plt.figure(figsize=(10, 6))
        plt.errorbar(recorded_timesteps, means, yerr=stds, marker='o', linestyle='-', capsize=5, label='Mean Reward ± Std Dev')
        plt.fill_between(recorded_timesteps, mins, maxs, alpha=0.2, color='lightblue', label='Min-Max Reward Range')

        plt.xlabel('Timestep (t+1, state before diffusion step)')
        plt.ylabel('Reward Estimate (from x0 pred)')
        plt.title(f'Reward Stats at Potential Resampling Steps ({self.potential_name} potential)')
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.gca().invert_xaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f"resampling_stats.png"), dpi=300)
        plt.close()