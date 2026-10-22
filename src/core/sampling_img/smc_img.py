import torch
import numpy as np
import math
import tqdm
import os
import warnings
import logging
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import inspect

from dataclasses import dataclass

from src.utils.pipeline_utils import ddim_step_with_mean, get_variance, _left_broadcast
from src.core.smc_weights import SMCWeights, gaussian_step_log_ratio, resampling_schedule

# Diffusers imports
from diffusers import DDIMPipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils import deprecate

# Setup logging and warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*convert_buffers_to_embedding.*") # Ignore specific warnings if needed


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """Retrieves the scheduler timesteps for the diffusion process."""
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"Scheduler {scheduler.__class__} does not support custom `timesteps`.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accepts_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_sigmas:
            raise ValueError(f"Scheduler {scheduler.__class__} does not support custom `sigmas`.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@dataclass
class SMCOutput:
    images: torch.Tensor        # [K, C, H, W] final particles in [-1, 1]
    log_weights: torch.Tensor   # [K] final log importance weights (zeros without SMC)
    nfes: int


class PotentialType(Enum):
    DIFF = "diff"
    MAX = "max"
    ADD = "add"

class FKD:
    """
    Implements the FKD steering mechanism for pixel-space diffusion.
    """
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
        state_to_reward_input_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
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
        self.time_steps = time_steps # Total number of timesteps

        self.reward_fn = reward_fn
        self.state_to_reward_input_fn = state_to_reward_input_fn # Renamed from latent_to_decode_fn

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform resampling based on rewards and weights.
        """
        # Prepare predicted x0 for reward calculation
        reward_input_images = self.state_to_reward_input_fn(x0_preds_for_reward)
        rs_candidates = tempering_factor * self.reward_fn(reward_input_images)

        # Compute importance weights based on potential type
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

        # Clamp weights and handle NaNs
        w = torch.clamp(w, min=0, max=1e10)
        w[torch.isnan(w)] = 0.0

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


# --- FKD DDIM Pipeline ---

class FKDDDIMPipeline(DDIMPipeline):
    """
    DDIMPipeline adapted for FKD-SMC steering.
    """
    def __init__(self, unet, scheduler):
        super().__init__(unet=unet, scheduler=scheduler)
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def _perform_rollout(
        self,
        start_image: torch.Tensor,
        start_timestep_index: int,
        all_timesteps: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Performs a full DDIM rollout from a given starting state and timestep.

        Args:
            start_image: The starting noisy image state (x_t) for a single particle.
            start_timestep_index: The index in `all_timesteps` corresponding to the starting state.
            all_timesteps: The full list of timesteps used in the main sampling loop.
            generator: Random generator.
            eta: DDIM eta parameter.

        Returns:
            The final denoised image (x_0) tensor for that particle. Shape [1, C, H, W].
        """
        image = start_image.clone()
        rollout_timesteps = all_timesteps[start_timestep_index:]

        for i, t in enumerate(rollout_timesteps):
            # Predict noise model output
            model_output = self.unet(image, t).sample

            # Compute previous sample
            step_output = self.scheduler.step(
                model_output, t, image, eta=eta, generator=generator, return_dict=True
            )
            image = step_output.prev_sample
            x0_pred = step_output.pred_original_sample

        return x0_pred

    @torch.no_grad()
    def _perform_batched_rollout(
        self,
        start_images: torch.Tensor,
        start_timestep_index: int,
        all_timesteps: torch.Tensor,
        processing_batch_size: int,
        generator: Optional[torch.Generator] = None,
        eta: float = 0.0,
        use_progress_bar: bool = False,
    ) -> torch.Tensor:
        """
        Performs DDIM rollouts in batches from given starting states and timestep index.
        (Used by sample_batched_smc)

        Args:
            start_images: Starting noisy images (x_t) for the full population. [N, C, H, W]
            start_timestep_index: Index in `all_timesteps` corresponding to the starting state.
            all_timesteps: Full list of timesteps for the main sampling loop.
            processing_batch_size: Batch size for processing rollouts.
            generator: Random generator.
            eta: DDIM eta parameter.
            use_progress_bar: Whether to show a progress bar for rollout batches.

        Returns:
            Final denoised images (x_0) for the full population. [N, C, H, W]
        """
        num_particles = start_images.shape[0]
        all_final_x0 = torch.zeros_like(start_images)
        rollout_timesteps = all_timesteps[start_timestep_index:]
        device = start_images.device

        iterator = range(0, num_particles, processing_batch_size)
        if use_progress_bar:
            iterator = tqdm.tqdm(iterator, desc=f"Rollout Step {start_timestep_index}", leave=False)

        for batch_start in iterator:
            batch_end = min(batch_start + processing_batch_size, num_particles)
            current_batch_images = start_images[batch_start:batch_end].clone().to(device)

            for i, t in enumerate(rollout_timesteps):
                # Predict noise model output
                model_output = self.unet(current_batch_images, t).sample

                # Compute previous sample
                step_output = self.scheduler.step(
                    model_output, t, current_batch_images, eta=eta, generator=generator, return_dict=True
                )
                current_batch_images = step_output.prev_sample
                batch_final_x0 = step_output.pred_original_sample
            
            all_final_x0[batch_start:batch_end] = current_batch_images.to(all_final_x0.device)

        return all_final_x0

    @torch.no_grad()
    def __call__(
        self,
        num_inference_steps: int = 50,
        fkd_args: Optional[Dict] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"], # Compatibility, 'latents' here means image state x_t
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:

        # Handle callbacks (similar to SD pipeline)
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        if callback is not None:
            deprecate("callback", "1.0.0", "Use `callback_on_step_end`.")
        if callback_steps is not None:
            deprecate("callback_steps", "1.0.0", "Use `callback_on_step_end`.")

        # 0. Check inputs (basic checks, more could be added)
        if fkd_args is None:
            raise ValueError("`fkd_args` must be provided to use FKDDDIMPipeline.")
        batch_size = fkd_args.get('num_particles')

        # 1. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # 2. Prepare initial noise (image space)
        num_channels = self.unet.config.in_channels
        shape = (
            batch_size,
            num_channels,
            self.unet.config.sample_size,
            self.unet.config.sample_size,
        )
        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(shape, generator=generator, device=self.device, dtype=self.unet.dtype)

        # 3. Instantiate FKD Steering mechanism
        # The reward function expects predicted clean images (x0)
        # state_to_reward_input_fn might be needed if reward_fn expects specific normalization
        # Make sure 'use_rollout' is included in fkd_args if needed
        fkd = FKD(**fkd_args)

        # 4. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Predict the noise residual
                model_output = self.unet(image, t).sample

                # Compute the previous noisy sample x_t -> x_t-1
                step_output = self.scheduler.step(
                    model_output, t, image, eta=eta, generator=generator, return_dict=True
                )
                prev_image = step_output.prev_sample
                x0_pred = step_output.pred_original_sample

                # Determine if this step *could* involve resampling
                should_resample = fkd.is_potential_resampling_step(i)
                if should_resample:
                    # Determine the x0 prediction to use for reward calculation
                    if fkd.use_rollout:
                        logger.info(f"Performing rollouts for resampling at step {i} (t={t.item()})...")
                        # Perform rollout
                        x0_for_reward_calculation = self._perform_rollout(
                            start_image=image,
                            start_timestep_index=i,
                            all_timesteps=timesteps,
                            generator=generator,
                            eta=0,
                        )
                        logger.info(f"Rollouts complete for step {i}.")
                    else:
                        x0_for_reward_calculation = x0_pred

                image = prev_image

                if should_resample:
                    # Perform the potential resampling using the calculated x0 for rewards
                    image = fkd.resample(
                        sampling_idx=i,
                        states=image,
                        x0_preds_for_reward=x0_for_reward_calculation
                    )

                # Handle callbacks
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    # Adapt keys if necessary, 'latents' becomes 'image'
                    for k in callback_on_step_end_tensor_inputs:
                         if k == "latents": callback_kwargs["image"] = image
                         else: callback_kwargs[k] = locals()[k] # Or adjust mapping
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    image = callback_outputs.pop("image", image) # Adapt key

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, image) # Pass image instead of latents

        # 5. Post-processing (already in image space)

        # Denormalize if necessary, based on how data was trained/normalized
        # image = (image / 2 + 0.5).clamp(0, 1)
        # image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    @torch.no_grad()
    def sample_batched_smc(
        self,
        num_particles: int,
        processing_batch_size: int,
        num_inference_steps: int = 50,
        use_smc: bool = False,
        use_guidance: bool = False,
        use_tempering: bool = False,
        use_rollout: bool = False,
        guidance_scale: float = 1.0,
        reward_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        fkd_config: Optional[Dict] = None,
        device: str = 'cpu',
        use_progress_bar: bool = True,
        eta: float = 0.0,
        das_gamma: float = 0.008,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> SMCOutput:
        """
        SMC/FK (use_smc), DPS-RG (use_guidance), TDS (both) and DAS (TDS + use_tempering) with the
        UNet evaluated in chunks of processing_batch_size.

        Each step evaluates the UNet once per particle (1 NFE); with guidance the same forward is
        differentiated for the reward gradient of x0_hat (1 more NFE). Potentials use the x0
        estimate of that forward pass. Importance weights follow src.core.smc_weights: they persist
        across skipped resampling steps, the terminal potential is r(x_0) of the final image with
        tempering factor 1, and the final log-weights are returned for a weighted draw.
        """
        device = self.device
        fkd_config = fkd_config or {}
        self.nfes = 0

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels = self.unet.config.in_channels
        height = width = self.unet.config.sample_size
        shape = (num_particles, num_channels, height, width)
        image = randn_tensor(shape, generator=generator, device=device, dtype=self.unet.dtype)

        schedule = resampling_schedule(fkd_config.get('resampling_t_start', 10), fkd_config.get('resampling_t_end', 40),
                                       fkd_config.get('resample_frequency', 10))
        potential_steps = set(schedule)
        weights = None
        if use_smc:
            if PotentialType(fkd_config.get('potential_type', 'diff')) != PotentialType.DIFF:
                raise NotImplementedError("Only the 'diff' potential is implemented with persistent importance weights.")
            weights = SMCWeights(num_particles, lmbda=fkd_config.get('lmbda', 1.0),
                                 adaptive_resampling=fkd_config.get('adaptive_resampling', False),
                                 resampling_schedule=schedule, device=device,
                                 reward_min_value=fkd_config.get('reward_min_value', 0.0))

        def batched_reward(x):
            return torch.cat([reward_fn(x[b:b + processing_batch_size]).detach().reshape(-1)
                              for b in range(0, x.shape[0], processing_batch_size)])

        iterator = enumerate(timesteps)
        if use_progress_bar:
            iterator = tqdm.tqdm(iterator, total=len(timesteps), desc="SMC sampling")

        for i, t in iterator:
            factor = min((1 + das_gamma) ** i - 1, 1.) if use_tempering else 1.0

            noise_preds = torch.zeros_like(image)
            rewards = None
            guidance_grads = None
            if use_guidance:
                rewards = torch.zeros(num_particles, device=device)
                guidance_grads = torch.zeros_like(image)
            for b0 in range(0, num_particles, processing_batch_size):
                b1 = min(b0 + processing_batch_size, num_particles)
                if use_guidance:
                    with torch.enable_grad():
                        image_batch = image[b0:b1].detach().requires_grad_(True)
                        noise_pred_batch = self.unet(image_batch, t).sample
                        alpha_prod_t = self.scheduler.alphas_cumprod[t]
                        x0_pred_batch = (image_batch - (1 - alpha_prod_t) ** 0.5 * noise_pred_batch) / alpha_prod_t ** 0.5
                        if self.scheduler.config.clip_sample:
                            x0_pred_batch = x0_pred_batch.clamp(-self.scheduler.config.clip_sample_range,
                                                                self.scheduler.config.clip_sample_range)
                        rewards_batch = reward_fn(x0_pred_batch)
                        grad = torch.autograd.grad(rewards_batch.sum(), image_batch)[0].detach()
                    guidance_grads[b0:b1] = torch.nan_to_num(grad) * factor
                    rewards[b0:b1] = rewards_batch.detach()
                    noise_preds[b0:b1] = noise_pred_batch.detach()
                    self.nfes += 2 * (b1 - b0)
                else:
                    noise_preds[b0:b1] = self.unet(image[b0:b1], t).sample
                    self.nfes += b1 - b0

            prev_image, x0_pred, prev_image_mean = ddim_step_with_mean(
                self.scheduler, noise_preds, t, image, eta=eta, generator=generator, return_dict=True
            )

            if use_smc and i in potential_steps:
                if use_rollout and not use_guidance:
                    x0_for_reward = self._perform_batched_rollout(
                        start_images=image, start_timestep_index=i, all_timesteps=timesteps,
                        processing_batch_size=processing_batch_size, generator=generator, eta=eta,
                        use_progress_bar=False)
                    self.nfes += num_particles * (len(timesteps) - i)
                    rewards = batched_reward(x0_for_reward)
                elif rewards is None:
                    rewards = batched_reward(x0_pred)
                indices = weights.potential_step(i, factor * rewards)
                if indices is not None:
                    prev_image = prev_image[indices]
                    prev_image_mean = prev_image_mean[indices]
                    if guidance_grads is not None:
                        guidance_grads = guidance_grads[indices]

            prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
            variance = get_variance(self.scheduler, t, torch.clamp(prev_timestep, 0, self.scheduler.config.num_train_timesteps - 1))
            variance = eta ** 2 * _left_broadcast(variance, prev_image.shape).to(device)
            if prev_timestep < 0:
                variance = torch.zeros_like(variance)   # last step: deterministic, no guidance shift

            if use_guidance:
                shift = variance * guidance_scale * guidance_grads
                prev_image = prev_image + shift
                if weights is not None:
                    weights.add_log_ratio(gaussian_step_log_ratio(prev_image, prev_image_mean, shift, variance))
            if weights is not None:
                weights.invalidate(~torch.isfinite(prev_image).flatten(1).all(dim=1))
            image = prev_image

        log_weights = torch.zeros(num_particles, device=device)
        if weights is not None:
            weights.final_step(batched_reward(image))
            log_weights = weights.log_w
            logger.info(f"Final ESS ratio {weights.ess_ratio():.3f}; resampling events {weights.num_resampling_events}; "
                        f"invalid particles {weights.num_invalid}")
        return SMCOutput(images=image, log_weights=log_weights, nfes=self.nfes)
