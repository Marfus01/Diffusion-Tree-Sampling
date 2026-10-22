"""
Feynman-Kac Diffusion (FKD) steering mechanism implementation.
"""

import torch
from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import Callable, Optional, Tuple
import logging


@dataclass
class SDSMCOutput:
    images: list                # final images (PIL with output_type="pil")
    log_weights: torch.Tensor   # [K] final log importance weights
    final_rewards: torch.Tensor # [K] guiding reward of the final images
    nfes: int


class PotentialType(Enum):
    DIFF = "diff"
    MAX = "max"
    ADD = "add"


class FKD:
    """
    Implements the FKD steering mechanism. Should be initialized along the diffusion process. .resample() should be invoked at each diffusion timestep.
    See FKD fkd_pipeline_sdxl
    Args:
        potential_type: Type of potential function must be one of PotentialType.
        lmbda: Lambda hyperparameter controlling weight scaling.
        num_particles: Number of particles to maintain in the population.
        adaptive_resampling: Whether to perform adaptive resampling.
        resample_frequency: Frequency (in timesteps) to perform resampling.
        resampling_t_start: Timestep to start resampling.
        resampling_t_end: Timestep to stop resampling.
        time_steps: Total number of timesteps in the sampling process.
        reward_fn: Function to compute rewards from decoded latents.
        reward_min_value: Minimum value for rewards (default: 0.0). Important for the Max potential type.
        latent_to_decode_fn: Function to decode latents to images, relevant for latent diffusion models (default: identity function).
        device: Device on which computations will be performed (default: CUDA).
        use_rollout: Whether to use rollout for resampling (default: False).
        **kwargs: Additional keyword arguments, unused.
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
        latent_to_decode_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        device: torch.device = torch.device('cuda'),
        use_rollout: bool = False,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        # Initialize hyperparameters and functions

        # if kwargs:
            # logging.warning(f"FKD Steering - Unused arguments: {kwargs}")

        self.potential_type = PotentialType(potential_type)
        self.lmbda = lmbda
        self.num_particles = num_particles
        self.adaptive_resampling = adaptive_resampling
        self.resample_frequency = resample_frequency
        self.resampling_t_start = resampling_t_start
        self.resampling_t_end = resampling_t_end
        self.time_steps = time_steps
        self.use_rollout = use_rollout
        self.batch_size = batch_size
        self.reward_fn = reward_fn
        self.latent_to_decode_fn = latent_to_decode_fn

        # Initialize device and population reward state
        self.device = device

        # initial rewards
        self.population_rs = (
            torch.ones(self.num_particles, device=self.device) * reward_min_value
        )
        self.product_of_potentials = torch.ones(self.num_particles).to(self.device)

    def is_potential_resampling_step(self, sampling_idx: int) -> bool:
        """
        Checks if the current sampling index is a step where resampling might occur.
        """
        current_timestep_index = sampling_idx
        is_last_step = current_timestep_index == self.time_steps - 1
        is_within_interval = self.resampling_t_start <= current_timestep_index <= self.resampling_t_end
        is_on_frequency = (current_timestep_index - self.resampling_t_start) % self.resample_frequency == 0

        return is_last_step or (is_within_interval and is_on_frequency)

    def resample(
        self, *, sampling_idx: int, latents: torch.Tensor, x0_preds_for_reward: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform resampling of particles (latents) based on calculated rewards and weights.

        Args:
            sampling_idx: Current sampling index (0 to time_steps-1).
            latents: Current noisy latent states x_t (Tensor[B, C, H, W]).
            x0_preds_for_reward: Predictions for clean latent x0 used for reward calculation
                                 (either direct prediction or from rollout). Tensor[B, C, H, W].

        Returns:
            A tuple containing resampled latents and optionally resampled images.
        """
        # Decode latents to population images and compute rewards using the provided x0 predictions
        # do this in batches
        rs_candidates = torch.zeros(self.num_particles, device=self.device)
        for i in range(0, self.num_particles, self.batch_size):
            batch_x0_preds_for_reward = x0_preds_for_reward[i:i+self.batch_size]
            population_images = self.latent_to_decode_fn(batch_x0_preds_for_reward)
            batch_rs_candidates = self.reward_fn(population_images)
            rs_candidates[i:i+self.batch_size] = batch_rs_candidates

        # Compute importance weights
        if self.potential_type == PotentialType.MAX:
            w = torch.exp(self.lmbda * torch.max(rs_candidates, self.population_rs))
        elif self.potential_type == PotentialType.ADD:
            rs_candidates_combined = rs_candidates + self.population_rs
            w = torch.exp(self.lmbda * rs_candidates_combined)
        elif self.potential_type == PotentialType.DIFF:
            diffs = rs_candidates - self.population_rs
            w = torch.exp(self.lmbda * diffs)
        else:
            raise ValueError(f"potential_type {self.potential_type} not recognized")

        if sampling_idx == self.time_steps - 1:
            if self.potential_type == PotentialType.MAX:
                w = torch.exp(self.lmbda * torch.max(rs_candidates, self.population_rs)) / self.product_of_potentials
            elif self.potential_type == PotentialType.ADD:
                w = torch.exp(self.lmbda * rs_candidates_combined) / self.product_of_potentials

        w = torch.clamp(w, 0, 1e10)
        w[torch.isnan(w)] = 0.0

        if self.adaptive_resampling or sampling_idx == self.time_steps - 1:
            # compute effective sample size
            normalized_w = w / w.sum()
            ess = 1.0 / (normalized_w.pow(2).sum())

            if ess < 0.5 * self.num_particles:
                print(f"Resampling at timestep {sampling_idx} with ESS: {ess}")
                # Resample indices based on weights
                indices = torch.multinomial(
                    w, num_samples=self.num_particles, replacement=True
                )
                resampled_latents = latents[indices]
                self.population_rs = rs_candidates[indices]

                # Update product of potentials
                self.product_of_potentials = (
                    self.product_of_potentials[indices] * w[indices]
                )
            else:
                # No resampling
                resampled_latents = latents
                self.population_rs = rs_candidates

        else:
            # Resample indices based on weights
            indices = torch.multinomial(
                w, num_samples=self.num_particles, replacement=True
            )
            resampled_latents = latents[indices]
            self.population_rs = rs_candidates[indices]

            # Update product of potentials
            self.product_of_potentials = (
                self.product_of_potentials[indices] * w[indices]
            )

        return resampled_latents

'''
########################################################
Diffusion pipeline
########################################################
'''

# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

from src.core.rewards import get_reward_function
from src.core.smc_weights import SMCWeights

import torch
from packaging import version
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class FKDStableDiffusion(
    StableDiffusionPipeline,
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    StableDiffusionLoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )

    @torch.no_grad()
    def _perform_rollout(
        self,
        start_latents: torch.Tensor, # Shape [1, C, H, W]
        start_timestep_index: int,
        all_timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        eta: float = 0.0,
        guidance_scale: float = 7.5,
        guidance_rescale: float = 0.0,
        added_cond_kwargs: Optional[Dict] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Performs a full SD rollout from a given starting latent state and timestep.

        Args:
            start_latents: The starting noisy latent state (x_t) for a single particle.
            start_timestep_index: The index in `all_timesteps` corresponding to the starting state.
            all_timesteps: The full list of timesteps used in the main sampling loop.
            prompt_embeds: Conditional and unconditional embeddings.
            generator: Random generator.
            eta: DDIM eta parameter.
            guidance_scale: Classifier-free guidance scale.
            guidance_rescale: Guidance rescale factor.
            added_cond_kwargs: Additional conditioning kwargs for the UNet.
            timestep_cond: Timestep conditioning.
            cross_attention_kwargs: Cross attention kwargs.

        Returns:
            The predicted original latent sample (x0_pred) tensor after the rollout completes.
            Shape [1, C, H, W].
        """
        latents = start_latents.clone()
        rollout_timesteps = all_timesteps[start_timestep_index:]
        do_classifier_free_guidance = guidance_scale > 1.0

        for i, t in enumerate(rollout_timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2)
                if do_classifier_free_guidance
                else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(
                    noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                )

            # compute the previous noisy sample x_t -> x_t-1
            # Important: We need eta and generator for scheduler step
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta) # Re-prepare for rollout steps
            step_output = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs, return_dict=True
            )
            latents = step_output.prev_sample
            x0_pred = step_output.pred_original_sample

        return x0_pred

    @torch.no_grad()
    def _perform_batched_rollout(
        self,
        start_latents_population: torch.Tensor,  # Shape [N, C, H, W]
        start_timestep_index: int,
        all_timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        processing_batch_size: int,
        generator: Optional[torch.Generator] = None,
        eta: float = 0.0,
        guidance_scale: float = 7.5,
        guidance_rescale: float = 0.0,
        added_cond_kwargs: Optional[Dict] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict] = None,
        use_progress_bar: bool = False,
    ) -> torch.Tensor:
        """
        Performs DDIM rollouts in batches from given starting latent states and timestep index.

        Args:
            start_latents_population: Starting noisy latents (x_t) for the full population. [N, C, H, W]
            start_timestep_index: Index in `all_timesteps` corresponding to the starting state.
            all_timesteps: Full list of timesteps for the main sampling loop.
            prompt_embeds: Conditional and unconditional embeddings.
            processing_batch_size: Batch size for processing rollouts.
            generator: Random generator.
            eta: DDIM eta parameter.
            guidance_scale: Classifier-free guidance scale.
            guidance_rescale: Guidance rescale factor.
            added_cond_kwargs: Additional conditioning kwargs for the UNet.
            timestep_cond: Timestep conditioning.
            cross_attention_kwargs: Cross attention kwargs.
            use_progress_bar: Whether to show a progress bar for rollout batches.

        Returns:
            Predicted original latent samples (x0_pred) for the full population. [N, C, H, W]
        """
        num_particles = start_latents_population.shape[0]
        all_final_x0_preds = torch.zeros_like(start_latents_population)
        rollout_timesteps = all_timesteps[start_timestep_index:]
        device = start_latents_population.device
        do_classifier_free_guidance = guidance_scale > 1.0

        iterator = range(0, num_particles, processing_batch_size)
        if use_progress_bar:
            from tqdm.auto import tqdm  # Import tqdm here to avoid making it a hard dependency
            iterator = tqdm(iterator, desc=f"Rollout Batches (Step {start_timestep_index})", leave=False)

        for batch_start_idx in iterator:
            batch_end_idx = min(batch_start_idx + processing_batch_size, num_particles)
            current_batch_latents = start_latents_population[batch_start_idx:batch_end_idx].clone().to(device)
            
            # Adapt prompt_embeds for the current batch size if necessary
            # prompt_embeds should be [num_particles*2, seq, dim] if CFG, or [num_particles, seq, dim] if no CFG
            if do_classifier_free_guidance:
                # For CFG, we need to slice both unconditional and conditional embeddings
                # prompt_embeds layout: [uncond_0, cond_0, uncond_1, cond_1, ...]
                current_batch_prompt_embeds = prompt_embeds[batch_start_idx*2 : batch_end_idx*2]
            else:
                current_batch_prompt_embeds = prompt_embeds[batch_start_idx:batch_end_idx]
            
            # Handle added_cond_kwargs and timestep_cond for batching if they are per-particle
            # For now, assume they are broadcastable or correctly sliced.
            current_batch_added_cond_kwargs = added_cond_kwargs
            current_batch_timestep_cond = timestep_cond
            if timestep_cond is not None and timestep_cond.shape[0] == num_particles * 2 : # CFG
                 current_batch_timestep_cond = timestep_cond[batch_start_idx*2 : batch_end_idx*2] if do_classifier_free_guidance else timestep_cond[batch_start_idx:batch_end_idx]
            elif timestep_cond is not None and timestep_cond.shape[0] == num_particles: # No CFG
                 current_batch_timestep_cond = timestep_cond[batch_start_idx:batch_end_idx]
            
            if added_cond_kwargs is not None and "image_embeds" in added_cond_kwargs:
                img_embeds = added_cond_kwargs["image_embeds"]
                if img_embeds.shape[0] == num_particles *2: # CFG
                    current_batch_img_embeds = img_embeds[batch_start_idx*2:batch_end_idx*2] if do_classifier_free_guidance else img_embeds[batch_start_idx:batch_end_idx]
                elif img_embeds.shape[0] == num_particles: # No CFG
                    current_batch_img_embeds = img_embeds[batch_start_idx:batch_end_idx]
                else: # Assume broadcast
                    current_batch_img_embeds = img_embeds
                current_batch_added_cond_kwargs = {"image_embeds": current_batch_img_embeds}


            for _, t in enumerate(rollout_timesteps):
                latent_model_input = (
                    torch.cat([current_batch_latents] * 2)
                    if do_classifier_free_guidance
                    else current_batch_latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred_batch = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=current_batch_prompt_embeds, # Use potentially batched prompt_embeds
                    timestep_cond=current_batch_timestep_cond, # Use potentially batched timestep_cond
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=current_batch_added_cond_kwargs, # Use potentially batched added_cond_kwargs
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond_batch, noise_pred_text_batch = noise_pred_batch.chunk(2)
                    noise_pred_batch = noise_pred_uncond_batch + guidance_scale * (
                        noise_pred_text_batch - noise_pred_uncond_batch
                    )

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    noise_pred_batch = rescale_noise_cfg(
                        noise_pred_batch, noise_pred_text_batch, guidance_rescale=guidance_rescale
                    )
                
                # Prepare extra_step_kwargs for this batch. Generator state needs to be handled carefully if used.
                # For simplicity, re-preparing, but for full determinism with a single generator, one might need to pass sub-generators or handle seed.
                batch_extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)


                step_output_batch = self.scheduler.step(
                    noise_pred_batch, t, current_batch_latents, **batch_extra_step_kwargs, return_dict=True
                )
                current_batch_latents = step_output_batch.prev_sample
                batch_x0_pred = step_output_batch.pred_original_sample
            
            all_final_x0_preds[batch_start_idx:batch_end_idx] = batch_x0_pred.to(all_final_x0_preds.device)

        return all_final_x0_preds

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        fkd_args: Optional[Dict[str, Any]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            
            FK Steering Addition:
            fkd_args (`dict`, *optional*):
                The arguments to be passed to the FKD class. If not defined, FKD will not be used.

            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # Added for FK Steering
        def postprocess_and_apply_reward_fn(x):
            # convert to pil image
            imagesx = self.image_processor.postprocess(x, output_type=output_type)
            imagesx = [image for image in imagesx]
            rewards = get_reward_function(
                fkd_args["guidance_reward_fn"], 
                images=imagesx, 
                prompts=prompt, 
                metric_to_chase=fkd_args.get("metric_to_chase", None)
            )
            return torch.tensor(rewards).to(x.device)

        if fkd_args is not None and fkd_args['use_smc']:
            fkd = FKD(
                latent_to_decode_fn=lambda x: latent_to_decode(
                    model=self, output_type=output_type, latents=x
                ),
                reward_fn=postprocess_and_apply_reward_fn,
                **fkd_args,
            )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1

                # FK Steering Change
                # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                step_dict = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=True
                )

                # FK Steering Change
                prev_latents = step_dict["prev_sample"]
                x0_preds = step_dict["pred_original_sample"]

                # FK Steering Change
                if fkd_args is not None and fkd_args["use_smc"]:
                    # Determine if this step *could* involve resampling
                    should_resample = fkd.is_potential_resampling_step(i)
                    if should_resample:
                        # Determine the x0 prediction to use for reward calculation
                        if fkd.use_rollout:
                            logger.info(f"Performing rollouts for resampling at step {i} (t={t.item()})...")
                            # Perform rollout
                            x0_for_reward_calculation = self._perform_rollout(
                                start_latents=latents,
                                start_timestep_index=i,
                                all_timesteps=timesteps,
                                prompt_embeds=prompt_embeds,
                                generator=generator,
                                eta=eta,
                                guidance_scale=self.guidance_scale,
                                guidance_rescale=self.guidance_rescale,
                                added_cond_kwargs=added_cond_kwargs,
                                timestep_cond=timestep_cond,
                                cross_attention_kwargs=self.cross_attention_kwargs,
                            )
                            logger.info(f"Rollouts complete for step {i}.")
                        else:
                            # Use the one-shot x0 prediction from the current step
                            x0_for_reward_calculation = x0_preds

                    latents = prev_latents

                    if should_resample:
                        latents = fkd.resample(
                            sampling_idx=i,
                            latents=latents,
                            x0_preds_for_reward=x0_for_reward_calculation
                        )

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[0]
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            has_nsfw_concept = None
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )

    @torch.no_grad()
    def sample_batched_smc(
        self,
        num_particles: int,
        prompt: Union[str, List[str]],
        fkd_args: Dict[str, Any], # Mandatory, must include num_particles
        processing_batch_size: int, # For UNet and rollout processing
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # num_images_per_prompt is implicitly 1 as fkd_args['num_particles'] defines the population size
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None, # Initial latents for the entire population
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        use_rollout_progress_bar: bool = False, # For batched rollout verbosity
        **kwargs,
    ):
        r"""
        Batched call function to the pipeline for generation using FKD-SMC.
        Args:
            prompt (`str` or `List[str]`): The prompt to guide image generation.
            fkd_args (`dict`): Arguments for FKD, must include 'num_particles'.
            processing_batch_size (`int`): Batch size for UNet calls and batched rollouts.
            Other arguments are similar to the standard __call__ method.
        """
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        if callback is not None: deprecate("callback", "1.0.0", "Use `callback_on_step_end`")
        if callback_steps is not None: deprecate("callback_steps", "1.0.0", "Use `callback_on_step_end`")

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        
        if not fkd_args:
            raise ValueError("`fkd_args` must be provided.")

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        fkd_args["num_particles"] = num_particles

        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt,
            prompt_embeds, negative_prompt_embeds, ip_adapter_image, ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        if isinstance(prompt, str):
            prompt_list = [prompt] * num_particles # Repeat prompt for each particle
        elif isinstance(prompt, list) and len(prompt) == 1:
            prompt_list = prompt * num_particles
        elif isinstance(prompt, list) and len(prompt) == num_particles:
            prompt_list = prompt # Each particle has its own prompt
        else:
            raise ValueError("Prompt must be a string, a list of one string, or a list of `num_particles` strings.")

        # For batched processing, batch_size is effectively num_particles for initial setup,
        # but prompt encoding should handle it. num_images_per_prompt is 1.
        # The prompt_embeds will be [num_particles * 2, seq, dim] if CFG, or [num_particles, seq, dim]
        
        device = self._execution_device
        lora_scale = self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None

        # Encode prompts for the entire population
        # If a single prompt is given, it's encoded once and then should be tiled.
        # If multiple prompts (one per particle), they are encoded.
        # The encode_prompt handles num_images_per_prompt, which we set to 1 here,
        # and batch_size for it is the number of unique prompts.
        # We then manage tiling for particles if needed.

        # Let's simplify: encode a single prompt and expect it to be used for all particles,
        # or expect pre-computed prompt_embeds for the population.
        # For now, assume `prompt` is a single string or list of 1, and we tile embeddings.
        # `encode_prompt` is called with `batch_size=1` effectively if prompt is `str`.
        # We need to ensure prompt_embeds are then [num_particles * 2, ...]

        single_prompt_str = prompt[0] if isinstance(prompt, list) else prompt
        single_negative_prompt_str = negative_prompt[0] if isinstance(negative_prompt, list) else negative_prompt
        
        _prompt_embeds, _negative_prompt_embeds = self.encode_prompt(
            single_prompt_str, device, 1, self.do_classifier_free_guidance, single_negative_prompt_str,
            prompt_embeds=None, negative_prompt_embeds=None, # Force re-encoding for simplicity here
            lora_scale=lora_scale, clip_skip=self.clip_skip
        )

        if self.do_classifier_free_guidance:
            # _prompt_embeds is [1, seq, dim], _negative_prompt_embeds is [1, seq, dim]
            # For CFG, we need: [uncond_0, cond_0, uncond_1, cond_1, ...]
            # First tile each embedding separately
            tiled_negative_embeds = _negative_prompt_embeds.repeat(num_particles, 1, 1)  # [num_particles, seq, dim]
            tiled_positive_embeds = _prompt_embeds.repeat(num_particles, 1, 1)  # [num_particles, seq, dim]
            # Then interleave: stack and reshape to get [uncond_0, cond_0, uncond_1, cond_1, ...]
            prompt_embeds_population = torch.stack([tiled_negative_embeds, tiled_positive_embeds], dim=1).reshape(-1, *_prompt_embeds.shape[1:])
        else:
            # Tile for population: [num_particles, seq, dim]
            prompt_embeds_population = _prompt_embeds.repeat(num_particles, 1, 1)


        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            # Assuming ip_adapter_image is single, tile image_embeds for population
            # prepare_ip_adapter_image_embeds takes batch_size, which should be num_particles
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image, ip_adapter_image_embeds, device,
                num_particles, # batch_size for IP adapter
                self.do_classifier_free_guidance
            ) # Output shape is [ (2 if CFG else 1) * num_particles, num_images, embed_dim ]
        else:
            image_embeds = None
            
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        num_channels_latents = self.unet.config.in_channels
        # latents here are for the whole population
        latents = self.prepare_latents(
            num_particles, num_channels_latents, height, width,
            prompt_embeds_population.dtype, device, generator, latents
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        added_cond_kwargs = {"image_embeds": image_embeds} if image_embeds is not None else {}

        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(num_particles)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
            if self.do_classifier_free_guidance: # timestep_cond is [num_particles, dim]
                # Needs to be [num_particles*2, dim] for CFG input to UNet if UNet expects batched cond
                # Or, it's handled inside UNet. Assuming it's added before CFG duplication.
                # Let's check how it's used in the original __call__
                # In original: guidance_scale_tensor.repeat(batch_size * num_images_per_prompt)
                # Here batch_size = num_particles, num_images_per_prompt = 1
                # So timestep_cond is already [num_particles, dim]
                pass

        # Added for FK Steering
        def postprocess_and_apply_reward_fn(x):
            # convert to pil image
            imagesx = self.image_processor.postprocess(x, output_type=output_type)
            imagesx = [image for image in imagesx]
            rewards = get_reward_function(
                fkd_args["guidance_reward_fn"], 
                images=imagesx, 
                prompts=[prompt_list[0]] * len(imagesx), 
                metric_to_chase=fkd_args.get("metric_to_chase", None)
            )
            return torch.tensor(rewards).to(x.device)

        reward_batch_size = fkd_args.get("batch_size", processing_batch_size)

        def batched_reward(x0_latents):
            out = []
            for b in range(0, x0_latents.shape[0], reward_batch_size):
                decoded = latent_to_decode(model=self, output_type=output_type, latents=x0_latents[b:b + reward_batch_size])
                out.append(postprocess_and_apply_reward_fn(decoded).float())
            self.reward_evals += x0_latents.shape[0]
            return torch.cat(out)

        if PotentialType(fkd_args.get("potential_type", "diff")) != PotentialType.DIFF:
            raise NotImplementedError("Only the 'diff' potential is implemented with persistent importance weights.")
        schedule = list(range(fkd_args["resampling_t_start"], fkd_args["resampling_t_end"] + 1, fkd_args["resample_frequency"]))
        weights = SMCWeights(num_particles, lmbda=fkd_args["lmbda"], adaptive_resampling=fkd_args["adaptive_resampling"],
                             resampling_schedule=schedule, device=device,
                             reward_min_value=fkd_args.get("reward_min_value", 0.0))
        use_rollout = fkd_args.get("use_rollout", False)
        self.nfes = 0
        self.reward_evals = 0

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt: continue

                # Batched UNet prediction for the whole population
                population_noise_pred = torch.zeros_like(latents)

                for batch_start_idx in range(0, num_particles, processing_batch_size):
                    batch_end_idx = min(batch_start_idx + processing_batch_size, num_particles)
                    current_latents_batch = latents[batch_start_idx:batch_end_idx]
                    
                    # Slice prompt embeds: each particle has 2 (uncond, cond)
                    current_prompt_embeds_batch = prompt_embeds_population[batch_start_idx*2 : batch_end_idx*2] if self.do_classifier_free_guidance else prompt_embeds_population[batch_start_idx:batch_end_idx]

                    current_added_cond_kwargs_batch = {}
                    if "image_embeds" in added_cond_kwargs and added_cond_kwargs["image_embeds"] is not None:
                        img_embeds_full_pop = added_cond_kwargs["image_embeds"] # num_particles * (2 if CFG)
                        current_img_embeds_batch = img_embeds_full_pop[batch_start_idx*2:batch_end_idx*2] if self.do_classifier_free_guidance else img_embeds_full_pop[batch_start_idx:batch_end_idx]
                        current_added_cond_kwargs_batch["image_embeds"] = current_img_embeds_batch
                    
                    current_timestep_cond_batch = None
                    if timestep_cond is not None: # [num_particles, dim]
                         _ts_cond_batch = timestep_cond[batch_start_idx:batch_end_idx]
                         current_timestep_cond_batch = _ts_cond_batch


                    latent_model_input_batch = (
                        torch.cat([current_latents_batch] * 2) if self.do_classifier_free_guidance else current_latents_batch
                    )
                    latent_model_input_batch = self.scheduler.scale_model_input(latent_model_input_batch, t)

                    noise_pred_batch = self.unet(
                        latent_model_input_batch, t,
                        encoder_hidden_states=current_prompt_embeds_batch,
                        timestep_cond=current_timestep_cond_batch, # Pass sliced batch
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=current_added_cond_kwargs_batch,
                        return_dict=False
                    )[0]

                    if self.do_classifier_free_guidance:
                        noise_pred_uncond_batch, noise_pred_text_batch = noise_pred_batch.chunk(2)
                        noise_pred_batch = noise_pred_uncond_batch + self.guidance_scale * (
                            noise_pred_text_batch - noise_pred_uncond_batch
                        )
                        if self.guidance_rescale > 0.0:
                            noise_pred_batch = rescale_noise_cfg(
                                noise_pred_batch, noise_pred_text_batch, guidance_rescale=self.guidance_rescale
                            )
                    
                    population_noise_pred[batch_start_idx:batch_end_idx] = noise_pred_batch
                    self.nfes += batch_end_idx - batch_start_idx   # one CFG pair = 1 NFE

                # Scheduler step for the whole population
                step_dict = self.scheduler.step(
                    population_noise_pred, t, latents, **extra_step_kwargs, return_dict=True
                )
                prev_latents = step_dict["prev_sample"]
                current_x0_preds = step_dict["pred_original_sample"] # x0_preds for all particles [N, C, H, W]

                if i in schedule:
                    x0_for_reward_calculation: torch.Tensor
                    if use_rollout:
                        logger.info(f"Performing batched rollouts for resampling at step {i} (t={t.item()})...")
                        # _perform_batched_rollout needs prompt_embeds_population and other relevant args
                        x0_for_reward_calculation = self._perform_batched_rollout(
                            start_latents_population=latents, # Current latents xt for the population
                            start_timestep_index=i,
                            all_timesteps=timesteps,
                            prompt_embeds=prompt_embeds_population, # Pass the population prompt embeds
                            processing_batch_size=processing_batch_size,
                            generator=generator, # Pass the main generator
                            eta=eta, # Rollout eta, could be different, using main eta for now
                            guidance_scale=self.guidance_scale,
                            guidance_rescale=self.guidance_rescale,
                            added_cond_kwargs=added_cond_kwargs, # Pass population added_cond_kwargs
                            timestep_cond=timestep_cond, # Pass population timestep_cond
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            use_progress_bar=use_rollout_progress_bar,
                        )
                        logger.info(f"Batched rollouts complete for step {i}.")
                        self.nfes += num_particles * (len(timesteps) - i)
                    else:
                        x0_for_reward_calculation = current_x0_preds

                    # the potential is evaluated at x_t; the pair (x_t, x_{t-1}) is resampled
                    indices = weights.potential_step(i, batched_reward(x0_for_reward_calculation))
                    latents = prev_latents[indices] if indices is not None else prev_latents
                else:
                    latents = prev_latents

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k_cb in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k_cb] = locals()[k_cb]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    # Potentially update other items like prompt_embeds if callback modifies them
                    prompt_embeds_population = callback_outputs.pop("prompt_embeds", prompt_embeds_population)


                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents) # Pass population latents

        # Post-processing
        if not output_type == "latent":
            # Decode all latents in the population
            image = torch.zeros(num_particles, 3, height, width).to(latents.device)
            for i in range(0, num_particles, processing_batch_size):
                batch_latents = latents[i:i+processing_batch_size]
                batch_images = self.vae.decode(batch_latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
                image[i:i+processing_batch_size] = batch_images
            # image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            has_nsfw_concept = None # Simplified, run_safety_checker can be batched if needed
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept] # per image

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # terminal potential lambda * (r(x_0) - r~_last) on the final images
        final_rewards = torch.tensor(get_reward_function(fkd_args["guidance_reward_fn"], images=list(image),
                                                         prompts=[prompt_list[0]] * len(image)), device=device).float()
        weights.final_step(final_rewards)
        self.reward_evals += len(image)

        self.maybe_free_model_hooks()
        return SDSMCOutput(images=image, log_weights=weights.log_w, final_rewards=final_rewards, nfes=self.nfes)


# FK Steering Change
def latent_to_decode(*, model, output_type, latents):
    if not output_type == "latent":
        # # make sure the VAE is in float32 mode, as it overflows in float16
        # needs_upcasting = model.vae.dtype == torch.float16 and model.vae.config.force_upcast

        # if needs_upcasting:
        #     model.upcast_vae()
        #     latents = latents.to(next(iter(model.vae.post_quant_conv.parameters())).dtype)
        if latents.dtype != model.vae.dtype:
            if torch.backends.mps.is_available():
                # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                model.vae = model.vae.to(latents.dtype)

        # unscale/denormalize the latents
        # denormalize with the mean and std if available and not None
        has_latents_mean = (
            hasattr(model.vae.config, "latents_mean")
            and model.vae.config.latents_mean is not None
        )
        has_latents_std = (
            hasattr(model.vae.config, "latents_std")
            and model.vae.config.latents_std is not None
        )
        if has_latents_mean and has_latents_std:
            latents_mean = (
                torch.tensor(model.vae.config.latents_mean)
                .view(1, 4, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(model.vae.config.latents_std)
                .view(1, 4, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = (
                latents * latents_std / model.vae.config.scaling_factor + latents_mean
            )
        else:
            latents = latents / model.vae.config.scaling_factor

        image = model.vae.decode(latents, return_dict=False)[0]
    else:
        image = latents

    return image




