import torch
import torch_tensorrt
import numpy as np
import math
import tqdm
import random
import os
import time
import sys
import warnings
import logging
from collections import defaultdict
from typing import Callable, List, Optional, Union
import matplotlib.pyplot as plt
import networkx as nx
import inspect


logging.getLogger("torch_tensorrt.dynamo.conversion._TRTInterpreter").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="Both operands of the binary elementwise op .* are constant")

from diffusers import DDIMPipeline, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from src.utils.pipeline_utils import get_variance, _left_broadcast


# Global cache for shared pipelines: key = (model_id, device)
_pipeline_cache = {}

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Retrieves the scheduler timesteps for the diffusion process.
    
    Args:
        scheduler: The diffusion scheduler
        num_inference_steps: Number of diffusion steps
        device: The device to put the timesteps on
        timesteps: Optional custom timesteps
        sigmas: Optional custom sigmas
    
    Returns:
        Tuple[torch.Tensor, int]: A tuple with the timesteps tensor and the actual number of steps
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accepts_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_sigmas:
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


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescales the predicted noise based on classifier-free guidance.
    """
    if guidance_rescale == 0:
        return noise_cfg
    
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class MCTSNode:
    """
    Each node represents a latent state x_t at diffusion timestep t.
    Children are possible next states x_{t-1}.
    """

    def __init__(self, x_t, t, parent=None, timestep_value=None):
        """
        Args:
          x_t (torch.Tensor): The noisy latent at diffusion timestep t.
          t (int): The current diffusion timestep index.
          parent (MCTSNode or None): The parent node. None if root.
          timestep_value (float or None): The actual scheduler timestep value if known.
        """
        self.x_t = x_t          # shape: [B, C, H, W] for images
        self.t = t               # This is the timestep index
        self.timestep_value = timestep_value  # This is the actual scheduler timestep value
        self.parent = parent
        self.children = []      # list of child nodes
        self.visit_count = 1    # N(x_t) in the algorithm
        self.value_est = 0.0    # Estimated value of this node
        self.noise_pred = None

    def is_leaf(self):
        return (self.t <= 0) or (len(self.children) == 0)
        
    def update_value(self, value):
        self.value_est = value


class MetaRootNode(MCTSNode):
    """
    A special meta-root node that doesn't correspond to any actual diffusion timestep.
    Its children are created by sampling from random noise.
    """
    def __init__(self, sampler):
        """
        Args:
            sampler: The MCTSStableDiffusionSampler instance to use for sampling children
        """
        # Use a timestep index that's one more than the maximum diffusion timestep
        virtual_timestep = len(sampler.timesteps)
        # Create a dummy tensor with the right shape based on args
        # Use pipeline.unet.config for shape
        unet_config = sampler.pipeline.unet.config
        image_shape = (
            1, 
            unet_config.in_channels, 
            unet_config.sample_size, 
            unet_config.sample_size
        )
        dummy_x = torch.zeros(image_shape, device=sampler.device)

        # Initialize with the virtual timestep and dummy tensor
        super().__init__(dummy_x, virtual_timestep, parent=None, timestep_value=None)


class MCTSImageDiffusionSampler:
    """
    MCTS-style sampler for pixel-space diffusion models with a black-box reward.
    """

    def __init__(self, 
                 pipeline: DDIMPipeline,
                 reward_fn: Callable,
                 expand_timesteps: List[int], 
                 num_nfes: int = 1000,
                 num_inference_steps: int = 50,
                 eta: float = 1.0,
                 device: str = 'cuda',
                 exploration_const: float = 1.0,
                 progressive_width_const: float = 1.0,
                 progressive_width_alpha: float = 0.5,
                 initial_exploration_prob: float = 0.0,
                 final_exploration_prob: float = 0.0,
                 use_progress_bar: bool = True,
                 use_rollout: bool = False,
                 use_guidance: bool = False,
                 guidance_scale: float = 1.0,
                 maximize: bool = False,
                 cache_noise_pred: bool = True,
                 exploration_type: str = 'epsilon',
                 batch_size: int = 4,
                 output_dir: str = None,
                 visualization_timesteps: List[int] = None):
        """
        Args:
          reward_fn: Function that takes images (Tensor[B, C, H, W]) and returns scalar rewards.
          expand_timesteps: Timesteps at which we do MCTS expansion.
          num_nfes: Total MCTS NFE.
          exploration_const: UCB exploration constant (lambda).
          progressive_width_const: C in progressive widening.
          progressive_width_alpha: Alpha in progressive widening.
          initial_exploration_prob: Initial epsilon for epsilon-greedy.
          final_exploration_prob: Final epsilon for epsilon-greedy.
          use_progress_bar: Whether to use tqdm progress bars.
          use_rollout: Whether to use full rollout or estimate value.
          exploration_type: 'epsilon' or 'ucb'.
          batch_size: Number of nodes to process in parallel per iteration.
          output_dir: Directory to save visualizations and results.
          visualization_timesteps: Timesteps to include in visualization.
        """
        self.pipeline = pipeline
        self.device = device
        self.reward_fn = reward_fn
        self.expand_timesteps = sorted(expand_timesteps, reverse=True)
        self.num_nfes = num_nfes
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        self.c_ucb = exploration_const
        self.progressive_width_const = progressive_width_const
        self.progressive_width_alpha = progressive_width_alpha
        self.initial_exploration_prob = initial_exploration_prob
        self.final_exploration_prob = final_exploration_prob
        self.current_exploration_prob = initial_exploration_prob
        self.use_progress_bar = use_progress_bar
        self.use_rollout = use_rollout
        self.use_guidance = use_guidance
        self.guidance_scale = guidance_scale
        self.maximize = maximize
        self.cache_noise_pred = cache_noise_pred
        self.exploration_type = exploration_type
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.visualization_timesteps = visualization_timesteps
        
        # These will be set up in setup_diffusion
        self.extra_step_kwargs = {}
        
        # Load pipeline components
        # self._load_sd_pipeline(model_name_or_path)
        self.setup_diffusion()

        self.nfes = 0
        self.meta_root = None

    def _load_sd_pipeline(self, model_name_or_path):
        """
        Load the Diffusion pipeline with DDIM scheduler.
        """
        # Reuse existing pipeline if already loaded for this model and device
        cache_key = (model_name_or_path, self.device)
        if cache_key in _pipeline_cache:
            self.pipeline = _pipeline_cache[cache_key]
            return
        # Otherwise, load and compile
        pipe = DDIMPipeline.from_pretrained(model_name_or_path)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to(self.device)

        backend = "torch_tensorrt"
        fast = True

        if fast:
            # Optimize the UNet portion with Torch-TensorRT
            pipe.unet = torch.compile(
                pipe.unet,
                backend=backend,
                options={
                    "truncate_long_and_double": True,
                    "enabled_precisions": {torch.float32, torch.float16},
                },
                dynamic=False,
            )
        
        # (After compile) cache for reuse
        _pipeline_cache[cache_key] = pipe
        self.pipeline = pipe
    
    def setup_diffusion(self):
        """
        Set up preprocessing for diffusion to avoid redundant calculations during tree expansion.
        """
        # print(f"Setting up diffusion process...")
        
        # Get the scheduler timesteps using proper pipeline method
        # timesteps, self.num_inference_steps = retrieve_timesteps(
        #     self.pipeline.scheduler, 
        #     self.num_inference_steps, 
        #     self.device,
        #     timesteps=None, 
        #     sigmas=None
        # )
        timesteps = self.pipeline.scheduler.timesteps
        self.timesteps = torch.flip(timesteps, [0])
        
    def get_timestep_from_index(self, index):
        """
        Get the actual diffusion timestep value from a timestep index.
        """
        # Make sure index is within bounds
        index = min(max(0, index), len(self.timesteps) - 1)
        return self.timesteps[index].item()
    
    def _initialize_meta_root(self):
        """
        Initialize a single meta-root node as the starting point for the search.
        This meta-root is at a virtual timestep beyond the normal diffusion process.
        """
        self.meta_root = MetaRootNode(self)
        return self.meta_root
    
    def _is_meta_root(self, node):
        return node.t >= len(self.timesteps)
    
    def sample_noise(self, batch_size=1):
        """
        Generate initial noise matching the pipeline's UNet configuration.
        """
        unet_config = self.pipeline.unet.config
        image_shape = (
            batch_size,
            unet_config.in_channels,
            unet_config.sample_size,
            unet_config.sample_size,
        )
        return randn_tensor(
            image_shape, generator=None, device=self.device, dtype=self.pipeline.unet.dtype,
        )
    
    def p_sample(self, x_t, t, predict_x0=False, eta=None):
        """
        Perform one step of the diffusion process using the pipeline's components.
        Args:
            x_t (torch.Tensor): Current noisy image tensor [B, C, H, W].
            t_idx (Union[int, torch.Tensor, list]): MCTS timestep *index* (0 to T-1).
            predict_x0 (bool): If True, return the predicted x0 instead of x_{t-1}.
        """
        batch_size = x_t.shape[0]
        if eta is None:
            eta = self.eta
        # choose branch
        if self.use_guidance and not predict_x0:
            # enable gradient inside guidance
            with torch.enable_grad():
                x_t_guided = x_t.clone().detach().requires_grad_(True)
                # determine timesteps
                t_idx = t if isinstance(t, int) else (t.item() if isinstance(t, torch.Tensor) and t.numel()==1 else t[0])
                timestep = self.get_timestep_from_index(t_idx)
                prev_t = timestep - self.pipeline.scheduler.config.num_train_timesteps // self.pipeline.scheduler.num_inference_steps
                # prev_idx = max(0, int(t_idx) + 1)
                # prev_t = self.get_timestep_from_index(prev_idx)
                timestep_tensor = torch.tensor([timestep], device=self.device)
                # UNet + scheduler
                noise_pred = self.pipeline.unet(x_t_guided, timestep_tensor).sample
                step_output = self.pipeline.scheduler.step(noise_pred, timestep, x_t_guided, eta=eta)
                prev_sample = step_output.prev_sample
                x0_pred = step_output.pred_original_sample
                # gradient of reward
                reward = self.reward_fn(x0_pred)
                guidance_grad = torch.autograd.grad(reward.sum(), x_t_guided)[0].detach()
                guidance_grad = torch.nan_to_num(guidance_grad)
                x_t_guided.requires_grad_(False)
                # variance scaling: convert indices to tensors
                t_var = torch.tensor([timestep], dtype=torch.long, device=self.device)
                prev_var = torch.tensor([prev_t], dtype=torch.long, device=self.device)
                var = get_variance(self.pipeline.scheduler, t_var, prev_var)
                var = ((eta**2) * _left_broadcast(var, prev_sample.shape)).to(self.device)
                pred = prev_sample + self.guidance_scale * var * guidance_grad
            # update count
            self.nfes += batch_size
            return pred.detach(), noise_pred.detach()
        else:
            # standard step without guidance
            with torch.no_grad():
                # timestep extraction
                if isinstance(t, (int, float)):
                    timestep = self.get_timestep_from_index(t)
                elif isinstance(t, torch.Tensor):
                    timestep = self.get_timestep_from_index(t.item() if t.numel()==1 else t[0].item())
                elif isinstance(t, list):
                    timestep = self.get_timestep_from_index(t[0])
                else:
                    raise ValueError(f"Unsupported timestep type: {type(t)}")
                timestep_tensor = torch.tensor([timestep], device=self.device)
                noise_pred = self.pipeline.unet(x_t, timestep_tensor).sample
                step_output = self.pipeline.scheduler.step(noise_pred, timestep, x_t, eta=eta)
                if not predict_x0:
                    pred = step_output.prev_sample
                else:
                    pred = step_output.pred_original_sample
            self.nfes += batch_size
            return pred, noise_pred
    
    def expand_nodes(self, nodes, t, predict_x0=False, eta=None):
        """
        Expand nodes by sampling from the diffusion model.
        """
        # Prepare parameters
        if eta is None:
            eta = self.eta
        # Determine timestep value
        if isinstance(t, (int, float)):
            timestep = self.get_timestep_from_index(t)
        elif isinstance(t, torch.Tensor):
            timestep = self.get_timestep_from_index(t.item() if t.numel() == 1 else t[0].item())
        elif isinstance(t, list):
            timestep = self.get_timestep_from_index(t[0])
        else:
            raise ValueError(f"Unsupported timestep type: {type(t)}")
        # Prepare batch
        x_t_batch = torch.cat([node.x_t for node in nodes], dim=0)
        batch_size = x_t_batch.shape[0]
        timestep_tensor = torch.tensor([timestep], device=self.device)
        # UNet and scheduler step
        if self.use_guidance:
            with torch.enable_grad():
                x_t_guided = x_t_batch.clone().detach().requires_grad_(True)
                noise_pred = self.pipeline.unet(x_t_guided, timestep_tensor).sample
                step_output = self.pipeline.scheduler.step(noise_pred, timestep, x_t_guided, eta=eta)
                prev_sample = step_output.prev_sample
                x0_pred = step_output.pred_original_sample
                # reward gradient
                reward = self.reward_fn(x0_pred)
                guidance_grad = torch.autograd.grad(reward.sum(), x_t_guided)[0].detach()
                guidance_grad = torch.nan_to_num(guidance_grad)
                x_t_guided.requires_grad_(False)
                # variance scaling: convert indices to tensors
                # t_idx = t if isinstance(t, int) else (t.item() if isinstance(t, torch.Tensor) and t.numel()==1 else t[0])
                # prev_idx = max(0, int(t_idx) + 1)
                # prev_t = self.get_timestep_from_index(prev_idx)
                prev_t = timestep - self.pipeline.scheduler.config.num_train_timesteps // self.pipeline.scheduler.num_inference_steps
                t_var = torch.tensor([timestep], dtype=torch.long, device=self.device)
                prev_var = torch.tensor([prev_t], dtype=torch.long, device=self.device)
                var = get_variance(self.pipeline.scheduler, t_var, prev_var)
                var = (eta**2 * _left_broadcast(var, prev_sample.shape)).to(self.device)
                pred_tm1 = prev_sample + self.guidance_scale * var * guidance_grad
        else:
            with torch.no_grad():
                noise_pred = self.pipeline.unet(x_t_batch, timestep_tensor).sample
                step_output = self.pipeline.scheduler.step(noise_pred, timestep, x_t_batch, eta=eta)
                prev_sample = step_output.prev_sample
                x0_pred = step_output.pred_original_sample
                pred_tm1 = prev_sample
        # Update count and cache noise predictions
        self.nfes += batch_size
        if self.cache_noise_pred:
            for i, node in enumerate(nodes):
                node.noise_pred = noise_pred[i:i+1]
        # Return either x_{t-1} or (x_{t-1}, x0)
        if not predict_x0:
            return pred_tm1
        else:
            return pred_tm1, x0_pred
    
    def ddim_denoise(self, x_t, t):
        """
        Deterministically denoise x_t to x_0 using DDIM scheduler from diffusers.
        Follows the same implementation as the pipeline.
        """
        return self.p_sample(x_t, t, predict_x0=True)
    
    def _ddim_rollout_value(self, x_t, t):
        """
        Evaluate intermediate nodes by deterministically rolling out x_t to predicted x_0
        and computing reward(x_0).
        Args:
            x_t (torch.Tensor): Current noisy image tensor [B, C, H, W].
            t_idx (Union[int, torch.Tensor, list]): MCTS timestep *index* (0 to T-1).
        """
        with torch.no_grad():
            x_0_approx, noise_pred = self.ddim_denoise(x_t, t)
            # Use raw log-space rewards instead of exponentiated rewards
            # r_values = np.exp(self.reward_fn(x_0_approx))
            r_values = self.reward_fn(x_0_approx)
            
        return r_values, noise_pred
    
    def _select_child_by_value(self, node, use_exploration=True):
        """
        Select a child of 'node' proportional to its value.
        With probability p, select a random child if exploration is enabled.
        Return the selected child_node.
        """
        if not node.children:
            return None
        
        if len(node.children) == 1:
            return node.children[0]
        
        if self.exploration_type == 'epsilon':
            # Epsilon-greedy exploration
            if use_exploration and random.random() < self.current_exploration_prob:
                return random.choice(node.children)
                
            # Otherwise, select based on value (exploitation)
            # Compute selection probabilities via softmax on log-space values
            # values = torch.tensor(np.array([max(1e-6, child.value_est) for child in node.children]))
            values = torch.tensor([child.value_est for child in node.children], dtype=torch.float32)
            max_val = values.max()
            exp_vals = torch.exp(values - max_val)
            if self.maximize:
                idx = torch.argmax(values).item()
            else:
                probs = exp_vals / exp_vals.sum()
                idx = torch.multinomial(probs, 1).item()
            return node.children[idx]
        
        elif self.exploration_type == 'ucb':
            # UCB exploration
            if not use_exploration:
                # Compute selection probabilities via softmax on log-space values
                # values = torch.tensor(np.array([max(1e-6, child.value_est) for child in node.children]))
                values = torch.tensor([child.value_est for child in node.children], dtype=torch.float32)
                max_val = values.max()
                exp_vals = torch.exp(values - max_val)
                probs = exp_vals / exp_vals.sum()
                idx = torch.multinomial(probs, 1).item()
                return node.children[idx]
            
            # Calculate UCB scores for each child
            parent_visits = max(1, node.visit_count)  # Avoid division by zero
            
            # Vectorized UCB calculation
            child_visits = torch.tensor(np.array([max(1, child.visit_count) for child in node.children]))
            child_values = torch.tensor(np.array([child.value_est for child in node.children]))
            
            # UCB formula: value + c * sqrt(ln(parent_visits) / child_visits)
            exploration_term = self.c_ucb * torch.sqrt(torch.log(torch.tensor(parent_visits)) / child_visits)
            ucb_scores = child_values + exploration_term
            
            # Find the child with the highest UCB score
            if self.maximize:
                idx = torch.argmax(ucb_scores).item()
            else:
                probs = ucb_scores / ucb_scores.sum()
                idx = torch.multinomial(probs, 1).item()
            return node.children[idx]
        
        else:
            raise ValueError(f"Unknown exploration type: {self.exploration_type}")

    def _can_expand(self, node):
        """
        Check if a node can be expanded based on progressive widening criteria.
        Also handles the special case of the meta-root node.
        """
        # Special case for meta-root
        if self._is_meta_root(node):
            effective_visits = max(1, node.visit_count)
            if self.use_rollout:
                meta_root_alpha = 0.8
            else:
                alpha_min = 0.2
                alpha_max = 0.5
                meta_root_alpha = alpha_min + (alpha_max - alpha_min) * (self.nfes / self.num_nfes)
            max_children = math.floor(self.progressive_width_const * (effective_visits ** meta_root_alpha))
            return len(node.children) < max_children
        
        # For normal nodes:
        if node.t <= 0:  # Can't expand terminal nodes
            return False
        
        if node.t not in self.expand_timesteps:  # Only expand at specified timesteps
            return False
        
        # Ensure at least 1 for the calculation
        effective_visits = max(1, node.visit_count)
        
        # Progressive widening: max_children = C * N(node)^alpha
        
        # progressive_width_alpha = alpha_min + (alpha_max - alpha_min) * (self.num_inference_steps - node.t) / self.num_inference_steps
        if self.use_rollout:
            progressive_width_alpha = 0.8
        else:
            alpha_min = 0.2
            alpha_max = 0.5
            progressive_width_alpha = alpha_min + (alpha_max - alpha_min) * (self.nfes / self.num_nfes)
        max_children = math.floor(self.progressive_width_const * (effective_visits ** progressive_width_alpha))
        
        # Ensure at least 1 child is allowed
        max_children = max(1, max_children)
        
        return len(node.children) < max_children

    def _backup(self, node):
        """
        Update the value of a node and all its ancestors.
        For non-leaf nodes, value is the average of its children's values.
        """
        # Traverse up the tree
        current = node
        while current.parent is not None:
            parent = current.parent
            
            # Calculate the average value of all children
            if parent.children:
                child_values = torch.tensor(np.array([child.value_est for child in parent.children]))
                # avg_value = child_values.mean().item()
                # log mean exp of child values
                avg_value = torch.logsumexp(child_values, dim=0).item() - torch.log(torch.tensor(len(child_values), dtype=child_values.dtype)).item()
                
                # Update the parent's value
                parent.update_value(avg_value)
            
            # Move up the tree
            current = parent

    def _update_exploration_probability(self):
        """
        Update the exploration probability based on current NFE.
        Linearly decay from initial_exploration_prob to final_exploration_prob.
        """
        if self.nfes >= self.num_nfes:
            self.current_exploration_prob = self.final_exploration_prob
        else:
            # Linear decay
            decay_rate = (self.initial_exploration_prob - self.final_exploration_prob) / self.num_nfes
            self.current_exploration_prob = self.initial_exploration_prob - decay_rate * self.nfes
    
    def _simulate_down_batched(self, nodes):
        """
        Batch simulation of a single step for multiple nodes.
        This allows multiple MCTS simulations to share computation.
        
        Args:
            nodes: List of MCTSNodes to simulate a step for
        """
        # Track nodes we're currently processing
        current_nodes = nodes.copy()
        max_timestep_idx = len(self.timesteps) - 1
        
        # Continue until all nodes reach t=0
        while any(node.t > 0 for node in current_nodes):
            # Dictionary to track node indices in current_nodes list
            node_indices = {id(node): i for i, node in enumerate(current_nodes)}
            
            # Group nodes by their current state
            terminal_nodes = []  # Nodes at t=0
            expansion_nodes = []  # Nodes that can be expanded at expansion timesteps
            selection_nodes = []  # Nodes with children to select from
            rollout_nodes = []   # Nodes that need rollout to next expansion point
            
            # Categorize each node
            for node in current_nodes:
                if node.t <= 0:
                    terminal_nodes.append(node)
                    continue
                    
                if node.t in self.expand_timesteps and self._can_expand(node):
                    expansion_nodes.append(node)
                elif not node.is_leaf():
                    selection_nodes.append(node)
                else:
                    rollout_nodes.append(node)
            
            # Track which nodes to remove from current_nodes
            nodes_to_remove = set()
            
            # 1. Process selection nodes (fast, no diffusion model calls)
            for node in selection_nodes:
                next_node = self._select_child_by_value(node, use_exploration=True)
                # Increment visit count of the selected child
                next_node.visit_count += 1
                current_nodes[node_indices[id(node)]] = next_node
            
            meta_expansion_nodes = [node for node in expansion_nodes if self._is_meta_root(node)]
            normal_expansion_nodes = [node for node in expansion_nodes if not self._is_meta_root(node)]

            if meta_expansion_nodes:
                x_T = self.sample_noise(len(meta_expansion_nodes))
                val, noise_pred = self._ddim_rollout_value(x_T, max_timestep_idx)
                for i, node in enumerate(meta_expansion_nodes):
                    x_T_batch = x_T[i:i+1]

                    child = MCTSNode(x_T_batch, max_timestep_idx, parent=node)
                    child.update_value(val[i].item())
                    if self.cache_noise_pred:
                        child.noise_pred = noise_pred[i:i+1]
                    node.children.append(child)

                    current_nodes[node_indices[id(node)]] = child

                    self._backup(child)

            # 2. Process expansion nodes (requires diffusion model calls)
            if normal_expansion_nodes:
                # Group by timestep for efficient batching
                nodes_by_t = {}
                for node in expansion_nodes:
                    if node.t not in nodes_by_t:
                        nodes_by_t[node.t] = []
                    nodes_by_t[node.t].append(node)
                
                # Process each timestep batch
                for t, batch_nodes in nodes_by_t.items():
                    x_tm1_batch, x_0_batch = self.expand_nodes(batch_nodes, t, predict_x0=True)
                    # Use raw log-space rewards instead of exponentiated rewards
                    # val_batch = np.exp(self.reward_fn(x_0_batch))
                    val_batch = self.reward_fn(x_0_batch)
                    
                    # Create child nodes and perform expansion
                    for i, node in enumerate(batch_nodes):
                        # Extract individual sample
                        x_tm1 = x_tm1_batch[i:i+1]
                        
                        # Create new child node
                        child = MCTSNode(x_tm1, t - 1, parent=node)
                        
                        child.update_value(val_batch[i].item())
                        
                        # Add child to parent's children list
                        node.children.append(child)
                        
                        # Update current node to the new child
                        current_nodes[node_indices[id(node)]] = child
                        
                        # Backup: update values up the tree (not visit counts)
                        self._backup(child)
                        
                        # Mark parent for removal since it's been expanded
                        # nodes_to_remove.add(id(node))
                        # nodes_to_remove.add(id(child))
            
            # 3. Process rollout nodes (requires diffusion model calls to next expansion point)
            if rollout_nodes:
                # Group rollout nodes by timestep
                nodes_by_t = {}
                for node in rollout_nodes:
                    if node.t not in nodes_by_t:
                        nodes_by_t[node.t] = []
                    nodes_by_t[node.t].append(node)
                
                # Process each timestep batch
                for t, batch_nodes in nodes_by_t.items():
                    # Collect x_t tensors for the batch
                    x_t_batch = torch.cat([node.x_t for node in batch_nodes], dim=0)
                    t_tensor = torch.tensor([t] * len(batch_nodes), device=self.device).long()
                    
                    # Batch sample from diffusion model
                    # with torch.no_grad():
                    x_tm1_batch, noise_pred_batch = self.p_sample(x_t_batch, t_tensor)
                    
                    # Create child nodes for rollout
                    for i, node in enumerate(batch_nodes):
                        x_tm1 = x_tm1_batch[i:i+1]
                        next_node = MCTSNode(x_tm1, t - 1, parent=node)
                        node.children.append(next_node)
                        if self.cache_noise_pred:
                            node.noise_pred = noise_pred_batch[i:i+1]
                        
                        # Update current node to child
                        current_nodes[node_indices[id(node)]] = next_node
                        
                        # if we've reached an expansion timestep, evaluate the node and do backup
                        if next_node.t in self.expand_timesteps:
                            val, noise_pred = self._ddim_rollout_value(next_node.x_t, next_node.t)
                            next_node.update_value(val.item())
                            if self.cache_noise_pred:
                                next_node.noise_pred = noise_pred
                            self._backup(next_node)
                            nodes_to_remove.add(id(next_node))

                        # If we've reached t=0, evaluate reward and do backup
                        if next_node.t <= 0:
                            # Pass x_t (final image) directly to reward function
                            # r_value = np.exp(self.reward_fn(next_node.x_t))
                            r_value = self.reward_fn(next_node.x_t).item()
                            next_node.update_value(r_value)
                            self._backup(next_node)

            # 4. Process terminal nodes (if any nodes reached t=0)
            for node in terminal_nodes:
                if node.visit_count == 1:  # Only evaluate if first visit
                    # Pass x_t (final image) directly to reward function
                    # r_value = np.exp(self.reward_fn(node.x_t))
                    r_value = self.reward_fn(node.x_t).item()
                    node.update_value(r_value)
                    self._backup(node)
                    nodes_to_remove.add(id(node))
            
            current_nodes = [node for node in current_nodes if id(node) not in nodes_to_remove]

    def _batch_rollout_to_terminal_for_paths(self, path_to_start_node_map: dict[int, MCTSNode]) -> dict[int, MCTSNode]:
        """
        Perform batched rollout of multiple simulation paths to t=0.
        Each path is identified by a path_idx.
        Sets value and performs backup for each terminal node reached.

        Args:
            path_to_start_node_map: Dictionary mapping path_idx to the MCTSNode where rollout begins for that path.

        Returns:
            Dictionary mapping path_idx to the terminal MCTSNode reached for that path.
        """
        active_rollout_sims = {
            path_idx: node for path_idx, node in path_to_start_node_map.items() if node.t > 0
        }
        finalized_sim_paths = {
            path_idx: node for path_idx, node in path_to_start_node_map.items() if node.t <= 0
        }

        # Handle paths that might already be terminal
        for path_idx, node in list(finalized_sim_paths.items()): # Iterate over a copy for modification
            if node.t <= 0: # Ensure it is indeed terminal
                # Assume reward calculation and backup should happen if it's a fresh terminal state.
                # This function is called after creating a child, so this child is new.
                # r_value = np.exp(self.reward_fn(node.x_t))
                r_value = self.reward_fn(node.x_t).item()
                node.update_value(r_value)
                self._backup(node)
            else: # Should not happen based on above separation
                del finalized_sim_paths[path_idx]


        while active_rollout_sims:
            nodes_by_t: dict[int, list[tuple[int, MCTSNode]]] = defaultdict(list)
            for path_idx, node in active_rollout_sims.items():
                nodes_by_t[node.t].append((path_idx, node))

            next_iter_active_rollout_sims = {}

            for t, sim_path_node_list in sorted(nodes_by_t.items(), key=lambda item: item[0], reverse=True):
                if t <= 0: # Should have been caught or already terminal
                    for path_idx, node in sim_path_node_list:
                        if path_idx not in finalized_sim_paths: # If not already processed
                            # r_value = np.exp(self.reward_fn(node.x_t))
                            r_value = self.reward_fn(node.x_t).item()
                            node.update_value(r_value)
                            self._backup(node)
                            finalized_sim_paths[path_idx] = node
                    continue

                x_t_batch = torch.cat([node.x_t for _, node in sim_path_node_list], dim=0)
                
                # Create a tensor for timesteps if model expects it, else scalar t might be fine
                # Assuming self.p_sample can handle scalar t for a batch, or adapts internally.
                # For safety, let's pass a tensor of timesteps.
                t_tensor = torch.full((len(sim_path_node_list),), t, device=self.device, dtype=torch.long)
                
                # with torch.no_grad():
                x_tm1_batch, noise_pred_batch = self.p_sample(x_t_batch, t_tensor)

                for i, (path_idx, parent_node_in_rollout) in enumerate(sim_path_node_list):
                    child_x_tm1 = x_tm1_batch[i:i+1]
                    next_node_in_rollout = MCTSNode(child_x_tm1, t - 1, parent=parent_node_in_rollout)
                    parent_node_in_rollout.children.append(next_node_in_rollout)
                    if self.cache_noise_pred:
                        parent_node_in_rollout.noise_pred = noise_pred_batch[i:i+1]

                    if next_node_in_rollout.t <= 0:
                        # r_value = np.exp(self.reward_fn(next_node_in_rollout.x_t))
                        r_value = self.reward_fn(next_node_in_rollout.x_t).item()
                        next_node_in_rollout.update_value(r_value)
                        self._backup(next_node_in_rollout)
                        finalized_sim_paths[path_idx] = next_node_in_rollout
                    else:
                        next_iter_active_rollout_sims[path_idx] = next_node_in_rollout
            
            active_rollout_sims = next_iter_active_rollout_sims
        
        return finalized_sim_paths

    def _simulate_down_full_batched(self, nodes_for_batch_start):
        """
        Batched version of full MCTS simulation that processes multiple nodes in parallel.
        Each element in `nodes_for_batch_start` is the starting MCTSNode for an independent simulation path.
        Typically, this will be [self.meta_root] * self.batch_size.
        Similar to the original _simulate_down_batched but continues rollout to t=0 after expansion.
        """
        # current_path_nodes[i] is the MCTSNode for the i-th simulation path.
        current_path_nodes = list(nodes_for_batch_start) # Make a mutable copy
        max_timestep_idx = len(self.timesteps) - 1

        while True:
            # Active paths for this iteration step: (original_path_idx, current_node_for_path)
            active_path_tuples_for_step: list[tuple[int, MCTSNode]] = []
            
            has_active_paths_gt_0 = False
            for i in range(len(current_path_nodes)):
                node = current_path_nodes[i]
                # A path is active if its node is not None (not yet fully processed/terminated from previous logic)
                # and its timestep is > 0.
                if node is not None and node.t > 0:
                    active_path_tuples_for_step.append((i, node))
                    has_active_paths_gt_0 = True
            
            if not has_active_paths_gt_0: # All paths have reached t <= 0 or marked None
                break

            # Categorize paths for operations. path_idx is the index in current_path_nodes.
            selection_ops: list[tuple[int, MCTSNode]] = []       # (path_idx, parent_node_obj)
            meta_expansion_ops: list[tuple[int, MCTSNode]] = []  # (path_idx, meta_node_obj)
            normal_expansion_ops: list[tuple[int, MCTSNode]] = []# (path_idx, parent_node_obj)
            single_step_ops: list[tuple[int, MCTSNode]] = []     # (path_idx, parent_node_obj)

            for path_idx, node in active_path_tuples_for_step:
                # Node is guaranteed to be node.t > 0 here
                is_meta = self._is_meta_root(node)
                can_expand_node = self._can_expand(node)

                if is_meta and can_expand_node:
                    meta_expansion_ops.append((path_idx, node))
                elif not is_meta and node.t in self.expand_timesteps and can_expand_node:
                    normal_expansion_ops.append((path_idx, node))
                elif node.children: # Has children, not expanding (due to _can_expand or not an expand_timestep)
                    selection_ops.append((path_idx, node))
                else: # Is a leaf (no children), and not an expansion candidate
                    single_step_ops.append((path_idx, node))

            # Stores updates: path_idx -> new_node_for_that_path
            next_path_nodes_update: dict[int, MCTSNode] = {}

            # 1. Process Selection Ops
            for path_idx, parent_node in selection_ops:
                chosen_child = self._select_child_by_value(parent_node, use_exploration=True)
                chosen_child.visit_count += 1
                next_path_nodes_update[path_idx] = chosen_child

            # 2. Process Meta-Expansion Ops
            if meta_expansion_ops:
                parent_meta_node = meta_expansion_ops[0][1] # All are self.meta_root
                num_meta_to_expand = len(meta_expansion_ops)
                
                x_T_batch = self.sample_noise(num_meta_to_expand)
                
                children_to_rollout_map: dict[int, MCTSNode] = {}
                for i in range(num_meta_to_expand):
                    path_idx, _ = meta_expansion_ops[i]
                    child_x_T = x_T_batch[i:i+1]
                    # Each path gets a NEW child, added to the shared meta_root's children list
                    new_child = MCTSNode(child_x_T, max_timestep_idx, parent=parent_meta_node)
                    parent_meta_node.children.append(new_child)
                    children_to_rollout_map[path_idx] = new_child
                
                finalized_paths_map = self._batch_rollout_to_terminal_for_paths(children_to_rollout_map)
                for path_idx, terminal_node in finalized_paths_map.items():
                    next_path_nodes_update[path_idx] = terminal_node # Path is now at this terminal node

            # 3. Process Normal-Expansion Ops
            normal_exp_grouped_by_parent: defaultdict[MCTSNode, list[int]] = defaultdict(list)
            for path_idx, parent_node_obj in normal_expansion_ops:
                normal_exp_grouped_by_parent[parent_node_obj].append(path_idx)

            for parent_node, path_indices_for_this_parent in normal_exp_grouped_by_parent.items():
                num_to_expand_from_parent = len(path_indices_for_this_parent)
                # All paths expanding this *same* parent_node object generate a new child from it.
                # x_t_for_parent_batch = parent_node.x_t.repeat(num_to_expand_from_parent, 1, 1, 1)
                current_t = parent_node.t
                
                # with torch.no_grad():
                #     # Timestep can be scalar if all 't' are same for this parent_node
                #     x_tm1_batch_output = self.p_sample(x_t_for_parent_batch, current_t)

                x_tm1_batch_output = self.expand_nodes([parent_node] * num_to_expand_from_parent, current_t)

                children_to_rollout_map: dict[int, MCTSNode] = {}
                for i in range(num_to_expand_from_parent):
                    path_idx = path_indices_for_this_parent[i]
                    child_x_tm1 = x_tm1_batch_output[i:i+1]
                    new_child = MCTSNode(child_x_tm1, current_t - 1, parent=parent_node)
                    parent_node.children.append(new_child) # Add to shared parent's children
                    children_to_rollout_map[path_idx] = new_child
                
                finalized_paths_map = self._batch_rollout_to_terminal_for_paths(children_to_rollout_map)
                for path_idx, terminal_node in finalized_paths_map.items():
                    next_path_nodes_update[path_idx] = terminal_node

            # 4. Process Single-Step Rollout Ops (leaf, t>0, not expanding)
            single_step_grouped_by_parent_t: defaultdict[tuple[MCTSNode, int], list[int]] = defaultdict(list)
            for path_idx, parent_node_obj in single_step_ops:
                single_step_grouped_by_parent_t[(parent_node_obj, parent_node_obj.t)].append(path_idx)

            for (parent_leaf_node, t_val), path_indices_for_this_leaf in single_step_grouped_by_parent_t.items():
                num_to_step_from_leaf = len(path_indices_for_this_leaf)
                x_t_for_leaf_batch = parent_leaf_node.x_t.repeat(num_to_step_from_leaf, 1, 1, 1)

                # with torch.no_grad():
                x_tm1_batch_output, noise_pred_batch = self.p_sample(x_t_for_leaf_batch, t_val)
                
                for i in range(num_to_step_from_leaf):
                    path_idx = path_indices_for_this_leaf[i]
                    child_x_tm1 = x_tm1_batch_output[i:i+1]
                    # Each path gets a new child representing one step forward
                    one_step_child = MCTSNode(child_x_tm1, t_val - 1, parent=parent_leaf_node)
                    parent_leaf_node.children.append(one_step_child) # Add to shared parent's children
                    next_path_nodes_update[path_idx] = one_step_child
                    if self.cache_noise_pred:
                        parent_leaf_node.noise_pred = noise_pred_batch[i:i+1]

                    if one_step_child.t <= 0: # Reached terminal by this single step
                        # r_value = np.exp(self.reward_fn(one_step_child.x_t))
                        r_value = self.reward_fn(one_step_child.x_t).item()
                        one_step_child.update_value(r_value)
                        self._backup(one_step_child)
                        # next_path_nodes_update already has this terminal node for path_idx

            # 5. Apply all updates to current_path_nodes
            for path_idx, new_node in next_path_nodes_update.items():
                current_path_nodes[path_idx] = new_node
            
            # 6. Final check for any paths that are now terminal and mark them as None for next loop iteration check
            # This is mostly to ensure paths that become terminal are correctly excluded from has_active_paths_gt_0 logic
            for i in range(len(current_path_nodes)):
                if current_path_nodes[i] is not None and current_path_nodes[i].t <= 0:
                    # Value setting for these terminal nodes should have occurred when they were created/finalized
                    # by expansion, _batch_rollout_to_terminal_for_paths, or single-step rollout.
                    # For the simulation loop, we just note they are no longer active > 0.
                    pass 
        # End of while True loop (all paths have reached t <= 0 or effectively completed)

    def visualize_tree(self, root_node, max_depth=4, max_nodes_per_level=100, it=0):
        """
        Visualize the MCTS tree with node values.
        
        Args:
            root_node: The root node to start visualization from
            max_depth: Maximum depth to visualize (to prevent overcrowding)
            max_nodes_per_level: Maximum number of nodes to show per level
            
        Returns:
            None (displays a plot)
        """
        if self.visualization_timesteps is None:
            return
        # Create a directed graph
        G = nx.DiGraph()
        
        # Only show root node, expansion timesteps, and t=0 nodes
        # important_timesteps = set(list(self.expand_timesteps) + [0, root_node.t])
        important_timesteps = set(self.visualization_timesteps)
        
        # Add nodes and edges using BFS traversal
        queue = [(root_node, 0)]  # (node, depth)
        nodes_to_display = {}  # node_id -> (node, depth)
        edges_to_add = []  # list of (parent_id, child_id) tuples
        
        while queue:
            node, depth = queue.pop(0)
            node_id = str(id(node))
            
            # Add node to display if it's at an important timestep
            if node.t in important_timesteps:
                nodes_to_display[node_id] = (node, depth)
                
                # If this node has a parent that should be displayed, track the edge
                if node.parent is not None:
                    parent_node = node.parent
                    parent_id = str(id(parent_node))
                    
                    # If parent is at an important timestep, add direct edge
                    if parent_node.t in important_timesteps:
                        edges_to_add.append((parent_id, node_id))
                    else:
                        # Find the closest ancestor at an important timestep
                        current_parent = parent_node
                        while current_parent is not None:
                            if current_parent.t in important_timesteps:
                                edges_to_add.append((str(id(current_parent)), node_id))
                                break
                            current_parent = current_parent.parent
            
            # Stop at max depth
            if depth >= max_depth:
                continue
                
            # Process children
            for i, child in enumerate(node.children):
                if i >= max_nodes_per_level:
                    break
                queue.append((child, depth + 1))
        
        # Add nodes to graph
        for node_id, (node, depth) in nodes_to_display.items():
            G.add_node(node_id, 
                       depth=depth, 
                       timestep=node.t,
                       value=float(node.value_est),  # Convert to float
                       visits=node.visit_count)
        
        # Add edges to graph
        for parent_id, child_id in edges_to_add:
            if parent_id in G.nodes and child_id in G.nodes:
                G.add_edge(parent_id, child_id)
        
        # Position nodes using hierarchical layout
        # Group nodes by depth
        nodes_by_depth = defaultdict(list)
        for node_id in G.nodes:
            depth = G.nodes[node_id]['depth']
            nodes_by_depth[depth].append(node_id)
        
        # Position nodes with increased spacing
        pos = {}
        for depth, nodes in nodes_by_depth.items():
            level_width = len(nodes)
            spacing_factor = 2.0
            for i, node_id in enumerate(sorted(nodes, key=lambda n: G.nodes[n]['value'])):
                pos[node_id] = ((i - level_width/2) * spacing_factor, -depth * 1.5)
        
        # Create figure and axis with larger figure size
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # Draw nodes with size proportional to visit count and color by value
        node_sizes = [100 + 10 * G.nodes[n]['visits'] for n in G.nodes]
        node_colors = [float(G.nodes[n]['value']) for n in G.nodes]  # Convert to float
        
        # Draw the graph
        nx.draw(G, pos, 
                ax=ax,
                node_size=node_sizes,
                node_color=node_colors,
                cmap=plt.cm.viridis,
                with_labels=False,
                arrows=True,
                edge_color='gray',
                width=1.0,
                alpha=0.8)
        
        # Add node labels with better formatting
        labels = {}
        for node_id in G.nodes:
            node = nodes_to_display[node_id][0]            
            # Create a more compact label with visit count and number of children
            labels[node_id] = f"t:{G.nodes[node_id]['timestep']}\nv:{G.nodes[node_id]['value']:.2f}\nn:{G.nodes[node_id]['visits']}\nc:{len(node.children)}"
        
        # Draw labels with bbox to prevent overlap
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax,
                               bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=3))
        
        # Add color bar with explicit axes reference
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        sm.set_array(node_colors)
        fig.colorbar(sm, ax=ax, label='Node Value')
        
        ax.set_title(f"MCTS Tree Visualization (max depth {max_depth})")
        plt.tight_layout()
        ax.axis('off')
        plt.savefig(os.path.join(self.output_dir, f"mcts_tree_visualization_{it}.png"), dpi=300)
        plt.close()
    
    @torch.no_grad()
    def initialize_root_nodes(self):
        self.start_time = time.time()
        self.meta_root = self._initialize_meta_root()

    @torch.no_grad()
    def build_search_tree(self, num_nfes=None):
        """
        Perform multiple MCTS iterations starting from the meta-root node.
        The meta-root allows dynamic generation of tree roots during the search.
        """
        # Initialize meta-root if not provided
        if self.meta_root is None:
            self.start_time = time.time()
            self.meta_root = self._initialize_meta_root()
        
        # Set up iteration range and possibly wrap with tqdm
        # if num_nfes is not None:
        #     self.num_nfes = num_nfes
        max_nfes = num_nfes
        if self.use_progress_bar:
            pbar = tqdm.tqdm(total=max_nfes, desc="Building search tree")
            pbar.update(self.nfes)
        
        # Main MCTS loop
        while self.nfes < max_nfes:
            self.meta_root.visit_count += self.batch_size

            # Update exploration probability
            self._update_exploration_probability()
            
            # Use batched simulation
            if self.use_rollout:
                self._simulate_down_full_batched([self.meta_root] * self.batch_size)
            else:
                self._simulate_down_batched([self.meta_root] * self.batch_size)
            
            # Update progress bar or print progress
            if self.use_progress_bar:
                if self.nfes < num_nfes:
                    pbar.update(self.nfes - pbar.n)
                else:
                    pbar.update(num_nfes - pbar.n)
            elif (self.nfes + 1) % (num_nfes // 10) == 0:
                progress = (self.nfes + 1) / num_nfes * 100
                print(f"Tree building progress: {progress:.1f}% (NFE {self.nfes+1}/{num_nfes})")
                sys.stdout.flush()
            
        if self.use_progress_bar:
            pbar.close()
        
        # # Visualize the tree for debugging
        # if self.visualization_timesteps:
        #     self.visualize_tree(self.meta_root, max_depth=50, it=0)
        
        print(f"MCTS tree building {self.num_nfes} NFEs in {time.time() - self.start_time:.2f} seconds")

    @torch.no_grad()
    def sample_from_tree(self, population_size=1):
        self.start_time2 = time.time()
        max_timestep_idx = len(self.timesteps) - 1

        final_nodes = [None] * population_size # Store the t=0 node for each path

        # Start population_size paths, each initially at the meta root
        current_nodes = [self.meta_root] * population_size
        active_path_indices = list(range(population_size)) # Indices of paths still processing

        while active_path_indices:
            # --- Prepare for next step ---
            next_nodes = current_nodes[:] # Copy current state to modify for the next step
            next_active_path_indices = [] # Track which paths continue to the next step
            processed_indices_this_iter = set() # Ensure each path is processed only once per iteration

            # Group active nodes by their current timestep 't'
            nodes_by_t = defaultdict(list)
            min_active_t = float('inf')
            for idx in active_path_indices:
                node = current_nodes[idx]
                if node.t > 0: # Only consider nodes not yet at t=0
                    nodes_by_t[node.t].append((idx, node)) # Store (original_path_index, node)
                    min_active_t = min(min_active_t, node.t)
                else:
                     # Should normally not happen here if node.t > 0 check is done before loop
                     # But handle defensively: if a node is already at t=0, store it
                     if final_nodes[idx] is None: # Store only once
                         final_nodes[idx] = node


            if min_active_t == float('inf'): # No active nodes left with t > 0
                break # All paths have reached t=0

            # --- Process nodes starting from the highest timestep ---
            # Ensures meta-root (virtual timestep) is handled first if present
            for t_idx, node_list in sorted(nodes_by_t.items(), key=lambda item: item[0], reverse=True):

                # Separate nodes based on children / meta status for batch processing
                nodes_with_children = []
                nodes_without_children_normal = []
                nodes_without_children_meta = []

                for path_idx, node in node_list:
                    # Skip if this path index was already processed in this iteration
                    # (e.g., if meta and normal nodes exist at the same high timestep, unlikely but possible)
                    if path_idx in processed_indices_this_iter: continue

                    if len(node.children) == 0:
                        if self._is_meta_root(node):
                            nodes_without_children_meta.append((path_idx, node))
                        else: # Normal node without children requires a deterministic step
                            nodes_without_children_normal.append((path_idx, node))
                    else: # Node has children in the tree, select the best one
                        nodes_with_children.append((path_idx, node))

                # 1. Handle Meta Root "expansion" -> actually just the first step T -> T-1
                # This happens if population_size > number of initial children explored from meta_root
                if nodes_without_children_meta:
                    path_indices = [idx for idx, _ in nodes_without_children_meta]

                    # Sample initial noise (x_T) for these paths
                    x_T_batch = self.sample_noise(len(path_indices))

                    # Use p_sample for the first step T -> T-1 (deterministic rollout)
                    t_tensor = torch.tensor([max_timestep_idx] * len(path_indices), device=self.device).long()
                    # with torch.no_grad():
                    x_Tm1_batch, noise_pred_batch = self.p_sample(x_T_batch, t_tensor)

                    next_t_idx = max_timestep_idx - 1
                    next_t_value = self.get_timestep_from_index(next_t_idx)

                    for i, path_idx in enumerate(path_indices):
                        parent_node = nodes_without_children_meta[i][1] # The meta root instance for this path
                        # Create the transient node for the next step (x_{T-1} at t=T-1)
                        # Parent is conceptually the meta_root, but we don't modify meta_root.children
                        next_node = MCTSNode(x_Tm1_batch[i:i+1], next_t_idx, parent=parent_node, timestep_value=next_t_value)
                        next_nodes[path_idx] = next_node # Update the state for this specific path

                        if next_node.t > 0:
                            next_active_path_indices.append(path_idx)
                        else: # Handle unlikely case where T-1 <= 0 (num_timesteps=1)
                            final_nodes[path_idx] = next_node
                        processed_indices_this_iter.add(path_idx)


                # 2. Handle normal nodes without children (deterministic step t -> t-1)
                # These are leaf nodes encountered before t=0 during tree build.
                if nodes_without_children_normal:
                    normal_nodes = [n for _, n in nodes_without_children_normal]
                    path_indices = [idx for idx, _ in nodes_without_children_normal]

                    x_t_batch = torch.cat([node.x_t for node in normal_nodes], dim=0)
                    t_tensor = torch.tensor([t_idx] * len(path_indices), device=self.device).long()

                    # with torch.no_grad():
                    x_tm1_batch, noise_pred_batch = self.p_sample(x_t_batch, t_tensor)

                    next_t_idx = t_idx - 1
                    next_t_value = self.get_timestep_from_index(next_t_idx)

                    for i, path_idx in enumerate(path_indices):
                        parent_node = nodes_without_children_normal[i][1] # The parent node for this path step
                        x_tm1 = x_tm1_batch[i:i+1]
                        # Create transient node for the next step, don't add to parent.children
                        next_node = MCTSNode(x_tm1, next_t_idx, parent=parent_node, timestep_value=next_t_value)
                        next_nodes[path_idx] = next_node # Update the state for this specific path
                        if self.cache_noise_pred:
                            parent_node.noise_pred = noise_pred_batch[i:i+1]

                        if next_node.t > 0:
                            next_active_path_indices.append(path_idx)
                        else: # Path reached t=0
                            final_nodes[path_idx] = next_node
                        processed_indices_this_iter.add(path_idx)


                # 3. Handle nodes with children (select best based on value/maximize)
                for path_idx, node in nodes_with_children:
                    if path_idx in processed_indices_this_iter: continue # Skip if already processed

                    chosen_child = self._select_child_by_value(node, use_exploration=False)
                    next_nodes[path_idx] = chosen_child # Update the state for this specific path

                    if chosen_child.t > 0:
                        next_active_path_indices.append(path_idx)
                    else: # Path reached t=0
                        final_nodes[path_idx] = chosen_child
                    processed_indices_this_iter.add(path_idx)


            # Update state for the next iteration
            current_nodes = next_nodes
            # Update the list of indices for paths that are still active (t > 0)
            active_path_indices = sorted(list(set(next_active_path_indices)))

        # --- Final Processing ---
        # All paths should have reached t=0 and their final nodes stored in final_nodes
        sampled_nodes = [node for node in final_nodes if node is not None]

        if not sampled_nodes:
             print("Warning: No samples generated in sample_from_tree. Returning empty.")
             return torch.empty((0, self.dims), device=self.device), np.array([])

        # Handle the 'maximize' case: return only the single best node
        if self.maximize:
             # Find the node with the highest value estimate
             # Ensure value_est exists and handle potential missing values defensively
             valid_final_nodes = [n for n in sampled_nodes if hasattr(n, 'value_est')]
             if not valid_final_nodes:
                  print("Error: Maximize=True but no final nodes with value_est found. Returning first sample.")
                  # Fallback: return the first node found, or handle error differently
                  sampled_nodes = sampled_nodes[0:1] if sampled_nodes else []
             else:
                  best_node = max(valid_final_nodes, key=lambda x: x.value_est)
                  sampled_nodes = [best_node]
        # else: sampled_nodes contains all paths

        # Check if any samples were actually produced (especially after maximize)
        if not sampled_nodes:
             print("Warning: No samples selected after processing. Returning empty.")
             return torch.empty((0, self.dims), device=self.device), np.array([])

        final_images = torch.cat([node.x_t for node in sampled_nodes], dim=0)
        final_images_reward = self.reward_fn(final_images)
        
        print(f"MCTS population rollout {population_size} samples in {time.time() - self.start_time2:.2f} seconds, {self.nfes} NFEs")

        return final_images, final_images_reward