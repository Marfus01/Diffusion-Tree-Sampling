import torch
import math
import tqdm
import matplotlib.pyplot as plt
import random
import networkx as nx
import os
from collections import defaultdict
import sys
import time
import numpy as np
import logging
import warnings


logging.getLogger("torch_tensorrt.dynamo.conversion._TRTInterpreter").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="Both operands of the binary elementwise op .* are constant")

from src.models.diffusion_2d import DiffusionModel
from src.utils.datasets import get_2d_dataset
from src.utils.plotting_utils import visualize_dataset_comparison, plot_density_evolution
from src.evaluation.distribution_metrics import *
    

VISIT_BINS = ((1, 1), (2, 9), (10, 99), (100, 999), (1000, None))


def _visit_bin(v):
    for lo, hi in VISIT_BINS:
        if v >= lo and (hi is None or v <= hi):
            return f"{lo}-{hi if hi is not None else 'inf'}"


def _summary(values):
    a = np.asarray(values, dtype=np.float64)
    return {'n': int(a.size), 'mean': float(a.mean()), 'min': float(a.min()), 'q50': float(np.quantile(a, 0.5)),
            'q90': float(np.quantile(a, 0.9)), 'q99': float(np.quantile(a, 0.99)), 'max': float(a.max())}


def tree_statistics(meta_root, expand_timesteps, child_log_value=None):
    """Shape and value statistics of a built rollout tree. Read-only: no random numbers, no change to the tree.

    Layers are the meta-root ('meta') and the expansion timesteps below it ('80', '60', ... for the
    2D schedule); a branch node is a node with at least two children, the meta-root always counts as one.
    Per layer: node count, branch-node count, histogram of the number of children and summary of the
    visit counts. Per terminal leaf: the number of branch ancestors below the meta-root (branch depth)
    and the lowest layer among them. With child_log_value(x [B, d], t [B]) -> log M_t (float64), the
    value error value_est - log M_t of every child of a branch node, per layer of the parent and per
    visit-count bin of the child (VISIT_BINS).

    Returns (stats, leaves) with leaves a list of (leaf, log_w): log_w = value_est(leaf) - sum of
    log n_children over its branch ancestors. With the log-mean-exp backup the traversal of
    sample_from_tree picks each leaf with probability proportional to exp(log_w)."""
    layer_ts = [t for t in expand_timesteps if t < meta_root.t]
    layers = {'meta': {'children': [len(meta_root.children)], 'visits': [meta_root.visit_count], 'nodes': 1}}
    layers.update({str(t): {'children': [], 'visits': [], 'nodes': 0} for t in layer_ts})
    depth_hist, lowest_hist = defaultdict(int), defaultdict(int)
    branch_children = []                                    # (parent layer, child)
    leaves, nonterminal = [], 0
    stack = [(meta_root, 0.0, 0, 'none')]
    while stack:
        node, log_n, depth, lowest = stack.pop()
        k = len(node.children)
        is_meta = node is meta_root
        if not is_meta and str(node.t) in layers:
            layers[str(node.t)]['nodes'] += 1
            layers[str(node.t)]['children'].append(k)
            layers[str(node.t)]['visits'].append(node.visit_count)
        if k == 0:
            if node.t <= 0:
                leaves.append((node, float(node.value_est) - log_n))
                depth_hist[depth] += 1
                lowest_hist[lowest] += 1
            else:
                nonterminal += 1
            continue
        if k > 1 or is_meta:
            label = 'meta' if is_meta else str(node.t)
            branch_children.extend((label, c) for c in node.children)
            log_n += math.log(k)
            if not is_meta:
                depth, lowest = depth + 1, label
        stack.extend((c, log_n, depth, lowest) for c in node.children)

    stats = {'n_roots': len(meta_root.children), 'n_leaves': len(leaves), 'n_nonterminal_leaves': nonterminal,
             'leaf_branch_depth': {str(d): n for d, n in sorted(depth_hist.items())},
             'leaf_lowest_branch_layer': dict(lowest_hist), 'layers': {}}
    for label, L in layers.items():
        ch = np.asarray(L['children'])
        stats['layers'][label] = {
            'nodes': L['nodes'], 'branch_nodes': int((ch > 1).sum()) if label != 'meta' else 1,
            'children_hist': {str(c): int(n) for c, n in zip(*np.unique(ch, return_counts=True))} if ch.size else {},
            'children_of_branch_nodes': _summary(ch[ch > 1]) if (ch > 1).any() else None,
            'visits': _summary(L['visits']) if L['visits'] else None}
    if child_log_value is not None and branch_children:
        x = torch.cat([c.x_t for _, c in branch_children]).double()
        t = torch.tensor([c.t for _, c in branch_children])
        log_m = torch.cat([child_log_value(x[i:i + 4096], t[i:i + 4096]) for i in range(0, len(x), 4096)]).numpy()
        rows = defaultdict(list)
        for (label, c), lm in zip(branch_children, log_m):
            err = float(c.value_est) - float(lm)
            rows[(label, 'all')].append(err)
            rows[(label, _visit_bin(c.visit_count))].append(err)
        value_error = defaultdict(dict)
        for (label, b), e in rows.items():
            e = np.asarray(e)
            value_error[label][b] = {'n': int(e.size), 'bias': float(e.mean()),
                                     'var': float(e.var(ddof=1)) if e.size > 1 else None,
                                     'rmse': float(np.sqrt(np.mean(e ** 2)))}
        stats['value_error'] = dict(value_error)
    return stats, leaves


class MCTSNode:
    """
    Each node represents a state x_t at diffusion timestep t.
    Children are possible next states x_{t-1}.
    Slots instead of a per-instance __dict__: a tree holds one node per NFE (1e7 at the largest GMM budget).
    """
    __slots__ = ('x_t', 't', 'timestep_value', 'parent', 'children', 'visit_count', 'value_est', 'noise_pred',
                 'is_final')

    def __init__(self, x_t, t, parent=None, timestep_value=None):
        """
        Args:
          x_t (torch.Tensor): The noisy sample at diffusion timestep t. [B, D] for 2D data
          t (int): The current diffusion timestep index.
          parent (MCTSNode or None): The parent node. None if root.
          timestep_value (float or None): The actual scheduler timestep value if known.
        """
        self.x_t = x_t          # shape: [B, D] for 2D data
        self.t = t               # This is the timestep index
        self.timestep_value = timestep_value  # This is the actual scheduler timestep value
        self.parent = parent
        self.children = []      # list of child nodes
        self.visit_count = 1    # N(x_t) in the algorithm - Initialize to 1
        self.value_est = 0.0    # Estimated value of this node
        self.noise_pred = None  # Noise prediction for this node

    def is_leaf(self):
        return (self.t <= 0) or (len(self.children) == 0)
        
    def update_value(self, value):
        """
        Directly set the node's value to the given value.
        """
        self.value_est = value


class MetaRootNode(MCTSNode):
    """
    A special meta-root node that doesn't correspond to any actual diffusion timestep.
    Its children are created by sampling from random noise.
    """
    def __init__(self, sampler):
        """
        Args:
            sampler: The MCTSDiffusionSampler instance to use for sampling children
        """
        # Use a timestep index that's one more than the maximum diffusion timestep
        virtual_timestep = len(sampler.timesteps)
        # Create a dummy tensor with the right shape based on args
        # Use sampler.dims for shape
        data_shape = (1, sampler.dims)
        dummy_x = torch.zeros(data_shape, device=sampler.device)

        # Initialize with the virtual timestep and dummy tensor
        super().__init__(dummy_x, virtual_timestep, parent=None, timestep_value=None)


class MCTSDiffusionSampler:
    """
    MCTS-style sampler for a pre-trained diffusion model with a black-box reward.
    Adapted from MCTSImageDiffusionSampler.
    """

    def __init__(self, 
                 diffusion_model: DiffusionModel, 
                 reward_fn: callable,
                 expand_timesteps: list, 
                 num_nfes: int = 1000,
                 num_inference_steps: int = 50,
                 eta: float = 1.0,
                 device: str = 'cpu',
                 exploration_const: float = 1.0,
                 progressive_width_const: float = 1.0,
                 progressive_width_alpha: float = 0.5,
                 initial_exploration_prob: float = 0.1,
                 final_exploration_prob: float = 0.05,
                 use_progress_bar: bool = True,
                 use_rollout: bool = False,
                 maximize: bool = False,
                 cache_noise_pred: bool = True,
                 exploration_type: str = 'epsilon',
                 batch_size: int = 4,
                 output_dir: str = None,
                 visualization_timesteps: list = None,
                 dims: int = None,
                 pw_alpha: float = None,
                 pw_alpha_root: float = None,
                 keep_chain_states: bool = True):
        """
        Args:
          diffusion_model: Pretrained diffusion model providing:
            - p_sample(x_t, t): samples x_{t-1} from the model
            - ddim_denoise(x_t, t): deterministically denoises x_t -> x_0 (approx)
          reward_fn: Function that takes samples (Tensor[B, D]) and returns scalar rewards.
          expand_timesteps: Timesteps indices at which we do MCTS expansion.
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
          dims: Dimensionality of the data (defaults to diffusion_model.input_dim).
          pw_alpha: Progressive-widening exponent for expansion nodes and the meta-root. None keeps
            the original hard-coded values (0.3 for nodes and 0.4 for the meta-root with rollouts,
            0.4 for both without).
          pw_alpha_root: Exponent for the meta-root only; overrides pw_alpha there.
          keep_chain_states: False drops x_t and the cached noise_pred of rollout-chain nodes once their
            child exists, except at the expansion timesteps, directly below them and at t = T - 1 (the
            states that expansion, sampling and tree_statistics read). Chain nodes elsewhere are never
            expanded, so the tree, the NFE count and the samples are unchanged; memory per node drops.

        Node values are stored in log space: value_est = log M_hat, the soft backup is
        log-mean-exp over children, and selection samples children with probability
        softmax(value_est). Terminal nodes (t = 0) hold the clean sample after the last,
        deterministic denoising step at index 0.
        """
        self.model = diffusion_model
        self.reward_fn = reward_fn
        self.expand_timesteps = sorted(expand_timesteps, reverse=True)
        self.num_nfes = num_nfes
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        self.c_ucb = exploration_const
        self.device = device
        self.progressive_width_const = progressive_width_const
        self.progressive_width_alpha = progressive_width_alpha
        self.initial_exploration_prob = initial_exploration_prob
        self.final_exploration_prob = final_exploration_prob
        self.current_exploration_prob = initial_exploration_prob
        self.use_progress_bar = use_progress_bar
        self.use_rollout = use_rollout
        self.maximize = maximize
        self.cache_noise_pred = cache_noise_pred
        self.exploration_type = exploration_type
        self.batch_size = batch_size
        self.dims = dims if dims is not None else diffusion_model.input_dim
        self.pw_alpha = pw_alpha
        self.pw_alpha_root = pw_alpha_root
        self.keep_chain_states = keep_chain_states
        self.output_dir = output_dir
        self.visualization_timesteps = visualization_timesteps if visualization_timesteps is not None else []

        # Setup diffusion related variables
        self.setup_diffusion()

        self.nfes = 0
        self.meta_root = None

    def setup_diffusion(self):
        """
        Set up diffusion variables.
        """
        self.num_timesteps = len(self.model.betas)
        self.timesteps = torch.arange(self.num_timesteps, device=self.device).long()
        self.num_inference_steps = self.num_timesteps

    def get_timestep_from_index(self, index):
        """
        Get the actual diffusion timestep value (which is the index itself in this case).
        """
        # Make sure index is within bounds
        index = min(max(0, index), self.num_timesteps - 1)
        return index

    def _initialize_meta_root(self):
        """
        Initialize a single meta-root node as the starting point for the search.
        This meta-root is at a virtual timestep beyond the normal diffusion process.
        """
        self.meta_root = MetaRootNode(self)
        return self.meta_root
    
    def _state_is_read(self, t):
        return t in self.expand_timesteps or (t + 1) in self.expand_timesteps or t == self.num_timesteps - 1

    def _is_meta_root(self, node):
        return node.t >= self.num_timesteps
    
    def sample_noise(self, batch_size=1):
        """
        Generate initial noise matching the data dimensionality.
        """
        return self.model.sample_initial_noise(batch_size).to(self.device)

    def p_sample(self, x_t, t, predict_x0=False, eta=None):
        """
        Perform one step of the diffusion process using the model's methods.
        Args:
            x_t (torch.Tensor): Current noisy data tensor [B, D].
            t_idx (Union[int, torch.Tensor, list]): MCTS timestep *index* (0 to T-1).
            predict_x0 (bool): If True, use ddim_denoise to predict x0.
        Returns:
            Tensor: Either x_{t-1} or predicted x0.
        """
        batch_size = x_t.shape[0]
        if eta is None:
            eta = self.eta

        # Call method from model
        with torch.no_grad():
            if predict_x0:
                pred, noise_pred = self.model.ddim_denoise(x_t, t)
            else:
                pred, noise_pred = self.model.p_sample(x_t, t, eta)

        self.nfes += batch_size
    
        return pred, noise_pred

    def expand_nodes(self, nodes, t, predict_x0=False, eta=None):
        """
        Expand nodes by sampling from the diffusion model.
        """
        # Get all nodes that don't have a noise prediction
        noise_pred_nodes = []
        for node in nodes:
            if node.noise_pred is None:
                noise_pred_nodes.append(node)

        if eta is None:
            eta = self.eta

        if noise_pred_nodes:
            x_t = torch.cat([node.x_t for node in noise_pred_nodes], dim=0)
            batch_size = x_t.shape[0]
            self.nfes += batch_size

            # Get noise prediction with UNet
            with torch.no_grad():
                noise_pred = self.model.predict_noise(x_t, t)
            
            if self.cache_noise_pred:
                for i, node in enumerate(noise_pred_nodes):
                    node.noise_pred = noise_pred[i:i+1]

        # Now take scheduler step for all nodes
        x_t = torch.cat([node.x_t for node in nodes], dim=0)
        if self.cache_noise_pred:
            noise_pred = torch.cat([node.noise_pred for node in nodes], dim=0)

        x_t_prev, predicted_x0 = self.model.scheduler_step(
            noise_pred,
            t,
            x_t, 
            eta=eta,
        )

        # Return either the previous sample or the predicted original sample
        if not predict_x0:
            return x_t_prev
        else:
            return x_t_prev, predicted_x0

    def ddim_denoise(self, x_t, t):
        """
        Deterministically denoise x_t to x_0 using the model's ddim_denoise.
        """
        return self.p_sample(x_t, t, predict_x0=True)

    def _finalize_terminal(self, node):
        """Applies the last denoising step (index 0, zero variance) to a node that reached t = 0,
        so the terminal node holds the same clean sample the SMC samplers return. Costs 1 NFE."""
        if getattr(node, 'is_final', False):
            return
        x_clean, _ = self.p_sample(node.x_t, 0)
        node.x_t = x_clean
        node.is_final = True

    def _terminal_value(self, node):
        """Finalizes a new terminal node and sets its log value r(x_0)."""
        self._finalize_terminal(node)
        node.update_value(float(np.asarray(self.reward_fn(node.x_t)).reshape(-1)[0]))

    def _ddim_rollout_value(self, x_t, t):
        """
        Evaluate intermediate nodes by deterministically rolling out x_t to predicted x_0
        and computing reward(x_0).
        Args:
            x_t (torch.Tensor): Current noisy data tensor [B, D].
            t_idx (Union[int, torch.Tensor, list]): MCTS timestep *index* (0 to T-1).
        Returns:
            Tensor: reward values for the batch [B].
        """
        with torch.no_grad():
            x_0_approx, noise_pred = self.ddim_denoise(x_t, t)
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
                
            # Otherwise sample a child with probability proportional to exp(log value)
            values = torch.tensor([child.value_est for child in node.children], dtype=torch.float64)
            if self.maximize:
                idx = torch.argmax(values).item()
            else:
                probs = torch.softmax(values, dim=0)
                idx = torch.multinomial(probs, 1).item()
            return node.children[idx]
        
        elif self.exploration_type == 'ucb':
            # UCB exploration
            if not use_exploration:
                values = torch.tensor([child.value_est for child in node.children], dtype=torch.float64)
                probs = torch.softmax(values, dim=0)
                idx = torch.multinomial(probs, 1).item()
                return node.children[idx]
            
            # Calculate UCB scores for each child
            parent_visits = max(1, node.visit_count)  # Avoid division by zero
            
            # Vectorized UCB calculation
            child_visits = torch.tensor(np.array([max(1, child.visit_count) for child in node.children]))
            child_values = torch.exp(torch.tensor([child.value_est for child in node.children], dtype=torch.float64))
            
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
        Also handles the special case of the meta-root node. (Copied from mcts_img.py)
        """
        # Special case for meta-root
        if self._is_meta_root(node):
            effective_visits = max(1, node.visit_count)
            self.progressive_width_const = 2
            if self.pw_alpha_root is not None:
                meta_root_alpha = self.pw_alpha_root
            elif self.pw_alpha is not None:
                meta_root_alpha = self.pw_alpha
            else:
                meta_root_alpha = 0.4
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
        self.progressive_width_const = 2
        if self.pw_alpha is not None:
            progressive_width_alpha = self.pw_alpha
        else:
            progressive_width_alpha = 0.3 if self.use_rollout else 0.4
        max_children = math.floor(self.progressive_width_const * (effective_visits ** progressive_width_alpha))
        
        # Ensure at least 1 child is allowed
        max_children = max(1, max_children)

        return len(node.children) < max_children

    def _backup(self, node):
        """
        Update the value of a node and all its ancestors.
        For non-leaf nodes, the log value is the log-mean-exp of the children's log values,
        i.e. M_hat(parent) is the average of the children's M_hat.
        """
        # Traverse up the tree
        current = node
        while current.parent is not None:
            parent = current.parent
            
            if parent.children:
                child_values = torch.tensor([child.value_est for child in parent.children], dtype=torch.float64)
                avg_value = (torch.logsumexp(child_values, dim=0) - math.log(len(parent.children))).item()
                parent.update_value(avg_value)

            # Move up the tree
            current = parent

    def _update_exploration_probability(self):
        """
        Update the exploration probability based on current iteration.
        Linearly decay from initial_exploration_prob to final_exploration_prob. (Copied from mcts_img.py)
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
        max_timestep_idx = self.num_timesteps - 1
        
        # Continue until all nodes reach t=0
        # while any(node.t > 0 for node in current_nodes):
        while current_nodes:
            # Dictionary to track node indices in current_nodes list
            node_indices = {id(node): i for i, node in enumerate(current_nodes)}
            # Group nodes by their current state
            terminal_nodes = []  # Nodes at t=0
            expansion_nodes = []  # Nodes that can be expanded at expansion timesteps
            selection_nodes = []  # Nodes with children to select from
            rollout_nodes = []   # Nodes that need rollout to next expansion point

            # Categorize each node
            for i, node in enumerate(current_nodes):
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

                for t, batch_nodes in nodes_by_t.items():
                    # x_t_batch = torch.cat([node.x_t for node in batch_nodes], dim=0)
                    # with torch.no_grad():
                    #     x_tm1_batch = self.p_sample(x_t_batch, t)
                    #     val_batch = self._ddim_rollout_value(x_tm1_batch, t - 1)
                    x_tm1_batch, x_0_batch = self.expand_nodes(batch_nodes, t, predict_x0=True)
                    val_batch = self.reward_fn(x_0_batch)

                    for i, node in enumerate(batch_nodes):
                        x_tm1 = x_tm1_batch[i:i+1]
                        child = MCTSNode(x_tm1, t - 1, parent=node)
                        child.update_value(val_batch[i].item())
                        node.children.append(child)
                        current_nodes[node_indices[id(node)]] = child
                        self._backup(child)
                        nodes_to_remove.add(id(child))

            # 3. Process rollout nodes (requires diffusion model calls to next expansion point)
            if rollout_nodes:
                nodes_by_t = {}
                for node in rollout_nodes:
                    if node.t not in nodes_by_t:
                        nodes_by_t[node.t] = []
                    nodes_by_t[node.t].append(node)
                
                for t, batch_nodes in nodes_by_t.items():
                    x_t_batch = torch.cat([node.x_t for node in batch_nodes], dim=0)
                    with torch.no_grad():
                        x_tm1_batch, noise_pred_batch = self.p_sample(x_t_batch, t)

                    for i, node in enumerate(batch_nodes):
                        x_tm1 = x_tm1_batch[i:i+1]
                        next_node = MCTSNode(x_tm1, t - 1, parent=node)
                        node.children.append(next_node)
                        if self.cache_noise_pred:
                            node.noise_pred = noise_pred_batch[i:i+1]

                        current_nodes[node_indices[id(node)]] = next_node

                        # if we've reached an expansion timestep, evaluate the node and do backup
                        if next_node.t in self.expand_timesteps:
                            val, noise_pred = self._ddim_rollout_value(next_node.x_t, next_node.t - 1)
                            next_node.update_value(val.item())
                            if self.cache_noise_pred:
                                next_node.noise_pred = noise_pred
                            self._backup(next_node)
                            nodes_to_remove.add(id(next_node))

                        # If we've reached t=0, evaluate reward and do backup
                        if next_node.t <= 0:
                            # Pass x_t (final image) directly to reward function
                            self._terminal_value(next_node)
                            self._backup(next_node)

            # 4. Process terminal nodes (if any nodes reached t=0)
            for node in terminal_nodes:
                if node.visit_count == 1:  # Only evaluate if first visit
                    # Pass x_t (final image) directly to reward function
                    self._terminal_value(node)
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
                self._terminal_value(node)
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
                            self._terminal_value(node)
                            self._backup(node)
                            finalized_sim_paths[path_idx] = node
                    continue

                x_t_batch = torch.cat([node.x_t for _, node in sim_path_node_list], dim=0)
                
                # Create a tensor for timesteps if model expects it, else scalar t might be fine
                # Assuming self.p_sample can handle scalar t for a batch, or adapts internally.
                # For safety, let's pass a tensor of timesteps.
                t_tensor = torch.full((len(sim_path_node_list),), t, device=self.device, dtype=torch.long)
                
                with torch.no_grad():
                    x_tm1_batch, noise_pred_batch = self.p_sample(x_t_batch, t_tensor)

                for i, (path_idx, parent_node_in_rollout) in enumerate(sim_path_node_list):
                    child_x_tm1 = x_tm1_batch[i:i+1]
                    next_node_in_rollout = MCTSNode(child_x_tm1, t - 1, parent=parent_node_in_rollout)
                    parent_node_in_rollout.children.append(next_node_in_rollout)
                    if not self.keep_chain_states and not self._state_is_read(t):
                        parent_node_in_rollout.x_t = None
                    elif self.cache_noise_pred:
                        parent_node_in_rollout.noise_pred = noise_pred_batch[i:i+1]

                    if next_node_in_rollout.t <= 0:
                        self._terminal_value(next_node_in_rollout)
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
                # x_t_for_parent_batch = parent_node.x_t.repeat(num_to_expand_from_parent, 1)
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
                x_t_for_leaf_batch = parent_leaf_node.x_t.repeat(num_to_step_from_leaf, 1)

                with torch.no_grad():
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
                        self._terminal_value(one_step_child)
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
            elif (self.nfes + 1) % max(1, num_nfes // 10) == 0:
                progress = (self.nfes + 1) / num_nfes * 100
                print(f"Tree building progress: {progress:.1f}% (NFE {self.nfes+1}/{num_nfes})")
                sys.stdout.flush()
                        
        if self.use_progress_bar:
            pbar.close()
        
        # # Visualize the tree for debugging
        # if self.visualization_timesteps:
        #     self.visualize_tree(self.meta_root, max_depth=100, it=0)
        
        print(f"MCTS tree building {num_nfes} NFEs in {time.time() - self.start_time:.2f} seconds")

    @torch.no_grad()
    def sample_from_tree(self, population_size=1, max_restarts_per_path=100): # Added max_restarts
        """
        Sample multiple final states by traversing the built MCTS tree using rejection sampling.

        If a path hits a non-terminal leaf node (t>0), it restarts from the meta-root.
        Continues until population_size paths successfully reach t=0 through the tree.

        Args:
            population_size (int): Number of independent paths to sample.
            max_restarts_per_path (int): Maximum number of times a single path can be restarted
                                         before failing. Prevents infinite loops if t=0 is unreachable.

        Returns:
            tuple: (final_samples, final_samples_reward, nodes_by_timestep)
                - final_samples (torch.Tensor): Tensor of final generated samples [N <= population_size, D].
                - final_samples_reward (np.ndarray): Array of rewards for the final samples.
                - nodes_by_timestep (dict): Dictionary mapping visualization timesteps
                                            to lists of nodes reached at those timesteps
                                            across all sampled *and attempted* paths.
        """
        self.start_time2 = time.time()

        final_nodes = {}  # path_idx -> final MCTSNode at t=0
        nodes_by_timestep = defaultdict(list)  # For visualization
        current_nodes = {idx: self.meta_root for idx in range(population_size)} # path_idx -> current MCTSNode
        path_restarts = defaultdict(int) # path_idx -> number of restarts
        active_path_indices = set(range(population_size)) # Indices of paths still running or restarting

        total_iterations = 0
        # Heuristic upper bound: avg path length * restarts * population
        max_total_iterations = population_size * (max_restarts_per_path + 1) * (self.num_timesteps + 1)

        while len(final_nodes) < population_size and active_path_indices and total_iterations < max_total_iterations:
            total_iterations += 1
            nodes_to_process_this_iter = {idx: current_nodes[idx] for idx in active_path_indices}
            next_iter_nodes = {} # Stores the node state for the *next* iteration for active paths
            processed_indices_this_iter = set() # Track indices handled in this iteration

            # --- Visualization snapshot (before processing step) ---
            if self.visualization_timesteps:
                current_active_nodes_list = list(nodes_to_process_this_iter.values())
                for viz_t in self.visualization_timesteps:
                    nodes_at_viz_t = []
                    if viz_t == len(self.timesteps): # Virtual timestep T
                        nodes_at_viz_t = [node for node in current_active_nodes_list if self._is_meta_root(node)]
                    elif viz_t >= 0: # Actual timesteps 0 to T-1
                        nodes_at_viz_t = [node for node in current_active_nodes_list if not self._is_meta_root(node) and node.t == viz_t]

                    # Add nodes that *finished* in the previous step if visualizing t=0
                    if viz_t == 0:
                        finished_nodes_now_at_t0 = [n for idx, n in final_nodes.items() if idx not in processed_indices_this_iter]
                        nodes_at_viz_t.extend(finished_nodes_now_at_t0)

                    if nodes_at_viz_t:
                        nodes_by_timestep[viz_t].extend(nodes_at_viz_t)

            # --- Process nodes step by step ---
            # Group active nodes by their current timestep 't' to process consistently (optional, could process dict directly)
            nodes_by_t = defaultdict(list)
            for idx, node in nodes_to_process_this_iter.items():
                 t_group = node.t # Use the node's timestep for grouping
                 nodes_by_t[t_group].append((idx, node))

            # Process nodes starting from the highest timestep (ensures meta-root handled if needed)
            for t_idx, node_list in sorted(nodes_by_t.items(), key=lambda item: item[0], reverse=True):
                for path_idx, node in node_list:
                    if path_idx in processed_indices_this_iter: continue # Already handled

                    # Check if path exceeded restarts
                    if path_restarts[path_idx] >= max_restarts_per_path:
                        # print(f"Warning: Path {path_idx} exceeded max restarts ({max_restarts_per_path}). Discarding.")
                        active_path_indices.discard(path_idx) # Stop processing this path
                        processed_indices_this_iter.add(path_idx)
                        continue

                    # Case 1: Node has children -> Select next node based on tree values
                    if node.children:
                        # Always exploit the learned tree values during sampling by picking the best child locally.
                        chosen_child = self._select_child_by_value(node, use_exploration=False)

                        if chosen_child.t <= 0: # Reached terminal node via tree
                            final_nodes[path_idx] = chosen_child
                            active_path_indices.discard(path_idx) # Path finished successfully
                        else: # Continue traversal
                            next_iter_nodes[path_idx] = chosen_child # Set node for next iteration
                    # Case 2: Node is a leaf node (no children)
                    else:
                        # Subcase 2a: Leaf is non-terminal (t > 0 or meta-root) -> Reject and Restart
                        if node.t > 0 or self._is_meta_root(node):
                            # Check if meta-root *actually* has children before restarting
                            # If meta_root has no children, the tree is empty/not built, cannot sample.
                            if self._is_meta_root(node) and not self.meta_root.children:
                                 print(f"Error: Path {path_idx} hit meta-root leaf. Tree empty or not built. Cannot sample. Discarding path.")
                                 active_path_indices.discard(path_idx) # Stop processing this path
                            else:
                                # print(f"Info: Path {path_idx} hit non-terminal leaf at t={node.t}. Restarting.")
                                path_restarts[path_idx] += 1
                                next_iter_nodes[path_idx] = self.meta_root # Reset to meta-root for next iteration
                                # Keep path_idx in active_path_indices
                        # Subcase 2b: Leaf is terminal (t <= 0)
                        # This means we reached t=0 directly without a child selection (e.g., T=1 and first step went to t=0)
                        # Or it's a t=0 node that was somehow added as a leaf without a parent link?? (Shouldn't happen)
                        else:
                             # print(f"Info: Path {path_idx} reached terminal leaf node t={node.t}.")
                             final_nodes[path_idx] = node
                             active_path_indices.discard(path_idx) # Path finished successfully

                    processed_indices_this_iter.add(path_idx) # Mark as processed for this iteration

            # Update current_nodes state for the next iteration based on transitions
            current_nodes = next_iter_nodes
            # Clean active_path_indices: keep only those present in next_iter_nodes keys
            active_path_indices = active_path_indices.intersection(current_nodes.keys())


        # --- Final Processing ---
        if total_iterations >= max_total_iterations:
             print(f"Warning: Reached max total iterations ({max_total_iterations}) before collecting {population_size} samples. Collected {len(final_nodes)}.")
        elif not active_path_indices and len(final_nodes) < population_size:
             num_failed = population_size - len(final_nodes)
             print(f"Warning: {num_failed} paths failed (likely hit max restarts or empty meta-root). Collected {len(final_nodes)} out of {population_size} requested.")

        # Collect the successfully completed final nodes
        sampled_nodes = list(final_nodes.values()) # These are nodes at t=0

        if not sampled_nodes:
             print("Warning: No samples generated successfully in sample_from_tree. Returning empty.")
             # Add final t=0 nodes to visualization even if empty, to maintain structure
             if 0 in self.visualization_timesteps:
                 nodes_by_timestep[0] # Ensure the key exists

             return torch.empty((0, self.dims), device=self.device), np.array([]), nodes_by_timestep

        # Handle the function's 'maximize' argument based on tree estimates *before* final reward calc
        if self.maximize:
            valid_final_nodes = [n for n in sampled_nodes if hasattr(n, 'value_est')] # Check tree estimates
            if not valid_final_nodes:
                print("Warning: Maximize=True but no final nodes with value_est from tree found. Returning first sample.")
                sampled_nodes = sampled_nodes[0:1] if sampled_nodes else []
            elif valid_final_nodes:
                try:
                    best_node = max(valid_final_nodes, key=lambda x: x.value_est)
                    sampled_nodes = [best_node]
                except Exception as e:
                    print(f"Warning: Error finding best node during maximize: {e}. Returning first node.")
                    sampled_nodes = sampled_nodes[0:1] if sampled_nodes else []

            else: # Should not happen if sampled_nodes was not empty
                sampled_nodes = []

        # Check if any samples remain after filtering
        if not sampled_nodes:
             print("Warning: No samples selected after processing (maximize might have removed all). Returning empty.")
             # Add final t=0 nodes to visualization even if empty
             if 0 in self.visualization_timesteps:
                 nodes_by_timestep[0] # Ensure the key exists
             return torch.empty((0, self.dims), device=self.device), np.array([]), nodes_by_timestep

        # Concatenate final samples (x_0) and calculate their *actual* rewards
        final_samples_tensor = torch.cat([node.x_t for node in sampled_nodes], dim=0)
        try:
            final_samples_reward = self.reward_fn(final_samples_tensor) # Calculate final reward
        except Exception as e:
            print(f"Error calculating final rewards: {e}. Returning empty rewards.")
            final_samples_reward = np.array([])


        # Optional: Update node values with final reward (for potential future use/logging)
        # This happens *after* selection based on tree estimates if maximize=True
        if isinstance(final_samples_reward, (np.ndarray, torch.Tensor)) and final_samples_reward.shape[0] == len(sampled_nodes):
             for i, node in enumerate(sampled_nodes):
                 try:
                     node.update_value(float(final_samples_reward[i]))
                 except (IndexError, TypeError, ValueError) as e:
                     print(f"Warning: Reward array processing error: {e}. Rewards shape: {getattr(final_samples_reward, 'shape', 'N/A')}, index: {i}")
                     pass # Handle error appropriately
        elif len(sampled_nodes) > 0 :
             print(f"Warning: Shape mismatch between samples ({len(sampled_nodes)}) and rewards ({getattr(final_samples_reward, 'shape', 'N/A')}). Skipping node value update.")


        # Ensure reward is numpy array for consistent return type
        if not isinstance(final_samples_reward, np.ndarray):
            try:
                final_samples_reward = np.array(final_samples_reward)
                if final_samples_reward.ndim == 0 and len(sampled_nodes) == 1: # Handle single scalar reward
                     final_samples_reward = final_samples_reward.reshape(1)
                elif final_samples_reward.ndim == 0 and len(sampled_nodes) != 1:
                     print("Warning: Scalar reward returned for multiple samples. Broadcasting.")
                     final_samples_reward = np.repeat(final_samples_reward, len(sampled_nodes))

            except Exception as e:
                print(f"Warning: Could not convert final rewards to numpy array: {e}")
                final_samples_reward = np.array([]) # Return empty array on failure


        # Add final t=0 nodes to visualization dict if requested (ensure no duplicates)
        if 0 in self.visualization_timesteps:
             final_node_ids = {id(n) for n in nodes_by_timestep[0]}
             nodes_to_add = [n for n in sampled_nodes if id(n) not in final_node_ids]
             nodes_by_timestep[0].extend(nodes_to_add)
        elif 0 not in nodes_by_timestep and 0 in self.visualization_timesteps:
             # Ensure the key exists even if no nodes were added during loop
             nodes_by_timestep[0] = []


        print(f"MCTS population sampling (rejection) {len(sampled_nodes)} samples in {time.time() - self.start_time2:.2f} seconds. Total NFEs during build: {self.nfes}. Total iterations: {total_iterations}.")

        return final_samples_tensor, final_samples_reward, nodes_by_timestep
