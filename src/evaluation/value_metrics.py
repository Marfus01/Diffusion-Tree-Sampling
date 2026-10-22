"""
Evaluation metrics for value estimates in MCTS and SMC.
This module provides functions to evaluate the accuracy of value estimates
compared to ground truth for different timesteps in diffusion models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
import seaborn as sns
import argparse


def sample_nodes_from_tree(nodes_by_timestep, timesteps, samples_per_timestep=10):
    """
    Sample nodes from multiple MCTS trees at specific timesteps.
    
    Args:
        nodes_by_timestep: Dictionary mapping timesteps to lists of nodes
        timesteps: List of timesteps to sample nodes from
        samples_per_timestep: Number of nodes to sample per timestep
        
    Returns:
        Dictionary mapping timesteps to lists of sampled nodes
    """
    sampled_nodes = {t: [] for t in timesteps}
    
    # Now sample nodes uniformly from the collected nodes for each timestep
    for t in timesteps:
        available_nodes = nodes_by_timestep.get(t, [])
        if not available_nodes:
            print(f"Warning: No nodes found for timestep {t}")
            continue
            
        # If we have fewer nodes than requested, use all available nodes
        if len(available_nodes) <= samples_per_timestep:
            sampled_nodes[t] = available_nodes
        else:
            # Randomly sample the requested number of nodes
            sampled_nodes[t] = random.sample(available_nodes, samples_per_timestep)
        
        values_to_print = []
        for node in sampled_nodes[t]:
            values_to_print.append(node.value_est)
    
    # Print how many nodes were found for each timestep
    for t, nodes in sampled_nodes.items():
        if len(nodes) < samples_per_timestep:
            print(f"Warning: Only found {len(nodes)} nodes for timestep {t}")
    
    return sampled_nodes


def sample_particles_from_smc(smc_particles, timesteps, samples_per_timestep=10):
    """
    Sample particles from SMC at specific timesteps.
    
    Args:
        smc_particles: Dict mapping timesteps to lists or dicts of particles from SMC
        timesteps: List of timesteps to sample particles from
        samples_per_timestep: Number of particles to sample per timestep
        
    Returns:
        Dictionary mapping timesteps to lists of sampled particles
    """
    sampled_particles = {}
    for t in timesteps:
        if t in smc_particles:
            particles = smc_particles[t]['samples']
            rewards = smc_particles[t]['rewards']
            
            # Skip if particles is None or empty
            if particles is None or particles.size(0) == 0:
                print(f"Warning: No particles found for timestep {t}")
                sampled_particles[t] = {}
                continue
            
            # # Convert 2D tensor to list of particles (each row is a particle)
            # particles = [row for row in particles]
            
            # Sample min(samples_per_timestep, available_particles) particles
            num_samples = min(samples_per_timestep, len(particles))
            # Sample indices to maintain correspondence between particles and rewards
            sampled_indices = random.sample(range(len(particles)), num_samples)
            sampled_particles[t] = [
                {'samples': particles[i], 'rewards': rewards[i]} 
                for i in sampled_indices
            ]
            
        else:
            print(f"Warning: No particles found for timestep {t}")
            sampled_particles[t] = {}
            
    return sampled_particles


def compute_ground_truth_values(samples, t, diffusion_model, reward_fn, num_trajectories=100, device='cuda'):
    """
    Compute ground truth value estimates for a set of samples (nodes or particles) at a single timestep.
    
    Args:
        samples: List of samples (MCTS nodes or SMC particles) all at the same timestep
        t: model time index of the samples' state; t < 0 means the samples are already clean.
           Rollouts apply p_sample at t, t-1, ..., 0, ending at the clean sample.
        diffusion_model: The diffusion model to sample trajectories
        reward_fn: Function to compute rewards at t=0
        num_trajectories: Number of trajectories to sample per sample
        device: Device to perform computations on
        
    Returns:
        Dictionary mapping sample IDs to ground truth value estimates
    """
    ground_truth_values = {}

    if t < 0:
        x_batch = torch.stack([sample.x_t[0] if hasattr(sample, 'x_t') else sample['samples'] for sample in samples])
        all_rewards = np.exp(reward_fn(x_batch))
        # Store rewards in ground truth values dictionary
        for i, sample in enumerate(samples):
            ground_truth_values[id(sample)] = float(all_rewards[i])
        return ground_truth_values
    
    # Create a batch of all samples
    x_batch = torch.stack([sample.x_t[0] if hasattr(sample, 'x_t') else sample['samples'] for sample in samples])
    # Repeat the batch num_trajectories times
    # [s1, s2, s3] with num_trajectories=2 becomes [s1, s1, s2, s2, s3, s3]
    x_batch = torch.repeat_interleave(x_batch, num_trajectories, dim=0)
    
    # Run the diffusion model until t=0
    current_t = t
    while current_t >= 0:
        t_tensor = torch.tensor([current_t] * x_batch.size(0), device=device).long()
        with torch.no_grad():
            x_batch, _ = diffusion_model.p_sample(x_batch, t_tensor)
        current_t -= 1
    
    # Now x_batch contains states at t=0, compute rewards
    all_rewards = np.exp(reward_fn(x_batch))
    
    # Reshape rewards to group by sample: (num_samples * num_trajectories) -> (num_samples, num_trajectories)
    rewards_grouped = all_rewards.reshape(-1, num_trajectories)
    
    # Compute average reward for each sample
    avg_rewards = rewards_grouped.mean(axis=1)
    
    # Store in the ground truth values dictionary
    for i, sample in enumerate(samples):
        ground_truth_values[id(sample)] = float(avg_rewards[i])
    
    return ground_truth_values


def compute_relative_errors(samples, ground_truth_values):
    """
    Compute relative errors of value estimates compared to ground truth.
    
    Args:
        samples: List of samples (MCTS nodes or SMC particles) with value estimates
        ground_truth_values: Dictionary mapping sample IDs to ground truth values
        
    Returns:
        List of relative errors
    """
    relative_errors = []

    for sample in samples:
        sample_id = id(sample)
        if sample_id in ground_truth_values:
            # MCTS nodes store log values; SMC particles store e^{r(x0_hat)}
            if hasattr(sample, 'value_est'):
                log_estimate = sample.value_est
            else:
                log_estimate = np.log(sample['rewards'] + 1e-10)
            ground_truth = ground_truth_values[sample_id]
            
            # Skip if ground truth is zero to avoid division by zero
            if abs(ground_truth) < 1e-10:
                continue
            
            rel_error = (log_estimate - np.log(ground_truth+1e-10)) / np.log(ground_truth+1e-10)
            # rel_error = (np.log(value_estimate+1e-10) - np.log(ground_truth+1e-10))
            # rel_rmse = np.sqrt(np.mean(rel_error ** 2))
            relative_errors.append(rel_error)
    
    return relative_errors


def evaluate_mcts_value_estimates(nodes_by_timestep, diffusion_model, reward_fn, timesteps, 
                                  samples_per_timestep=10, num_trajectories=100, device='cuda'):
    """
    Evaluate MCTS value estimates for different timesteps.
    
    Args:
        nodes_by_timestep: Dictionary mapping timesteps to lists of nodes
        diffusion_model: The diffusion model to sample trajectories
        reward_fn: Function to compute rewards at t=0
        timesteps: List of timesteps to evaluate
        samples_per_timestep: Number of nodes to sample per timestep
        num_trajectories: Number of trajectories to sample per node
        device: Device to perform computations on
        
    Returns:
        Dictionary mapping timesteps to lists of relative errors
    """
    all_errors = {t: [] for t in timesteps}
    
    # Sample nodes from all trees at once
    sampled_nodes = sample_nodes_from_tree(nodes_by_timestep, timesteps, samples_per_timestep)
    
    # For each timestep
    for t, nodes in sampled_nodes.items():
        if not nodes:
            continue
            
        # Compute ground truth values
        # node.t is the model index of its state; terminal nodes (t = 0) hold the clean sample
        state_index = t if t > 0 else -1
        ground_truth_values = compute_ground_truth_values(nodes, state_index, diffusion_model, reward_fn, 
                                                        num_trajectories, device)
        # Compute relative errors
        errors = compute_relative_errors(nodes, ground_truth_values)
        all_errors[t].extend(errors)

    return all_errors


def evaluate_smc_value_estimates(smc_particles, diffusion_model, reward_fn, timesteps,
                                samples_per_timestep=10, num_trajectories=100, device='cuda'):
    """
    Evaluate SMC value estimates for different timesteps.
    
    Args:
        smc_particles: Dict mapping timesteps to lists of particles from SMC
        diffusion_model: The diffusion model to sample trajectories
        reward_fn: Function to compute rewards at t=0
        timesteps: List of timesteps to evaluate
        samples_per_timestep: Number of particles to sample per timestep
        num_trajectories: Number of trajectories to sample per particle
        device: Device to perform computations on
        
    Returns:
        Dictionary mapping timesteps to lists of relative errors
    """
    all_errors = {t: [] for t in timesteps}
    
    # Sample particles from SMC
    sampled_particles = sample_particles_from_smc(smc_particles, timesteps, samples_per_timestep)
    
    # For each timestep
    for t, particles in sampled_particles.items():
        if not particles:
            continue
            
        # Compute ground truth values
        # SMC history keys are t+1 for the state at model index t; key 0 is the clean sample
        ground_truth_values = compute_ground_truth_values(particles, t - 1, diffusion_model, reward_fn, 
                                                        num_trajectories, device)
        
        # Compute relative errors
        errors = compute_relative_errors(particles, ground_truth_values)
        all_errors[t].extend(errors)

    return all_errors


def plot_value_error_vs_timestep(mcts_iterations_errors=None, smc_particles_errors=None, 
                               plot_dir='plots', 
                               filename='value_error_vs_timestep.png'):
    """
    Plot relative value error vs. diffusion timestep for MCTS and/or SMC.
    
    Args:
        mcts_iterations_errors: Dict mapping iteration counts to error dicts for MCTS
        smc_particles_errors: Dict mapping particle counts to error dicts for SMC
        plot_dir: Directory to save the plot
        filename: Filename for the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot MCTS errors for all iterations
    if mcts_iterations_errors:
        timesteps = sorted(next(iter(mcts_iterations_errors.values())).keys())
        for iterations, errors in sorted(mcts_iterations_errors.items()):
            mean_errors = [np.mean(errors[t]) if errors[t] else 0 for t in timesteps]
            std_errors = [np.std(errors[t]) if errors[t] else 0 for t in timesteps]
            
            plt.errorbar(timesteps, mean_errors, yerr=std_errors, marker='o', linestyle='-', 
                       label=f'MCTS ({iterations} iterations)')
    
    # Plot SMC errors for all particle counts
    if smc_particles_errors:
        timesteps = sorted(next(iter(smc_particles_errors.values())).keys())
        for particles, errors in sorted(smc_particles_errors.items()):
            mean_errors = [np.mean(errors[t]) if errors[t] else 0 for t in timesteps]
            std_errors = [np.std(errors[t]) if errors[t] else 0 for t in timesteps]
            
            plt.errorbar(timesteps, mean_errors, yerr=std_errors, marker='s', linestyle='--', 
                       label=f'SMC ({particles} particles)')
    
    plt.xlabel('Diffusion Timestep')
    plt.ylabel('Relative Value Error')
    plt.title('Value Estimation Error vs. Diffusion Timestep')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


def plot_value_error_comparison(mcts_iterations_errors, smc_particles_errors, 
                              timesteps, plot_dir='plots', 
                              filename='value_error_comparison.png'):
    """
    Plot relative value error vs. diffusion timestep for different numbers of
    MCTS iterations and SMC particles.
    
    Args:
        mcts_iterations_errors: Dict mapping iteration counts to error dicts
        smc_particles_errors: Dict mapping particle counts to error dicts
        timesteps: List of timesteps to plot
        plot_dir: Directory to save the plot
        filename: Filename for the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Create subplots for each timestep
    num_timesteps = len(timesteps)
    num_cols = min(3, num_timesteps)
    num_rows = (num_timesteps + num_cols - 1) // num_cols
    
    for i, t in enumerate(sorted(timesteps)):
        plt.subplot(num_rows, num_cols, i + 1)
        
        # Plot MCTS errors for different iterations
        mcts_iterations = sorted(mcts_iterations_errors.keys())
        mcts_means = []
        mcts_stds = []
        
        for iterations in mcts_iterations:
            errors = mcts_iterations_errors[iterations].get(t, [])
            if errors:
                mcts_means.append(np.mean(errors))
                mcts_stds.append(np.std(errors))
            else:
                mcts_means.append(0)
                mcts_stds.append(0)
        
        plt.errorbar(mcts_iterations, mcts_means, yerr=mcts_stds, marker='o', linestyle='-', 
                   label='MCTS')
        
        # Plot SMC errors for different particle counts
        smc_particles = sorted(smc_particles_errors.keys())
        smc_means = []
        smc_stds = []
        
        for particles in smc_particles:
            errors = smc_particles_errors[particles].get(t, [])
            if errors:
                smc_means.append(np.mean(errors))
                smc_stds.append(np.std(errors))
            else:
                smc_means.append(0)
                smc_stds.append(0)
        
        plt.errorbar(smc_particles, smc_means, yerr=smc_stds, marker='s', linestyle='--', 
                   label='SMC')
        
        plt.xlabel('Iterations (MCTS) / Particles (SMC)')
        plt.ylabel('Relative Value Error')
        plt.title(f'Value Error at Timestep {t}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
    
    plt.tight_layout()
    
    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


def plot_heatmap(mcts_iterations_errors, smc_particles_errors, timesteps, 
               plot_dir='plots', filename='value_error_heatmap.png'):
    """
    Plot heatmaps of relative value error for different numbers of
    MCTS iterations and SMC particles across diffusion timesteps.
    
    Args:
        mcts_iterations_errors: Dict mapping iteration counts to error dicts
        smc_particles_errors: Dict mapping particle counts to error dicts
        timesteps: List of timesteps to plot
        plot_dir: Directory to save the plot
        filename: Filename for the plot
    """
    plt.figure(figsize=(20, 8))
    
    # Create MCTS heatmap
    plt.subplot(1, 2, 1)
    mcts_iterations = sorted(mcts_iterations_errors.keys())
    timesteps = sorted(timesteps)
    
    mcts_data = np.zeros((len(timesteps), len(mcts_iterations)))
    
    for i, t in enumerate(timesteps):
        for j, iterations in enumerate(mcts_iterations):
            errors = mcts_iterations_errors[iterations].get(t, [])
            if errors:
                mcts_data[i, j] = np.mean(errors)
    
    sns.heatmap(mcts_data, annot=True, fmt='.3f', cmap='viridis',
              xticklabels=mcts_iterations, yticklabels=timesteps)
    plt.xlabel('MCTS Iterations')
    plt.ylabel('Diffusion Timestep')
    plt.title('MCTS Value Error')
    
    # Create SMC heatmap
    plt.subplot(1, 2, 2)
    smc_particles = sorted(smc_particles_errors.keys())
    
    smc_data = np.zeros((len(timesteps), len(smc_particles)))
    
    for i, t in enumerate(timesteps):
        for j, particles in enumerate(smc_particles):
            errors = smc_particles_errors[particles].get(t, [])
            if errors:
                smc_data[i, j] = np.mean(errors)
    
    sns.heatmap(smc_data, annot=True, fmt='.3f', cmap='viridis',
              xticklabels=smc_particles, yticklabels=timesteps)
    plt.xlabel('SMC Particles')
    plt.ylabel('Diffusion Timestep')
    plt.title('SMC Value Error')
    
    plt.tight_layout()
    
    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

