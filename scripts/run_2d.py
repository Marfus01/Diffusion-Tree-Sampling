import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
from pathlib import Path
import logging
import colorlog
import pprint
import argparse
from dataclasses import asdict
import time
import tqdm
import seaborn as sns # Added for KDE plots
from scipy.stats import gaussian_kde # Added for manual ridge plots
import pickle # Added for saving plot data

from src.models.diffusion_2d import DiffusionModel
from src.utils.datasets import get_2d_dataset
from src.core.config import ToyConfig, get_default_toy_config, process_toy_config
from src.core.sampling.mcts_parallel import MCTSDiffusionSampler
# from src.core.sampling.fkd_2d import FKDDiffusion2DSampler, PotentialType
from src.core.sampling.smc_2d import SMCDiffusion2DSampler, PotentialType
from src.evaluation.distribution_metrics import compute_mmd
from src.evaluation.value_metrics import evaluate_mcts_value_estimates, evaluate_smc_value_estimates
# from scripts.plotter_2d import (
#     create_distribution_plots, create_reward_kde_plots,
#     create_all_methods_distribution_plot, create_plots
# )


# Setup logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)-2s [%(asctime)s]%(reset)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG':    'cyan', 'INFO':     'green', 'WARNING':  'yellow',
        'ERROR':    'red', 'CRITICAL': 'red,bg_white',
    }
))
root_logger = logging.getLogger()
for h in root_logger.handlers[:]: root_logger.removeHandler(h)
root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def parse_script_args():
    """Parses command-line overrides for ToyConfig."""
    parser = argparse.ArgumentParser(description='Run guided 2D Diffusion experiments', add_help=True)

    # Common Overrides
    cgroup = parser.add_argument_group('Common Overrides')
    cgroup.add_argument('--dataset', type=str, help='Dataset override.')
    cgroup.add_argument('--exp_name', type=str, help='Experiment name prefix.')
    cgroup.add_argument('--seed', type=int, help='Random seed.')
    cgroup.add_argument('--device', type=str, help='Device (cuda/cpu).')
    cgroup.add_argument('--no_progress_bar', action='store_true', help='Disable progress bars.')
    cgroup.add_argument('--hf_home', type=str, help='HuggingFace cache dir.')

    # Method Selection
    mgroup = parser.add_argument_group('Method Selection')
    mgroup.add_argument('--do_mcts', action='store_true', help='Enable MCTS (standard).')
    mgroup.add_argument('--do_mcts_rollout', action='store_true', help='Enable MCTS with Rollout reward estimation.')
    mgroup.add_argument('--do_guidance', action='store_true', help='Enable standalone reward-based guidance.')
    mgroup.add_argument('--do_smc', action='store_true', help='Enable standard SMC (FKD-style).')
    mgroup.add_argument('--do_smc_guidance', action='store_true', help='Enable SMC with reward-based guidance.')
    mgroup.add_argument('--do_smc_guidance_tempering', action='store_true', help='Enable SMC with reward-based guidance and Tempering.')
    mgroup.add_argument('--do_smc_rollout', action='store_true', help='Enable SMC with Rollout reward estimation.')

    # Sampling Overrides
    sgroup = parser.add_argument_group('Sampling Overrides')
    sgroup.add_argument('--compute_budget', nargs='+', type=int, help='Compute budget in NFEs.')
    sgroup.add_argument('--population_size', type=int, help='Final population size.')
    sgroup.add_argument('--potential_type', type=str, help='Potential type for FKD/SMC.')
    sgroup.add_argument('--lambda_param', type=float, help='Temperature for FKD/SMC.')
    sgroup.add_argument('--guidance_scale', type=float, default=1.0, help='Scale for reward-based guidance.')

    # Evaluation Overrides
    egroup = parser.add_argument_group('Evaluation Overrides')
    egroup.add_argument('--no_value_eval', action='store_true', help='Disable value function evaluation.')
    egroup.add_argument('--value_timesteps', nargs='+', type=int, help='Timesteps for value evaluation.')

    args = parser.parse_args()
    config = get_default_toy_config()

    # Apply overrides
    if args.dataset is not None: config.dataset.name = args.dataset
    if args.exp_name is not None: config.common.exp_name = args.exp_name
    if args.seed is not None: config.common.seed = args.seed
    if args.device is not None: config.common.device = args.device
    if args.no_progress_bar is not None: config.common.use_progress_bar = not args.no_progress_bar
    if args.hf_home is not None: config.common.hf_home = args.hf_home

    # Enable methods based on flags
    if args.do_mcts: config.mcts.enabled = True
    if args.do_mcts_rollout: config.mcts_rollout = True # Temp flag

    if args.compute_budget is not None:
        config.sampling.compute_budget = args.compute_budget # NFE budget
    if args.population_size is not None: config.sampling.population_size = args.population_size
    if args.potential_type is not None: config.fkd.potential_type = args.potential_type
    if args.lambda_param is not None:
        config.fkd.lambda_param = args.lambda_param
    if args.guidance_scale is not None:
        config.guidance_scale = args.guidance_scale # Store guidance scale in config

    if args.no_value_eval is not None: config.eval.evaluate_value = not args.no_value_eval
    if args.value_timesteps is not None: config.eval.value_eval_timesteps = args.value_timesteps

    config = process_toy_config(config)
    return config, args

def load_diffusion_model(config):
    """Loads the 2D diffusion model."""
    logger.info(f"Loading diffusion model for {config.dataset.name}...")
    model = DiffusionModel(
        input_dim=2,
        hidden_dim=config.model.hidden_dim,
        n_timesteps=config.model.n_timesteps,
        beta_start=config.model.beta_start,
        beta_end=config.model.beta_end,
        device=config.common.device
    )
    model_path = config.paths.diffusion_load_path
    if model_path and Path(model_path).exists():
        try:
            model.noise_predictor.load_state_dict(torch.load(model_path, map_location=config.common.device))
            logger.info(f"Loaded pre-trained model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}. Using untrained model.")
    else:
        logger.warning(f"Diffusion model path not found ({model_path}). Using untrained model.")

    model.noise_predictor.eval()
    return model

def run_mcts(config, dataset_name, diffusion_model, dataset, method_name="MCTS"):
    """Runs MCTS or MCTS-R experiment."""
    logger.info(f"--- Running {method_name} for {dataset_name} ---")
    metrics_data = {}
    value_errors = {}

    # Set rollout based on method name
    use_rollout = ("-R" in method_name or "Rollout" in method_name) # Updated check

    # Generate ground truth samples
    # gt_samples, gt_rewards = dataset.sample_product_distribution_diff(n_samples=config.sampling.population_size)
    # gt_samples_np = gt_samples.cpu().numpy()
    # gt_rewards_np = gt_rewards
    gt_samples_np = dataset.gt_samples
    gt_rewards_np = dataset.gt_rewards
    gt_mean_reward = gt_rewards_np.mean()

    all_mcts_samples = {}
    all_mcts_rewards = {}
    value_eval_timesteps = config.eval.value_eval_timesteps if config.eval.evaluate_value else None
    # value_eval_timesteps = list(config.mcts.expand_timesteps[1:] + [0]) if config.eval.evaluate_value else None
    visualization_timesteps = list(config.mcts.expand_timesteps + [0])

    reward_fn = lambda x: dataset.calculate_reward_diff(x).cpu().numpy()

    # Create MCTS sampler
    mcts_sampler = MCTSDiffusionSampler(
        diffusion_model=diffusion_model,
        # reward_fn=lambda x: dataset.calculate_reward(x),
        reward_fn=reward_fn,
        expand_timesteps=config.mcts.expand_timesteps,
        num_nfes=max(config.sampling.compute_budget),
        num_inference_steps=config.sampling.num_inference_steps,
        eta=config.sampling.eta,
        device=config.common.device,
        exploration_const=config.mcts.exploration_const,
        progressive_width_const=config.mcts.progressive_width_const,
        progressive_width_alpha=config.mcts.progressive_width_alpha,
        use_progress_bar=config.common.use_progress_bar,
        use_rollout=use_rollout,
        batch_size=config.mcts.batch_size,
        output_dir=config.paths.plot_dir,
        visualization_timesteps=visualization_timesteps,
        dims=2
    )

    mcts_sampler.initialize_root_nodes()
    nodes_by_timestep_all_iters = {}
    start_run_time = time.time()

    # Run MCTS progressively for each NFE target
    for target_nfes in sorted(config.sampling.compute_budget):
        logger.info(f"{method_name} NFE Target {target_nfes} for {dataset_name}")
        iter_start_time = time.time()

        # Build search tree up to the target NFE
        mcts_sampler.build_search_tree(num_nfes=target_nfes) # Pass NFE budget

        # Sample final population
        # Adjust sampling population size based on NFE? Or keep fixed? Keep fixed for now.
        samples, _, nodes_by_timestep = mcts_sampler.sample_from_tree(population_size=config.sampling.population_size)

        # Evaluate samples
        # rewards = dataset.calculate_reward(samples)
        rewards = dataset.calculate_reward_diff(samples).cpu().numpy()
        samples_np = samples.cpu().numpy()
        rewards_np = rewards
        # Calculate ESS based on sample counts
        if samples_np.size > 0: # Check if the array is not empty
            _, counts = np.unique(samples_np, axis=0, return_counts=True)
            total_samples = len(samples_np) # More direct way to get total count

            if total_samples > 0:
                # Treat counts as unnormalized weights
                normalized_weights = counts / total_samples
                # Calculate ESS: 1 / sum(normalized_weights^2)
                sum_sq_weights = np.sum(normalized_weights**2)
                ess = 1.0 / sum_sq_weights if sum_sq_weights > 0 else 0.0
            else:
                ess = 0.0
        else:
            ess = 0.0
        ess /= config.sampling.population_size

        # Log the calculated ESS value (replace or add to existing logging)
        # logger.log({'ESS': ess}) # Example logging
        logger.info(f"Calculated ESS: {ess:.4f}")

        mmd = compute_mmd(gt_samples_np, samples_np)
        mean_reward = rewards_np.mean()
        iter_run_time = time.time() - iter_start_time

        actual_nfes = mcts_sampler.nfes
        # Store metrics
        metrics_data[target_nfes] = {
            'MMD': mmd, 'MeanReward': mean_reward, 'GT_MeanReward': gt_mean_reward,
            'ESS': ess, 'NFEs': target_nfes, 'Time': iter_run_time, 'ActualNFEs': actual_nfes
        }
        all_mcts_samples[target_nfes] = samples_np
        all_mcts_rewards[target_nfes] = rewards_np
        nodes_by_timestep_all_iters[target_nfes] = nodes_by_timestep

        logger.info(f"  NFE Target: {target_nfes}, Actual NFEs: {actual_nfes}")
        logger.info(f"  MMD: {mmd:.4f}, Mean Reward: {mean_reward:.4f}, GT Mean Reward: {gt_mean_reward:.4f}, ESS: {ess:.4f}, Time: {iter_run_time:.2f}s")

    total_run_time = time.time() - start_run_time
    logger.info(f"{method_name} total run time for {dataset_name}: {total_run_time:.2f}s")

    # # Create distribution plots
    # create_distribution_plots(
    #     samples_dict=all_mcts_samples, rewards_dict=all_mcts_rewards,
    #     gt_samples=gt_samples_np, gt_rewards=gt_rewards_np,
    #     method_name=method_name, plot_dir=config.paths.plot_dir, dataset_name=dataset_name
    # )

    # Value Evaluation
    if config.eval.evaluate_value:
        logger.info(f"Evaluating {method_name} value estimates...")
        value_eval_start_time = time.time()
        for target_nfes in sorted(config.sampling.compute_budget):
            logger.info(f"  Value eval for {target_nfes} NFEs...")
            iter_value_errors = evaluate_mcts_value_estimates(
                nodes_by_timestep=nodes_by_timestep_all_iters[target_nfes],
                diffusion_model=diffusion_model, reward_fn=reward_fn,
                timesteps=value_eval_timesteps,
                samples_per_timestep=config.eval.value_samples_per_timestep,
                num_trajectories=config.eval.value_num_trajectories,
                device=config.common.device
            )
            value_errors[target_nfes] = iter_value_errors
        value_eval_time = time.time() - value_eval_start_time
        logger.info(f"Value evaluation completed in {value_eval_time:.2f}s")

    # Convert to DataFrames
    sample_metrics_list = [{'Method': method_name, 'N': n, **m} for n, m in metrics_data.items()]
    sample_metrics_df = pd.DataFrame(sample_metrics_list)
    value_metrics_list = [{'Method': method_name, 'N': n, 'Timestep': t, 'ValueError': np.mean(err), 'ValueStd': np.var(err, ddof=1)}
                          for n, errs in value_errors.items() for t, err in errs.items()]
    value_metrics_df = pd.DataFrame(value_metrics_list)

    return sample_metrics_df, value_metrics_df, all_mcts_samples, all_mcts_rewards # Added all_mcts_rewards

def run_fkd(config, dataset_name, diffusion_model, dataset, method_name="SMC"):
    """Runs SMC (FKD-style), SMC-Guidance, or SMC-Rollout experiment."""
    logger.info(f"--- Running {method_name} for {dataset_name} ---")
    metrics_data = {}
    value_errors = {}

    # Set rollout based on method name
    use_guidance = "guidance" in method_name.lower()
    use_smc = "smc" in method_name.lower()
    use_rollout = "rollout" in method_name.lower()
    use_tempering = "tempering" in method_name.lower()

    logger.info(f"  Config: use_smc={use_smc}, use_guidance={use_guidance}, use_rollout={use_rollout}, use_tempering={use_tempering}")

    # Generate ground truth samples
    # gt_samples, gt_rewards = dataset.sample_product_distribution_diff(n_samples=config.sampling.population_size)
    # gt_samples_np = gt_samples.cpu().numpy()
    # gt_rewards_np = gt_rewards
    gt_samples_np = dataset.gt_samples
    gt_rewards_np = dataset.gt_rewards
    gt_mean_reward = gt_rewards_np.mean()

    all_fkd_samples = {} # Keep variable names consistent with original function for now
    all_fkd_rewards = {}
    # value_eval_timesteps = config.eval.value_eval_timesteps if config.eval.evaluate_value else None
    # Define value timesteps based on resampling schedule
    value_eval_timesteps_set = set(range(config.fkd.resampling_t_start, config.fkd.resampling_t_end + 1, config.fkd.resample_frequency))
    value_eval_timesteps_set.add(config.fkd.resampling_t_start) # Ensure start is included
    value_eval_timesteps_set.add(config.fkd.resampling_t_end) # Ensure end is included
    value_eval_timesteps_set.add(0) # Ensure final timestep is included
    value_eval_timesteps = sorted(list(value_eval_timesteps_set), reverse=True)
    value_eval_timesteps = config.eval.value_eval_timesteps if config.eval.evaluate_value else None
    visualization_timesteps = value_eval_timesteps # Use same timesteps for visualization

    # Map potential type string to enum
    

    # FKD/SMC config setup using the refactored sampler
    # Use the correct reward function based on guidance flag
    # reward_function = dataset.calculate_reward_diff if use_guidance else dataset.calculate_reward
    reward_function = dataset.calculate_reward_diff
    reward_fn = lambda x: reward_function(x).cpu().numpy()
    fkd_config = {
        'reward_fn': lambda x: reward_function(x.to(config.common.device)), # Ensure tensor on correct device
        'potential_type': config.fkd.potential_type,
        'lmbda': config.fkd.lambda_param,
        'adaptive_resampling': config.fkd.adaptive_resampling,
        'resample_frequency': config.fkd.resample_frequency,
        'resampling_t_start': config.fkd.resampling_t_start,
        'resampling_t_end': config.fkd.resampling_t_end,
        'reward_min_value': config.fkd.reward_min_val,
        # 'device': config.common.device, # Device is passed to sampler directly
    }

    nfe_per_particle = config.sampling.num_inference_steps # p_sample
    if use_guidance:
        nfe_per_particle += config.sampling.num_inference_steps # ddim_denoise for grad (at each denoising step)
    if use_smc and not use_guidance:
        nfe_per_particle += len(list(range(config.fkd.resampling_t_start, config.fkd.resampling_t_end + 1, config.fkd.resample_frequency))) # ddim_denoise for reward (at each resampling step)
    if use_rollout:
        # This counts NFEs *within* the _perform_rollout calls initiated during the main loop
        rollout_nfe_count = 0
        resampling_indices = range(config.fkd.resampling_t_start, config.fkd.resampling_t_end + 1, config.fkd.resample_frequency)
        for i in resampling_indices:
            current_timestep_idx = config.sampling.num_inference_steps - i # Map sampling index to timestep_idx
            rollout_nfe_count += max(0, current_timestep_idx) # NFE = number of steps remaining
        nfe_per_particle += rollout_nfe_count
        logger.info(f"Estimated additional NFEs per particle from rollout: {rollout_nfe_count}")

    logger.info(f"Estimated NFEs per particle for {method_name}: {nfe_per_particle}")
    # Determine particle counts required for each NFE target
    eval_particles = sorted(list(set([max(1, nfe // nfe_per_particle) for nfe in config.sampling.compute_budget])))
    logger.info(f"Will run with particle counts: {eval_particles}")
    max_n_particles = max(eval_particles) if eval_particles else 1

    all_fkd_particles_dict = {}
    start_run_time = time.time()

    # Run FKD for each required particle count
    for i, n_particles in enumerate(eval_particles):
        logger.info(f"{method_name} Particle Count {n_particles} for {dataset_name}")
        iter_start_time = time.time()

        # Create FKD sampler instance
        sampler = SMCDiffusion2DSampler(
            diffusion_model=diffusion_model,
            use_smc=use_smc,
            use_guidance=use_guidance,
            use_tempering=use_tempering,
            guidance_scale=config.guidance_scale, # Use guidance scale from config
            reward_fn=reward_function, # Pass the chosen reward function
            fkd_config=fkd_config,
            device=config.common.device,
            visualization_timesteps=visualization_timesteps,
            dataset_name=dataset_name,
            use_progress_bar=config.common.use_progress_bar,
            use_rollout=use_rollout,
        )

        # Run sampling
        return_all = config.eval.evaluate_value
        sampler_output = sampler.sample(
            # fkd_config=fkd_config, # fkd_config now passed in __init__
            num_particles=n_particles,
            eta=config.sampling.eta,
            return_all_particles=return_all
        )
        actual_nfes_run = sampler.nfes

        if return_all:
            particles_dict = sampler_output # Contains samples and rewards per timestep
            final_samples_t = particles_dict[0]['samples'] # Samples at t=0
            all_fkd_particles_dict[n_particles] = particles_dict # Store for value eval
        else:
            final_samples_t = sampler_output # Just final samples

        # Resample to desired population size if needed (e.g., if n_particles < population_size)
        if len(final_samples_t) > 0:
            if len(final_samples_t) < config.sampling.population_size:
                 logger.warning(f"Generated samples ({len(final_samples_t)}) < target population ({config.sampling.population_size}). Resampling with replacement.")
                 resample_indices = torch.randint(0, len(final_samples_t), (config.sampling.population_size,))
                 final_samples = final_samples_t[resample_indices]
            elif len(final_samples_t) > config.sampling.population_size:
                 logger.info(f"Generated samples ({len(final_samples_t)}) > target population ({config.sampling.population_size}). Subsampling.")
                 resample_indices = torch.randperm(len(final_samples_t))[:config.sampling.population_size]
                 final_samples = final_samples_t[resample_indices]
            else:
                 final_samples = final_samples_t
        else:
            final_samples = final_samples_t # Handle empty case

        # Evaluate final samples
        # rewards = dataset.calculate_reward(final_samples.to(config.common.device))
        rewards = reward_fn(final_samples)
        samples_np = final_samples.cpu().numpy()
        rewards_np = rewards

        # Calculate ESS based on final sample counts
        ess = 0.0
        if samples_np.size > 0:
            _, counts = np.unique(samples_np, axis=0, return_counts=True)
            total_samples = len(samples_np)
            if total_samples > 0:
                normalized_weights = counts / total_samples
                sum_sq_weights = np.sum(normalized_weights**2)
                if sum_sq_weights > 0:
                    ess = 1.0 / sum_sq_weights
            ess /= config.sampling.population_size # Normalize by population size
        # print(f"Calculated ESS (final samples): {ess:.4f}")


        # Evaluate the resampled population
        # eval_rewards = dataset.calculate_reward(final_samples) if len(final_samples) > 0 else np.array([]) # Handle empty case
        eval_rewards = reward_fn(final_samples)
        # samples_np already computed
        rewards_np = eval_rewards

        mmd = compute_mmd(gt_samples_np, samples_np) if len(samples_np) > 0 else float('inf') # Handle empty case
        mean_reward = rewards_np.mean() if len(rewards_np) > 0 else float('-inf') # Handle empty case
        iter_run_time = time.time() - iter_start_time

        nfe_target = config.sampling.compute_budget[i]
        # Store metrics for the NFE target
        metrics_data[nfe_target] = {
            'MMD': mmd, 'MeanReward': mean_reward, 'GT_MeanReward': gt_mean_reward,
            'ESS': ess, 'NFEs': nfe_target, 'Time': iter_run_time,
            'ActualParticles': n_particles, 'ActualNFEs': actual_nfes_run
        }
        # Store raw samples for plotting associated with NFE target
        all_fkd_samples[nfe_target] = samples_np
        all_fkd_rewards[nfe_target] = rewards_np
        logger.info(f"  NFE Target: {nfe_target} (using run with {n_particles} particles, Actual NFEs: {actual_nfes_run})")
        logger.info(f"    MMD: {mmd:.4f}, Mean Reward: {mean_reward:.4f}, GT Mean Reward: {gt_mean_reward:.4f}, ESS: {ess:.4f}, Time: {iter_run_time:.2f}s")


    total_run_time = time.time() - start_run_time
    logger.info(f"{method_name} total run time for {dataset_name}: {total_run_time:.2f}s")

    # # Create distribution plots
    # create_distribution_plots(
    #     samples_dict=all_fkd_samples, rewards_dict=all_fkd_rewards,
    #     gt_samples=gt_samples_np, gt_rewards=gt_rewards_np,
    #     method_name=method_name, plot_dir=config.paths.plot_dir, dataset_name=dataset_name
    # )

    # Value Evaluation
    if config.eval.evaluate_value and return_all and use_smc and not use_guidance:
        logger.info(f"Evaluating {method_name} value estimates...")
        value_eval_start_time = time.time()
        for nfe_target in sorted(metrics_data.keys()): # Evaluate based on NFE targets
             actual_particles_used = metrics_data[nfe_target]['ActualParticles']
             logger.info(f"  Value eval for NFE target {nfe_target} (using run with {actual_particles_used} particles)...")
             if actual_particles_used in all_fkd_particles_dict:
                 iter_value_errors = evaluate_smc_value_estimates( # Use the SMC evaluator
                     smc_particles=all_fkd_particles_dict[actual_particles_used], # Pass the history dict from the correct run
                     diffusion_model=diffusion_model, reward_fn=reward_fn, # Use non-diff reward for eval
                     timesteps=value_eval_timesteps,
                     samples_per_timestep=config.eval.value_samples_per_timestep,
                     num_trajectories=config.eval.value_num_trajectories,
                     device=config.common.device
                 )
                 value_errors[nfe_target] = iter_value_errors # Store errors associated with NFE target
             else:
                 logger.warning(f"  Skipping value eval for NFE {nfe_target}: particle data not found for {actual_particles_used} particles.")
        value_eval_time = time.time() - value_eval_start_time
        logger.info(f"Value evaluation completed in {value_eval_time:.2f}s")

    # Convert to DataFrames
    sample_metrics_list = [{'Method': method_name, 'N': n, **m} for n, m in metrics_data.items()]
    sample_metrics_df = pd.DataFrame(sample_metrics_list)
    value_metrics_list = [{'Method': method_name, 'N': n, 'Timestep': t, 'ValueError': np.mean(err), 'ValueStd': np.var(err, ddof=1)}
                           for n, errs in value_errors.items() for t, err in errs.items()]
    value_metrics_df = pd.DataFrame(value_metrics_list)

    return sample_metrics_df, value_metrics_df, all_fkd_samples, all_fkd_rewards # Added all_fkd_rewards

def aggregate_and_save_summary(config): # Removed list arguments
     """Aggregates metrics across seeds by loading individual CSVs and saves summary CSVs."""
     logger.info("Loading and aggregating results from individual run CSVs...")
     metrics_dir = Path(config.paths.metrics_dir)
     # Ensure metrics directory exists (though it should have been created during runs)
     metrics_dir.mkdir(parents=True, exist_ok=True)

     # --- Load and Aggregate Sample Metrics --- #
     sample_metric_files = list(metrics_dir.glob("sample_metrics_*.csv"))
     summary_df = None # Initialize
     summary_agg_df = None # Initialize aggregated df
     if not sample_metric_files:
         logger.error("No sample metric CSV files found in %s. Cannot aggregate or plot.", metrics_dir)
         summary_df = pd.DataFrame() # Empty DF for plot function
     else:
         logger.info(f"Found {len(sample_metric_files)} sample metric files.")
         sample_dfs = []
         for f in sample_metric_files:
             try:
                 sample_dfs.append(pd.read_csv(f))
             except Exception as e:
                 logger.error(f"Failed to read sample metrics file {f}: {e}")
         if not sample_dfs:
              logger.error("Failed to read any sample metric CSV files. Cannot proceed.")
              summary_df = pd.DataFrame()
         else:
            summary_df = pd.concat(sample_dfs, ignore_index=True)
            summary_raw_path = metrics_dir / "all_runs_raw_summary.csv" # Save raw data
            summary_df = summary_df.sort_values(by=['Seed', 'Dataset', 'Method', 'N'])
            try:
                 summary_df.to_csv(summary_raw_path, index=False)
                 logger.info(f"Saved combined raw metrics summary to {summary_raw_path}")
            except Exception as e:
                 logger.error(f"Failed to save combined raw metrics summary: {e}")

            # Calculate mean and std/sem across seeds (same logic as before)
            grouping_cols = ['Dataset', 'Method', 'N']
            metric_cols = ['MMD', 'MeanReward', 'GT_MeanReward', 'ESS', 'Time', 'ActualParticles', 'ActualCandidates', 'ActualNFEs'] # Added 'ActualNFEs'
            # Filter out columns that might not exist in all DFs (e.g., ActualParticles/Candidates)
            valid_metric_cols = [col for col in metric_cols if col in summary_df.columns]

            # Define aggregation functions
            agg_funcs = {col: ['mean', 'std'] for col in valid_metric_cols}
            # Special handling for NFEs if needed (should be constant within group)
            # if 'NFEs' in summary_df.columns: agg_funcs['NFEs'] = 'first' # Take first NFE value --> Remove this, NFEs is target N, already in grouping_cols

            try:
                summary_agg_df = summary_df.groupby(grouping_cols).agg(agg_funcs).reset_index()
                # Flatten MultiIndex columns (e.g., ('MMD', 'mean') -> 'MMD_mean')
                summary_agg_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in summary_agg_df.columns.values]

                # Calculate Standard Error (SEM) = std / sqrt(n)
                num_seeds = summary_df.groupby(grouping_cols).size().reset_index(name='Count')
                summary_agg_df = pd.merge(summary_agg_df, num_seeds, on=grouping_cols)
                for col in valid_metric_cols:
                    if f'{col}_std' in summary_agg_df.columns:
                        summary_agg_df[f'{col}_sem'] = summary_agg_df[f'{col}_std'] / np.sqrt(summary_agg_df['Count'])

                summary_agg_path = metrics_dir / "all_runs_aggregated_summary.csv" # Save aggregated data
                summary_agg_df = summary_agg_df.sort_values(by=['Dataset', 'Method', 'N'])
                summary_agg_df.to_csv(summary_agg_path, index=False)
                logger.info(f"Saved aggregated metrics summary (mean/std/sem across seeds) to {summary_agg_path}")

            except Exception as e:
                 logger.error(f"Error during sample metrics aggregation: {e}")
                 summary_agg_df = None # Ensure it's None if aggregation fails

     # --- Load and Aggregate Value Metrics --- #
     value_metric_files = list(metrics_dir.glob("value_metrics_*.csv"))
     value_df = None # Initialize
     value_agg_df = None # Initialize aggregated df
     if config.eval.evaluate_value:
         if not value_metric_files:
             logger.info("No value metric CSV files found in %s.", metrics_dir)
             value_df = pd.DataFrame()
         else:
             logger.info(f"Found {len(value_metric_files)} value metric files.")
             value_dfs = []
             for f in value_metric_files:
                  try:
                     value_dfs.append(pd.read_csv(f))
                  except Exception as e:
                     logger.error(f"Failed to read value metrics file {f}: {e}")

             if not value_dfs:
                 logger.error("Failed to read any value metric CSV files.")
                 value_df = pd.DataFrame()
             else:
                 value_df = pd.concat(value_dfs, ignore_index=True)
                 value_raw_path = metrics_dir / "all_runs_raw_value_summary.csv" # Save raw data
                 value_df = value_df.sort_values(by=['Seed', 'Dataset', 'Method', 'N', 'Timestep'])
                 try:
                     value_df.to_csv(value_raw_path, index=False)
                     logger.info(f"Saved combined raw value error summary to {value_raw_path}")
                 except Exception as e:
                     logger.error(f"Failed to save combined raw value error summary: {e}")


                 # Calculate mean and std/sem across seeds (same logic as before)
                 value_grouping_cols = ['Dataset', 'Method', 'N', 'Timestep']
                 value_metric_cols = ['ValueError', 'ValueStd']

                 try:
                     value_agg_df = value_df.groupby(value_grouping_cols).agg({'ValueError': ['mean', 'std'], 'ValueStd': ['mean', 'std']}).reset_index()
                     value_agg_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in value_agg_df.columns.values]

                     # Calculate SEM
                     value_num_seeds = value_df.groupby(value_grouping_cols).size().reset_index(name='Count')
                     value_agg_df = pd.merge(value_agg_df, value_num_seeds, on=value_grouping_cols)
                     if 'ValueError_std' in value_agg_df.columns:
                          value_agg_df['ValueError_sem'] = value_agg_df['ValueError_std'] / np.sqrt(value_agg_df['Count'])

                     value_agg_path = metrics_dir / "all_runs_aggregated_value_summary.csv" # Save aggregated data
                     value_agg_df = value_agg_df.sort_values(by=['Dataset', 'Method', 'N', 'Timestep'])
                     value_agg_df.to_csv(value_agg_path, index=False)
                     logger.info(f"Saved aggregated value error summary (mean/std/sem across seeds) to {value_agg_path}")
                 except Exception as e:
                     logger.error(f"Error during value metrics aggregation: {e}")
                     value_agg_df = None # Ensure it's None if aggregation fails
         # else: (case handled by initial check)
         #     logger.info("No value error data generated.")
     else:
          logger.info("Value evaluation was disabled, skipping value metrics loading/aggregation.")


    #  # Create plots using loaded and aggregated data
    #  create_plots(summary_df, summary_agg_df, value_df, value_agg_df, config)


def main():
    config, args = parse_script_args() # Get args alongside config
    logger.info("--- Initial Configuration ---")
    logger.info("\n" + pprint.pformat(asdict(config)))
    logger.info("--- Script Arguments ---")
    logger.info("\n" + pprint.pformat(vars(args)))


    # Define the seeds to run
    seeds_to_run = [config.common.seed, config.common.seed + 1, config.common.seed + 2, config.common.seed + 3, config.common.seed + 4]
    # seeds_to_run = [config.common.seed]
    logger.info(f"Running experiments for seeds: {seeds_to_run}")

    datasets_to_run = [config.dataset.name] if config.dataset.name in config.dataset.supported_datasets else config.dataset.supported_datasets
    logger.info(f"Running experiments for datasets: {datasets_to_run}")

    # Determine enabled methods based on script arguments
    enabled_methods = []
    if args.do_mcts: enabled_methods.append("MCTS")
    if args.do_mcts_rollout: enabled_methods.append("MCTS-Rollout") # Changed name for clarity
    if args.do_guidance: enabled_methods.append("Guidance") # Standalone Guidance
    if args.do_smc: enabled_methods.append("SMC")
    if args.do_smc_guidance: enabled_methods.append("SMC-Guidance")
    if args.do_smc_guidance_tempering: enabled_methods.append("SMC-Guidance-Tempering")
    if args.do_smc_rollout: enabled_methods.append("SMC-Rollout")

    if not enabled_methods:
         logger.warning("No methods specified via command-line flags. Exiting.")
         return

    logger.info(f"Enabled methods: {enabled_methods}")

    # Outer loop for seeds
    for seed in seeds_to_run:
        logger.info(f"######### Starting Seed: {seed} #########")
        # Make a deep copy or re-parse args if process_toy_config modifies config inplace
        # Assuming process_toy_config works correctly when called multiple times with updated seed
        current_config_base = get_default_toy_config() # Start fresh to avoid side effects
        # Re-apply initial cmd line overrides (except seed)
        if args.dataset is not None: current_config_base.dataset.name = args.dataset
        if args.exp_name is not None: current_config_base.common.exp_name = args.exp_name
        # SEED IS SET HERE
        current_config_base.common.seed = seed
        if args.device is not None: current_config_base.common.device = args.device
        if args.no_progress_bar is not None: current_config_base.common.use_progress_bar = not args.no_progress_bar
        if args.hf_home is not None: current_config_base.common.hf_home = args.hf_home
        if args.compute_budget is not None: current_config_base.sampling.compute_budget = args.compute_budget
        if args.population_size is not None: current_config_base.sampling.population_size = args.population_size
        if args.potential_type is not None: current_config_base.fkd.potential_type = args.potential_type
        if args.lambda_param is not None:
            current_config_base.fkd.lambda_param = args.lambda_param
        if args.guidance_scale is not None:
            current_config_base.guidance_scale = args.guidance_scale
        if args.no_value_eval is not None: current_config_base.eval.evaluate_value = not args.no_value_eval
        if args.value_timesteps is not None: current_config_base.eval.value_eval_timesteps = args.value_timesteps

        for dataset_name in datasets_to_run:
            logger.info(f"===== Starting Dataset: {dataset_name} (Seed: {seed}) =====")
            # Process config for this specific dataset AND SEED run
            # Ensure paths/etc are updated based on the current seed and dataset
            current_config_base.dataset.name = dataset_name
            processed_config = process_toy_config(current_config_base) # Get processed config for this run
            logger.info(f"  Run directory: {processed_config.paths.run_dir}")

            # Set random seeds for this specific run
            random.seed(processed_config.common.seed)
            np.random.seed(processed_config.common.seed)
            torch.manual_seed(processed_config.common.seed)
            if processed_config.common.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.manual_seed_all(processed_config.common.seed)

            # Load resources ONCE per dataset/seed
            diffusion_model = load_diffusion_model(processed_config)
            dataset = get_2d_dataset(config, dataset_name, device=processed_config.common.device)
            # Generate GT samples ONCE per dataset/seed
            # gt_samples, _ = dataset.sample_product_distribution_diff(n_samples=processed_config.sampling.population_size)
            # gt_samples_np = gt_samples.cpu().numpy()
            gt_samples_np = dataset.gt_samples[:processed_config.sampling.population_size]
            gt_rewards_np = dataset.gt_rewards[:processed_config.sampling.population_size]

            # Store samples and rewards for the combined plot for this dataset/seed run
            all_method_samples_for_plot = {}
            all_method_rewards_for_plot = {}

            for method in enabled_methods:
                logger.info(f"<<< Running {method} for Dataset: {dataset_name} (Seed: {seed}) >>>")

                # Select Run Function based on method name
                run_fn = None
                if method.startswith("MCTS"):
                    run_fn = run_mcts
                elif method.startswith("SMC"):
                    run_fn = run_fkd
                elif method == "Guidance":
                    run_fn = run_fkd
                else:
                    logger.error(f"Unknown method prefix for {method}. Skipping.")
                    continue

                # Run experiment
                method_samples = None
                try:
                    sample_metrics_df, value_metrics_df, method_samples, method_rewards = run_fn(processed_config, dataset_name, diffusion_model, dataset, method_name=method)
                except Exception as e:
                     logger.error(f"!!! ERROR during run_fn for {method}/{dataset_name}/seed {seed} !!!")
                     logger.error(e, exc_info=True) # Log traceback
                     sample_metrics_df = pd.DataFrame() # Create empty DFs to avoid downstream errors
                     value_metrics_df = pd.DataFrame()
                     method_samples = {} # Empty dict if run failed

                # Store samples for combined plot
                if method_samples:
                    all_method_samples_for_plot[method] = method_samples
                if method_rewards:
                    all_method_rewards_for_plot[method] = method_rewards

                # Add Dataset and Seed columns
                if sample_metrics_df is not None and not sample_metrics_df.empty:
                    sample_metrics_df['Dataset'] = dataset_name
                    sample_metrics_df['Seed'] = seed
                if value_metrics_df is not None and not value_metrics_df.empty:
                    value_metrics_df['Dataset'] = dataset_name
                    value_metrics_df['Seed'] = seed

                # Save intermediate results for this method/dataset/seed run
                metrics_dir = Path(processed_config.paths.metrics_dir)
                metrics_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
                if sample_metrics_df is not None and not sample_metrics_df.empty:
                    sample_filename = metrics_dir / f"sample_metrics_{dataset_name}_{method}_seed{seed}.csv"
                    try:
                        sample_metrics_df.to_csv(sample_filename, index=False)
                        logger.debug(f"Saved intermediate sample metrics to {sample_filename}")
                    except Exception as e:
                        logger.error(f"Failed to save intermediate sample metrics to {sample_filename}: {e}")

                if value_metrics_df is not None and not value_metrics_df.empty:
                    value_filename = metrics_dir / f"value_metrics_{dataset_name}_{method}_seed{seed}.csv"
                    try:
                        value_metrics_df.to_csv(value_filename, index=False)
                        logger.debug(f"Saved intermediate value metrics to {value_filename}")
                    except Exception as e:
                        logger.error(f"Failed to save intermediate value metrics to {value_filename}: {e}")

                logger.info(f">>> Finished {method} for Dataset: {dataset_name} (Seed: {seed}) <<<")

            # # --- Create Combined Distribution Plot for this Dataset/Seed ---
            # create_all_methods_distribution_plot(
            #     all_samples_by_method=all_method_samples_for_plot,
            #     gt_samples=gt_samples_np,
            #     plot_dir=processed_config.paths.plot_dir,
            #     dataset_name=dataset_name,
            #     seed=seed
            # )

            # # Combine rewards into a single DataFrame for KDE plotting
            # kde_plot_data = []
            # for method, rewards_dict in all_method_rewards_for_plot.items():
            #     for nfe, rewards_array in rewards_dict.items():
            #         if rewards_array is not None and rewards_array.size > 0:
            #              for reward in rewards_array:
            #                  kde_plot_data.append({'Method': method, 'NFE': nfe, 'Reward': reward})
            # all_rewards_df = pd.DataFrame(kde_plot_data)

            # create_reward_kde_plots(
            #     all_rewards_df=all_rewards_df,
            #     gt_rewards=gt_rewards_np,
            #     plot_dir=processed_config.paths.plot_dir,
            #     dataset_name=dataset_name,
            #     seed=seed
            # )

            # --- SAVE DATA FOR LATER PLOTTING ---
            data_save_dir = Path(processed_config.paths.image_dir)
            data_save_dir.mkdir(parents=True, exist_ok=True) # Ensure data_save_dir exists

            # Save all method samples (dict: {method: {nfe: samples_array}})
            method_samples_save_path = data_save_dir / f"all_method_samples_{dataset_name}_seed{seed}.pkl"
            with open(method_samples_save_path, "wb") as f:
                pickle.dump(all_method_samples_for_plot, f)
            logger.info(f"Saved all method samples to {method_samples_save_path}")

            # Save all method rewards (dict: {method: {nfe: rewards_array}})
            method_rewards_save_path = data_save_dir / f"all_method_rewards_{dataset_name}_seed{seed}.pkl"
            with open(method_rewards_save_path, "wb") as f:
                pickle.dump(all_method_rewards_for_plot, f)
            logger.info(f"Saved all method rewards to {method_rewards_save_path}")

            samples = diffusion_model.sample_initial_noise(processed_config.sampling.population_size)
            for t in range(diffusion_model.n_timesteps-1, 0, -1):
                samples, _ = diffusion_model.p_sample(samples, t)
            samples_save_path = data_save_dir / f"prior_samples_{dataset_name}.pkl"
            with open(samples_save_path, "wb") as f:
                pickle.dump(samples, f)
            logger.info(f"Saved all samples to {samples_save_path}")

            # Clean up dataset-specific resources
            del diffusion_model, dataset, all_method_samples_for_plot, all_method_rewards_for_plot

            logger.info(f"===== Finished Dataset: {dataset_name} (Seed: {seed}) =====")

        # Clean up seed-specific config if necessary (or just let the loop overwrite)
        logger.info(f"######### Finished Seed: {seed} #########")

    # Aggregate and plot results by loading CSVs AFTER all seeds/methods/datasets are run
    # Process base config one last time to get the correct final paths for aggregation/plotting
    final_processed_config = process_toy_config(config)
    aggregate_and_save_summary(final_processed_config) # Pass the final processed config

    logger.info(f"=== Experiment Complete. Base directory: {final_processed_config.common.base_dir} ===")


if __name__ == '__main__':
    main() 
