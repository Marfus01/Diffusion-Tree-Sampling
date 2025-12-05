import argparse
import tqdm
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision.utils import save_image
import math
import logging
import colorlog
import pprint
from dataclasses import asdict
import multiprocessing
import copy
import sys
import time

from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.utils import logging as dlogging
from diffusers.utils import make_image_grid

from src.core.sampling_sd.mcts_sd import MCTSStableDiffusionSampler
from src.core.sampling_sd.fkd_sd import FKDStableDiffusion
from src.core.rewards import get_reward_function, do_clip_score_diversity
from src.core.config import get_default_sd_config, process_sd_config

# Setup logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)-2s [%(asctime)s]%(reset)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    }
))

root_logger = logging.getLogger()
for h in root_logger.handlers[:]:
    root_logger.removeHandler(h)

root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
dlogging.set_verbosity_error()


def parse_script_args():
    """Parses command-line overrides for SDConfig."""
    parser = argparse.ArgumentParser(description='Run guided Stable Diffusion experiments', add_help=True)

    # --- Common Overrides ---
    cgroup = parser.add_argument_group('Common Overrides')
    cgroup.add_argument('--prompts', nargs='+', help='List of prompts to use for generation.')
    cgroup.add_argument('--prompts_file', type=str, help='JSON file containing prompts.')
    cgroup.add_argument('--negative_prompt', type=str, help='Negative prompt override.')
    cgroup.add_argument('--sd_model_id', type=str, help='Stable Diffusion model ID override.')
    cgroup.add_argument('--exp_name', type=str, help='Override experiment name prefix.')
    cgroup.add_argument('--seed', type=int, help='Override random seed.')
    cgroup.add_argument('--device', type=str, help='Override device.')
    cgroup.add_argument('--no_progress_bar', action='store_true', help='Override whether to use progress bar.')    
    cgroup.add_argument('--hf_home', type=str, help='Override HuggingFace cache directory (HF_HOME).')

    # --- Method Enabling ---
    mgroup = parser.add_argument_group('Method Selection')
    mgroup.add_argument('--do_mcts', action='store_true', help='Enable MCTS (standard). Note: Excludes rollout version.')
    mgroup.add_argument('--do_mcts_rollout', action='store_true', help='Enable MCTS with Rollout.')
    mgroup.add_argument('--do_fkd', action='store_true', help='Enable FKD (standard). Note: Excludes rollout version.')
    mgroup.add_argument('--do_bon', action='store_true', help='Enable Best-of-N (rejection sampling).')

    # --- Sampling Overrides ---
    sgroup = parser.add_argument_group('Sampling Overrides')
    sgroup.add_argument('--compute_budget', nargs='+', type=int, help='Override compute budget in NFEs.')
    sgroup.add_argument('--population_size', type=int, help='Number of final # samples per prompt.')
    sgroup.add_argument('--potential_type', type=str, help='Override potential type for FKD.')
    sgroup.add_argument('--lambda_param', type=float, help='Override lambda parameter for all methods.')
    sgroup.add_argument('--guiding_reward_fns', nargs='+', help='Override guiding reward function.')

    args = parser.parse_args()
    config = get_default_sd_config()

    if args.prompts is not None and args.prompts_file is not None:
        raise ValueError("Only one of --prompts or --prompts_file can be specified")
    if args.prompts is not None: config.dataset.prompts = args.prompts
    elif args.prompts_file is not None: config.dataset.prompts_file = args.prompts_file
    else: raise ValueError("Either --prompts or --prompts_file must be specified")
    if args.negative_prompt is not None: config.dataset.negative_prompt = args.negative_prompt
    if args.sd_model_id is not None: config.model.sd_model_id = args.sd_model_id

    if args.exp_name is not None: config.common.exp_name = args.exp_name
    if args.seed is not None: config.common.seed = args.seed
    if args.device is not None: config.common.device = args.device
    if args.no_progress_bar is not None: config.common.use_progress_bar = not args.no_progress_bar
    if args.hf_home is not None: config.common.hf_home = args.hf_home

    # Enable methods based on specific flags
    if args.do_mcts: config.mcts.enabled = True
    if args.do_mcts_rollout: config.mcts_rollout = True # Using a temporary attribute
    if args.do_fkd: config.fkd.enabled = True
    if args.do_bon: config.bon.enabled = True


    if args.compute_budget is not None:
        config.sampling.compute_budget = args.compute_budget
    if args.population_size is not None: config.sampling.population_size = args.population_size
    if args.potential_type is not None: config.fkd.potential_type = args.potential_type
    if args.lambda_param is not None:
        config.fkd.lambda_param = args.lambda_param
    if args.guiding_reward_fns is not None: config.guiding_reward_fns = args.guiding_reward_fns

    if 'AestheticScore' in config.guiding_reward_fns:
        config.mcts.exploration_const = 40.0
    else:
        config.mcts.exploration_const = 0.1

    config = process_sd_config(config)

    return config


@torch.no_grad()
def evaluate_images(images_pil, prompt, config):
    """Evaluate a set of generated PIL images on configured reward functions."""
    results = {}
    num_images = len(images_pil)
    eval_batch_size = config.eval.eval_batch_size

    # Reward functions
    for reward_name in config.eval.reward_fns:
        all_rewards = []
        # Process in batches for reward models
        for i in range(0, num_images, eval_batch_size):
            batch_images = images_pil[i:min(i + eval_batch_size, num_images)]
            batch_prompts = [prompt] * len(batch_images)
            rewards = get_reward_function(
                reward_name=reward_name,
                images=batch_images,
                prompts=batch_prompts,
            )
            all_rewards.extend(rewards)

        results[f'{reward_name}_Mean'] = np.mean(all_rewards) if all_rewards else 0.0
        results[f'{reward_name}_Std'] = np.std(all_rewards) if all_rewards else 0.0
        results[f'{reward_name}_Max'] = np.max(all_rewards) if all_rewards else 0.0
        results[f'{reward_name}_Min'] = np.min(all_rewards) if all_rewards else 0.0
        results[f'{reward_name}_Samples'] = all_rewards

    # CLIP score and diversity
    _, clip_diversity = do_clip_score_diversity(
        images=images_pil,
        prompts=[prompt] * num_images,
    )
    results['CLIP_Diversity'] = clip_diversity

    return results


def save_results_to_csv(metrics_data, prompt, output_path, method_name, config, save=False):
    """
    Save metrics to CSV for a specific method, prompt, and iteration/N.
    metrics_data is a dict: {target_nfe: {metric_name: float_or_list}}
    """
    rows = []
    # The key is now the target NFE
    for nfe_target, metrics in metrics_data.items():
        sample_metrics = {k.replace('_Samples', ''): v for k, v in metrics.items() if k.endswith('_Samples') and isinstance(v, list)}
        num_samples = len(next(iter(sample_metrics.values()))) if sample_metrics else 1

        for i in range(num_samples):
            row_base = {
                'Method': method_name,
                'Prompt': prompt,
                'N': nfe_target,
                'NFEs': metrics.get('NFEs', nfe_target),
                'Population_Size': config.sampling.population_size,
                'Sample_Index': i
            }
            for k, v in metrics.items():
                if not k.endswith('_Samples'):
                    row_base[k] = v
            for k, v_list in sample_metrics.items():
                row_base[k] = v_list[i] if i < len(v_list) else float('nan')
            rows.append(row_base)

    if not rows:
        logger.warning(f"No data rows generated for method {method_name}.")
        return None

    df = pd.DataFrame(rows)
    cols_order = ['Method', 'Prompt', 'N', 'NFEs', 'Population_Size', 'Sample_Index']
    reward_cols = []
    for r in config.eval.reward_fns:
        for suffix in ['_Mean', '_Std', '_Max', '_Min', '']:
             col = f"{r}{suffix}"
             if col in df.columns: reward_cols.append(col)
    cols_order += reward_cols + ['CLIP_Diversity']

    extra_cols = [col for col in df.columns if col not in cols_order]
    final_cols = [c for c in cols_order if c in df.columns] + extra_cols
    df = df[final_cols]
    if save:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved metrics for {method_name} to {output_path}")
    return df


@torch.no_grad()
def run_mcts(config, prompt, reward_name, pipeline):
    """Run MCTS-guided Stable Diffusion generation."""
    logger.info(f"--- Running MCTS for prompt: '{prompt}' ---")
    # Setup memory logging
    method_name = "MCTS-R" if config.mcts.use_rollout else "MCTS"
    log_file_prefix = Path(config.paths.metrics_dir) / f"gpu_memory_{method_name}.txt"
    log_file_prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics_data = {}

    reward_fn_mcts = lambda images_pil: get_reward_function(
        reward_name=reward_name,
        images=images_pil,
        prompts=[prompt] * len(images_pil),
    )

    # Create output directory for MCTS images
    dir_name = "mcts-r" if config.mcts.use_rollout else "mcts"
    output_dir_mcts_images = Path(config.paths.image_dir) / reward_name / dir_name
    output_dir_mcts_images.mkdir(parents=True, exist_ok=True)

    mcts_sampler = MCTSStableDiffusionSampler(
        pipeline=pipeline,
        prompt=prompt,
        reward_fn=reward_fn_mcts,
        expand_timesteps=config.mcts.expand_timesteps,
        num_inference_steps=config.sampling.num_inference_steps,
        eta=config.sampling.eta,
        num_nfes=max(config.sampling.compute_budget),
        device=config.common.device,
        guidance_scale=config.sampling.guidance_scale,
        exploration_const=config.mcts.exploration_const,
        progressive_width_const=config.mcts.progressive_width_const,
        progressive_width_alpha=config.mcts.progressive_width_alpha,
        use_progress_bar=config.common.use_progress_bar,
        use_rollout=config.mcts.use_rollout,
        maximize=False,
        exploration_type='ucb',
        batch_size=config.mcts.batch_size,
        output_dir=str(output_dir_mcts_images),
        visualization_timesteps=list(config.mcts.expand_timesteps + [0]),
    )

    mcts_sampler.initialize_root_nodes()

    # Run MCTS progressively for each target iteration count
    for target_nfes in sorted(config.sampling.compute_budget):
        # Reset memory stats for this NFE iteration
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        # logger.info(f"MCTS NFE {target_nfes}")

        # Timing the sampling phase
        start_time = time.time()

        # Build search tree
        mcts_sampler.build_search_tree(num_nfes=target_nfes)

        # Sample images
        sampled_images, _ = mcts_sampler.sample_from_tree(
            population_size=config.sampling.population_size,
        )

        # Log sampling time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start_time
        with open(log_file_prefix, 'a') as f:
            f.write(f"{method_name} NFE {target_nfes}: sampling time {elapsed:.4f}s\n")

        # Log memory usage after this NFE iteration
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            curr = torch.cuda.memory_allocated()
            peak = torch.cuda.max_memory_allocated()
            with open(log_file_prefix, 'a') as f:
                f.write(f"{method_name} NFE {target_nfes}: current {curr/1024**2:.2f}MB, peak {peak/1024**2:.2f}MB\n")

        # Evaluate metrics
        metrics = evaluate_images(sampled_images, prompt, config)
        metrics['NFEs'] = target_nfes
        metrics_data[target_nfes] = metrics

        # Save image grid
        grid_path = output_dir_mcts_images / f"{prompt.replace(' ', '_')}_{target_nfes}.png"
        rows = int(math.ceil(np.sqrt(len(sampled_images))))
        cols = int(math.ceil(len(sampled_images) / rows))
        image_grid = make_image_grid(sampled_images, rows=rows, cols=cols)
        image_grid.save(grid_path)

    return metrics_data


@torch.no_grad()
def run_bon(config, prompt, base_pipeline):
    """Run Best-of-N generation."""
    metrics_data = {}  # {n_candidates: metrics_dict}
    all_base_samples_pil = {}  # {n_candidates: list_of_pil_images}
    all_base_samples_pt = {}   # {n_candidates: tensor_of_images}
    generated_samples_pt = [] # Store all generated tensors incrementally

    if base_pipeline is None:
        logger.error("Base pipeline is None, cannot run Best-of-N.")
        return metrics_data, all_base_samples_pil, all_base_samples_pt

    # Use candidate counts from FKD config for N values
    n_candidate_list = sorted(config.fkd.particles)
    max_n_candidates = max(n_candidate_list)
    # Use a reasonable batch size for generation
    gen_batch_size = config.eval.eval_batch_size # Reuse eval batch size

    logger.info(f"Generating up to {max_n_candidates} base samples...")
    for i in range(0, max_n_candidates, gen_batch_size):
        current_batch_size = min(gen_batch_size, max_n_candidates - i)
        logger.debug(f"Generating base batch {i // gen_batch_size + 1}/{(max_n_candidates + gen_batch_size - 1) // gen_batch_size} ({current_batch_size} samples)")
        try:
            output = base_pipeline(
                prompt=[prompt] * current_batch_size,
                negative_prompt=[config.dataset.negative_prompt] * current_batch_size,
                num_inference_steps=config.sampling.num_inference_steps,
                guidance_scale=config.sampling.guidance_scale,
                eta=config.sampling.eta,
                output_type="pt",
            ).images
            generated_samples_pt.append(output.to(config.common.device))
        except Exception as e:
            logger.error(f"Error during base sample generation batch: {e}")
            # Should we continue or stop? Stop for now.
            return metrics_data, all_base_samples_pil, all_base_samples_pt

    if not generated_samples_pt:
         logger.error("No base samples were generated.")
         return metrics_data, all_base_samples_pil, all_base_samples_pt

    all_generated_pt = torch.cat(generated_samples_pt, dim=0)
    logger.info(f"Generated {all_generated_pt.shape[0]} base samples total.")

    # Evaluate for each N in the candidate list
    for n_candidates in n_candidate_list:
        logger.info(f"Evaluating Best-of-N for N = {n_candidates}")
        if n_candidates > all_generated_pt.shape[0]:
            logger.warning(f"Requested N={n_candidates} but only {all_generated_pt.shape[0]} samples generated. Using all generated.")
            n_candidates = all_generated_pt.shape[0]
        if n_candidates == 0:
             continue

        current_samples_pt = all_generated_pt[:n_candidates]
        # Convert needed tensors to PIL for evaluation
        current_samples_pil = base_pipeline.image_processor.postprocess(
                current_samples_pt, output_type="pil", do_denormalize=[True] * n_candidates # Diffusers standard is denormalize
        )
        # Store for potential reuse
        all_base_samples_pil[n_candidates] = current_samples_pil
        all_base_samples_pt[n_candidates] = current_samples_pt

        # NFE calculation for base generation
        nfe = config.sampling.num_inference_steps * n_candidates

        # Select final population for evaluation/saving grid
        if config.sampling.population_size < n_candidates:
            # Uniform random sampling from the N candidates
            indices = torch.randperm(n_candidates)[:config.sampling.population_size]
            eval_samples_pil = [current_samples_pil[i] for i in indices]
            save_samples_pt = current_samples_pt[indices]
        else:
            eval_samples_pil = current_samples_pil
            save_samples_pt = current_samples_pt

        # Evaluate metrics on the selected population
        metrics = evaluate_images(eval_samples_pil, prompt, config)
        metrics['NFEs'] = nfe # Assign NFE for generating N candidates
        metrics_data[n_candidates] = metrics

        # Save image grid of the evaluated population
        grid_dir = Path(config.paths.image_dir) / "bon"
        grid_dir.mkdir(parents=True, exist_ok=True)
        grid_path = grid_dir / f"bon_N{n_candidates}_{prompt[:30].replace(' ', '_')}.png"
        try:
            save_samples_pt_norm = (save_samples_pt / 2 + 0.5).clamp(0, 1)
            save_image(save_samples_pt_norm.cpu(), grid_path, nrow=int(math.ceil(np.sqrt(len(eval_samples_pil)))))
            logger.debug(f"Saved BON image grid to {grid_path}")
        except Exception as e:
            logger.error(f"Failed to save BON image grid: {e}")

    return metrics_data, all_base_samples_pil, all_base_samples_pt


@torch.no_grad()
def run_fkd(config, prompt, reward_name, pipeline):
    """Run FKD-guided Stable Diffusion generation."""
    metrics_data = {}
    # Setup memory logging
    method_name = "FKD-R" if config.fkd.use_rollout else "FKD"
    log_file_prefix = Path(config.paths.metrics_dir) / f"gpu_memory_{method_name}.txt"
    log_file_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Create output directory for FKD images
    dir_name = "fkd-r" if config.fkd.use_rollout else "fkd"
    output_dir_fkd_images = Path(config.paths.image_dir) / reward_name / dir_name
    output_dir_fkd_images.mkdir(parents=True, exist_ok=True)

    reward_fn_fkd = lambda images_pil: get_reward_function(
        reward_name=reward_name,
        images=images_pil,
        prompts=[prompt] * len(images_pil),
    )

    fkd_args = {
        "use_smc": True,
        "potential_type": config.fkd.potential_type,
        "lmbda": config.fkd.lambda_param,
        "adaptive_resampling": config.fkd.adaptive_resampling,
        "resample_frequency": config.fkd.resample_frequency,
        "resampling_t_start": config.fkd.resampling_t_start,
        "resampling_t_end": config.fkd.resampling_t_end,
        "time_steps": config.sampling.num_inference_steps,
        "guidance_reward_fn": reward_name,
        "reward_min_value": config.fkd.reward_min_val,
        "device": config.common.device,
        "use_rollout": config.fkd.use_rollout,
        "batch_size": config.fkd.batch_size,
    }

    # Determine max particle count and run FKD once
    if config.fkd.use_rollout:
        nfe_per_particle = config.sampling.num_inference_steps      # for main denoising loop
        for t in range(config.fkd.resampling_t_start, config.fkd.resampling_t_end + 1, config.fkd.resample_frequency):
            nfe_per_particle += config.sampling.num_inference_steps - t     # for rollouts
    else:
        nfe_per_particle = config.sampling.num_inference_steps
    eval_particles =  sorted(list(set([max(1, nfe // nfe_per_particle) for nfe in config.sampling.compute_budget])))
    max_n_particles = max(eval_particles)

    # # Process in batches to handle large particle counts
    # all_sampled_images = []
    # for batch_start in tqdm.tqdm(range(0, max_n_particles, config.fkd.batch_size), 
    #                       desc="Generating FKD samples",
    #                       disable=not config.common.use_progress_bar):
    #     batch_size = min(config.fkd.batch_size, max_n_particles - batch_start)
    #     # logger.info(f"Processing generation batch {batch_start//config.fkd.batch_size + 1}/{(max_n_particles + config.fkd.batch_size - 1)//config.fkd.batch_size} " +
    #     #       f"(particles {batch_start+1}-{batch_start+batch_size}/{max_n_particles})")
        
    #     # Run generation with FKD steering
    #     batch_fkd_args = fkd_args.copy()
    #     batch_fkd_args["num_particles"] = batch_size

    #     output = pipeline(
    #         prompt=[prompt] * batch_size,
    #         negative_prompt=[config.dataset.negative_prompt] * batch_size,
    #         fkd_args=batch_fkd_args,
    #         num_inference_steps=config.sampling.num_inference_steps,
    #         guidance_scale=config.sampling.guidance_scale,
    #         eta=config.sampling.eta,
    #         output_type="pil",
    #     ).images

    #     all_sampled_images.extend(output)
    
    # Evaluate for each target NFE level by subsetting
    for i, n_particles in enumerate(eval_particles):
        # Reset memory stats for this particle count
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        logger.info(f"FKD Particle Count {n_particles}")

        # Timing the sampling phase
        start_time = time.time()

        # Call the new sample_batched_smc method
        current_samples = pipeline.sample_batched_smc(
            num_particles=n_particles,
            prompt=[prompt] * n_particles,
            processing_batch_size=config.fkd.batch_size,
            negative_prompt=[config.dataset.negative_prompt] * n_particles,
            fkd_args=fkd_args,
            num_inference_steps=config.sampling.num_inference_steps,
            guidance_scale=config.sampling.guidance_scale,
            eta=config.sampling.eta,
            output_type="pil",
        ).images

        # Log sampling time for FKD
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start_time
        with open(log_file_prefix, 'a') as f:
            f.write(f"{method_name} Particles {n_particles}: sampling time {elapsed:.4f}s\n")

        # Log memory usage after this particle count
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            curr = torch.cuda.memory_allocated()
            peak = torch.cuda.max_memory_allocated()
            with open(log_file_prefix, 'a') as f:
                f.write(f"{method_name} Particles {n_particles}: current {curr/1024**2:.2f}MB, peak {peak/1024**2:.2f}MB\n")


        logger.info(f"Evaluating FKD for N = {n_particles} particles")

        # if config.sampling.population_size < len(current_samples):
        #     indices = np.random.choice(len(current_samples), size=config.sampling.population_size)
        #     eval_images = [current_samples[i] for i in indices]
        # else:
        #     eval_images = current_samples
        num_to_resample = min(config.sampling.population_size, len(current_samples))
        # Get indices of top num_to_resample samples based on rewards
        rewards = np.array(reward_fn_fkd(current_samples))
        top_indices = np.argsort(rewards)[-num_to_resample:]
        eval_images = [current_samples[i] for i in top_indices]
            
        # Evaluate metrics
        metrics = evaluate_images(eval_images, prompt, config)
        metrics['NFEs'] = config.sampling.compute_budget[i]
        metrics_data[config.sampling.compute_budget[i]] = metrics

        # Save image grid
        grid_path = output_dir_fkd_images / f"{prompt.replace(' ', '_')}_{config.sampling.compute_budget[i]}.png"
        rows = int(math.ceil(np.sqrt(len(eval_images))))
        cols = int(math.ceil(len(eval_images) / rows))
        image_grid = make_image_grid(eval_images, rows=rows, cols=cols)
        image_grid.save(grid_path)

    if not config.common.use_progress_bar:
         dlogging.set_verbosity_warning()

    return metrics_data

def aggregate_and_save_summary_sd(all_run_metrics_dfs, reward_name, output_dir):
    """
    Aggregates metrics from multiple DataFrames (one per method/prompt/N run) and saves summary CSVs.
    all_run_metrics_dfs: Dict[method_name, List[pd.DataFrame]]
    """
    logger.info(f"Aggregating SD results across prompts for {reward_name}")
    output_dir = Path(output_dir)

    for method, df_list in all_run_metrics_dfs.items():
        if not df_list:
            # logger.warning(f"No dataframes found for method {method}, skipping aggregation.")
            continue

        summary_df = pd.concat(df_list, ignore_index=True)
        avg_cols = [col for col in summary_df.columns if '_Mean' in col or col in ['CLIP_Diversity']]
        avg_cols += [col for col in summary_df.columns if '_Max' in col]
        avg_cols += [col for col in summary_df.columns if '_Min' in col]
        avg_cols += ['NFEs']

        valid_avg_cols = []
        for col in avg_cols:
            if col in summary_df.columns:
                summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')
                if not summary_df[col].isnull().all(): # Only include if not all NaN
                     valid_avg_cols.append(col)
            else:
                 logger.warning(f"Summary column {col} not found in {method} results, cannot average.")

        # Group by Iteration/Particles
        group_col = 'N'
        summary_agg = summary_df.groupby(group_col)[valid_avg_cols].mean().reset_index()
        if not summary_agg.empty:
            summary_agg_path = output_dir / f"{method}_{reward_name}_summary_avg_across_prompts.csv"
            summary_agg.to_csv(summary_agg_path, index=False)
            logger.info(f"Saved aggregated metrics for {method} (averaged across prompts) to {summary_agg_path}")
        else:
            logger.warning(f"Aggregation resulted in empty dataframe for method {method}.")


def load_pipeline(config, method):
    if method == 'FKD' or method == 'FKD-R':
        pipeline = FKDStableDiffusion.from_pretrained(config.model.sd_model_id)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(config.model.sd_model_id)
    
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(
        num_inference_steps=config.sampling.num_inference_steps, device=config.common.device
    )
    pipeline.to(config.common.device)
    pipeline.set_progress_bar_config(disable=True)
    if not config.common.use_progress_bar:
        dlogging.set_verbosity_error()

    logger.info(f"[{config.common.device}] Attempting to compile UNet...")
    sys.stdout.flush()
    try:
        pipeline.unet = torch.compile(
            pipeline.unet,
            backend="tensorrt",
            options={
                "truncate_long_and_double": True,
                "enabled_precisions": {torch.float32, torch.float16},
                # "triton.cudagraphs": True,
                # "max_autotune": True,
            },
            dynamic=False,
        )
    except Exception as e:
        logger.error(f"[{config.common.device}] Failed to compile UNet: {e}", exc_info=True)
    sys.stdout.flush()

    logger.info(f"[{config.common.device}] Attempting to compile VAE...")
    sys.stdout.flush()
    try:
        pipeline.vae = torch.compile(
            pipeline.vae,
            backend="tensorrt",
            options={
                "truncate_long_and_double": True,
                "enabled_precisions": {torch.float32, torch.float16},
                # "triton.cudagraphs": True,
                # "max_autotune": True,
            },
            dynamic=False,
        )
    except Exception as e:
        logger.error(f"[{config.common.device}] Failed to compile VAE: {e}", exc_info=True)
    sys.stdout.flush()

    return pipeline


def worker_process(gpu_id, config, prompt_subset):
    """Worker function to process a subset of prompts on a specific GPU."""
    # Set device for this process
    device = f'cuda:{gpu_id}'
    try:
        torch.cuda.set_device(device)
        config.common.device = device # Update config for this process
        logger.info(f"[GPU {gpu_id}] Set device to {device}")
    except Exception as e:
        logger.error(f"[GPU {gpu_id}] Failed to set device {device}: {e}")
        return {} # Return empty results if device setup fails

    logger.info(f"[GPU {gpu_id}] Starting worker for {len(prompt_subset)} prompts: {prompt_subset}")

    # Initialize results storage for this worker
    worker_all_run_metrics_dfs = {} # MODIFIED: Initialize as empty dict

    # Make a copy of the original config to modify within the worker for each method
    original_config_for_worker = copy.deepcopy(config)

    enabled_methods = []
    if config.mcts.enabled: enabled_methods.append("MCTS")
    if hasattr(config, 'mcts_rollout') and config.mcts_rollout: enabled_methods.append("MCTS-R")
    if config.fkd.enabled: enabled_methods.append("FKD")
    if hasattr(config, 'fkd_rollout') and config.fkd_rollout: enabled_methods.append("FKD-R")
    if config.bon.enabled: enabled_methods.append("BoN")

    for method in enabled_methods:
        # Reset GPU memory stats before running each method
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        # Restore and modify config for the current method
        current_config = copy.deepcopy(original_config_for_worker)
        logger.info(f"[GPU {gpu_id}] Method: {method}")

        # Set rollout flags based on method name
        current_config.mcts.use_rollout = False
        current_config.fkd.use_rollout = False

        if method == "MCTS":
            pass
        elif method == "MCTS-R":
            current_config.mcts.use_rollout = True
            current_config.mcts.enabled = True
        elif method == "FKD":
            pass
        elif method == "FKD-R":
            current_config.fkd.use_rollout = True
            current_config.fkd.enabled = True
        pipeline = None
        try:
            pipeline = load_pipeline(current_config, method)
            logger.info(f"[GPU {gpu_id}] Loaded pipeline for {method} on {current_config.common.device}")
        except Exception as e:
            logger.error(f"[GPU {gpu_id}] Failed to load pipeline for method {method}: {e}", exc_info=True)
            continue

        run_fn = None
        if method == "MCTS" or method == "MCTS-R":
            run_fn = run_mcts
        elif method == "FKD" or method == "FKD-R":
            run_fn = run_fkd
        elif method == 'BoN':
            run_fn = run_bon
        else:
            logger.warning(f"[GPU {gpu_id}] Unknown method {method}, skipping.")
            continue

        # for reward_name in current_config.eval.reward_fns:
        for reward_name in current_config.guiding_reward_fns:
            logger.info(f"[GPU {gpu_id}] Guiding with Reward: {reward_name} for {method}")
            method_guidance_id = f"{method}_guided_by_{reward_name}" # NEW: Create unique ID
            prompt_results_dfs_for_reward = [] # Store DFs for this specific method, reward, and its prompts
            for prompt in prompt_subset:
                logger.info(f"[GPU {gpu_id}] Prompt: '{prompt}'")
                try:
                    results_dict = run_fn(current_config, prompt, reward_name, pipeline)
                    df = save_results_to_csv(results_dict, prompt, None, method_guidance_id, current_config, save=False)
                    if df is not None:
                        prompt_results_dfs_for_reward.append(df)
                        logger.debug(f"[GPU {gpu_id}] Collected results for {method_guidance_id}, '{prompt[:20]}...'" )
                    else:
                         logger.warning(f"[GPU {gpu_id}] No dataframe generated for {method_guidance_id}, '{prompt[:20]}...'" )
                except Exception as e:
                    logger.error(f"[GPU {gpu_id}] Error running {method_guidance_id} for prompt '{prompt}': {e}", exc_info=True)
                finally:
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
            
            # After iterating through prompts for this method/reward combination:
            if prompt_results_dfs_for_reward:
                # Store for returning to main process
                if method_guidance_id not in worker_all_run_metrics_dfs:
                    worker_all_run_metrics_dfs[method_guidance_id] = []
                worker_all_run_metrics_dfs[method_guidance_id].extend(prompt_results_dfs_for_reward)

                # Intermediate save for this method_guidance_id on this worker
                try:
                    method_reward_df = pd.concat(prompt_results_dfs_for_reward, ignore_index=True)
                    intermediate_dir = Path(current_config.paths.metrics_dir) / "intermediate_per_method_reward"
                    intermediate_dir.mkdir(parents=True, exist_ok=True)
                    output_path_csv = intermediate_dir / f"{method_guidance_id}_gpu{gpu_id}_prompts.csv"
                    method_reward_df.to_csv(output_path_csv, index=False)
                    logger.info(f"[GPU {gpu_id}] Saved intermediate results for {method_guidance_id} to {output_path_csv}")
                except Exception as e:
                    logger.error(f"[GPU {gpu_id}] Failed to save intermediate CSV for {method_guidance_id}: {e}", exc_info=True)

        # Log GPU memory usage after method completion
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_alloc = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            peak_alloc_mb = peak_alloc / 1024**2
            peak_reserved_mb = peak_reserved / 1024**2
            log_file = Path(current_config.paths.metrics_dir) / f'gpu_memory_gpu{gpu_id}.txt'
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, 'a') as f:
                f.write(f'Method {method}: Peak Allocated {peak_alloc_mb:.2f}MB, Peak Reserved {peak_reserved_mb:.2f}MB\n')
        del pipeline
        pipeline = None
        logger.info(f"[GPU {gpu_id}] Cleaned up pipeline for {method}")
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    logger.info(f"[GPU {gpu_id}] Worker finished.")
    return worker_all_run_metrics_dfs


def main():
    config = parse_script_args()

    logger.info("--- Final Configuration ---")
    logger.info("\n" + pprint.pformat(asdict(config)))

    # Check for GPUs
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Cannot distribute across GPUs. Exiting.")
        return
    num_gpus = torch.cuda.device_count()
    logger.info(f"Found {num_gpus} GPUs.")

    if not config.dataset.prompts:
         logger.error("No prompts provided or loaded. Exiting.")
         return

    # Distribute prompts
    prompts = config.dataset.prompts
    if not prompts:
        logger.error("Prompt list is empty after loading. Exiting.")
        return

    prompts_per_gpu = [[] for _ in range(num_gpus)]
    for i, prompt in enumerate(prompts):
        prompts_per_gpu[i % num_gpus].append(prompt)

    logger.info("Distributing prompts across GPUs:")
    for i, p_list in enumerate(prompts_per_gpu):
        logger.info(f"  GPU {i}: {len(p_list)} prompts {p_list if len(p_list) < 5 else '[...]'}") # Log subset for clarity

    # Prepare arguments for each process, ensuring config is deepcopied
    pool_args = []
    for gpu_id, prompt_subset in enumerate(prompts_per_gpu):
        if prompt_subset: # Only create args if there are prompts for this GPU
            pool_args.append((gpu_id, copy.deepcopy(config), prompt_subset))
        else:
            logger.warning(f"GPU {gpu_id} has no prompts assigned, skipping process creation.")

    if not pool_args:
        logger.error("No processes to launch (check prompt list and GPU count). Exiting.")
        return

    # Set start method for multiprocessing with CUDA
    try:
        multiprocessing.set_start_method('spawn', force=True)
        logger.info("Set multiprocessing start method to 'spawn'.")
    except RuntimeError as e:
        logger.warning(f"Could not set multiprocessing start method ('spawn'): {e}. Might be already set.")
    except AttributeError:
         logger.warning("Cannot set start_method, multiprocessing might not behave correctly with CUDA.")

    # Run processes in parallel using a Pool
    logger.info(f"Launching {len(pool_args)} worker processes...")
    results_list = []
    try:
        with multiprocessing.Pool(processes=len(pool_args)) as pool:
            # Use starmap to pass multiple arguments
            results_list = pool.starmap(worker_process, pool_args)
        logger.info("All worker processes finished.")
    except Exception as e:
        logger.error(f"Multiprocessing Pool failed: {e}", exc_info=True)
        # Optionally attempt cleanup or just exit
        return

    # Aggregate results from all workers
    logger.info("Aggregating results from workers...")
    final_all_run_metrics_dfs = {} # MODIFIED: Initialize as empty dict
    for worker_results in results_list:
        if isinstance(worker_results, dict): # Check if worker returned a dict
            for method_guidance_id_key, dfs_from_worker in worker_results.items(): # MODIFIED: Iterate over new keys
                if method_guidance_id_key not in final_all_run_metrics_dfs:
                    final_all_run_metrics_dfs[method_guidance_id_key] = []
                final_all_run_metrics_dfs[method_guidance_id_key].extend(dfs_from_worker)
        else:
            logger.warning(f"Received unexpected result type from worker: {type(worker_results)}")


    total_dfs = sum(len(dfs_list) for dfs_list in final_all_run_metrics_dfs.values()) # MODIFIED sum over values
    if total_dfs == 0:
        logger.warning("No metric dataframes were collected from any workers. Skipping final aggregation.")
    else:
        logger.info(f"Collected {total_dfs} total dataframes across all method-guidance combinations and workers.")
        # Final summary saving
        for current_guiding_reward in config.eval.reward_fns: # This is the reward we want a summary for
             logger.info(f"Aggregating and saving summary for guidance with reward: {current_guiding_reward}")
             
             dfs_for_this_guidance = {} # Dict[base_method_name, List[pd.DataFrame]]
             for method_guidance_id_key, all_dfs_for_id in final_all_run_metrics_dfs.items():
                 expected_suffix = f"_guided_by_{current_guiding_reward}"
                 if method_guidance_id_key.endswith(expected_suffix):
                     base_method_name = method_guidance_id_key[:-len(expected_suffix)]
                     # all_dfs_for_id contains all DFs for base_method_name when guided by current_guiding_reward
                     dfs_for_this_guidance[base_method_name] = all_dfs_for_id
             
             if not dfs_for_this_guidance:
                 logger.warning(f"No data found for any method guided by {current_guiding_reward}. Skipping summary.")
                 continue
                 
             try:
                # Pass the filtered DFs and the guiding reward name for correct processing and filename
                aggregate_and_save_summary_sd(dfs_for_this_guidance, current_guiding_reward, config.paths.metrics_dir)
             except Exception as e:
                logger.error(f"Failed to aggregate/save summary for guidance reward {current_guiding_reward}: {e}", exc_info=True)


    logger.info(f"=== SD Experiment Complete. Results saved in {config.paths.run_dir} ===")


if __name__ == '__main__':
    # Ensure the start method is set before any processes are created if possible
    # Note: The set_start_method call is now inside main() before Pool creation
    main() 
