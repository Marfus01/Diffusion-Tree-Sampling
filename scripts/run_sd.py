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
import os
import sys
import time

from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.utils import logging as dlogging
from diffusers.utils import make_image_grid

from src.core.sampling_sd.mcts_sd import MCTSStableDiffusionSampler
from src.core.sampling_sd.fkd_sd import FKDStableDiffusion
from src.core.rewards import get_reward_function, do_clip_score_diversity
from src.core.config import get_default_sd_config, process_sd_config, seed_experiment
from src.utils.run_logging import RewardCounter, write_run_meta, reset_peak_memory, peak_memory_mb

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
    cgroup.add_argument('--prompt_ids', nargs='+', type=int,
                        help='Run only these indices of the prompt list; the per-prompt seed stays seed + index.')
    cgroup.add_argument('--negative_prompt', type=str, help='Negative prompt override.')
    cgroup.add_argument('--sd_model_id', type=str, help='Stable Diffusion model ID override.')
    cgroup.add_argument('--exp_name', type=str, help='Override experiment name prefix.')
    cgroup.add_argument('--seed', type=int, help='Override random seed.')
    cgroup.add_argument('--device', type=str, help='Override device.')
    cgroup.add_argument('--no_progress_bar', action='store_true', help='Override whether to use progress bar.')    
    cgroup.add_argument('--hf_home', type=str, help='Override HuggingFace cache directory (HF_HOME).')
    cgroup.add_argument('--base_dir', type=str, help='Experiment root (default $DTS_BASE_DIR or ~/mcts/experiments).')
    cgroup.add_argument('--no_compile', action='store_true', help='Do not compile the UNet / VAE / ImageReward with TensorRT.')

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
    sgroup.add_argument('--lambda', dest='reward_lambda', type=float, default=1.0,
                        help='Inverse temperature of the target p e^{lambda r}: DTS uses lambda * r, FK weights use lambda.')
    sgroup.add_argument('--guiding_reward_fns', nargs='+', default=['ImageReward'], help='Guiding reward function(s).')
    sgroup.add_argument('--maximize', action='store_true',
                        help='DTS*: argmax selection in tree construction and traversal (default: DTS, sampling by value).')
    sgroup.add_argument('--pw_alpha_root', type=float, default=0.8,
                        help='Meta-root widening exponent of DTS/DTS* (0.8 as used by the authors for T2I, issue #2; code default 0.4).')
    sgroup.add_argument('--pw_alpha', type=float, default=0.4,
                        help='Widening exponent of the non-root nodes of DTS/DTS* (code value 0.4; paper 0.8).')
    sgroup.add_argument('--adaptive_resampling', type=lambda v: v.lower() in ('1', 'true', 'yes'),
                        help='ESS-adaptive resampling for FK (true/false).')

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
    config.bon_enabled = args.do_bon
    config.maximize = args.maximize
    config.pw_alpha_root = args.pw_alpha_root
    config.pw_alpha = args.pw_alpha
    config.prompt_ids = args.prompt_ids
    config.no_compile = args.no_compile
    if args.no_compile: os.environ["DTS_NO_COMPILE"] = "1"
    if args.base_dir is not None: config.common.base_dir = args.base_dir
    if args.adaptive_resampling is not None: config.fkd.adaptive_resampling = args.adaptive_resampling


    if args.compute_budget is not None:
        config.sampling.compute_budget = args.compute_budget
    if args.population_size is not None: config.sampling.population_size = args.population_size
    if args.potential_type is not None: config.fkd.potential_type = args.potential_type
    config.fkd.lambda_param = args.reward_lambda
    config.reward_lambda = args.reward_lambda
    config.guiding_reward_fns = args.guiding_reward_fns

    if 'AestheticScore' in config.guiding_reward_fns:
        config.mcts.exploration_const = 40.0
    else:
        config.mcts.exploration_const = 0.1

    config = process_sd_config(config)
    write_run_meta(config.paths.run_dir, config, args)

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


def _prompt_slug(prompt):
    return "".join(c if c.isalnum() else "_" for c in prompt)[:60]


def finish_nfe_point(images, metrics, method_name, prompt, reward_name, target_nfes, config):
    """Saves the evaluation images (PNG) and logs one NFE point."""
    out_dir = Path(config.paths.image_dir) / reward_name / method_name / _prompt_slug(prompt) / f"N{target_nfes}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, img in enumerate(images):
        img.save(out_dir / f"{k:03d}.png")
    metrics['Distinct'] = len({img.tobytes() for img in images})
    logger.info(f"  {method_name} '{prompt[:40]}' NFE {target_nfes}: " +
                ", ".join(f"{k}={v:.4g}" for k, v in metrics.items() if isinstance(v, (int, float, np.floating))))


@torch.no_grad()
def run_mcts(config, prompt, reward_name, pipeline):
    """DTS (rollouts) or DTS* (rollouts + maximize); NFE budgets are snapshots of one growing tree."""
    method_name = ("MCTS-R" if config.mcts.use_rollout else "MCTS") + ("-max" if config.maximize else "")
    logger.info(f"--- Running {method_name} for prompt: '{prompt}' ---")
    metrics_data = {}

    reward = RewardCounter(lambda images_pil: get_reward_function(
        reward_name=reward_name, images=images_pil, prompts=[prompt] * len(images_pil)), scale=config.reward_lambda)

    mcts_sampler = MCTSStableDiffusionSampler(
        pipeline=pipeline,
        prompt=prompt,
        reward_fn=reward,
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
        maximize=config.maximize,
        exploration_type='ucb',
        batch_size=config.mcts.batch_size,
        output_dir=str(Path(config.paths.image_dir)),
        visualization_timesteps=list(config.mcts.expand_timesteps + [0]),
        pw_alpha_root=config.pw_alpha_root,
        pw_alpha=config.pw_alpha,
        negative_prompt=config.dataset.negative_prompt,
    )
    mcts_sampler.initialize_root_nodes()
    reset_peak_memory()
    build_time, build_reward_evals = 0.0, 0

    for target_nfes in config.sampling.compute_budget:
        reward.reset()
        t0 = time.time()
        mcts_sampler.build_search_tree(num_nfes=target_nfes)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        build_time += time.time() - t0
        build_reward_evals += reward.n_evals
        t1 = time.time()
        sampled_images, _ = mcts_sampler.sample_from_tree(population_size=config.sampling.population_size)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        sample_time = time.time() - t1
        peak_mem = peak_memory_mb()

        metrics = evaluate_images(sampled_images, prompt, config)
        metrics.update({'NFEs': target_nfes, 'ActualNFEs': mcts_sampler.nfes, 'BuildTime': build_time,
                        'SampleTime': sample_time, 'Time': build_time + sample_time,
                        'RewardEvals': build_reward_evals, 'PeakMemMB': peak_mem})
        finish_nfe_point(sampled_images, metrics, method_name, prompt, reward_name, target_nfes, config)
        metrics_data[target_nfes] = metrics

    return metrics_data


@torch.no_grad()
def run_bon(config, prompt, reward_name, pipeline):
    """Best-of-N: N = NFE // T prior samples (prefixes of one generated set for smaller budgets).
    The sample with the highest guiding reward is the output; it is repeated population_size times
    so every method is evaluated on the same number of images (Distinct = 1)."""
    metrics_data = {}
    num_steps = config.sampling.num_inference_steps
    n_list = [max(1, nfe // num_steps) for nfe in config.sampling.compute_budget]
    # FK's UNet batch: every budget is a multiple of it, so the compiled TensorRT engine is reused
    gen_batch_size = config.fkd.batch_size
    reset_peak_memory()
    images, rewards, gen_time = [], [], 0.0
    for nfe, n_candidates in zip(config.sampling.compute_budget, n_list):
        t0 = time.time()
        while len(images) < n_candidates:
            b = min(gen_batch_size, n_candidates - len(images))
            out = pipeline(prompt=[prompt] * b, negative_prompt=[config.dataset.negative_prompt] * b,
                           num_inference_steps=num_steps, guidance_scale=config.sampling.guidance_scale,
                           eta=config.sampling.eta, output_type="pil").images
            images.extend(out)
            rewards.extend(get_reward_function(reward_name=reward_name, images=out, prompts=[prompt] * b))
        if torch.cuda.is_available(): torch.cuda.synchronize()
        gen_time += time.time() - t0
        peak_mem = peak_memory_mb()
        best = int(np.argmax(rewards[:n_candidates]))
        eval_images = [images[best]] * config.sampling.population_size
        metrics = evaluate_images(eval_images, prompt, config)
        metrics.update({'NFEs': nfe, 'ActualNFEs': n_candidates * num_steps, 'ActualParticles': n_candidates,
                        'Time': gen_time, 'RewardEvals': n_candidates, 'PeakMemMB': peak_mem})
        finish_nfe_point(eval_images, metrics, "BoN", prompt, reward_name, nfe, config)
        metrics_data[nfe] = metrics
    return metrics_data


@torch.no_grad()
def run_fkd(config, prompt, reward_name, pipeline):
    """FK steering (SMC). Each NFE point is a separate run with K = NFE // T particles.

    'diff' potential (sampling): the population_size evaluation images are drawn with replacement
    by the final importance weights. 'max' potential (search, FKD-max): the final particle with the
    highest guiding reward is the output, repeated population_size times as for Best-of-N."""
    metrics_data = {}
    search = config.fkd.potential_type == "max"
    method_name = ("FKD-R" if config.fkd.use_rollout else "FKD") + ("-max" if search else "")
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
    nfe_per_particle = config.sampling.num_inference_steps
    if config.fkd.use_rollout:
        for t in range(config.fkd.resampling_t_start, config.fkd.resampling_t_end + 1, config.fkd.resample_frequency):
            nfe_per_particle += config.sampling.num_inference_steps - t

    for nfe_target in config.sampling.compute_budget:
        n_particles = max(1, nfe_target // nfe_per_particle)
        reset_peak_memory()
        t0 = time.time()
        out = pipeline.sample_batched_smc(
            num_particles=n_particles,
            prompt=[prompt] * n_particles,
            processing_batch_size=config.fkd.batch_size,
            negative_prompt=[config.dataset.negative_prompt] * n_particles,
            fkd_args=fkd_args,
            num_inference_steps=config.sampling.num_inference_steps,
            guidance_scale=config.sampling.guidance_scale,
            eta=config.sampling.eta,
            output_type="pil",
        )
        if torch.cuda.is_available(): torch.cuda.synchronize()
        elapsed = time.time() - t0
        peak_mem = peak_memory_mb()
        w = torch.softmax(out.log_weights.double(), dim=0)
        if search:
            idx = [int(torch.argmax(out.final_rewards))] * config.sampling.population_size
        else:
            idx = torch.multinomial(w, config.sampling.population_size, replacement=True).tolist()
        eval_images = [out.images[i] for i in idx]
        metrics = evaluate_images(eval_images, prompt, config)
        metrics.update({'NFEs': nfe_target, 'ActualNFEs': out.nfes, 'ActualParticles': n_particles,
                        'FinalWeightESS': float(1.0 / (w.pow(2).sum() * n_particles)),
                        'FinalRewardMax': float(out.final_rewards.max()),
                        'Time': elapsed, 'RewardEvals': pipeline.reward_evals, 'PeakMemMB': peak_mem})
        finish_nfe_point(eval_images, metrics, method_name, prompt, reward_name, nfe_target, config)
        metrics_data[nfe_target] = metrics

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
    # no safety checker: all methods decode latents without it, so Best-of-N must not either
    if method == 'FKD' or method == 'FKD-R':
        pipeline = FKDStableDiffusion.from_pretrained(config.model.sd_model_id, safety_checker=None, requires_safety_checker=False)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(config.model.sd_model_id, safety_checker=None, requires_safety_checker=False)
    
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(
        num_inference_steps=config.sampling.num_inference_steps, device=config.common.device
    )
    pipeline.to(config.common.device)
    pipeline.set_progress_bar_config(disable=True)
    if not config.common.use_progress_bar:
        dlogging.set_verbosity_error()

    if config.no_compile:
        logger.info(f"[{config.common.device}] TensorRT compilation disabled (--no_compile).")
        return pipeline

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
    if config.bon_enabled: enabled_methods.append("BoN")

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
            method_label = method + ("-max" if (method.startswith("MCTS") and current_config.maximize)
                                     or (method.startswith("FKD") and current_config.fkd.potential_type == "max") else "")
            method_guidance_id = f"{method_label}_guided_by_{reward_name}"
            prompt_results_dfs_for_reward = [] # Store DFs for this specific method, reward, and its prompts
            for prompt in prompt_subset:
                logger.info(f"[GPU {gpu_id}] Prompt: '{prompt}'")
                # same seed for every method on a prompt, independent of the GPU assignment
                seed_experiment(current_config.common.seed + current_config.dataset.prompts.index(prompt))
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
    if config.prompt_ids:
        prompts = [prompts[i] for i in config.prompt_ids]
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
