import argparse
import collections
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

from diffusers import DDIMPipeline, DDIMScheduler
from diffusers.utils import logging as dlogging
from pytorch_fid.inception import InceptionV3

from src.core.sampling_img.mcts_img import MCTSImageDiffusionSampler
from src.core.sampling_img.smc_img import FKDDDIMPipeline
from src.core.rewards import classifier_log_likelihood_reward
from src.utils.datasets import get_img_dataset
from src.models.classifier import get_train_classifier
from src.models.diffusion_img import get_diffusion_model
from src.core.config import get_default_image_config, process_image_config
from src.evaluation.image_metrics import (
    compute_fid,
    compute_mmd,
    compute_diversity,
    get_inception_features,
    get_inception_features_from_tensors,
    get_classifier_features_from_dataloader,
    get_classifier_features_from_tensors,
)

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


def parse_script_args():
    """Parses command-line overrides for ImageConfig."""
    # Only define overrides here. Defaults are in ImageConfig.
    parser = argparse.ArgumentParser(description='Run guided Image Diffusion experiments (MNIST/CIFAR)', add_help=True)

    # --- Generic Overrides ---
    cgroup = parser.add_argument_group('Common Overrides')
    cgroup.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], help='Dataset override.')
    cgroup.add_argument('--sample_classes', nargs='+', help='Specify target classes: "all", "odd", "even", "single", or a list of ints (e.g., 1 3 5)')
    cgroup.add_argument('--exp_name', type=str, help='Override experiment name prefix.')
    cgroup.add_argument('--seed', type=int, help='Override random seed.')
    cgroup.add_argument('--device', type=str, help='Override device.')
    cgroup.add_argument('--no_progress_bar', action='store_true', help='Override whether to use progress bar.')
    cgroup.add_argument('--hf_home', type=str, help='Override HuggingFace cache directory (HF_HOME).')
    cgroup.add_argument('--num_images', type=int, default=1, help='Number of images (N) to inpaint.')
    cgroup.add_argument('--sigma', type=float, default=0.1, help='Standard deviation for Gaussian forward model.')

    # --- Method Enabling ---
    mgroup = parser.add_argument_group('Method Selection')
    mgroup.add_argument('--do_mcts', action='store_true', help='Enable MCTS (standard). Note: Excludes rollout version.')
    mgroup.add_argument('--do_mcts_rollout', action='store_true', help='Enable MCTS with Rollout.')
    mgroup.add_argument('--do_guidance', action='store_true', help='Enable standalone reward-based guidance.')
    mgroup.add_argument('--do_smc', action='store_true', help='Enable standard SMC (FKD-style).')
    mgroup.add_argument('--do_smc_guidance', action='store_true', help='Enable SMC with reward-based guidance.')
    mgroup.add_argument('--do_smc_guidance_tempering', action='store_true', help='Enable SMC with reward-based guidance and Tempering.')
    mgroup.add_argument('--do_smc_rollout', action='store_true', help='Enable SMC with Rollout reward estimation.')
    mgroup.add_argument('--do_iss', action='store_true', help='Enable Importance Sampling.')

    # --- Sampling Overrides ---
    sgroup = parser.add_argument_group('Sampling Overrides')
    sgroup.add_argument('--compute_budget', nargs='+', type=int, help='Override compute budget in NFEs.')
    sgroup.add_argument('--population_size', type=int, help='Override final population size.')
    sgroup.add_argument('--potential_type', type=str, help='Override potential type for FKD.')
    sgroup.add_argument('--lambda_param', type=float, help='Override temperature parameter for all methods.')
    sgroup.add_argument('--guidance_scale', type=float, default=1.0, help='Scale for reward-based guidance.')

    args = parser.parse_args()
    config = get_default_image_config()
    # Use a separate task type for inpainting experiments
    config.task_type = 'inpainting'

    if args.dataset is not None: config.dataset.name = args.dataset
    if args.sample_classes is not None: config.dataset.sample_classes = args.sample_classes
    if args.exp_name is not None: config.common.exp_name = args.exp_name
    if args.seed is not None: config.common.seed = args.seed
    if args.device is not None: config.common.device = args.device
    if args.no_progress_bar is not None: config.common.use_progress_bar = not args.no_progress_bar
    if args.hf_home is not None: config.common.hf_home = args.hf_home
    if args.num_images is not None: config.num_images = args.num_images
    if args.sigma is not None: config.sigma = args.sigma

    # Enable methods based on specific flags
    if args.do_mcts: config.mcts.enabled = True
    if args.do_mcts_rollout: config.mcts_rollout = True # Using a temporary attribute
    # if args.do_fkd: config.fkd.enabled = True
    # if args.do_fkd_rollout: config.fkd_rollout = True # Using a temporary attribute
    if args.do_iss: config.iss.enabled = True

    # # Check if at least one method is enabled
    # if not config.mcts.enabled and not hasattr(config, 'mcts_rollout') and \
    #    not config.fkd.enabled and not hasattr(config, 'fkd_rollout') and \
    #    not config.iss.enabled:
    #     raise ValueError("At least one method must be enabled.")

    if args.compute_budget is not None:
        config.sampling.compute_budget = args.compute_budget
    if args.population_size is not None: config.sampling.population_size = args.population_size
    if args.potential_type is not None: config.fkd.potential_type = args.potential_type
    if args.lambda_param is not None:
        config.fkd.lambda_param = args.lambda_param
        config.iss.lambda_param = args.lambda_param
    if args.guidance_scale is not None:
        config.guidance_scale = args.guidance_scale # Store guidance scale in config

    config = process_image_config(config)
    return config, args


@torch.no_grad()
def evaluate_images(
    images,
    real_class_loader,
    target_class_list,
    classifier,
    inception_model,
    config,
):
    """Evaluate a set of generated images on log reward, FID, and diversity."""
    results = {}
    device = config.common.device

    # Average Reward
    log_rewards = classifier_log_likelihood_reward(images, target_class_list, classifier, device)
    results['LogProb_Mean'] = log_rewards.mean().item()
    results['LogProb_Std'] = log_rewards.std().item()

    # FID
    num_real_samples = len(real_class_loader.dataset)
    num_fake_samples = len(images)
    num_eval_samples = min(num_real_samples, num_fake_samples, config.eval.num_fid_samples)
    # logger.info(f"Calculating FID using {num_eval_samples} samples.")
    logger.info(f"Calculating FID using {num_eval_samples} samples (Real: {num_real_samples}, Fake: {num_fake_samples}, ConfigLimit: {config.eval.num_fid_samples}).")

    if config.dataset.name == 'mnist':
        logger.info("Using MNIST classifier for feature extraction.")
        # Use classifier features for MNIST
        fake_features = get_classifier_features_from_tensors(
            images, # Assumed [-1, 1]
            classifier, # Pass the classifier model
            device,
            config.sampling.eval_batch_size,
            "Generated (Classifier)",
        )
        real_features = get_classifier_features_from_dataloader(
            real_class_loader, # Assumes loader gives [-1, 1]
            num_eval_samples,
            classifier, # Pass the classifier model
            device,
            "Real Class (Classifier)",
        )
    else:
        logger.info("Using InceptionV3 for feature extraction.")
        # Use Inception features for other datasets (e.g., CIFAR-10)
        # Note: InceptionV3 in image_metrics assumes input is [-1, 1] and handles normalization internally if normalize_input=True
        # However, the inception model here is loaded with normalize_input=False, so it expects [0, 1]
        # images_0_1 = (images / 2 + 0.5).clamp(0, 1) # Convert fake images to [0, 1]
        fake_features = get_inception_features_from_tensors(
            images,
            inception_model,
            device,
            config.sampling.eval_batch_size,
            config.dataset.channels,
            "Generated (Inception)",
        )
        # Need to ensure real loader also provides [0, 1] for Inception or modify get_inception_features
        # Assuming get_inception_features handles the [-1, 1] from loader and converts internally if needed
        # Let's modify get_inception_features to explicitly handle the normalization based on model
        real_features = get_inception_features(
            real_class_loader, # Loader gives [-1, 1]
            num_eval_samples,
            inception_model, # Expects [0, 1]
            device,
            config.dataset.channels,
            "Real Class (Inception)",
        )

    # Select features for FID (potentially fewer than total generated)
    fid_fake_features = fake_features[:num_eval_samples]
    diversity_fake_features = fake_features

    fid_value = compute_fid(real_features, fid_fake_features)
    mmd_value = compute_mmd(real_features, fid_fake_features)
    avg_cosine_distance = compute_diversity(diversity_fake_features, config)
    results['FID'] = fid_value
    results['MMD'] = mmd_value
    results['Diversity_AvgCosDist'] = avg_cosine_distance

    return results

def save_results_to_csv(metrics_data, run_class_name, output_path, method_name, config, save=False):
    """
    Save metrics to CSV for a specific method and class run.
    metrics_data is a dict: {iteration/particle: {metric_name: float}}
    """
    rows = []
    for nfe_target, metrics in metrics_data.items():
        row_base = {
            'Method': method_name,
            'Run_Class': run_class_name,
            'N': nfe_target,
            'NFEs': metrics.get('NFEs', nfe_target),
            'Population_Size': config.sampling.population_size,
            'Dataset': config.dataset.name,
        }
        for k, v in metrics.items():
            row_base[k] = v
        rows.append(row_base)

    if not rows:
        logger.warning(f"No data rows generated for method {method_name} run {run_class_name}.")
        return None

    df = pd.DataFrame(rows)
    cols_order = [
        'Method', 'Run_Class', 'N', 'NFEs', 'Population_Size',
        'Dataset', 'FID', 'MMD', 'Diversity_AvgCosDist', 'LogProb_Mean', 'LogProb_Std',
    ]

    extra_cols = [col for col in df.columns if col not in cols_order]
    final_cols = [c for c in cols_order if c in df.columns] + extra_cols
    df = df[final_cols]
    if save:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved metrics for {method_name} ({run_class_name}) to {output_path}")
    return df


@torch.no_grad()
def run_mcts(config, run_class_name, target_class_list, classifier, 
             inception_model, real_loader, pipeline, method_name="MCTS"):
    """Run MCTS experiments."""
    logger.info(f"--- Running MCTS for {run_class_name} ---")
    metrics_data = {}

    use_rollout = "rollout" in method_name.lower()

    reward_fn_mcts = lambda images: classifier_log_likelihood_reward(
        images, target_class_list, classifier, config.common.device
    ).detach().cpu().numpy()

    # Create output directories
    dir_name = "mcts-rollout" if use_rollout else "mcts"
    output_dir_mcts_images = Path(config.paths.image_dir) / dir_name
    output_dir_mcts_images.mkdir(parents=True, exist_ok=True)
    # output_dir_mcts_metrics = Path(config.paths.metrics_dir) / "mcts"
    # output_dir_mcts_metrics.mkdir(parents=True, exist_ok=True)

    mcts_sampler = MCTSImageDiffusionSampler(
        pipeline=pipeline,
        reward_fn=reward_fn_mcts,
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
        output_dir=str(output_dir_mcts_images),
        visualization_timesteps=list(config.mcts.expand_timesteps + [0]),
    )

    mcts_sampler.initialize_root_nodes()

    # Run MCTS progressively for each target iteration count
    for target_nfes in sorted(config.sampling.compute_budget):
        logger.info(f"MCTS NFE {target_nfes} for {run_class_name}")

        # Build search tree
        mcts_sampler.build_search_tree(num_nfes=target_nfes)

        # Sample images
        sampled_images, _ = mcts_sampler.sample_from_tree(
            population_size=config.sampling.population_size,
        )

        # Evaluate metrics
        metrics = evaluate_images(
            sampled_images,
            real_loader,
            target_class_list,
            classifier,
            inception_model,
            config,
        )
        metrics['NFEs'] = target_nfes
        metrics['ActualNFEs'] = mcts_sampler.nfes
        metrics_data[target_nfes] = metrics
        logger.info(f"  NFE Target: {target_nfes} (using run with {config.sampling.population_size} particles, Actual NFEs: {mcts_sampler.nfes})")
        logger.info(f"    FID: {metrics['FID']:.4f}, MMD: {metrics['MMD']:.4f}, LogProb: {metrics['LogProb_Mean']:.4f}")


        if sampled_images.shape[0] > config.sampling.gen_batch_size:
            sampled_images = sampled_images[torch.randperm(sampled_images.shape[0])[:config.sampling.gen_batch_size]]

        # Save image grid
        sampled_images_0_1 = (sampled_images / 2 + 0.5).clamp(0, 1)
        grid_path = output_dir_mcts_images / f"{run_class_name}_{target_nfes}.png"
        save_image(sampled_images_0_1.cpu(), grid_path, nrow=int(math.ceil(np.sqrt(sampled_images_0_1.shape[0]))))

    # # Save results for this specific run class
    # csv_path = output_dir_mcts_metrics / f"MCTS_{run_class_name}_metrics.csv"
    # df = save_results_to_csv(metrics_data, csv_path, "MCTS", run_class_name, config)

    return metrics_data


@torch.no_grad()
def run_smc(config, run_class_name, target_class_list, classifier,
            inception_model, real_loader, pipeline, method_name="SMC"): # Add method_name
    """Run SMC, SMC-Guidance, SMC-Guidance-Tempering, or SMC-Rollout experiment."""
    logger.info(f"--- Running {method_name} for {run_class_name} ---")
    metrics_data = {}

    # Determine flags based on method_name
    use_smc = "smc" in method_name.lower() # Base SMC functionality
    use_guidance = "guidance" in method_name.lower()
    use_tempering = "tempering" in method_name.lower()
    use_rollout = "rollout" in method_name.lower()
    logger.info(f"  Config: use_smc={use_smc}, use_guidance={use_guidance}, use_tempering={use_tempering}, use_rollout={use_rollout}")


    # Reward function: Needs to return tensor for FKD/SMC internal use
    reward_fn_smc = lambda images: classifier_log_likelihood_reward(
        images, target_class_list, classifier, config.common.device
    ).to(config.common.device)

    # Create output directories (use method_name for folder)
    output_dir_method = Path(config.paths.image_dir) / method_name.lower().replace('-','_')
    output_dir_method.mkdir(parents=True, exist_ok=True)

    # --- FKD/SMC Config Dictionary ---
    fkd_config_dict = {
        'reward_fn': reward_fn_smc, # Use the tensor-returning function
        'potential_type': config.fkd.potential_type, # Keep name 'fkd' in config for now
        'lmbda': config.fkd.lambda_param,
        'adaptive_resampling': config.fkd.adaptive_resampling,
        'resample_frequency': config.fkd.resample_frequency,
        'resampling_t_start': config.fkd.resampling_t_start,
        'resampling_t_end': config.fkd.resampling_t_end,
        'reward_min_value': config.fkd.reward_min_val,
        # 'device' is handled by the pipeline
    }
    
    # --- Calculate NFEs per particle ---
    resampling_schedule_indices = range(
        config.fkd.resampling_t_start,
        config.fkd.resampling_t_end + 1,
        config.fkd.resample_frequency
    ) if use_smc else []

    nfe_per_particle = config.sampling.num_inference_steps
    if use_guidance:
        nfe_per_particle += config.sampling.num_inference_steps
    if use_smc and use_rollout:
        rollout_nfe_count = 0
        for i in resampling_schedule_indices:
            rollout_nfe_count += max(0, config.sampling.num_inference_steps - i)
        nfe_per_particle += rollout_nfe_count
        logger.info(f"Estimated additional NFEs per particle from rollout: {rollout_nfe_count}")

    logger.info(f"Estimated NFEs per particle for {method_name}: {nfe_per_particle}")
    if nfe_per_particle <= 0:
        logger.warning(f"Calculated NFE per particle is {nfe_per_particle}. Setting to 1.")
        nfe_per_particle = 1

    # Determine particle counts required for each NFE target
    eval_particles = sorted(list(set([max(1, nfe // nfe_per_particle) for nfe in config.sampling.compute_budget])))
    logger.info(f"Will run with particle counts: {eval_particles}")
    max_n_particles = max(eval_particles) if eval_particles else 1

    all_smc_samples = {} # Store final samples per NFE target for eval/plotting

    # Run SMC sampler once for each required particle count
    for i, n_particles in enumerate(eval_particles):
        logger.info(f"{method_name} Particle Count {n_particles} for {run_class_name}")

        # Call the new sample_batched_smc method
        current_samples = pipeline.sample_batched_smc(
            num_particles=n_particles,
            processing_batch_size=config.fkd.batch_size,
            num_inference_steps=config.sampling.num_inference_steps,
            use_smc=use_smc,
            use_guidance=use_guidance,
            use_tempering=use_tempering,
            use_rollout=use_rollout,
            guidance_scale=config.guidance_scale,
            reward_fn=reward_fn_smc,
            fkd_config=fkd_config_dict,
            device=config.common.device,
            eta=config.sampling.eta,
            output_type="pt",
            use_progress_bar=config.common.use_progress_bar,
        ).images

        actual_nfes_run = n_particles * nfe_per_particle # Estimate based on calculation

        # Resample generated population if needed to match eval population size
        if current_samples.shape[0] > config.sampling.population_size:
             logger.info(f"Subsampling {current_samples.shape[0]} generated images to {config.sampling.population_size} for evaluation.")
             indices = torch.randperm(current_samples.shape[0])[:config.sampling.population_size]
             eval_samples = current_samples[indices]
        elif current_samples.shape[0] < config.sampling.population_size:
            logger.warning(f"Generated samples ({len(current_samples)}) < target population ({config.sampling.population_size}). Resampling with replacement.")
            resample_indices = torch.randint(0, len(current_samples), (config.sampling.population_size,))
            eval_samples = current_samples[resample_indices]
        else:
            eval_samples = current_samples

        # Evaluate metrics
        metrics = evaluate_images(
            eval_samples,
            real_loader,
            target_class_list,
            classifier,
            inception_model,
            config,
        )
        nfe_target = config.sampling.compute_budget[i] # Corresponding NFE target
        metrics['NFEs'] = nfe_target # Use target NFE for reporting
        metrics['ActualNFEs'] = actual_nfes_run # Store estimated actual NFEs
        metrics['ActualParticles'] = n_particles # Store particles used for this run
        metrics_data[nfe_target] = metrics
        logger.info(f"  NFE Target: {nfe_target} (using run with {n_particles} particles, Actual NFEs: {actual_nfes_run})")
        logger.info(f"    FID: {metrics['FID']:.4f}, MMD: {metrics['MMD']:.4f}, LogProb: {metrics['LogProb_Mean']:.4f}")

        # Save image grid (use the evaluated samples)
        save_samples = eval_samples
        if save_samples.shape[0] > config.sampling.gen_batch_size:
            save_samples = save_samples[torch.randperm(save_samples.shape[0])[:config.sampling.gen_batch_size]]

        eval_images_0_1 = (save_samples / 2 + 0.5).clamp(0, 1)
        grid_path = output_dir_method / f"{run_class_name}_{nfe_target}.png"
        save_image(eval_images_0_1.cpu(), grid_path, nrow=int(math.ceil(np.sqrt(eval_images_0_1.shape[0]))))

    return metrics_data


@torch.no_grad()
def run_iss(config, run_class_name, target_class_list, classifier, 
            inception_model, real_loader, pipeline):
    """Run ISS experiments."""
    logger.info(f"--- Running ISS for {run_class_name} ---")
    metrics_data = {}

    reward_fn_iss = lambda images: classifier_log_likelihood_reward(
        images, target_class_list, classifier, config.common.device
    ).detach().cpu().numpy()

    # Create output directories
    output_dir_iss_images = Path(config.paths.image_dir) / "iss"
    output_dir_iss_images.mkdir(parents=True, exist_ok=True)
    # output_dir_iss_metrics = Path(config.paths.metrics_dir) / "iss"
    # output_dir_iss_metrics.mkdir(parents=True, exist_ok=True)

    # Determine max candidate count and generate base samples once
    nfe_per_particle = config.sampling.num_inference_steps
    eval_particles =  [nfe // nfe_per_particle for nfe in config.sampling.compute_budget]
    max_n_particles = max(eval_particles)
    logger.info(f"Running ISS base generation for max N = {max_n_particles} for {run_class_name}")

    # Process in batches to handle large candidate counts
    all_base_samples = []
    for batch_start in tqdm.tqdm(range(0, max_n_particles, config.iss.batch_size), 
                          desc="Generating base samples",
                          disable=not config.common.use_progress_bar):
        batch_size = min(config.iss.batch_size, max_n_particles - batch_start)
        logger.info(f"Processing generation batch {batch_start//config.iss.batch_size + 1}/{(max_n_particles + config.iss.batch_size - 1)//config.iss.batch_size} " +
              f"(candidates {batch_start+1}-{batch_start+batch_size}/{max_n_particles})")

        # Run generation with base pipeline
        output = pipeline(
            batch_size=batch_size,
            num_inference_steps=config.sampling.num_inference_steps,
            eta=config.sampling.eta,
            output_type="pt"
        ).images

        all_base_samples.extend(output)

    all_base_samples = np.stack(all_base_samples, axis=0) if len(all_base_samples) > 1 else all_base_samples[0]
    all_base_samples = torch.from_numpy(all_base_samples).permute(0, 3, 1, 2).to(config.common.device)
    for i, n_particles in enumerate(eval_particles):
        logger.info(f"--- Evaluating ISS for N = {n_particles} ---")

        # Select the first n_particles base samples
        current_base_samples = all_base_samples[:min(n_particles, max_n_particles)]

        # Calculate weights for the current subset of base samples
        rewards = reward_fn_iss(current_base_samples)
        weights = np.exp(config.iss.lambda_param * rewards)
        weights /= weights.sum()
        weights = np.nan_to_num(weights, nan=0.0)
        if weights.sum() <= 0:
            logger.warning(f"ISS resampling failed for N={n_particles} (all weights zero). Selecting random samples.")
            weights = np.ones(len(current_base_samples)) / len(current_base_samples)
        
        # Resample with replacement from the current subset
        population_size = min(config.sampling.compute_budget[i] // 150, config.eval.num_fid_samples)
        num_to_resample = min(population_size, len(current_base_samples), config.eval.num_fid_samples)
        resample_indices = np.random.choice(len(current_base_samples), size=num_to_resample, p=weights)
        eval_images = current_base_samples[resample_indices]
        eval_images = (2 * eval_images - 1).clamp(-1, 1)

        # Evaluate metrics on the resampled images
        metrics = evaluate_images(
            eval_images,
            real_loader,
            target_class_list,
            classifier,
            inception_model,
            config,
        )
        metrics['NFEs'] = config.sampling.compute_budget[i]
        metrics_data[config.sampling.compute_budget[i]] = metrics

        if eval_images.shape[0] > config.sampling.gen_batch_size:
            eval_images = eval_images[torch.randperm(eval_images.shape[0])[:config.sampling.gen_batch_size]]

        # Save image grid
        eval_images_0_1 = (eval_images / 2 + 0.5).clamp(0, 1)
        grid_path = output_dir_iss_images / f"{run_class_name}_{config.sampling.compute_budget[i]}.png"
        save_image(eval_images_0_1.cpu(), grid_path, nrow=int(math.ceil(np.sqrt(eval_images_0_1.shape[0]))))

    # # Save results for this specific run class
    # csv_path = output_dir_iss_metrics / f"ISS_{run_class_name}_metrics.csv"
    # df = save_results_to_csv(metrics_data, csv_path, "ISS", run_class_name, config)

    return metrics_data


def aggregate_and_save_summary(all_run_metrics_dfs, dataset_name, output_dir):
    """
    Aggregates metrics from multiple DataFrames (one per method/class run) and saves summary CSVs.
    all_run_metrics_dfs: Dict[method_name, List[pd.DataFrame]]
    """
    logger.info(f"Aggregating results across runs")
    output_dir = Path(output_dir)
    mnist_odd_classes = [f"class{c}" for c in [1, 3, 5, 7, 9]]
    mnist_even_classes = [f"class{c}" for c in [0, 2, 4, 6, 8]]

    # Make sure keys match the new method names
    valid_methods = [m for m in all_run_metrics_dfs.keys() if all_run_metrics_dfs[m]]

    for method in valid_methods: # Iterate only over methods with data
        df_list = all_run_metrics_dfs[method]
        if not df_list:
            logger.warning(f"No dataframes found for method {method}, skipping aggregation.")
            continue

        summary_df = pd.concat(df_list, ignore_index=True)
        # Include potential new columns from SMC runs
        avg_cols = ['FID', 'MMD', 'Diversity_AvgCosDist', 'LogProb_Mean', 'LogProb_Std', 'NFEs', 'ActualNFEs', 'ActualParticles']

        valid_avg_cols = []
        for col in avg_cols:
            if col in summary_df.columns:
                summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')
                valid_avg_cols.append(col)
            else:
                 logger.warning(f"Summary column {col} not found in {method} results, cannot average.")

        # Group by Iteration/Particles
        group_col = 'N'
        summary_all = summary_df.groupby(group_col)[valid_avg_cols].mean().reset_index()
        if not summary_all.empty:
            summary_all['Group'] = 'all_single_classes'
            summary_all_path = output_dir / f"{method}_summary_all.csv"
            summary_all.to_csv(summary_all_path, index=False)
            logger.info(f"Saved aggregated metrics for {method} (all classes) to {summary_all_path}")

        # Average across ODD/EVEN MNIST classes (if applicable)
        if dataset_name == 'mnist':
            for group_name, class_list in [('odd', mnist_odd_classes), ('even', mnist_even_classes)]:
                sub_df = summary_df[summary_df['Run_Class'].isin(class_list)]
                if not sub_df.empty:
                    summary_group = sub_df.groupby(group_col)[valid_avg_cols].mean().reset_index()
                    summary_group['Group'] = f'mnist_{group_name}'
                    summary_group_path = output_dir / f"{method}_summary_{group_name}.csv"
                    summary_group.to_csv(summary_group_path, index=False)
                    logger.info(f"Saved aggregated metrics for {method} (MNIST {group_name}) to {summary_group_path}")


def load_pipeline(config, method):
    pipeline = get_diffusion_model(config)
    if 'smc' in method.lower() or 'guidance' in method.lower():
        pipeline = FKDDDIMPipeline.from_pretrained(config.paths.diffusion_load_path)

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(
        num_inference_steps=config.sampling.num_inference_steps, device=config.common.device
    )
    pipeline.to(config.common.device)
    pipeline.set_progress_bar_config(disable=True)
    if not config.common.use_progress_bar:
        dlogging.set_verbosity_error()    

    logger.info(f"Loaded diffusion model from {config.paths.diffusion_load_path}")
    return pipeline


def main():
    config, args = parse_script_args()
    logger.info("--- Final Configuration ---")
    logger.info("\n" + pprint.pformat(asdict(config)))

    # Load dataset and prepare batch of images
    logger.info("Loading dataset for inpainting...")
    image_data = get_img_dataset(config, splits=['train'])
    train_loader = image_data['train_loader']
    batch = next(iter(train_loader))
    images = batch['images']
    num_images = min(args.num_images, images.shape[0])
    from collections import defaultdict
    metrics_collection = defaultdict(lambda: defaultdict(list))
    # Loop over first N images
    for img_idx in range(num_images):
        logger.info(f"Processing image {img_idx+1}/{num_images} for inpainting")
        original_image = images[img_idx].to(config.common.device)
        H, W = original_image.shape[1], original_image.shape[2]
        mh, mw = int(H * 0.3), int(W * 0.3)
        y1, x1 = H // 2 - mh // 2, W // 2 - mw // 2
        mask = torch.ones_like(original_image)
        mask[:, y1:y1+mh, x1:x1+mw] = 0
        noisy_measurement = (original_image * mask + torch.randn_like(original_image) * args.sigma * mask).to(config.common.device)
        # Define reward functions
        def gaussian_reward(images):
            se = -((images.to(config.common.device) * mask - noisy_measurement) ** 2).view(images.shape[0], -1).sum(dim=1)
            return se
        def gaussian_reward_numpy(images):
            return gaussian_reward(images).detach().cpu().numpy()
        # Determine which inpainting methods to run
        enabled_methods = []
        if args.do_mcts: enabled_methods.append("MCTS")
        if args.do_mcts_rollout: enabled_methods.append("MCTS-Rollout")
        if args.do_guidance: enabled_methods.append("Guidance")
        if args.do_smc: enabled_methods.append("SMC")
        if args.do_smc_guidance: enabled_methods.append("SMC-Guidance")
        if args.do_smc_guidance_tempering: enabled_methods.append("SMC-Guidance-Tempering")
        if args.do_smc_rollout: enabled_methods.append("SMC-Rollout")
        if args.do_iss: enabled_methods.append("ISS")
        # Loop each method for this image
        for method in enabled_methods:
            logger.info(f"===== Starting Inpainting Method: {method} ======")
            pipeline = load_pipeline(config, method)
            if method.startswith("MCTS"):
                # Use normalized Gaussian reward for MCTS inpainting
                reward_fn = gaussian_reward
                use_rollout = "rollout" in method.lower()

                sampler = MCTSImageDiffusionSampler(
                    pipeline=pipeline,
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
                    batch_size=config.mcts.batch_size,
                    use_rollout=use_rollout,
                    use_guidance=False,
                    output_dir=str(Path(config.paths.image_dir) / "mcts"),
                    visualization_timesteps=list(config.mcts.expand_timesteps + [0]),
                )
                sampler.initialize_root_nodes()
                for target_nfes in sorted(config.sampling.compute_budget):
                    sampler.build_search_tree(num_nfes=target_nfes)
                    sampled_images, _ = sampler.sample_from_tree(population_size=config.sampling.population_size)
                    save_imgs = sampled_images
                    if save_imgs.shape[0] > config.sampling.gen_batch_size:
                        save_imgs = save_imgs[torch.randperm(save_imgs.shape[0])[:config.sampling.gen_batch_size]]
                    imgs_0_1 = (save_imgs / 2 + 0.5).clamp(0,1)
                    grid_path = Path(config.paths.image_dir) / f"{method}_{target_nfes}.png"
                    save_image(imgs_0_1.cpu(), grid_path, nrow=int(math.ceil(np.sqrt(imgs_0_1.shape[0]))))
                    logger.info(f"Saved MCTS sample grid to {grid_path}")
                    # Compute MSE metrics
                    diff = (save_imgs - original_image.unsqueeze(0)) ** 2
                    full_mse = diff.mean().item()
                    masked_region = (1 - mask)
                    masked_diff = diff * masked_region.unsqueeze(0)
                    masked_mse = masked_diff.sum().item() / masked_region.sum().item()
                    logger.info(f"MCTS NFE {target_nfes} - Full MSE: {full_mse:.4f}, Masked MSE: {masked_mse:.4f}")
                    # Evaluate and log rewards for MCTS samples
                    rewards = reward_fn(save_imgs)
                    logger.info(f"MCTS NFE {target_nfes} Rewards: {rewards}")
                    metrics_collection[method][target_nfes].append({'Full_MSE': full_mse, 'Masked_MSE': masked_mse, 'LogProb_Mean': rewards.mean().item(), 'LogProb_Std': rewards.std().item(), 'FID': 0, 'MMD': 0, 'Diversity_AvgCosDist': 0}) # Placeholder for metrics
            elif method == "ISS":
                logger.warning("ISS method not yet supported for inpainting; skipping")
                continue
            else:
                # Gaussian reward on masked region with normalization
                fkd_config_dict = {
                    'reward_fn': gaussian_reward,
                    'potential_type': config.fkd.potential_type,  
                    'lmbda': config.fkd.lambda_param,
                    'adaptive_resampling': config.fkd.adaptive_resampling,
                    'resample_frequency': config.fkd.resample_frequency,
                    'resampling_t_start': config.fkd.resampling_t_start,
                    'resampling_t_end': config.fkd.resampling_t_end,
                    'reward_min_value': config.fkd.reward_min_val,
                }
                # Determine flags based on method name
                use_smc = method.lower().startswith("smc")
                use_guidance = "guidance" in method.lower()
                use_tempering = "tempering" in method.lower()
                use_rollout = "rollout" in method.lower()
                # --- SMC per-compute-budget sampling ---
                # Compute NFEs per particle
                nfe_per_particle = config.sampling.num_inference_steps
                if use_guidance:
                    nfe_per_particle += config.sampling.num_inference_steps
                if use_smc and use_rollout:
                    rollout_nfe_count = 0
                    for t in range(config.fkd.resampling_t_start, config.fkd.resampling_t_end + 1, config.fkd.resample_frequency):
                        rollout_nfe_count += max(0, config.sampling.num_inference_steps - t)
                    nfe_per_particle += rollout_nfe_count
                if nfe_per_particle <= 0:
                    nfe_per_particle = 1

                # Loop over target NFEs
                for nfe_target in sorted(config.sampling.compute_budget):
                    n_particles = max(1, nfe_target // nfe_per_particle)
                    # Run SMC sampling for this particle count
                    samples = pipeline.sample_batched_smc(
                        num_particles=n_particles,
                        processing_batch_size=config.fkd.batch_size,
                        num_inference_steps=config.sampling.num_inference_steps,
                        use_smc=use_smc,
                        use_guidance=use_guidance,
                        use_tempering=use_tempering,
                        use_rollout=use_rollout,
                        guidance_scale=config.guidance_scale,
                        reward_fn=fkd_config_dict['reward_fn'],
                        fkd_config=fkd_config_dict,
                        device=config.common.device,
                        eta=config.sampling.eta,
                        output_type="pt",
                        use_progress_bar=config.common.use_progress_bar,
                    ).images

                    save_imgs = samples
                    # Subsample for grid
                    if save_imgs.shape[0] > config.sampling.gen_batch_size:
                        save_imgs = save_imgs[torch.randperm(save_imgs.shape[0])[:config.sampling.gen_batch_size]]
                    imgs_0_1 = (save_imgs / 2 + 0.5).clamp(0,1)
                    grid_path = Path(config.paths.image_dir) / f"{method}_{nfe_target}.png"
                    save_image(imgs_0_1.cpu(), grid_path, nrow=int(math.ceil(np.sqrt(imgs_0_1.shape[0]))))
                    logger.info(f"Saved {method} sample grid to {grid_path}")
                    # Compute and log metrics
                    diff = (save_imgs - original_image.unsqueeze(0)) ** 2
                    full_mse = diff.mean().item()
                    masked_region = (1 - mask)
                    masked_diff = diff * masked_region.unsqueeze(0)
                    masked_mse = masked_diff.sum().item() / masked_region.sum().item()
                    logger.info(f"{method} NFE {nfe_target} - Full MSE: {full_mse:.4f}, Masked MSE: {masked_mse:.4f}")
                    # Evaluate and log rewards
                    rewards = gaussian_reward_numpy(save_imgs)
                    logger.info(f"{method} NFE {nfe_target} Rewards: {rewards}")
                    metrics_collection[method][nfe_target].append({'Full_MSE': full_mse, 'Masked_MSE': masked_mse, 'LogProb_Mean': rewards.mean().item(), 'LogProb_Std': rewards.std().item(), 'FID': 0, 'MMD': 0, 'Diversity_AvgCosDist': 0}) # Placeholder for metrics

    # After processing all images, average metrics and write CSVs
    for method, nfe_dict in metrics_collection.items():
        rows = []
        for nfe, metrics_list in sorted(nfe_dict.items()):
            avg_metrics = {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in metrics_list[0]}
            avg_metrics['NFEs'] = nfe
            rows.append(avg_metrics)
        df = pd.DataFrame(rows).sort_values('NFEs')
        csv_path = Path(config.paths.image_dir) / f"{method}_summary.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved summary metrics for {method} to {csv_path}")
    return metrics_collection


if __name__ == '__main__':
    main()