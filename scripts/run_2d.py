import torch
import numpy as np
import random
import pandas as pd
from pathlib import Path
import logging
import colorlog
import pprint
import argparse
from dataclasses import asdict
import time
import pickle
import json

from src.models.diffusion_2d import DiffusionModel
from src.models.gmm_analytic import GMMInverseProblem, AnalyticGMMDiffusion, GMMTask, sliced_wasserstein
from src.utils.datasets import get_2d_dataset
from src.core.config import get_default_toy_config, process_toy_config
from src.core.sampling.mcts_parallel import MCTSDiffusionSampler, tree_statistics
from src.core.sampling.smc_2d import SMCDiffusion2DSampler
from src.evaluation.distribution_metrics import compute_mmd
from src.evaluation.value_metrics import evaluate_mcts_value_estimates, evaluate_smc_value_estimates
from src.core.smc_weights import SMCDivergedError
from src.utils.run_logging import (
    RewardCounter, write_run_meta, reset_peak_memory, peak_memory_mb, distinct_count,
)


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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes'):
        return True
    if v.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {v}")


def parse_script_args():
    parser = argparse.ArgumentParser(description='Run guided 2D / GMM diffusion experiments', add_help=True)

    cgroup = parser.add_argument_group('Common Overrides')
    cgroup.add_argument('--dataset', type=str, required=True,
                        help="A 2D toy dataset name, or 'gmm' for the analytic GMM inverse problem.")
    cgroup.add_argument('--exp_name', type=str, help='Experiment name prefix.')
    cgroup.add_argument('--base_dir', type=str, help='Experiment root (default $DTS_BASE_DIR or ~/mcts/experiments).')
    cgroup.add_argument('--seed', type=int, help='First random seed.')
    cgroup.add_argument('--num_seeds', type=int, default=5, help='Number of consecutive seeds starting at --seed.')
    cgroup.add_argument('--device', type=str, help='Device (cuda/cpu).')
    cgroup.add_argument('--no_progress_bar', action='store_true', help='Disable progress bars.')
    cgroup.add_argument('--hf_home', type=str, help='HuggingFace cache dir.')

    ggroup = parser.add_argument_group('GMM inverse problem (--dataset gmm)')
    ggroup.add_argument('--gmm_dim', type=int, help='d_x of the GMM prior.')
    ggroup.add_argument('--gmm_instance', type=int, default=0,
                        help='Problem-instance seed: redraws prior weights, A, sigma_y and y.')
    ggroup.add_argument('--sw_projections', type=int, default=10_000, help='Random directions for sliced Wasserstein.')

    mgroup = parser.add_argument_group('Method Selection')
    mgroup.add_argument('--do_mcts', action='store_true', help='Enable MCTS (standard).')
    mgroup.add_argument('--do_mcts_rollout', action='store_true', help='Enable MCTS with Rollout reward estimation (DTS).')
    mgroup.add_argument('--do_guidance', action='store_true', help='Enable standalone reward-based guidance (DPS-RG).')
    mgroup.add_argument('--do_smc', action='store_true', help='Enable standard SMC (FKD-style).')
    mgroup.add_argument('--do_smc_guidance', action='store_true', help='Enable SMC with reward-based guidance (TDS).')
    mgroup.add_argument('--do_smc_guidance_tempering', action='store_true', help='Enable SMC with reward-based guidance and Tempering (DAS).')
    mgroup.add_argument('--do_smc_rollout', action='store_true', help='Enable SMC with Rollout reward estimation.')

    sgroup = parser.add_argument_group('Sampling Overrides')
    sgroup.add_argument('--compute_budget', nargs='+', type=int, help='Compute budget in NFEs.')
    sgroup.add_argument('--population_size', type=int, help='Number of evaluation samples per NFE point (default 1000; 2000 for gmm).')
    sgroup.add_argument('--potential_type', type=str, help='Potential type for FKD/SMC.')
    sgroup.add_argument('--lambda', dest='reward_lambda', type=float, default=1.0,
                        help='Inverse temperature of the target p e^{lambda r}: every method uses lambda * r.')
    sgroup.add_argument('--guidance_scale', type=float, default=1.0, help='Scale for reward-based guidance.')
    sgroup.add_argument('--das_gamma', type=float, default=0.008, help='DAS tempering rate gamma.')
    sgroup.add_argument('--adaptive_resampling', type=str2bool, help='ESS-adaptive resampling for SMC/TDS/DAS.')
    sgroup.add_argument('--resampling_steps', nargs=3, type=int, metavar=('START', 'END', 'FREQ'),
                        help='SMC/FK, TDS and DAS potential and resampling steps: sampling indices range(START, END + 1, FREQ), 0 = first denoising step (default: the DTS branching schedule; 0 T-1 1 = every step). DTS branching is set separately.')
    sgroup.add_argument('--pw_alpha', type=float, help='DTS progressive-widening alpha (default: original hard-coded values).')
    sgroup.add_argument('--pw_alpha_root', type=float, help='DTS progressive-widening alpha of the meta-root only.')
    sgroup.add_argument('--eps_explore', type=float, help='Constant epsilon for DTS tree construction (default: linear 0.1 -> 0.05).')
    sgroup.add_argument('--mcts_batch_size', type=int, help='DTS internal batch size.')
    sgroup.add_argument('--compact_tree', action='store_true',
                        help='DTS: drop the states of rollout-chain nodes that are never read again '
                             '(MCTSDiffusionSampler keep_chain_states=False); same tree and samples, less memory.')

    egroup = parser.add_argument_group('Evaluation Overrides')
    egroup.add_argument('--no_value_eval', action='store_true', help='Disable value function evaluation.')
    egroup.add_argument('--value_timesteps', nargs='+', type=int, help='Timesteps for value evaluation.')
    egroup.add_argument('--no_save_samples', action='store_true', help='Do not save evaluation samples.')
    egroup.add_argument('--tree_stats', action='store_true',
                        help='DTS: after every NFE snapshot write the tree statistics of mcts_parallel.tree_statistics '
                             '(and, for --dataset gmm, value errors against the closed-form log M_t and the exact leaf '
                             'distribution) to tree_stats_<method>_seed<seed>.json. Read-only, samples are unchanged.')

    args = parser.parse_args()
    if args.dataset == 'gmm' and args.gmm_dim is None:
        parser.error("--dataset gmm requires --gmm_dim")
    return args


def build_config(args, seed):
    """Fresh, processed ToyConfig for one seed (paths, seeding and config.json)."""
    config = get_default_toy_config()
    if args.dataset == 'gmm':
        config.dataset.name = f"gmm_d{args.gmm_dim}_i{args.gmm_instance}"
        config.eval.evaluate_value = False
        config.sampling.population_size = 2000
    else:
        if args.dataset not in config.dataset.supported_datasets:
            raise ValueError(f"Unknown dataset {args.dataset}; supported: {config.dataset.supported_datasets} or 'gmm'")
        config.dataset.name = args.dataset
    if args.exp_name is not None: config.common.exp_name = args.exp_name
    if args.base_dir is not None: config.common.base_dir = args.base_dir
    config.common.seed = seed
    if args.device is not None: config.common.device = args.device
    config.common.use_progress_bar = not args.no_progress_bar
    if args.hf_home is not None: config.common.hf_home = args.hf_home
    if args.compute_budget is not None: config.sampling.compute_budget = sorted(args.compute_budget)
    if args.population_size is not None: config.sampling.population_size = args.population_size
    if args.potential_type is not None: config.fkd.potential_type = args.potential_type
    if args.adaptive_resampling is not None: config.fkd.adaptive_resampling = args.adaptive_resampling
    if args.resampling_steps is not None:
        config.fkd.resampling_t_start, config.fkd.resampling_t_end, config.fkd.resample_frequency = args.resampling_steps
    if args.mcts_batch_size is not None: config.mcts.batch_size = args.mcts_batch_size
    if args.no_value_eval: config.eval.evaluate_value = False
    if args.value_timesteps is not None: config.eval.value_eval_timesteps = args.value_timesteps
    config.guidance_scale = args.guidance_scale
    return process_toy_config(config)


def load_diffusion_model(config):
    """Loads the trained 2D diffusion model; a missing checkpoint is an error."""
    model_path = config.paths.diffusion_load_path
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(
            f"No diffusion checkpoint in {config.paths.checkpoint_dir}; train it with "
            f"`python -m src.models.diffusion_2d --dataset {config.dataset.name}` first.")
    model = DiffusionModel(
        input_dim=2,
        hidden_dim=config.model.hidden_dim,
        n_timesteps=config.model.n_timesteps,
        beta_start=config.model.beta_start,
        beta_end=config.model.beta_end,
        device=config.common.device
    )
    model.noise_predictor.load_state_dict(torch.load(model_path, map_location=config.common.device))
    logger.info(f"Loaded pre-trained model from {model_path}")
    model.noise_predictor.eval()
    return model


def load_task(config, args):
    """Returns (diffusion_model, task) where task provides gt_samples, gt_rewards and
    calculate_reward_diff, plus the GMM problem (or None)."""
    device = config.common.device
    if args.dataset == 'gmm':
        problem = GMMInverseProblem(dim=args.gmm_dim, instance_seed=args.gmm_instance,
                                    lmbda=args.reward_lambda, device=device)
        model = AnalyticGMMDiffusion(problem, n_timesteps=config.model.n_timesteps,
                                     beta_start=config.model.beta_start, beta_end=config.model.beta_end,
                                     device=device)
        task = GMMTask(problem, n_target=config.sampling.population_size,
                       target_seed=10_000 + config.common.seed, device=device)
        logger.info(f"GMM instance {args.gmm_instance}: d_x={problem.dim}, sigma_y={problem.sigma_y:.4f}, "
                    f"y={problem.y.tolist()}, top posterior masses={np.sort(problem.posterior_weights)[::-1][:5]}")
        return model, task, problem
    model = load_diffusion_model(config)
    task = get_2d_dataset(config, args.dataset, device=device)
    if task.gt_samples is None:
        raise FileNotFoundError(
            f"No target samples in {config.paths.checkpoint_dir}; generate them with "
            f"`python -m scripts.gen_2d_targets --dataset {args.dataset}` first.")
    return model, task, None


def sample_unguided(model, n):
    """n samples of p_theta,0: the full reverse chain t = T-1, ..., 0."""
    x = model.sample_initial_noise(n)
    for t in range(model.n_timesteps - 1, -1, -1):
        x, _ = model.p_sample(x, t)
    return x


def distribution_metrics(samples_np, task, problem, sw_projections):
    out = {'MMD': compute_mmd(task.gt_samples, samples_np) if samples_np.shape[1] == 2 else float('nan')}
    if problem is not None:
        out['SW'] = sliced_wasserstein(samples_np, task.gt_samples, n_projections=sw_projections)
        out['ModeTV'] = problem.mode_mass_tv(samples_np)
    return out


def dup_ess(samples_np, population_size):
    _, counts = np.unique(samples_np, axis=0, return_counts=True)
    w = counts / len(samples_np)
    return float(1.0 / np.sum(w ** 2)) / population_size


def snapshot_tree_stats(sampler, diffusion_model, problem):
    """tree_statistics of the current tree; on the GMM also the value error against the closed-form
    log M_t and the exact leaf distribution of sample_from_tree (E[r] - E_{pi*_0}[r], mode-mass TV, ESS)."""
    log_value = None
    if problem is not None:
        abar = diffusion_model.alphas_cumprod.double().cpu()
        log_value = lambda x, t: problem.log_soft_value(x, abar[t])
    stats, leaves = tree_statistics(sampler.meta_root, sampler.expand_timesteps, log_value)
    if problem is not None and leaves:
        x = torch.cat([n.x_t for n, _ in leaves]).float()
        lw = np.array([w for _, w in leaves])
        p = np.exp(lw - np.logaddexp.reduce(lw))
        r = problem.log_likelihood(x).double().cpu().numpy()
        mass = np.bincount(problem.map_component(x.cpu().numpy()), weights=p, minlength=len(problem.weights))
        stats['leaf_distribution'] = {'reward_gap': float(p @ r - problem.target_mean_reward()),
                                      'mode_tv': 0.5 * float(np.abs(mass - problem.posterior_weights).sum()),
                                      'ess': float(1.0 / np.sum(p ** 2))}
    return stats


def make_mcts_sampler(config, args, diffusion_model, reward_np, use_rollout):
    eps_init = args.eps_explore if args.eps_explore is not None else 0.1
    eps_final = args.eps_explore if args.eps_explore is not None else 0.05
    return MCTSDiffusionSampler(
        diffusion_model=diffusion_model,
        reward_fn=reward_np,
        expand_timesteps=config.mcts.expand_timesteps,
        num_nfes=max(config.sampling.compute_budget),
        num_inference_steps=config.sampling.num_inference_steps,
        eta=config.sampling.eta,
        device=config.common.device,
        exploration_const=config.mcts.exploration_const,
        progressive_width_const=config.mcts.progressive_width_const,
        progressive_width_alpha=config.mcts.progressive_width_alpha,
        initial_exploration_prob=eps_init,
        final_exploration_prob=eps_final,
        use_progress_bar=config.common.use_progress_bar,
        use_rollout=use_rollout,
        batch_size=config.mcts.batch_size,
        output_dir=config.paths.plot_dir,
        visualization_timesteps=list(config.mcts.expand_timesteps + [0]),
        pw_alpha=args.pw_alpha,
        pw_alpha_root=args.pw_alpha_root,
        keep_chain_states=not args.compact_tree,
    )


def run_mcts(config, args, dataset_name, diffusion_model, task, problem, reward, method_name="MCTS"):
    """Runs MCTS or DTS (MCTS-Rollout). Smaller NFE budgets are snapshots of one growing tree."""
    logger.info(f"--- Running {method_name} for {dataset_name} ---")
    metrics_data, value_errors, samples_by_nfe, nodes_by_timestep_all_iters = {}, {}, {}, {}
    use_rollout = ("-R" in method_name or "Rollout" in method_name)
    value_eval_timesteps = config.eval.value_eval_timesteps if config.eval.evaluate_value else None
    reward_np = lambda x: reward(x).detach().cpu().numpy()
    mcts_sampler = make_mcts_sampler(config, args, diffusion_model, reward_np, use_rollout)
    mcts_sampler.initialize_root_nodes()
    reset_peak_memory()
    build_time, build_reward_evals = 0.0, 0
    tree_stats = []

    for target_nfes in config.sampling.compute_budget:
        reward.reset()
        t0 = time.time()
        mcts_sampler.build_search_tree(num_nfes=target_nfes)
        build_time += time.time() - t0
        build_reward_evals += reward.n_evals
        t1 = time.time()
        samples, _, nodes_by_timestep = mcts_sampler.sample_from_tree(population_size=config.sampling.population_size)
        sample_time = time.time() - t1
        peak_mem = peak_memory_mb()
        samples_np = samples.cpu().numpy()
        rewards_np = task.calculate_reward_diff(samples).detach().cpu().numpy()

        m = {
            'NFEs': target_nfes, 'ActualNFEs': mcts_sampler.nfes,
            'MeanReward': rewards_np.mean(), 'GT_MeanReward': task.gt_rewards.mean(),
            'ESS': dup_ess(samples_np, config.sampling.population_size), 'Distinct': distinct_count(samples_np),
            'BuildTime': build_time, 'SampleTime': sample_time, 'Time': build_time + sample_time,
            'RewardEvals': build_reward_evals, 'PeakMemMB': peak_mem,
        }
        m.update(distribution_metrics(samples_np, task, problem, args.sw_projections))
        metrics_data[target_nfes] = m
        samples_by_nfe[target_nfes] = samples_np
        nodes_by_timestep_all_iters[target_nfes] = nodes_by_timestep
        logger.info(f"  {method_name} NFE {target_nfes} (actual {mcts_sampler.nfes}): " +
                    ", ".join(f"{k}={v:.4g}" for k, v in m.items() if isinstance(v, (float, np.floating))))
        if args.tree_stats:
            t2 = time.time()
            st = snapshot_tree_stats(mcts_sampler, diffusion_model, problem)
            tree_stats.append({'NFEs': target_nfes, 'ActualNFEs': mcts_sampler.nfes, 'BuildTime': build_time,
                               'StatsTime': time.time() - t2, **st})
            logger.info(f"  tree: roots {st['n_roots']}, leaves {st['n_leaves']}, branch depth {st['leaf_branch_depth']}")
            Path(config.paths.metrics_dir, f"tree_stats_{method_name}_seed{config.common.seed}.json").write_text(
                json.dumps(tree_stats, indent=1))

    if config.eval.evaluate_value:
        for target_nfes in config.sampling.compute_budget:
            value_errors[target_nfes] = evaluate_mcts_value_estimates(
                nodes_by_timestep=nodes_by_timestep_all_iters[target_nfes],
                diffusion_model=diffusion_model, reward_fn=reward_np,
                timesteps=value_eval_timesteps,
                samples_per_timestep=config.eval.value_samples_per_timestep,
                num_trajectories=config.eval.value_num_trajectories,
                device=config.common.device
            )

    sample_metrics_df = pd.DataFrame([{'Method': method_name, 'N': n, **m} for n, m in metrics_data.items()])
    value_metrics_df = pd.DataFrame([{'Method': method_name, 'N': n, 'Timestep': t, 'ValueError': np.mean(err), 'ValueStd': np.var(err, ddof=1)}
                                     for n, errs in value_errors.items() for t, err in errs.items()])
    return sample_metrics_df, value_metrics_df, samples_by_nfe


def run_fkd(config, args, dataset_name, diffusion_model, task, problem, reward, method_name="SMC"):
    """Runs SMC/FK, DPS-RG ('Guidance'), TDS ('SMC-Guidance'), DAS ('SMC-Guidance-Tempering') or
    SMC-Rollout. Every NFE point is a separate run with K = NFE // (NFE per particle) particles;
    SMC methods draw the evaluation samples with replacement by their final importance weights."""
    logger.info(f"--- Running {method_name} for {dataset_name} ---")
    metrics_data, value_errors, samples_by_nfe, history_by_nfe = {}, {}, {}, {}

    use_guidance = "guidance" in method_name.lower()
    use_smc = "smc" in method_name.lower()
    use_rollout = "rollout" in method_name.lower()
    use_tempering = "tempering" in method_name.lower()
    value_eval_timesteps = config.eval.value_eval_timesteps if config.eval.evaluate_value else None

    fkd_config = {
        'potential_type': config.fkd.potential_type,
        'lmbda': config.fkd.lambda_param,
        'adaptive_resampling': config.fkd.adaptive_resampling,
        'resample_frequency': config.fkd.resample_frequency,
        'resampling_t_start': config.fkd.resampling_t_start,
        'resampling_t_end': config.fkd.resampling_t_end,
        'reward_min_value': config.fkd.reward_min_val,
    }

    num_steps = config.sampling.num_inference_steps
    nfe_per_particle = num_steps * (2 if use_guidance else 1)   # forward, plus backward with guidance
    if use_smc and use_rollout and not use_guidance:
        nfe_per_particle += sum(num_steps - i for i in range(config.fkd.resampling_t_start, config.fkd.resampling_t_end + 1, config.fkd.resample_frequency))
    logger.info(f"NFEs per particle for {method_name}: {nfe_per_particle}")

    return_all = config.eval.evaluate_value and use_smc and not use_guidance
    n_eval = config.sampling.population_size
    for nfe_target in config.sampling.compute_budget:
        n_particles = max(1, nfe_target // nfe_per_particle)
        sampler = SMCDiffusion2DSampler(
            diffusion_model=diffusion_model,
            use_smc=use_smc, use_guidance=use_guidance, use_tempering=use_tempering,
            guidance_scale=config.guidance_scale, reward_fn=reward, fkd_config=fkd_config,
            device=config.common.device, dataset_name=dataset_name,
            use_progress_bar=config.common.use_progress_bar, use_rollout=use_rollout,
            das_gamma=args.das_gamma,
        )
        reward.reset()
        reset_peak_memory()
        t0 = time.time()
        try:
            out = sampler.sample(num_particles=n_particles, eta=config.sampling.eta, return_all_particles=return_all)
            if use_smc and not torch.isfinite(out['log_weights']).any():
                raise SMCDivergedError("SMC population has no particle with a finite weight (diverged).")
        except SMCDivergedError as e:
            logger.error(f"  {method_name} NFE {nfe_target} ({n_particles} particles): {e}")
            metrics_data[nfe_target] = {'NFEs': nfe_target, 'ActualParticles': n_particles, 'Diverged': 1,
                                        'Time': time.time() - t0}
            continue
        run_time = time.time() - t0
        run_reward_evals = reward.n_evals
        peak_mem = peak_memory_mb()
        particles = out['samples']

        if use_smc:
            idx = torch.multinomial(torch.softmax(out['log_weights'].double(), dim=0), n_eval, replacement=True)
        elif n_particles >= n_eval:
            idx = torch.randperm(n_particles, device=particles.device)[:n_eval]
        else:
            idx = torch.randint(0, n_particles, (n_eval,), device=particles.device)
        final_samples = particles[idx.to(particles.device)]
        samples_np = final_samples.cpu().numpy()
        rewards_np = task.calculate_reward_diff(final_samples).detach().cpu().numpy()

        w = torch.softmax(out['log_weights'].double(), dim=0)
        m = {
            'NFEs': nfe_target, 'ActualNFEs': sampler.nfes, 'ActualParticles': n_particles,
            'MeanReward': rewards_np.mean(), 'GT_MeanReward': task.gt_rewards.mean(),
            'ESS': dup_ess(samples_np, n_eval), 'Distinct': distinct_count(samples_np),
            'FinalWeightESS': float(1.0 / (w.pow(2).sum() * n_particles)),
            'InvalidParticles': int((~torch.isfinite(out['log_weights'])).sum()) if use_smc
                                else int((~torch.isfinite(particles).all(dim=-1)).sum()),
            'Time': run_time, 'RewardEvals': run_reward_evals, 'PeakMemMB': peak_mem,
        }
        m.update(distribution_metrics(samples_np, task, problem, args.sw_projections))
        metrics_data[nfe_target] = m
        samples_by_nfe[nfe_target] = samples_np
        if return_all:
            history_by_nfe[nfe_target] = out['history']
        logger.info(f"  {method_name} NFE {nfe_target} ({n_particles} particles, actual {sampler.nfes}): " +
                    ", ".join(f"{k}={v:.4g}" for k, v in m.items() if isinstance(v, (float, np.floating))))

    if return_all:
        reward_np = lambda x: reward(x).detach().cpu().numpy()
        for nfe_target, history in history_by_nfe.items():
            value_errors[nfe_target] = evaluate_smc_value_estimates(
                smc_particles=history, diffusion_model=diffusion_model, reward_fn=reward_np,
                timesteps=value_eval_timesteps,
                samples_per_timestep=config.eval.value_samples_per_timestep,
                num_trajectories=config.eval.value_num_trajectories,
                device=config.common.device
            )

    sample_metrics_df = pd.DataFrame([{'Method': method_name, 'N': n, **m} for n, m in metrics_data.items()])
    value_metrics_df = pd.DataFrame([{'Method': method_name, 'N': n, 'Timestep': t, 'ValueError': np.mean(err), 'ValueStd': np.var(err, ddof=1)}
                                     for n, errs in value_errors.items() for t, err in errs.items()])
    return sample_metrics_df, value_metrics_df, samples_by_nfe


def aggregate_and_save_summary(metrics_dir, evaluate_value):
    """Concatenates the per-seed CSVs and writes mean/std/sem across seeds."""
    metrics_dir = Path(metrics_dir)
    for prefix, group_cols, raw_name, agg_name in [
        ("sample_metrics_", ['Dataset', 'Method', 'N'], "all_runs_raw_summary.csv", "all_runs_aggregated_summary.csv"),
        ("value_metrics_", ['Dataset', 'Method', 'N', 'Timestep'], "all_runs_raw_value_summary.csv", "all_runs_aggregated_value_summary.csv"),
    ]:
        if prefix == "value_metrics_" and not evaluate_value:
            continue
        files = sorted(metrics_dir.glob(f"{prefix}*.csv"))
        if not files:
            logger.info(f"No {prefix}*.csv files in {metrics_dir}")
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df.to_csv(metrics_dir / raw_name, index=False)
        metric_cols = [c for c in df.columns if c not in group_cols + ['Seed'] and pd.api.types.is_numeric_dtype(df[c])]
        agg = df.groupby(group_cols)[metric_cols].agg(['mean', 'std'])
        agg.columns = ['_'.join(c) for c in agg.columns]
        agg['Count'] = df.groupby(group_cols).size()
        for c in metric_cols:
            agg[f'{c}_sem'] = agg[f'{c}_std'] / np.sqrt(agg['Count'])
        agg.reset_index().to_csv(metrics_dir / agg_name, index=False)
        logger.info(f"Saved {metrics_dir / agg_name}")


def main():
    args = parse_script_args()
    base_config = build_config(args, args.seed if args.seed is not None else get_default_toy_config().common.seed)
    first_seed = base_config.common.seed
    seeds_to_run = [first_seed + k for k in range(args.num_seeds)]
    logger.info("\n" + pprint.pformat(asdict(base_config)))
    logger.info("\n" + pprint.pformat(vars(args)))

    enabled_methods = []
    if args.do_mcts: enabled_methods.append("MCTS")
    if args.do_mcts_rollout: enabled_methods.append("MCTS-Rollout")
    if args.do_guidance: enabled_methods.append("Guidance")
    if args.do_smc: enabled_methods.append("SMC")
    if args.do_smc_guidance: enabled_methods.append("SMC-Guidance")
    if args.do_smc_guidance_tempering: enabled_methods.append("SMC-Guidance-Tempering")
    if args.do_smc_rollout: enabled_methods.append("SMC-Rollout")
    if not enabled_methods:
        logger.warning("No methods specified via command-line flags. Exiting.")
        return
    logger.info(f"Seeds {seeds_to_run}; methods {enabled_methods}")

    for seed in seeds_to_run:
        config = build_config(args, seed)
        dataset_name = config.dataset.name
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        diffusion_model, task, problem = load_task(config, args)
        reward = RewardCounter(task.calculate_reward_diff, scale=args.reward_lambda)
        metrics_dir = Path(config.paths.metrics_dir)
        extra = {}

        prior_samples = sample_unguided(diffusion_model, config.sampling.population_size)
        if problem is not None:
            q0 = problem.sample_prior(config.sampling.population_size, seed=20_000 + seed)
            extra['discretization_SW'] = sliced_wasserstein(prior_samples.cpu().numpy(), q0, n_projections=args.sw_projections)
            # same-size Monte-Carlo floor of discretization_SW: two exact prior draws
            extra['prior_split_SW'] = sliced_wasserstein(problem.sample_prior(config.sampling.population_size, seed=40_000 + seed),
                                                         q0, n_projections=args.sw_projections)
            target_split = problem.sample_target(config.sampling.population_size, seed=30_000 + seed)
            extra['target_split_SW'] = sliced_wasserstein(target_split, task.gt_samples, n_projections=args.sw_projections)
            extra['gmm'] = {'dim': problem.dim, 'instance': args.gmm_instance, 'sigma_y': problem.sigma_y,
                            'y': problem.y.tolist(), 'posterior_weights': problem.posterior_weights.tolist()}
            logger.info(f"SW(p_theta,0, q_0) = {extra['discretization_SW']:.4f} (two exact q_0 draws: {extra['prior_split_SW']:.4f}); "
                        f"SW between two exact target draws = {extra['target_split_SW']:.4f}")
        write_run_meta(config.paths.run_dir, config, args, extra=extra, name=f"run_meta_seed{seed}.json")

        all_samples = {}
        for method in enabled_methods:
            run_fn = run_mcts if method.startswith("MCTS") else run_fkd
            sample_df, value_df, samples_by_nfe = run_fn(config, args, dataset_name, diffusion_model, task, problem,
                                                         reward, method_name=method)
            for df in (sample_df, value_df):
                if not df.empty:
                    df['Dataset'] = dataset_name
                    df['Seed'] = seed
            if not sample_df.empty:
                sample_df.to_csv(metrics_dir / f"sample_metrics_{dataset_name}_{method}_seed{seed}.csv", index=False)
            if not value_df.empty:
                value_df.to_csv(metrics_dir / f"value_metrics_{dataset_name}_{method}_seed{seed}.csv", index=False)
            all_samples[method] = samples_by_nfe

        data_save_dir = Path(config.paths.image_dir)
        if not args.no_save_samples:
            np.savez_compressed(data_save_dir / f"samples_{dataset_name}_seed{seed}.npz",
                                **{f"{m}__N{n}": s for m, d in all_samples.items() for n, s in d.items()},
                                target=task.gt_samples, prior=prior_samples.cpu().numpy())
        with open(data_save_dir / f"all_method_samples_{dataset_name}_seed{seed}.pkl", "wb") as f:
            pickle.dump(all_samples, f)
        logger.info(f"===== Finished {dataset_name} seed {seed} =====")

    aggregate_and_save_summary(base_config.paths.metrics_dir, base_config.eval.evaluate_value)
    logger.info(f"=== Experiment Complete. Run directory: {base_config.paths.run_dir} ===")


if __name__ == '__main__':
    main()
