import os
import json
import logging
import random
import numpy as np
import torch
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime


logger = logging.getLogger(__name__)

# Helper function for default paths relative to home
def default_path(subpath: str) -> str:
    return str(Path.home() / subpath)

def env_or_default_path(env_var: str, subpath: str) -> str:
    """Path from `env_var` if set, otherwise `~/<subpath>`."""
    return os.environ.get(env_var) or default_path(subpath)

def cache_path(subdir: str) -> str:
    """Directory for downloaded reward / feature models (CLIP, ImageReward, aesthetic head).
    Root is $DTS_CACHE_DIR, defaulting to ~/scratch/.cache."""
    return str(Path(env_or_default_path("DTS_CACHE_DIR", "scratch/.cache")).expanduser() / subdir)

def seed_experiment(seed):
    """Set the seed for reproducibility in PyTorch runs.

    Args:
        seed (int): The seed number.
    """
    # Set the seed for Python's 'random' module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)

    # If using CUDA (PyTorch with GPU support)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --- Path Management ---
@dataclass
class PathsConfig:
    """Holds all derived paths for an experiment run."""
    # Base Dirs (Derived)
    output_dir: Optional[str] = None    # e.g., ./experiments/toy/results/gaussian_mixture
    checkpoint_dir: Optional[str] = None # e.g., ./experiments/toy/checkpoints/gaussian_mixture (or None)

    # Run Specific Dirs (Derived under output_dir)
    run_dir: Optional[str] = None           # e.g., ./experiments/toy/results/gaussian_mixture/{exp_name}_{timestamp}
    plot_dir: Optional[str] = None          # e.g., {run_dir}/plots
    metrics_dir: Optional[str] = None       # e.g., {run_dir}/metrics
    image_dir: Optional[str] = None         # e.g., {run_dir}/images

    # Specific Files (Derived)
    summary_metrics_path: Optional[str] = None # e.g., {metrics_dir}/summary_metrics.csv
    config_path: Optional[str] = None    # e.g., {run_dir}/config.json

    # Resolved Checkpoint Paths (Derived based on inputs and checkpoint_dir)
    diffusion_load_path: Optional[str] = None  # Resolved path or HF ID for diffusion model
    classifier_load_path: Optional[str] = None # Resolved path for classifier model

# --- Common Settings ---
@dataclass
class CommonConfig:
    """Common arguments for experiment setup, paths, logging."""
    # base_dir: str = default_path("scratch/mcts/tamia_exp") # Top-level directory (e.g., ./experiments)
    base_dir: str = field(default_factory=lambda: env_or_default_path("DTS_BASE_DIR", "mcts/experiments"))
    exp_name: Optional[str] = None # If None, derived from dataset/task
    seed: Optional[int] = 42
    device: str = "cuda"
    push_to_wandb: bool = False
    hf_home: Optional[str] = field(default_factory=lambda: env_or_default_path("HF_HOME", "scratch/.cache/huggingface"))
    use_progress_bar: bool = True

# --- Base classes for method configs --- 
@dataclass
class BaseMCTSConfig:
    enabled: bool = False

@dataclass
class BaseFKDConfig:
    enabled: bool = False

# --- Toy Task Specific Config ---

@dataclass
class ToyDatasetConfig:
    name: str = "gaussian_mixture" # Could be a list if running multiple
    supported_datasets: List[str] = field(default_factory=lambda: ["gaussian_mixture", "checkerboard", "spiral", 
                                                                   "two_moons", "checkerboard_stripe", "two_moons_bridge", "banana_rosenbrock"])

@dataclass
class ToyModelConfig:
    # Diffusion model settings common across datasets
    hidden_dim: int = 128
    n_timesteps: int = 100
    beta_start: float = 1e-3
    beta_end: float = 0.07
    ckpt_name: str = "diffusion_model.pt"

@dataclass
class ToySamplingConfig:
    num_inference_steps: int = 100 # Should match model_n_timesteps usually
    compute_budget: List[int] = field(default_factory=lambda: [1000, 5000, 10000, 50000, 100000])
    population_size: int = 1000 # Number of final samples
    eta: float = 1.0

@dataclass
class ToyMCTSConfig(BaseMCTSConfig):
    expand_timesteps: List[int] = field(default_factory=lambda: [100, 80, 60, 40, 20])
    # expand_timesteps: List[int] = field(default_factory=lambda: list(range(100, 0, -1)))
    use_rollout: bool = False
    exploration_const: float = 1.0
    progressive_width_const: float = 2.0
    progressive_width_alpha: float = 0.5
    batch_size: int = 1 # MCTS internal batch size

@dataclass
class ToyFKDConfig(BaseFKDConfig):
    particles: List[int] = None # defined in run script
    potential_type: str = "diff" # choices=['max', 'add', 'diff']
    use_rollout: bool = False
    lambda_param: float = 1.0
    adaptive_resampling: bool = True
    resample_frequency: int = 20
    resampling_t_start: int = 20
    resampling_t_end: int = 80
    reward_min_val: float = 0.0 # Updated in processing

@dataclass
class ToyEvalConfig:
    # MMD/Reward Eval handled implicitly with population_size
    # Value Evaluation Settings
    evaluate_value: bool = True # Flag to run value evaluation
    value_num_trajectories: int = 1000 # For ground truth estimation
    value_samples_per_timestep: int = 10 # Num samples to check value error at each step
    value_eval_timesteps: List[int] = field(default_factory=lambda: [80, 60, 40, 20, 0])

@dataclass
class ToyConfig:
    task_type: str = field(default="toy", init=False)
    common: CommonConfig = field(default_factory=CommonConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    dataset: ToyDatasetConfig = field(default_factory=ToyDatasetConfig)
    model: ToyModelConfig = field(default_factory=ToyModelConfig)
    sampling: ToySamplingConfig = field(default_factory=ToySamplingConfig)
    eval: ToyEvalConfig = field(default_factory=ToyEvalConfig)
    # Method configs (Reusing definitions from earlier)
    mcts: ToyMCTSConfig = field(default_factory=ToyMCTSConfig)
    fkd: ToyFKDConfig = field(default_factory=ToyFKDConfig) # Note: Renamed from smc


# --- Image Task Specific Config ---
DATASETS_INFO = {
    'mnist': {
        'splits': ['train', 'test'],
        'image_size': 32,
        'channels': 1,
        'num_classes': 10,
        'x_tensor': 'images',
        'y_tensor': 'labels',
        'block_out_channels': (32, 64, 128, 256),
        'classifier_model': 'cnn',
        'classifier_depth': 10,
        'fid_inception_block': 768,
    },
    'cifar10': {
        'splits': ['train', 'test'],
        'image_size': 32,
        'channels': 3,
        'num_classes': 10,
        'x_tensor': 'images',
        'y_tensor': 'labels',
        'block_out_channels': (32, 64, 128, 256),
        'classifier_model': 'resnet',
        'classifier_depth': 18,
        'fid_inception_block': 768,
    },
}

@dataclass
class ImageDatasetConfig:
    name: str = "mnist"
    data_path: str = field(default_factory=lambda: env_or_default_path("DTS_DATA_DIR", "scratch/.cache/deeplake/data"))
    splits: List[str] = field(default_factory=lambda: ['train', 'test'])
    image_size: int = 32
    channels: int = 1
    num_classes: int = 10
    x_tensor: str = 'images'
    y_tensor: str = 'labels'
    block_out_channels: Optional[Tuple[int, ...]] = None
    batch_size: int = 128 # Batch size for *training* (classifier or diffusion)
    workers: int = 4
    sample_classes: Union[List[int], str] = field(default_factory=lambda: list(range(10)))
    class_name: Optional[str] = None

@dataclass
class ImageModelConfig: # For classifier model
    model_id: Optional[str] = None # Path or HF ID for DDIM pipeline
    classifier_path: Optional[str] = None # Path to classifier state_dict
    classifier_model: Optional[str] = None
    classifier_depth: Optional[int] = None
    classifier_pretrained: bool = False
    classifier_epochs: int = 100
    classifier_patience: int = 20
    classifier_default_ckpt_name: str = "classifier.pth"

@dataclass
class ImageSamplingConfig:
    num_inference_steps: int = 50
    compute_budget: List[int] = field(default_factory=lambda: [1000, 5000, 10000, 50000, 100000])
    population_size: int = 5000 # Final # samples to evaluate
    eval_batch_size: int = 128 # Batch size for FID/metrics
    gen_batch_size: int = 16 # Final # samples to save images
    eta: float = 1.0

@dataclass
class ImageTrainingConfig: # For diffusion model training (used by diffusion.py)
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    save_image_epochs: int = 20
    save_model_epochs: int = 50
    mixed_precision: str = "no"
    num_epochs: int = 100

@dataclass
class ImageMCTSConfig(BaseMCTSConfig):
    expand_timesteps: List[int] = field(default_factory=lambda: [50, 40, 30, 20, 10])
    # expand_timesteps: List[int] = field(default_factory=lambda: list(range(50, 0, -1)))
    use_rollout: bool = True
    exploration_const: float = 1.0
    progressive_width_const: float = 2.0
    progressive_width_alpha: float = 0.5
    batch_size: int = 1 # MCTS internal batch size

@dataclass
class ImageFKDConfig(BaseFKDConfig):
    particles: List[int] = None # defined in run script
    potential_type: str = "diff" # choices=['reward', 'max', 'add', 'diff']
    use_rollout: bool = False
    lambda_param: float = 1.0
    adaptive_resampling: bool = False
    resample_frequency: int = 10
    resampling_t_start: int = 10
    resampling_t_end: int = 40
    reward_min_val: float = 0.0 # Updated in processing
    batch_size: int = 128 # FKD internal batch size

@dataclass
class ImageEvalConfig:
    num_fid_samples: int = 10000
    fid_inception_block: int = 768

@dataclass
class ImageConfig:
    task_type: str = field(default="img", init=False)
    common: CommonConfig = field(default_factory=CommonConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    dataset: ImageDatasetConfig = field(default_factory=ImageDatasetConfig)
    model: ImageModelConfig = field(default_factory=ImageModelConfig)
    sampling: ImageSamplingConfig = field(default_factory=ImageSamplingConfig)
    training: ImageTrainingConfig = field(default_factory=ImageTrainingConfig)
    mcts: ImageMCTSConfig = field(default_factory=ImageMCTSConfig)
    fkd: ImageFKDConfig = field(default_factory=ImageFKDConfig)
    eval: ImageEvalConfig = field(default_factory=ImageEvalConfig)


# --- SD Task Specific Config ---
@dataclass
class SDDatasetConfig:
    prompts: List[str] = field(default_factory=list)    # List of prompts to use for generation OR
    prompts_file: Optional[str] = None                  # Path to JSON file containing prompts
    negative_prompt: str = ""

@dataclass
class SDModelConfig:
    sd_model_id: str = "runwayml/stable-diffusion-v1-5"

@dataclass
class SDSamplingConfig:
    num_inference_steps: int = 50
    compute_budget: List[int] = field(default_factory=lambda: [10, 50, 100])
    population_size: int = 16 # Final # samples per prompt to save/evaluate
    guidance_scale: float = 7.5
    eta: float = 1.0 # eta=1.0 for DDIM scheduler

@dataclass
class SDMCTSConfig(BaseMCTSConfig):
    expand_timesteps: List[int] = field(default_factory=lambda: [50, 40, 30, 20, 10])
    # expand_timesteps: List[int] = field(default_factory=lambda: list(range(50, 0, -1)))
    use_rollout: bool = False
    exploration_const: float = 1.0
    progressive_width_const: float = 2.0
    progressive_width_alpha: float = 0.0
    batch_size: int = 1 # MCTS internal batch size (fixed to 1 for torch.compile)

@dataclass
class SDFKDConfig(BaseFKDConfig):
    particles: List[int] = None # defined in run script
    potential_type: str = "diff" # choices=['reward', 'max', 'add', 'diff']
    use_rollout: bool = False
    lambda_param: float = 1.0
    adaptive_resampling: bool = False
    resample_frequency: int = 10
    resampling_t_start: int = 10
    resampling_t_end: int = 40
    reward_min_val: float = 0.0 # Updated in processing
    batch_size: int = 20 # FKD internal batch size


@dataclass
class SDGreedySearchConfig:
    enabled: bool = False
    use_rollout: bool = False
    search_frequency: int = 5
    search_t_start: int = 0
    search_t_end: int = 45
    batch_size: int = 20 # Greedy internal batch size
    variant_name_for_path: Optional[str] = "greedy_code" # Added for naming consistency

@dataclass
class SDEvalConfig:
    # reward_fns: List[str] = field(default_factory=lambda: ["ImageReward"])
    reward_fns: List[str] = field(default_factory=lambda: ["ImageReward"])
    eval_batch_size: int = 64 # Batch size for reward function evaluation

@dataclass
class SDConfig:
    task_type: str = field(default="sd", init=False)
    common: CommonConfig = field(default_factory=CommonConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    dataset: SDDatasetConfig = field(default_factory=SDDatasetConfig)
    model: SDModelConfig = field(default_factory=SDModelConfig)
    sampling: SDSamplingConfig = field(default_factory=SDSamplingConfig)
    mcts: SDMCTSConfig = field(default_factory=SDMCTSConfig)
    fkd: SDFKDConfig = field(default_factory=SDFKDConfig)
    greedy: SDGreedySearchConfig = field(default_factory=SDGreedySearchConfig)
    eval: SDEvalConfig = field(default_factory=SDEvalConfig)


def get_default_toy_config():
    return ToyConfig()

def get_default_image_config():
    return ImageConfig()

def get_default_sd_config():
    return SDConfig()

# --- Finalize Config Functions ---
def _process_path_setup(config, class_name=None):
    """Sets up the run directory structure and populates paths."""
    # Determine Base Dirs
    task_type = config.task_type
    base_dir = Path(config.common.base_dir)
    if not base_dir.is_absolute():
        base_dir = Path.cwd() / base_dir

    task_base_dir = base_dir / task_type
    task_base_dir.mkdir(parents=True, exist_ok=True)

    # Determine dataset name for subfolders
    ds_name_for_paths = getattr(config.dataset, 'name', getattr(config, 'dataset', 'unknown_dataset'))

    # Checkpoint directory
    task_checkpoint_base = task_base_dir / "checkpoints"
    create_ckpt_results_dir = False
    if task_type in ["toy", "img"]:
        create_ckpt_results_dir = True

    if create_ckpt_results_dir:
        specific_ckpt_dir = task_checkpoint_base / ds_name_for_paths
        specific_ckpt_dir.mkdir(parents=True, exist_ok=True)
        config.paths.checkpoint_dir = str(specific_ckpt_dir)
        logger.info(f"Checkpoint directory set to: {config.paths.checkpoint_dir}")
    else:
        config.paths.checkpoint_dir = None
        logger.info(f"Local checkpoint directory skipped for task '{task_type}'.")

    # Results directory
    if create_ckpt_results_dir:
        task_results_base = task_base_dir / "results_tree_viz" / ds_name_for_paths
        task_results_base.mkdir(parents=True, exist_ok=True)
    else:
        task_results_base = task_base_dir      # for SD, don't create a separate results dir
    config.paths.output_dir = str(task_results_base)

    # Run-Specific Directory
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_final = task_results_base
    if class_name is not None:
        run_name_prefix = class_name
        if config.common.exp_name:
            run_name_prefix = f"{run_name_prefix}_{config.common.exp_name}"
        # run_dir_final = task_results_base / f"{run_name_prefix}_{timestamp}"
        run_dir_final = task_results_base / f"{run_name_prefix}"
    elif config.common.exp_name:
        run_dir_final = task_results_base / config.common.exp_name

    # Create Run Subdirectories
    run_dir_final.mkdir(parents=True, exist_ok=True)
    plot_dir = run_dir_final / "plots"
    plot_dir.mkdir(exist_ok=True)
    metrics_dir = run_dir_final / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    image_dir = run_dir_final / "images"
    image_dir.mkdir(exist_ok=True)

    # Populate PathsConfig
    config.paths.run_dir = str(run_dir_final)
    config.paths.plot_dir = str(plot_dir)
    config.paths.metrics_dir = str(metrics_dir)
    config.paths.image_dir = str(image_dir)
    config.paths.summary_metrics_path = str(metrics_dir / "summary_metrics.csv")
    config.paths.config_path = str(run_dir_final / "config.json")


def _resolve_checkpoint_path(user_path_or_id, task_ckpt_dir, ckpt_filename):
    """Resolves a user-provided path/ID to a full path or returns the ID."""
    if not user_path_or_id:
        # User didn't provide a path, check the default location
        if task_ckpt_dir and ckpt_filename:
            default_path = Path(task_ckpt_dir) / ckpt_filename
            return str(default_path) if default_path.exists() else None
        return None

    # Check if user provided path exists
    path_obj = Path(user_path_or_id)
    if path_obj.is_absolute() and path_obj.exists():
        return str(path_obj)
    if task_ckpt_dir and (task_ckpt_dir_path := Path(task_ckpt_dir) / path_obj).exists():
        return str(task_ckpt_dir_path)

    # Check relative to current working directory
    if not path_obj.is_absolute() and path_obj.exists():
        logger.debug(f"Resolved user path relative to CWD: {path_obj.resolve()}")
        return str(path_obj.resolve())

    return user_path_or_id


def process_toy_config(config):
    """Post-processes ToyConfig object."""
    seed_experiment(config.common.seed)
    _process_path_setup(config)

    # Set HF_HOME environment variable if specified
    if config.common.hf_home:
        hf_home_path = Path(config.common.hf_home).expanduser().resolve()
        os.environ['HF_HOME'] = str(hf_home_path)

    # Resolve diffusion checkpoint path
    config.paths.diffusion_load_path = _resolve_checkpoint_path(
        None,
        config.paths.checkpoint_dir,
        'diffusion_model.pt'
    )
    logger.info(f"Diffusion model path: {config.paths.diffusion_load_path}")

    if config.fkd.enabled:
        config.fkd.reward_min_val = float('-inf') if config.fkd.potential_type == "max" else 0.0

    with open(config.paths.config_path, 'w') as f:
        json.dump(asdict(config), f, indent=4)
    logger.info(f"Saved finalized toy config to {config.paths.config_path}")

    return config


def process_image_config(config):
    """Post-processes ImageConfig object."""
    seed_experiment(config.common.seed)

    # Set HF_HOME environment variable if specified
    if config.common.hf_home:
        hf_home_path = Path(config.common.hf_home).expanduser().resolve()
        os.environ['HF_HOME'] = str(hf_home_path)

    sample_classes = config.dataset.sample_classes
    dataset_name = config.dataset.name
    num_classes = config.dataset.num_classes
    if len(sample_classes) == 1:
        class_name = sample_classes[0]
        if class_name.isdigit():
            sample_classes = [int(class_name)]
            assert 0 <= sample_classes[0] < num_classes
        else:
            # Handle named classes or multi-label datasets
            if dataset_name == 'mnist' and class_name == 'odd':
                sample_classes = [1, 3, 5, 7, 9]
            elif dataset_name == 'mnist' and class_name == 'even':
                sample_classes = [0, 2, 4, 6, 8]
            elif class_name == 'all':
                sample_classes = list(range(num_classes))
            else:
                logger.warning(f"Class name {class_name} not recognized for dataset '{dataset_name}'. Defaulting to 'all'.")
                class_name = 'all'
                sample_classes = list(range(num_classes))
    else:
        class_name = 'subset'
        sample_classes = [int(c) for c in sample_classes]

    config.dataset.sample_classes = sample_classes
    config.dataset.class_name = class_name
    _process_path_setup(config, class_name)

    # Dataset specific processing
    ds_name = config.dataset.name
    if ds_name in DATASETS_INFO:
         info = DATASETS_INFO[ds_name]
         config.dataset.splits = info['splits']
         config.dataset.image_size = info['image_size']
         config.dataset.channels = info['channels']
         config.dataset.num_classes = info['num_classes']
         config.dataset.x_tensor = info['x_tensor']
         config.dataset.y_tensor = info['y_tensor']
         config.eval.fid_inception_block = info['fid_inception_block']
         config.dataset.block_out_channels = info.get('block_out_channels', None)

         # Set model defaults if not specified by user
         if config.model.classifier_model is None:
             config.model.classifier_model = info['classifier_model']
         if config.model.classifier_depth is None:
              config.model.classifier_depth = info['classifier_depth']
    else:
         logger.warning(f"Dataset '{ds_name}' not in DATASETS_INFO. Using defaults.")

    # Resolve diffusion model path/ID
    user_diffusion_path = config.model.model_id
    task_ckpt_dir = config.paths.checkpoint_dir
    default_diffusion_dir = f"{config.dataset.name}_unet_{config.dataset.image_size}px" if task_ckpt_dir else None
    resolved_user_path = _resolve_checkpoint_path(user_diffusion_path, task_ckpt_dir, None)

    if resolved_user_path is None:
        # User provided nothing. Check default.
        if config.dataset.name == 'cifar10':
            final_diffusion_path = "google/ddpm-cifar10-32"
        else:
            final_diffusion_path = _resolve_checkpoint_path(None, task_ckpt_dir, default_diffusion_dir)
    elif Path(resolved_user_path).exists():
        # User path resolved to an existing local directory/file.
        final_diffusion_path = resolved_user_path
    else:
        # User path provided, but doesn't exist locally. Assume it's an HF ID.
        logger.info(f"User-provided diffusion path '{resolved_user_path}' does not exist locally. Assuming it is an HF ID or remote path.")
        final_diffusion_path = user_diffusion_path

    config.paths.diffusion_load_path = final_diffusion_path

    # Resolve classifier path/ID
    user_classifier_path = config.model.classifier_path
    default_classifier_file = f"{config.dataset.name}_{config.model.classifier_model}_best.pt" if task_ckpt_dir else None
    config.paths.classifier_load_path = _resolve_checkpoint_path(user_classifier_path, task_ckpt_dir, default_classifier_file)

    if config.fkd.enabled:
        config.fkd.reward_min_val = float('-inf') if config.fkd.potential_type == "max" else 0.0

    with open(config.paths.config_path, 'w') as f:
        json.dump(asdict(config), f, indent=4)
    logger.info(f"Saved finalized image config to {config.paths.config_path}")

    return config


def process_sd_config(config):
    """Post-processes SDConfig object."""
    seed_experiment(config.common.seed)

    # Set HF_HOME environment variable if specified
    if config.common.hf_home:
        hf_home_path = Path(config.common.hf_home).expanduser().resolve()
        os.environ['HF_HOME'] = str(hf_home_path)

    prompts_name = Path(config.dataset.prompts_file).stem if config.dataset.prompts_file else "user_prompts"
    _process_path_setup(config, prompts_name)

    # Diffusion model is always loaded from HF ID for SD
    config.paths.diffusion_load_path = config.model.sd_model_id
    logger.info(f"SD model ID set to: {config.paths.diffusion_load_path}")
    config.paths.classifier_load_path = None

    # Prompts setup
    if config.dataset.prompts_file and not config.dataset.prompts:
        with open(config.dataset.prompts_file, 'r') as f:
            prompts_data = json.load(f)
        if isinstance(prompts_data, list): config.dataset.prompts = prompts_data
        elif isinstance(prompts_data, dict) and 'prompts' in prompts_data: config.dataset.prompts = prompts_data['prompts']
        else: raise ValueError("Prompts file structure not recognized.")
        logger.info(f"Loaded {len(config.dataset.prompts)} prompts from {config.dataset.prompts_file}")

    if not config.dataset.prompts:
         raise ValueError("Prompts are required for SD task. Provide via --prompts or --prompts_file.")
    else:
         logger.info(f"Using user provided prompts: {len(config.dataset.prompts)}")

    if config.fkd.enabled:
        config.fkd.reward_min_val = float('-inf') if config.fkd.potential_type == "max" else 0.0

    with open(config.paths.config_path, 'w') as f:
        json.dump(asdict(config), f, indent=4)
    logger.info(f"Saved finalized SD config to {config.paths.config_path}")

    return config
