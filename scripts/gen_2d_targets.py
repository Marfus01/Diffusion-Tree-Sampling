"""Writes exact-target samples p(x) e^{r(x)} / Z for a 2D toy dataset.

run_2d.py reads `samples.npy` and `rewards.npy` from <base_dir>/toy/checkpoints/<dataset>/.
The samples come from importance resampling of 10x oversampled draws of p(x) with the
differentiable reward, the same reward that guides all samplers.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from src.core.config import get_default_toy_config, process_toy_config
from src.utils.datasets import get_2d_dataset, DATASETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=str, default='gaussian_mixture', choices=list(DATASETS.keys()))
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--base_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    config = get_default_toy_config()
    config.dataset.name = args.dataset
    config.common.seed = args.seed
    if args.base_dir is not None:
        config.common.base_dir = args.base_dir
    config = process_toy_config(config)

    dataset = get_2d_dataset(config, args.dataset, device='cpu')
    samples, rewards = dataset.sample_product_distribution_diff(n_samples=args.n_samples)
    samples = samples.cpu().numpy() if isinstance(samples, torch.Tensor) else np.asarray(samples)
    rewards = np.asarray(rewards)

    out_dir = Path(config.paths.checkpoint_dir)
    np.save(out_dir / "samples.npy", samples)
    np.save(out_dir / "rewards.npy", rewards)
    print(f"Saved {len(samples)} target samples to {out_dir}")


if __name__ == '__main__':
    main()
