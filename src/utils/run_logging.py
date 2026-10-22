"""Run metadata shared by the run scripts: code version, configuration, compute counters."""
import json
import os
import resource
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def git_state():
    """(commit, dirty) of the repository the code runs from; (None, None) outside git."""
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=REPO_ROOT,
                                         text=True, stderr=subprocess.DEVNULL)
        return commit, bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def _jsonable(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_run_meta(run_dir, config, args=None, extra=None, name="run_meta.json"):
    commit, dirty = git_state()
    meta = {
        'git_commit': commit,
        'git_dirty': dirty,
        'argv': sys.argv,
        'args': vars(args) if args is not None else None,
        'config': asdict(config) if is_dataclass(config) else config,
        'hostname': os.uname().nodename,
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'torch_version': torch.__version__,
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    if extra:
        meta.update(extra)
    path = Path(run_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2, default=_jsonable)
    return path


class RewardCounter:
    """Wraps a reward function and counts evaluated samples (reward calls are logged, not
    charged as NFE). `scale` multiplies the reward: the tilt e^{lambda r} enters every method
    through the reward r_lambda = lambda * r."""

    def __init__(self, fn, scale: float = 1.0):
        self.fn = fn
        self.scale = scale
        self.n_evals = 0

    def __call__(self, x, *args, **kwargs):
        out = self.fn(x, *args, **kwargs)
        self.n_evals += len(x)
        if self.scale == 1.0:
            return out
        if isinstance(out, list):
            return [self.scale * v for v in out]
        return self.scale * out

    def reset(self):
        self.n_evals = 0


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb():
    """Peak CUDA memory allocated since the last reset; peak host RSS on CPU-only runs."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024 ** 2 if sys.platform == 'darwin' else rss / 1024


def distinct_count(samples) -> int:
    arr = samples.detach().cpu().numpy() if isinstance(samples, torch.Tensor) else np.asarray(samples)
    arr = arr.reshape(arr.shape[0], -1)
    return int(np.unique(arr, axis=0).shape[0])
