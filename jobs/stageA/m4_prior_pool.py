"""A-R012: a pool of prior samples of google/ddpm-cifar10-32, the pi*_0 reference and prior
importance resampling.

generate  DDIM, eta = 1, 50 steps (50 NFE per sample), --n_chunks chunks of --chunk_size samples;
          chunk c uses seed --seed + c, so chunks are independent and a finished chunk is skipped on
          restart. Each chunk_<c>.npz holds the uint8 images [n, 3, 32, 32] and, per classifier,
          log p(. | x) [n, 10] on the final float image in [-1, 1] (logp_<name>, the reward the
          samplers see) and on its uint8 quantization (logp_<name>_u8, what offline evaluation sees).
draw      With lambda = 1 the target is pi*_0(x) ~ p_theta,0(x) p(c | x), so prior samples carry
          self-normalized weights w_i ~ p(c | x_i).
          pi*_0 reference: per classifier and class, weights over the whole pool; ESS; the
          self-normalized E_pi[r] = sum_i w_i r_i / sum_i w_i with r = log p(c | x) of the same
          classifier and standard error sqrt(Var_w(r) / ESS); 5000 draws with replacement.
          Importance resampling: chunk s is seed s; at budget NFE the first NFE / 50 samples of the
          chunk (nested, like DTS snapshots), weights from --ir_classifier, 5000 draws with
          replacement.
          Draws are written as run directories <results>/<name>/metrics/samples_<method>_class_<c>_N<nfe>.npz
          for eval/eval_cifar.py, with a CSV of the weight statistics and E[r] of the draws under every
          classifier (float rewards).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
NFE_PER_SAMPLE = 50


def parse_classifiers(items):
    return [(name, kind, path) for name, kind, path in items]


@torch.no_grad()
def log_probs(model, x, batch=2000):
    return torch.cat([F.log_softmax(model(x[b:b + batch]), dim=-1).float() for b in range(0, len(x), batch)])


@torch.no_grad()
def generate(a):
    from diffusers import DDIMPipeline, DDIMScheduler
    from diffusers.utils.torch_utils import randn_tensor
    from src.models.classifier import load_cifar10_classifier

    out = Path(a.pool_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    pipe = DDIMPipeline.from_pretrained(a.model)
    unet = pipe.unet.to(device).eval()
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(NFE_PER_SAMPLE, device=device)
    models = {name: load_cifar10_classifier(kind, path, device) for name, kind, path in parse_classifiers(a.classifier)}

    for c in range(a.n_chunks):
        path = out / f'chunk_{c:02d}.npz'
        if path.exists():
            print(f'{path} exists, skipped', flush=True)
            continue
        t0 = time.time()
        g = torch.Generator(device=device).manual_seed(a.seed + c)
        imgs = []
        for b in range(0, a.chunk_size, a.batch_size):
            n = min(a.batch_size, a.chunk_size - b)
            x = randn_tensor((n, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size),
                             generator=g, device=device, dtype=unet.dtype)
            for t in scheduler.timesteps:
                x = scheduler.step(unet(x, t).sample, t, x, eta=1.0, generator=g).prev_sample
            imgs.append(x)
        x = torch.cat(imgs)
        u8 = ((x.clamp(-1, 1) / 2 + 0.5) * 255).round().to(torch.uint8)
        xq = u8.float() / 255.0 * 2 - 1
        arrays = {'images': u8.cpu().numpy()}
        for name, m in models.items():
            arrays[f'logp_{name}'] = log_probs(m, x).cpu().numpy()
            arrays[f'logp_{name}_u8'] = log_probs(m, xq).cpu().numpy()
        np.savez(path, **arrays)
        print(f'chunk {c}: {a.chunk_size} samples in {time.time() - t0:.1f} s '
              f'({a.chunk_size * NFE_PER_SAMPLE / (time.time() - t0):.0f} NFE/s)', flush=True)


def weight_stats(logw, r):
    """Self-normalized weights from log-weights logw [n]; returns ESS, E_w[r], sqrt(Var_w(r) / ESS)."""
    w = np.exp(logw - logw.max())
    w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)
    mean = float(np.sum(w * r))
    var = float(np.sum(w * (r - mean) ** 2))
    return w, float(ess), mean, float(np.sqrt(var / ess))


def draw(a):
    pool = [np.load(p) for p in sorted(Path(a.pool_dir).glob('chunk_*.npz'))]
    names = sorted(k[5:] for k in pool[0].files if k.startswith('logp_') and not k.endswith('_u8'))
    images = np.concatenate([p['images'] for p in pool])
    logp = {n: np.concatenate([p[f'logp_{n}'] for p in pool]).astype(np.float64) for n in names}
    chunk = len(pool[0]['images'])
    res = Path(a.results_dir)
    print(f'pool: {len(images)} samples in {len(pool)} chunks of {chunk}; classifiers {names}', flush=True)

    def save(run, method, cls, nfe, idx, row):
        d = res / run / 'metrics'
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f'samples_{method}_class_{cls}_N{nfe}.npz', images=images[idx])
        row.update({'Method': method, 'Run_Class': f'class_{cls}', 'N': nfe, 'NumSamples': len(idx),
                    'Distinct': int(len(np.unique(idx))),
                    **{f'LogProb_{n}': float(logp[n][idx, cls].mean()) for n in names}})
        return row

    rows = {}
    for name in names:
        run = f'm4pi_PiStar_{name}'
        for cls in a.classes:
            r = logp[name][:, cls]
            w, ess, mean, se = weight_stats(r, r)
            rng = np.random.default_rng([a.seed, cls, names.index(name)])
            idx = rng.choice(len(w), size=a.n_eval, replace=True, p=w)
            rows.setdefault(run, []).append(save(run, 'PiStar', cls, len(images) * NFE_PER_SAMPLE, idx, {
                'Classifier': name, 'PoolSize': len(images), 'ESS': ess, 'E_pi_r': mean, 'E_pi_r_SE': se,
                'PriorMean_r': float(r.mean())}))
    for s in a.ir_seeds:
        run = f'm4ir_IR_s{s}'
        for cls in a.classes:
            for nfe in a.nfe:
                n = nfe // NFE_PER_SAMPLE
                assert n <= chunk, f'NFE {nfe} needs {n} samples, chunk has {chunk}'
                sel = np.arange(s * chunk, s * chunk + n)
                r = logp[a.ir_classifier][sel, cls]
                w, ess, mean, se = weight_stats(r, r)
                rng = np.random.default_rng([a.seed, s, cls, nfe])
                idx = sel[rng.choice(n, size=a.n_eval, replace=True, p=w)]
                rows.setdefault(run, []).append(save(run, 'IR', cls, nfe, idx, {
                    'Classifier': a.ir_classifier, 'PoolSize': n, 'ESS': ess, 'ESS_ratio': ess / n,
                    'E_w_r': mean, 'E_w_r_SE': se}))
    for run, rr in rows.items():
        pd.DataFrame(rr).to_csv(res / run / 'metrics' / 'pool_draws.csv', index=False)
        print(pd.DataFrame(rr).to_string(), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('generate')
    g.add_argument('--pool_dir', required=True)
    g.add_argument('--model', default='google/ddpm-cifar10-32')
    g.add_argument('--classifier', nargs=3, action='append', required=True, metavar=('NAME', 'KIND', 'PATH'),
                   help='KIND resnet or resnet18_cifar10_edadaltocg (src.models.classifier.load_cifar10_classifier)')
    g.add_argument('--n_chunks', type=int, default=10)
    g.add_argument('--chunk_size', type=int, default=20000)
    g.add_argument('--batch_size', type=int, default=2000)
    g.add_argument('--seed', type=int, default=0)
    g.add_argument('--device', default='cuda')
    d = sub.add_parser('draw')
    d.add_argument('--pool_dir', required=True)
    d.add_argument('--results_dir', required=True, help='CIFAR-10 results directory of the run directories')
    d.add_argument('--classes', type=int, nargs='+', default=[1, 2, 3])
    d.add_argument('--ir_classifier', default='hf')
    d.add_argument('--ir_seeds', type=int, nargs='+', default=[0, 1, 2])
    d.add_argument('--nfe', type=int, nargs='+', default=[10000, 30000, 100000, 300000, 1000000])
    d.add_argument('--n_eval', type=int, default=5000)
    d.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    generate(a) if a.cmd == 'generate' else draw(a)
