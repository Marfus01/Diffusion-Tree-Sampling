"""CMMD from each CIFAR-10 sample set to a pi*_0 reference drawn from pool chunks it does not share.

reference  Per class, --n_eval draws with replacement from the pool chunks --ref_chunks with
           self-normalized weights w_i ~ p(c | x_i) of --classifier (lambda = 1, as in
           m4_prior_pool.py draw). Importance resampling seed s uses chunk s, so with the default
           chunks 3-9 the reference is independent of the IR runs (seeds 0-2).
floor      CMMD between two references of the same size drawn from disjoint halves of
           --ref_chunks: the value two independent pi*_0 sample sets reach.
runs       CMMD (as in eval/eval_cifar.py: CLIP ViT-L/14-336px, Gaussian kernel sigma = 10, x1000)
           from every metrics/samples_<method>_class_<c>_N<nfe>.npz under the given run directories to
           the reference of class c. CLIP embeddings are cached under --cache_dir.

Usage: python -m jobs.stageA.m4_cmmd_pistar --pool_dir <dir> --out <csv> --cache_dir <dir> <run_dir> ...
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eval.eval_cifar import clip_embeddings, cmmd

SAMPLES = re.compile(r'samples_(?P<method>.+)_class_(?P<cls>\d+)_N(?P<nfe>\d+)\.npz')


def draw_reference(images, logp, cls, n, seed):
    r = logp[:, cls].astype(np.float64)
    w = np.exp(r - r.max())
    w /= w.sum()
    idx = np.random.default_rng([seed, cls]).choice(len(w), size=n, replace=True, p=w)
    return images[idx], float(1.0 / (w ** 2).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dirs', nargs='+')
    ap.add_argument('--pool_dir', required=True)
    ap.add_argument('--ref_chunks', type=int, nargs='+', default=[3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--classifier', default='hf')
    ap.add_argument('--classes', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--n_eval', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--clip_model', default='openai/clip-vit-large-patch14-336')
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    from transformers import CLIPModel, CLIPProcessor
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(a.device)
    model = CLIPModel.from_pretrained(a.clip_model).to(device).eval()
    proc = CLIPProcessor.from_pretrained(a.clip_model)
    cache = Path(a.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    def embed(images, key):
        f = cache / f'{key}.npy'
        if f.exists():
            return torch.from_numpy(np.load(f))
        e = clip_embeddings(torch.from_numpy(images), model, proc, device)
        np.save(f, e.numpy())
        return e

    chunks = {c: np.load(Path(a.pool_dir) / f'chunk_{c:02d}.npz') for c in a.ref_chunks}
    def pool(cs):
        return (np.concatenate([chunks[c]['images'] for c in cs]),
                np.concatenate([chunks[c][f'logp_{a.classifier}'] for c in cs]))
    half = len(a.ref_chunks) // 2
    tag = f'{a.classifier}_chunks' + '-'.join(map(str, a.ref_chunks))
    images, logp = pool(a.ref_chunks)
    ref, rows = {}, []
    for cls in a.classes:
        x, ess = draw_reference(images, logp, cls, a.n_eval, a.seed)
        ref[cls] = embed(x, f'pistar_{tag}_class{cls}')
        halves = []
        for i, cs in enumerate((a.ref_chunks[:half], a.ref_chunks[half:])):
            xi, ess_i = draw_reference(*pool(cs), cls, a.n_eval, a.seed + 1 + i)
            halves.append((embed(xi, f'pistar_{tag}_half{i}_class{cls}'), ess_i))
        rows.append({'Run': 'pistar_floor', 'Method': 'PiStar-halves', 'Class': cls, 'N': None,
                     'CMMD_to_pistar': cmmd(halves[0][0], halves[1][0]), 'RefESS': ess,
                     'HalfESS': f'{halves[0][1]:.0f}/{halves[1][1]:.0f}'})
        print(rows[-1], flush=True)

    for run_dir in map(Path, a.run_dirs):
        for f in sorted((run_dir / 'metrics').glob('samples_*.npz')):
            m = SAMPLES.fullmatch(f.name)
            cls = int(m['cls'])
            if cls not in ref:
                continue
            e = embed(np.load(f)['images'], f'{run_dir.name}__{f.stem}')
            rows.append({'Run': run_dir.name, 'Method': m['method'], 'Class': cls, 'N': int(m['nfe']),
                         'CMMD_to_pistar': cmmd(e, ref[cls])})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == '__main__':
    main()
