"""Representative CIFAR-10 samples from run_img.py evaluation sets (samples_<method>_<class>_N<nfe>.npz).

For every npz under the given run directories: an 8x8 PNG of its first 64 images (the evaluation
set is a random draw, so these are representative), named after its run directory. For every
class: one comparison PNG at a common NFE (--compare_nfe, default the largest NFE every method of
that class has), one labelled row of the first 16 images per method; with several runs of one
method at that NFE the first run directory given on the command line is used.

Usage: python -m jobs.stageA.m2_sample_grids --out DIR RUN_DIR [RUN_DIR ...]
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PATTERN = re.compile(r"samples_(?P<method>.+)_(?P<run_class>class_\d+)_N(?P<nfe>\d+)\.npz")
LABELS = {'MCTS-Rollout': 'DTS', 'SMC': 'SMC/FK', 'SMC-Guidance': 'TDS', 'SMC-Guidance-Tempering': 'DAS', 'Guidance': 'DPS-RG'}
ORDER = ['MCTS-Rollout', 'SMC', 'SMC-Guidance', 'SMC-Guidance-Tempering', 'Guidance']


def tile(images, ncol, pad=2):
    """images: uint8 [N, 3, H, W] -> one PIL image with ncol columns and a white gap of pad pixels."""
    n, _, h, w = images.shape
    nrow = int(np.ceil(n / ncol))
    canvas = np.full((nrow * (h + pad) + pad, ncol * (w + pad) + pad, 3), 255, np.uint8)
    for k, im in enumerate(images):
        r, c = divmod(k, ncol)
        canvas[pad + r * (h + pad):pad + r * (h + pad) + h, pad + c * (w + pad):pad + c * (w + pad) + w] = im.transpose(1, 2, 0)
    return Image.fromarray(canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True)
    parser.add_argument('--scale', type=int, default=2, help='Nearest-neighbour upscaling of the 32x32 images.')
    parser.add_argument('--compare_nfe', type=int, default=None)
    parser.add_argument('run_dirs', nargs='+')
    args = parser.parse_args()
    out = Path(args.out)
    (out / 'grids').mkdir(parents=True, exist_ok=True)

    by_class = defaultdict(lambda: defaultdict(dict))   # class -> method -> nfe -> first path
    for run_dir in args.run_dirs:
        for f in sorted(Path(run_dir, 'metrics').glob('samples_*.npz')):
            m = PATTERN.match(f.name)
            if m is None:
                continue
            images = np.load(f)['images'][:64]
            name = f"{Path(run_dir).name}__{LABELS.get(m['method'], m['method']).replace('/', '-')}_N{m['nfe']}.png"
            g = tile(images, 8)
            g.resize((g.width * args.scale, g.height * args.scale), Image.NEAREST).save(out / 'grids' / name)
            by_class[m['run_class']][m['method']].setdefault(int(m['nfe']), f)

    for run_class, methods in by_class.items():
        nfe = args.compare_nfe or max(set.intersection(*(set(v) for v in methods.values())), default=None)
        rows = [(meth, nfe, methods[meth][nfe]) for meth in ORDER if nfe in methods.get(meth, {})]
        if not rows:
            continue
        strips = [tile(np.load(p)['images'][:16], 16) for _, _, p in rows]
        strips = [s.resize((s.width * args.scale, s.height * args.scale), Image.NEAREST) for s in strips]
        label_w = 110
        fig = Image.new('RGB', (label_w + strips[0].width, sum(s.height for s in strips)), 'white')
        draw = ImageDraw.Draw(fig)
        y = 0
        for (meth, nfe, _), s in zip(rows, strips):
            fig.paste(s, (label_w, y))
            draw.text((6, y + s.height // 2 - 12), f"{LABELS.get(meth, meth)}\nNFE {nfe:.0e}".replace('e+0', 'e'), fill='black')
            y += s.height
        fig.save(out / f'compare_{run_class}_N{nfe}.png')
        print(f'compare_{run_class}_N{nfe}.png: ' + ', '.join(f'{LABELS.get(m, m)} {p.parent.parent.name}' for m, _, p in rows))


if __name__ == '__main__':
    main()
