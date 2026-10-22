"""Offline metrics for saved SD evaluation images (run_sd.py <image_dir>/<reward>/<method>/<prompt>/N<nfe>/*.png).

Per (method, prompt, NFE): mean and max of ImageReward, mean HPSv2 (v2.1), mean LAION aesthetic
score, CLIP pairwise diversity (mean squared distance of normalized CLIP ViT-L/14 image features,
the repository's do_clip_score_diversity) and LPIPS diversity (mean LPIPS-alex over image pairs),
and the number of distinct images. Prompts (--prompts_file or --prompts) map directory names back.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.core.rewards import get_reward_function, do_clip_score_diversity, do_aesthetic_score


def slug(prompt):
    return "".join(c if c.isalnum() else "_" for c in prompt)[:60]


@torch.no_grad()
def lpips_diversity(images, model, device):
    if len(images) < 2:
        return float('nan')
    x = torch.stack([torch.from_numpy(np.asarray(im.convert('RGB'))).permute(2, 0, 1) for im in images]).float()
    x = (x / 127.5 - 1).to(device)
    vals = [float(model(x[i:i + 1], x[j:j + 1])) for i, j in itertools.combinations(range(len(images)), 2)]
    return float(np.mean(vals))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image_dir', required=True, help='<run_dir>/images/<guiding reward>')
    parser.add_argument('--prompts_file', help='JSON prompts file used for the run')
    parser.add_argument('--prompts', nargs='+', help='Prompts given on the run_sd.py command line')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    import lpips
    lp = lpips.LPIPS(net='alex').to(args.device).eval()
    if args.prompts:
        prompts = args.prompts
    else:
        data = json.load(open(args.prompts_file))
        prompts = data['prompts'] if isinstance(data, dict) else data
    by_slug = {slug(p): p for p in prompts}

    rows = []
    for method_dir in sorted(p for p in Path(args.image_dir).iterdir() if p.is_dir()):
        for prompt_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            prompt = by_slug.get(prompt_dir.name)
            if prompt is None:
                continue
            for nfe_dir in sorted(prompt_dir.glob('N*')):
                images = [Image.open(f).convert('RGB') for f in sorted(nfe_dir.glob('*.png'))]
                ir = get_reward_function('ImageReward', images=images, prompts=[prompt] * len(images))
                hps = get_reward_function('HumanPreference', images=images, prompts=[prompt] * len(images))
                aes = do_aesthetic_score(images=images)
                _, clip_div = do_clip_score_diversity(images=images, prompts=[prompt] * len(images)) if len(images) > 1 else (None, float('nan'))
                rows.append({'Method': method_dir.name, 'Prompt': prompt, 'N': int(nfe_dir.name[1:]),
                             'NumSamples': len(images), 'ImageReward_Mean': float(np.mean(ir)),
                             'ImageReward_Max': float(np.max(ir)), 'HPSv2_Mean': float(np.mean(hps)),
                             'Aesthetic_Mean': float(np.mean(aes)), 'CLIP_Diversity': clip_div,
                             'LPIPS_Diversity': lpips_diversity(images, lp, args.device),
                             'Distinct': len({im.tobytes() for im in images})})
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
