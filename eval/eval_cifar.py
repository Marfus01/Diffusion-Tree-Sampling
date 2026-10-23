"""Offline metrics for saved CIFAR-10 evaluation samples (run_img.py samples_*.npz).

For each samples_<method>_<class>_N<nfe>.npz in --run_dir/metrics:
  FID2048   Inception-v3 pool features (pytorch-fid), real set = the class's test images
            without augmentation;
  CMMD      MMD with a Gaussian kernel (sigma = 10, x1000) on L2-normalized CLIP ViT-L/14-336px
            image embeddings (Jayasumana et al. 2024 reference setting), same real set;
  HeldOutAcc fraction classified as the target class by ahsanjavid/convnext-tiny-finetuned-cifar10;
  CLIPDiversity mean sqrt(2 (1 - cos)) over random pairs of generated CLIP embeddings;
  Distinct  number of distinct images.
Duplicates from with-replacement draws stay in the evaluated set.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PATTERN = re.compile(r"samples_(?P<method>.+)_(?P<run_class>class_\d+)_N(?P<nfe>\d+)\.npz")


def load_real(data_path, class_idx):
    import torchvision
    ds = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False)
    x = ds.data[np.asarray(ds.targets) == class_idx]          # [N, 32, 32, 3] uint8
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def inception_features(images_u8, model, device, batch=250):
    feats = []
    for b in range(0, len(images_u8), batch):
        x = images_u8[b:b + batch].to(device).float() / 255.0
        f = model(x)[0]
        feats.append(f.squeeze(-1).squeeze(-1).cpu().double())
    return torch.cat(feats).numpy()


@torch.no_grad()
def clip_embeddings(images_u8, model, processor, device, batch=250):
    out = []
    for b in range(0, len(images_u8), batch):
        inputs = processor(images=list(images_u8[b:b + batch].permute(0, 2, 3, 1).numpy()), return_tensors="pt").to(device)
        e = model.get_image_features(**inputs)
        out.append(F.normalize(e.float(), dim=-1).cpu())
    return torch.cat(out).double()


def cmmd(x, y, sigma=10.0, scale=1000.0):
    """Biased (V-statistic) MMD^2 with k(a, b) = exp(-|a - b|^2 / (2 sigma^2)), as in the CMMD
    reference implementation."""
    gamma = 1.0 / (2 * sigma ** 2)
    def k(a, b):
        d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
        return torch.exp(-gamma * d)
    return float(scale * (k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()))


def clip_diversity(e, max_pairs=10_000, seed=0):
    n = e.shape[0]
    g = torch.Generator().manual_seed(seed)
    i = torch.randint(0, n, (max_pairs,), generator=g)
    j = torch.randint(0, n, (max_pairs,), generator=g)
    keep = i != j
    cos = (e[i[keep]] * e[j[keep]]).sum(1).clamp(-1, 1)
    return float(torch.sqrt(torch.clamp(2 * (1 - cos), min=0)).mean())


@torch.no_grad()
def heldout_accuracy(images_u8, class_idx, model, processor, device, batch=250):
    correct = 0
    for b in range(0, len(images_u8), batch):
        inputs = processor(images=list(images_u8[b:b + batch].permute(0, 2, 3, 1).numpy()), return_tensors="pt").to(device)
        pred = model(**inputs).logits.argmax(-1)
        correct += int((pred == class_idx).sum())
    return correct / len(images_u8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--data_path', required=True, help='torchvision CIFAR-10 root')
    parser.add_argument('--clip_model', default='openai/clip-vit-large-patch14-336')
    parser.add_argument('--heldout_model', default='ahsanjavid/convnext-tiny-finetuned-cifar10')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_frechet_distance
    # The concrete ConvNeXt classes: the Auto classes import timm_wrapper, which needs timm >= 1.0.
    from transformers import CLIPModel, CLIPProcessor, ConvNextImageProcessor, ConvNextForImageClassification

    device = args.device
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained(args.clip_model)
    held_proc = ConvNextImageProcessor.from_pretrained(args.heldout_model)
    held_model = ConvNextForImageClassification.from_pretrained(args.heldout_model).to(device).eval()

    files = sorted((Path(args.run_dir) / 'metrics').glob('samples_*.npz'))
    real_cache = {}
    rows = []
    for f in files:
        m = PATTERN.match(f.name)
        if m is None:
            continue
        class_idx = int(m['run_class'].split('_')[1])
        if class_idx not in real_cache:
            real = load_real(args.data_path, class_idx)
            fr = inception_features(real, inception, device)
            real_cache[class_idx] = (np.mean(fr, 0), np.cov(fr, rowvar=False), clip_embeddings(real, clip_model, clip_proc, device))
        mu_r, sig_r, clip_r = real_cache[class_idx]
        fake = torch.from_numpy(np.load(f)['images'])
        ff = inception_features(fake, inception, device)
        clip_f = clip_embeddings(fake, clip_model, clip_proc, device)
        rows.append({
            'Method': m['method'], 'Run_Class': m['run_class'], 'N': int(m['nfe']), 'NumSamples': len(fake),
            'FID2048': float(calculate_frechet_distance(np.mean(ff, 0), np.cov(ff, rowvar=False), mu_r, sig_r)),
            'CMMD': cmmd(clip_f, clip_r),
            'HeldOutAcc': heldout_accuracy(fake, class_idx, held_model, held_proc, device),
            'CLIPDiversity': clip_diversity(clip_f),
            'Distinct': int(np.unique(fake.reshape(len(fake), -1).numpy(), axis=0).shape[0]),
        })
        print(rows[-1], flush=True)
    out = args.out or str(Path(args.run_dir) / 'metrics' / 'offline_eval.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
