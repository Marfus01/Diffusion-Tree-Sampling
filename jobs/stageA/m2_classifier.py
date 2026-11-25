"""Loads (or, for the repository's own ResNet, trains) the CIFAR-10 guiding classifier p_phi of
run_img.py and writes its test-set accuracy and mean log p_phi(c | x) per class, on the
unaugmented test images.

Without --classifier_path the checkpoint is the one run_img.py resolves for the same --base_dir
({base_dir}/img/checkpoints/cifar10/cifar10_resnet_best.pt).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.core.config import get_default_image_config, process_image_config
from src.models.classifier import get_train_classifier
from src.utils.datasets import get_img_dataset


@torch.no_grad()
def per_class_stats(classifier, loader, device, num_classes):
    correct = np.zeros(num_classes)
    count = np.zeros(num_classes)
    log_p = np.zeros(num_classes)
    for batch in loader:
        x = batch['images'].to(device)
        y = batch['labels'].to(device).flatten()
        lp = F.log_softmax(classifier(x), dim=-1)
        pred = lp.argmax(-1)
        for c in range(num_classes):
            m = y == c
            count[c] += int(m.sum())
            correct[c] += int((pred[m] == c).sum())
            log_p[c] += float(lp[m, c].sum())
    return correct, count, log_p


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--classifier_model', default=None)
    parser.add_argument('--classifier_path', default=None)
    args = parser.parse_args()

    config = get_default_image_config()
    config.dataset.name = 'cifar10'
    config.common.base_dir = args.base_dir
    config.common.exp_name = 'classifier'
    config.common.seed = args.seed
    config.model.classifier_model = args.classifier_model
    config.model.classifier_path = args.classifier_path
    config = process_image_config(config)

    device = config.common.device
    classifier = get_train_classifier(config)
    ckpt = config.paths.classifier_load_path or str(Path(config.paths.checkpoint_dir) / 'cifar10_resnet_best.pt')

    loader = get_img_dataset(config, splits=['test'], augment=False)['test_loader']
    correct, count, log_p = per_class_stats(classifier, loader, device, config.dataset.num_classes)
    acc = correct / count
    out = {
        'classifier_model': config.model.classifier_model,
        'checkpoint': ckpt,
        'overall_accuracy': float(correct.sum() / count.sum()),
        'per_class_accuracy': {str(c): float(a) for c, a in enumerate(acc)},
        'per_class_mean_log_p': {str(c): float(v) for c, v in enumerate(log_p / count)},
        'classes_by_accuracy': [int(c) for c in np.argsort(-acc)],
        'classes_by_mean_log_p': [int(c) for c in np.argsort(-(log_p / count))],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
