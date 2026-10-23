"""Bitwise check of the SMC implementations used by the A-R010 ablation.

Runs `run_smc` of the run_img module found under --code_root (the upstream checkout or a fork
worktree) for SMC/FK, TDS and DAS on a small population and saves, per case, the evaluation images
that run_smc hands to evaluate_images. The global RNGs are re-seeded right before every run_smc
call, so two code roots are compared from the same random state. Different code roots import
modules of the same names, so each side runs in its own process:

  python jobs/stageA/m4_smc_equiv.py run --code_root <root> --side upstream|fork [--smc_impl fixed|upstream] --out a.pt ...
  python jobs/stageA/m4_smc_equiv.py compare a.pt b.pt

Cases: hyperparameters 'paper' (steps 10..40 every 10, adaptive, scale 1) and 'frozen' (M2 frozen
configurations), DAS gamma 0.008 in both (the upstream constant), and two evaluation sizes (more and
fewer evaluation images than particles). --frozen_das_gamma sets DAS's gamma in the frozen case
for fork-only comparisons.
"""
import argparse
import os
import sys
from types import SimpleNamespace


def cases(frozen_das_gamma):
    # method -> (NFE, hyperparameter set -> (adaptive, (start, end, freq), guidance_scale, das_gamma))
    hp = {
        'SMC': {'paper': (True, (10, 40, 10), 1.0, 0.008), 'frozen': (False, (10, 40, 10), 1.0, 0.008)},
        'SMC-Guidance': {'paper': (True, (10, 40, 10), 1.0, 0.008), 'frozen': (False, (10, 40, 10), 0.5, 0.008)},
        'SMC-Guidance-Tempering': {'paper': (True, (10, 40, 10), 1.0, 0.008),
                                   'frozen': (False, (0, 49, 1), 0.5, frozen_das_gamma)},
    }
    nfe = {'SMC': 800, 'SMC-Guidance': 1600, 'SMC-Guidance-Tempering': 1600}   # 16 particles each
    for method, sets in hp.items():
        for h, cfg in sets.items():
            for n_eval in (24, 10):
                yield f'{method}|{h}|n{n_eval}', method, nfe[method], n_eval, cfg


def run(a):
    sys.path.insert(0, a.code_root)
    os.chdir(a.code_root)
    import torch
    if a.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
    import scripts.run_img as R
    from src.core.config import get_default_image_config, process_image_config, seed_experiment
    from src.models.classifier import get_train_classifier

    captured = {}
    def capture(images, *_, **__):
        captured['x'] = images.detach().cpu().clone()
        return {'FID': 0.0, 'MMD': 0.0, 'LogProb_Mean': 0.0}

    R.evaluate_images = capture
    R.save_image = lambda *_, **__: None
    if hasattr(R, 'finish_nfe_point'):
        R.finish_nfe_point = lambda *_, **__: None

    def make_config(method, nfe, n_eval, cfg):
        adaptive, (t0, t1, freq), scale, _ = cfg
        config = get_default_image_config()
        config.dataset.name = 'cifar10'
        config.dataset.sample_classes = [str(a.cls)]
        config.common.seed = a.seed
        config.common.device = a.device
        config.common.base_dir = a.base_dir
        config.common.hf_home = os.environ.get('HF_HOME')
        config.common.use_progress_bar = False
        config.model.classifier_path = a.classifier_path
        config.sampling.compute_budget = [nfe]
        config.sampling.population_size = n_eval
        config.fkd.adaptive_resampling = adaptive
        config.fkd.resampling_t_start, config.fkd.resampling_t_end, config.fkd.resample_frequency = t0, t1, freq
        config.guidance_scale = scale
        return process_image_config(config)

    out, classifier, pipelines = {}, None, {}
    for key, method, nfe, n_eval, cfg in cases(a.frozen_das_gamma):
        config = make_config(method, nfe, n_eval, cfg)
        if classifier is None:
            classifier = get_train_classifier(config)
        if method not in pipelines:
            pipelines[method] = R.load_pipeline(config, method)
        captured.clear()
        seed_experiment(a.seed)
        if a.side == 'upstream':
            R.run_smc(config, f'class_{a.cls}', [a.cls], classifier, None, None, pipelines[method], method_name=method)
        else:
            args = SimpleNamespace(reward_lambda=1.0, das_gamma=cfg[3], smc_impl=a.smc_impl, no_save_samples=True)
            R.run_smc(config, args, f'class_{a.cls}', [a.cls], classifier, None, None, pipelines[method], method_name=method)
        x = captured['x']
        out[key] = x
        print(f'{key}: {tuple(x.shape)} mean {x.double().mean():.6f} distinct '
              f'{torch.unique(x.flatten(1), dim=0).shape[0]}', flush=True)
    torch.save(out, a.out)


def compare(a):
    import torch
    x, y = torch.load(a.a), torch.load(a.b)
    ok = True
    for k in x:
        same = k in y and x[k].shape == y[k].shape and torch.equal(x[k], y[k])
        diff = '' if same or k not in y or x[k].shape != y[k].shape else f' max|d| {(x[k] - y[k]).abs().max():.3e}'
        print(f"{'IDENTICAL' if same else 'DIFFERENT'} {k}{diff}")
        ok &= same
    print('ALL IDENTICAL' if ok else 'MISMATCH')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--code_root', required=True)
    r.add_argument('--side', choices=['upstream', 'fork'], required=True)
    r.add_argument('--smc_impl', choices=['fixed', 'upstream'], default='fixed')
    r.add_argument('--out', required=True)
    r.add_argument('--classifier_path', required=True, help='self-trained ResNet-18 state_dict (cifar10_resnet_best.pt)')
    r.add_argument('--base_dir', required=True, help='scratch experiment root for the configs these runs write')
    r.add_argument('--device', default='cpu')
    r.add_argument('--cls', type=int, default=1)
    r.add_argument('--seed', type=int, default=0)
    r.add_argument('--frozen_das_gamma', type=float, default=0.008)
    r.add_argument('--deterministic', action='store_true', help='torch.use_deterministic_algorithms(True)')
    c = sub.add_parser('compare')
    c.add_argument('a')
    c.add_argument('b')
    a = p.parse_args()
    run(a) if a.cmd == 'run' else compare(a)
