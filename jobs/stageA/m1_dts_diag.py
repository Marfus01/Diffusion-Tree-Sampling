"""DTS diagnostics on the analytic GMM: the exact sampling distribution of the built tree.

The tree is grown exactly as `run_2d --do_mcts_rollout` grows it (same seeding, same snapshots,
the same 2000-sample draw at each snapshot), and at each NFE snapshot the following are computed
by traversing the tree, without sampling noise:

- leaves: number of terminal leaves, and of non-terminal leaves (t > 0 without children, the
  nodes at which `sample_from_tree` restarts a path);
- tree: the leaf distribution of `sample_from_tree`, P(leaf) = prod over branch ancestors of
  softmax(value_est of siblings); its E[r] minus E_{pi*_0}[r] (closed form) and its mode-mass TV;
- nested_is: the same leaves weighted by e^{r(leaf)} / prod n_children(branch ancestors). With the
  normalized backup (value_est = log-mean-exp of the children) the softmax products telescope, so
  this equals `tree`: the sampling distribution depends on the intermediate values only through
  the shape of the tree they steer during construction;
- value error of every child of a branch node: value_est - log M_t (closed form under q_0|t),
  per branch layer and per visit-count bin.

Usage: python jobs/stageA/m1_dts_diag.py --out <json> <run_2d arguments>
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import scripts.run_2d as run_2d
from src.utils.run_logging import RewardCounter

VISIT_BINS = [(1, 1), (2, 9), (10, 99), (100, None)]


def traverse(meta_root):
    """Returns (branch nodes with their children, terminal leaves with branch-ancestor lists,
    count of non-terminal leaves)."""
    branches, leaves, nonterminal = [], [], 0
    stack = [(meta_root, ())]
    while stack:
        node, anc = stack.pop()
        if not node.children:
            if node.t <= 0:
                leaves.append((node, anc))
            else:
                nonterminal += 1
            continue
        if len(node.children) > 1 or node is meta_root:
            branches.append(node)
            anc = anc + (node,)
        stack.extend((c, anc) for c in node.children)
    return branches, leaves, nonterminal


def leaf_log_probs(meta_root, logit):
    """log P(leaf) of the traversal that picks a child with probability softmax(logit(child))."""
    out = {}
    stack = [(meta_root, 0.0)]
    while stack:
        node, lp = stack.pop()
        if not node.children:
            if node.t <= 0:
                out[id(node)] = lp
            continue
        if len(node.children) == 1:
            stack.append((node.children[0], lp))
            continue
        z = np.array([logit(c) for c in node.children], dtype=np.float64)
        z = z - np.logaddexp.reduce(z)
        stack.extend((c, lp + zc) for c, zc in zip(node.children, z))
    return out


def distribution_summary(problem, leaves, log_p):
    x = torch.cat([n.x_t for n, _ in leaves]).double()
    lp = np.array([log_p[id(n)] for n, _ in leaves])
    p = np.exp(lp - np.logaddexp.reduce(lp))
    r = problem.log_likelihood(x.float()).double().numpy()
    comp = problem.map_component(x.numpy())
    mass = np.bincount(comp, weights=p, minlength=len(problem.weights))
    return {'reward_gap': float(p @ r - problem.target_mean_reward()),
            'mode_tv': 0.5 * float(np.abs(mass - problem.posterior_weights).sum()),
            'ess_over_leaves': float(1.0 / np.sum(p ** 2))}


def diagnose(sampler, problem, abar):
    branches, leaves, nonterminal = traverse(sampler.meta_root)
    children = [c for b in branches for c in b.children]
    levels = torch.tensor([abar[c.t] for c in children], dtype=torch.float64)
    x = torch.cat([c.x_t for c in children]).double()
    log_m = torch.cat([problem.log_soft_value(x[i:i + 4096], levels[i:i + 4096])
                       for i in range(0, len(children), 4096)]).numpy()
    log_m_of = {id(c): v for c, v in zip(children, log_m)}

    out = {'n_leaves': len(leaves), 'n_nonterminal_leaves': nonterminal,
           'n_roots': len(sampler.meta_root.children),
           'children_per_branch_layer': {}}
    tree = leaf_log_probs(sampler.meta_root, lambda c: c.value_est)
    nested = {id(n): float(n.value_est) - sum(np.log(len(b.children)) for b in anc) for n, anc in leaves}
    out['tree'] = distribution_summary(problem, leaves, tree)
    out['nested_is'] = distribution_summary(problem, leaves, nested)

    per_layer = defaultdict(list)
    for b in branches:
        layer = 'meta' if b is sampler.meta_root else str(b.t)
        out['children_per_branch_layer'].setdefault(layer, []).append(len(b.children))
        for c in b.children:
            per_layer[layer].append((c.visit_count, float(c.value_est) - log_m_of[id(c)]))
    out['children_per_branch_layer'] = {k: {'nodes': len(v), 'mean': float(np.mean(v)), 'max': int(np.max(v))}
                                        for k, v in out['children_per_branch_layer'].items()}
    value_error = {}
    for layer, rows in per_layer.items():
        visits = np.array([v for v, _ in rows])
        err = np.array([e for _, e in rows])
        entry = {'all': {'n': len(err), 'bias': float(err.mean()), 'var': float(err.var(ddof=1)) if len(err) > 1 else None}}
        for lo, hi in VISIT_BINS:
            sel = (visits >= lo) & (visits <= (hi if hi is not None else np.inf))
            if sel.sum():
                entry[f'visits_{lo}-{hi if hi is not None else "inf"}'] = {
                    'n': int(sel.sum()), 'bias': float(err[sel].mean()),
                    'var': float(err[sel].var(ddof=1)) if sel.sum() > 1 else None}
        value_error[layer] = entry
    out['value_error'] = value_error
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    own, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = run_2d.parse_script_args()
    seed = args.seed if args.seed is not None else 42
    config = run_2d.build_config(args, seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    diffusion_model, task, problem = run_2d.load_task(config, args)
    run_2d.sample_unguided(diffusion_model, config.sampling.population_size)   # same RNG stream as run_2d
    reward = RewardCounter(task.calculate_reward_diff, scale=args.reward_lambda)
    reward_np = lambda x: reward(x).detach().cpu().numpy()
    sampler = run_2d.make_mcts_sampler(config, args, diffusion_model, reward_np, use_rollout=True)
    sampler.initialize_root_nodes()
    abar = diffusion_model.alphas_cumprod.double().cpu().numpy()

    results = []
    for nfe in config.sampling.compute_budget:
        sampler.build_search_tree(num_nfes=nfe)
        samples, _, _ = sampler.sample_from_tree(population_size=config.sampling.population_size)
        row = {'nfe': nfe, 'actual_nfe': sampler.nfes,
               'sampled_mean_reward_gap': float(task.calculate_reward_diff(samples).mean() - task.gt_rewards.mean()),
               **diagnose(sampler, problem, abar)}
        results.append(row)
        print(json.dumps({k: row[k] for k in ('nfe', 'n_leaves', 'n_nonterminal_leaves', 'tree', 'nested_is')}))
        sys.stdout.flush()
    Path(own.out).parent.mkdir(parents=True, exist_ok=True)
    Path(own.out).write_text(json.dumps({'argv': rest, 'seed': seed, 'results': results}, indent=1))


if __name__ == '__main__':
    main()
