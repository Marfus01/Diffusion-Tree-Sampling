"""Stratified GenEval prompt subset.

Reads the GenEval metadata (FK-Steering `text_to_image/prompt_files/geneval_metadata.jsonl`),
allocates n prompts across the six tags proportionally to the tag sizes (largest remainder,
at least one per tag), draws each tag's prompts uniformly with a fixed seed, and writes
{"prompts": [...], "metadata": [...]} for run_sd.py --prompts_file.
"""
import argparse
import json
import random
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    by_tag = defaultdict(list)
    with open(args.metadata) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                by_tag[item['tag']].append(item)
    tags = sorted(by_tag)
    total = sum(len(v) for v in by_tag.values())
    quota = {t: max(1, int(args.n * len(by_tag[t]) / total)) for t in tags}
    remainders = sorted(tags, key=lambda t: args.n * len(by_tag[t]) / total - int(args.n * len(by_tag[t]) / total), reverse=True)
    i = 0
    while sum(quota.values()) < args.n:
        quota[remainders[i % len(tags)]] += 1
        i += 1
    while sum(quota.values()) > args.n:
        t = max(tags, key=lambda t: quota[t])
        quota[t] -= 1

    rng = random.Random(args.seed)
    chosen = []
    for t in tags:
        chosen.extend(rng.sample(by_tag[t], quota[t]))
    with open(args.out, 'w') as f:
        json.dump({'prompts': [c['prompt'] for c in chosen], 'metadata': chosen}, f, indent=1)
    print({t: quota[t] for t in tags})
    for c in chosen:
        print(c['tag'], '|', c['prompt'])


if __name__ == '__main__':
    main()
