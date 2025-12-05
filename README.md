# Diffusion Tree Sampling: Scalable inference-time alignment of diffusion models

Install the dependencies:
```
pip install -r requirements.txt
```

Run the experiments using the files in the `scripts` directory. For example, to run the 2D experiments, run:
```
python scripts/run_2d.py --dataset gaussian_mixture --do_mcts_rollout
```

To run the image experiments, run:
```
python scripts/run_img.py --dataset mnist --do_mcts_rollout
```

To run the SD experiments, run:
```
python scripts/run_sd.py --prompts "A red colored car." --do_mcts_rollout
```

To run the various baselines, you need to specify them as command-line arguments. Here is the mapping from baseline name to command-line argument:
- DTS: `--do_mcts_rollout`
- Best-of-N: `--do_bon`
- SMC: `--do_smc` or `--do_fkd`
- SMC-Rollout: `--do_smc_rollout`
- DPS or DPS-RG: `--do_guidance`
- TDS or TDS-RG:`--do_smc_guidance`
- DAS or DAS-RG: `--do_smc_guidance_tempering`
