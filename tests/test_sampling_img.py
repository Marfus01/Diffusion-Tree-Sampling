"""NFE accounting of the pixel-space DTS and SMC samplers on a tiny random UNet (CPU)."""
import pytest
import torch

diffusers = pytest.importorskip("diffusers")
from diffusers import UNet2DModel, DDIMScheduler

from src.core.sampling_img.mcts_img import MCTSImageDiffusionSampler
from src.core.sampling_img.smc_img import FKDDDIMPipeline

T = 10


@pytest.fixture(scope="module")
def pipeline():
    torch.manual_seed(0)
    unet = UNet2DModel(sample_size=8, in_channels=3, out_channels=3, layers_per_block=1,
                       block_out_channels=(8, 16), down_block_types=("DownBlock2D", "DownBlock2D"),
                       up_block_types=("UpBlock2D", "UpBlock2D"), norm_num_groups=4).eval()
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear", clip_sample=True)
    pipe = FKDDDIMPipeline(unet=unet, scheduler=scheduler)
    pipe.scheduler.set_timesteps(T)
    return pipe


def reward(x):
    return x.mean(dim=(1, 2, 3))


def make_dts(pipeline, **kw):
    return MCTSImageDiffusionSampler(pipeline, reward_fn=lambda x: reward(x).detach().numpy(),
                                     expand_timesteps=[10, 8, 6, 4, 2], num_nfes=10_000, num_inference_steps=T,
                                     device='cpu', use_progress_bar=False, use_rollout=True, batch_size=1, **kw)


def tree_nodes(root):
    stack, out = [root], []
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(n.children)
    return out


def test_dts_trajectory_costs_T(pipeline):
    torch.manual_seed(0)
    s = make_dts(pipeline)
    s.initialize_root_nodes()
    s.build_search_tree(num_nfes=1)
    assert s.nfes == T


def test_dts_sibling_cache_rule(pipeline):
    torch.manual_seed(0)
    s = make_dts(pipeline)
    s.initialize_root_nodes()
    s.build_search_tree(num_nfes=600)
    nodes = [n for n in tree_nodes(s.meta_root) if not s._is_meta_root(n)]
    expected = sum(1 for n in nodes if n.t > 0 and n.children) + sum(1 for n in nodes if n.t == 0)
    assert s.nfes == expected
    assert any(len(n.children) > 1 for n in nodes if 0 < n.t < T)
    terminals = [n for n in nodes if n.t == 0]
    assert all(getattr(n, 'is_final', False) for n in terminals)
    x, _ = s.sample_from_tree(population_size=8)
    assert x.shape == (8, 3, 8, 8)


def test_dts_sibling_recompute_charges_every_child(pipeline):
    torch.manual_seed(0)
    s = make_dts(pipeline, sibling_recompute=True)
    s.initialize_root_nodes()
    s.build_search_tree(num_nfes=600)
    nodes = [n for n in tree_nodes(s.meta_root) if not s._is_meta_root(n)]
    # every created child costs one forward of its parent; terminal nodes one more
    expected = sum(len(n.children) for n in nodes if n.t > 0) + sum(1 for n in nodes if n.t == 0)
    assert s.nfes == expected


@pytest.mark.parametrize("use_guidance", [False, True])
def test_smc_img_nfe_and_weights(pipeline, use_guidance):
    torch.manual_seed(0)
    fkd = {'potential_type': 'diff', 'lmbda': 1.0, 'adaptive_resampling': True, 'resample_frequency': 2,
           'resampling_t_start': 2, 'resampling_t_end': 8, 'reward_min_value': 0.0}
    out = pipeline.sample_batched_smc(num_particles=13, processing_batch_size=5, num_inference_steps=T,
                                      use_smc=True, use_guidance=use_guidance, use_tempering=use_guidance,
                                      reward_fn=reward, fkd_config=fkd, eta=1.0, use_progress_bar=False,
                                      das_gamma=0.3)
    assert out.nfes == 13 * T * (2 if use_guidance else 1)
    assert out.images.shape == (13, 3, 8, 8)
    assert torch.isfinite(out.log_weights).all() and out.log_weights.abs().sum() > 0
