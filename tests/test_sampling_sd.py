"""CFG embedding order of the SD FK sampler (cfg_pairing 'fixed' and 'upstream')."""
import pytest
import torch

pytest.importorskip("diffusers")
from src.core.sampling_sd.fkd_sd import cfg_batch_prompt_embeds


def population(num_particles, seq=3, dim=4):
    # same construction as sample_batched_smc: [u_0, c_0, u_1, c_1, ...]
    u = torch.randn(1, seq, dim).repeat(num_particles, 1, 1)
    c = torch.randn(1, seq, dim).repeat(num_particles, 1, 1)
    return torch.stack([u, c], dim=1).reshape(-1, seq, dim), u[0], c[0]


@pytest.mark.parametrize("num_particles,batch", [(40, 20), (60, 20), (7, 3)])
def test_upstream_is_the_upstream_slice(num_particles, batch):
    pop, _, _ = population(num_particles)
    for s in range(0, num_particles, batch):
        e = min(s + batch, num_particles)
        assert torch.equal(cfg_batch_prompt_embeds(pop, s, e, "upstream"), pop[s * 2: e * 2])


@pytest.mark.parametrize("num_particles,batch", [(40, 20), (7, 3)])
def test_fixed_pairs_each_latent_with_u_then_c(num_particles, batch):
    pop, u, c = population(num_particles)
    for s in range(0, num_particles, batch):
        e = min(s + batch, num_particles)
        out = cfg_batch_prompt_embeds(pop, s, e, "fixed")
        b = e - s
        assert torch.equal(out, torch.cat([pop[s * 2: e * 2: 2], pop[s * 2 + 1: e * 2: 2]]))
        assert all(torch.equal(out[i], u) for i in range(b))
        assert all(torch.equal(out[b + i], c) for i in range(b))


def test_upstream_even_batch_has_identical_halves():
    pop, u, c = population(40)
    out = cfg_batch_prompt_embeds(pop, 0, 20, "upstream")
    assert torch.equal(out[:20], out[20:])
    assert all(torch.equal(out[i], u if i % 2 == 0 else c) for i in range(20))
