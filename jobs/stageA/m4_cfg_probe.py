"""Records the UNet inputs of the first denoising step of FKDStableDiffusion.sample_batched_smc.

Run once per code checkout (--code_dir) and compare the saved tensors across checkouts:
the UNet is replaced by a recorder that returns zeros and stops after --calls calls, so only the
text encoder runs. Used to check that cfg_pairing='upstream' feeds the UNet exactly what the
upstream code does, and cfg_pairing='fixed' exactly what repro-m3 does.

  python jobs/stageA/m4_cfg_probe.py --code_dir <checkout> --out a.pt [--cfg_pairing upstream]
  python jobs/stageA/m4_cfg_probe.py --compare a.pt b.pt
"""
import argparse
import os
import sys


class StopProbe(Exception):
    pass


def record(args):
    sys.path.insert(0, args.code_dir)
    os.chdir(args.code_dir)
    import torch
    from src.core.sampling_sd.fkd_sd import FKDStableDiffusion

    pipe = FKDStableDiffusion.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5",
                                              safety_checker=None, requires_safety_checker=False).to("cuda")
    calls = []

    def recorder(sample, timestep, encoder_hidden_states=None, **kw):
        calls.append({"sample": sample.detach().cpu(), "timestep": torch.as_tensor(timestep).cpu(),
                      "encoder_hidden_states": encoder_hidden_states.detach().cpu()})
        if len(calls) >= args.calls:
            raise StopProbe
        return (torch.zeros_like(sample),)

    pipe.unet.forward = recorder
    fkd_args = {"use_smc": True, "potential_type": "max", "lmbda": 10.0, "adaptive_resampling": False,
                "resample_frequency": 10, "resampling_t_start": 10, "resampling_t_end": 40, "time_steps": 50,
                "guidance_reward_fn": "ImageReward", "reward_min_value": float("-inf"), "device": "cuda",
                "use_rollout": False, "batch_size": 20, "num_particles": args.num_particles}
    if args.cfg_pairing is not None:
        fkd_args["cfg_pairing"] = args.cfg_pairing
    torch.manual_seed(0)
    try:
        pipe.sample_batched_smc(num_particles=args.num_particles, prompt=[args.prompt] * args.num_particles,
                                processing_batch_size=20, negative_prompt=[""] * args.num_particles,
                                fkd_args=fkd_args, num_inference_steps=50, guidance_scale=7.5, eta=1.0,
                                output_type="pil")
    except StopProbe:
        pass
    torch.save(calls, args.out)
    print(f"{args.code_dir}: {len(calls)} UNet calls recorded, "
          f"encoder_hidden_states {tuple(calls[0]['encoder_hidden_states'].shape)}")


def compare(a, b):
    import torch
    ca, cb = torch.load(a), torch.load(b)
    assert len(ca) == len(cb)
    res = {k: all(torch.equal(x[k], y[k]) for x, y in zip(ca, cb)) for k in ca[0]}
    print(f"{os.path.basename(a)} vs {os.path.basename(b)}: "
          + ", ".join(f"{k} {'IDENTICAL' if v else 'DIFFERENT'}" for k, v in res.items()))
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--code_dir")
    p.add_argument("--out")
    p.add_argument("--cfg_pairing", default=None)
    p.add_argument("--prompt", default="a photo of a black teddy bear")
    p.add_argument("--num_particles", type=int, default=40)
    p.add_argument("--calls", type=int, default=2)
    p.add_argument("--compare", nargs=2)
    args = p.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        record(args)
