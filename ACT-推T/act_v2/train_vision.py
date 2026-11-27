"""Train improved Pure-Vision ACT on Stage 2 dual-camera dataset on RTX GPU."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from act_v2.data import ACT_V2, AsyncPrefetcher, VisionSampler, save_json
from act_v2.inference import VisionACTPolicy, evaluate_vision
from act_v2.model import VisionACT, weighted_l1_loss
from act_v2.train_state import update_ema


def build_config(args: argparse.Namespace) -> dict:
    return {
        "name": args.name,
        "algorithm": "act_v2_vision",
        "token_mode": args.token_mode,
        "n_obs_steps": args.n_obs_steps,
        "chunk_size": args.chunk_size,
        "n_action_steps": args.n_action_steps,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "use_vae": bool(args.use_vae),
        "latent_dim": args.latent_dim,
        "kl_weight": args.kl_weight,
        "kl_warmup": args.kl_warmup,
        "free_bits": args.free_bits,
        "time_decay": args.time_decay,
        "fine_prob": args.fine_prob,
        "crop_size": args.crop_size,
        "micro_batch": args.micro_batch,
        "accumulation": args.accumulation,
        "effective_batch": args.micro_batch * args.accumulation,
        "steps": args.steps,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup": args.warmup,
        "weight_decay": args.weight_decay,
        "ema_decay": args.ema_decay,
        "checkpoint_every": args.checkpoint_every,
        "eval_every": args.eval_every,
        "eval_scenes": args.eval_scenes,
        "seed": args.seed,
        "policy_inputs": ["robot_xy[2]", "birdview", "robot0_eye_in_hand"],
        "privileged_inputs": [],
    }


def train(cfg: dict) -> dict:
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    run = ACT_V2 / "runs" / cfg["name"]
    run.mkdir(parents=True, exist_ok=True)
    if (run / "complete.json").exists():
        return json.loads((run / "complete.json").read_text(encoding="utf-8"))

    save_json(run / "config.json", cfg)
    train_sampler = VisionSampler(
        stats=None,
        n_obs_steps=cfg["n_obs_steps"],
        chunk_size=cfg["chunk_size"],
        fine_prob=cfg["fine_prob"],
        seed=cfg["seed"],
        validation=False,
    )
    stats = train_sampler.stats
    val_sampler = VisionSampler(
        stats=stats,
        n_obs_steps=cfg["n_obs_steps"],
        chunk_size=cfg["chunk_size"],
        fine_prob=0.0,
        seed=cfg["seed"] + 999,
        validation=True,
    )
    save_json(run / "stats.json", stats)

    model = VisionACT(
        n_obs_steps=cfg["n_obs_steps"],
        chunk_size=cfg["chunk_size"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_encoder_layers=cfg["num_encoder_layers"],
        num_decoder_layers=cfg["num_decoder_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        use_vae=cfg["use_vae"],
        latent_dim=cfg["latent_dim"],
        crop_size=cfg["crop_size"],
        token_mode=cfg.get("token_mode", "hybrid"),
    ).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[VisionACT] {cfg['name']}: params={n_params/1e6:.2f}M, train_eps={len(train_sampler.episodes)}, val_eps={len(val_sampler.episodes)}", flush=True)

    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    start_step = 0
    best_key = (-1.0, -1.0, -float("inf"))
    best_step = 0
    if (run / "latest.pt").exists():
        ck = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["optimizer"])
        start_step = ck["step"]
        best_key = tuple(ck["best_key"])
        best_step = ck.get("best_step", start_step)
        train_sampler.rng.bit_generator.state = ck["train_rng"]
        val_sampler.rng.bit_generator.state = ck["val_rng"]
        torch.set_rng_state(ck["torch_rng"])
        torch.cuda.set_rng_state(ck["cuda_rng"])
        del ck
        print(f"[VisionACT] Resumed from step {start_step}, best={best_key}", flush=True)

    def save_ckpt(step: int, filename: str = "latest.pt") -> None:
        obj = {
            "config": cfg,
            "stats": stats,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "best_key": best_key,
            "best_step": best_step,
            "train_rng": train_sampler.rng.bit_generator.state,
            "val_rng": val_sampler.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        }
        tmp = run / (filename + ".tmp")
        torch.save(obj, tmp)
        tmp.replace(run / filename)

    def validate(step: int) -> None:
        nonlocal best_key, best_step
        ema.eval()
        val_l1s = []
        with torch.inference_mode():
            for _ in range(10):
                r_np, im_np, a_np, m_np = val_sampler.sample(cfg["micro_batch"])
                r_t = torch.as_tensor(r_np, device="cuda")
                im_t = torch.as_tensor(im_np, device="cuda")
                a_t = torch.as_tensor(a_np, device="cuda")
                m_t = torch.as_tensor(m_np, device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred, _, _ = ema(r_t, im_t, augment=False)
                l1 = ((pred.float() - a_t).abs() * m_t[..., None]).sum() / (m_t.sum() * 2.0)
                val_l1s.append(float(l1))
        val_l1 = float(np.mean(val_l1s))
        save_ckpt(step)

        policy = VisionACTPolicy(
            model=ema,
            stats=stats,
            cfg=cfg,
            n_action_steps=cfg["n_action_steps"],
        )
        seeds = range(10100000, 10100000 + cfg["eval_scenes"])
        eval_dir = run / f"eval_{step:06d}"
        res = evaluate_vision(policy, seeds, out_dir=eval_dir)
        res.update(step=step, validation_l1=val_l1)
        save_json(eval_dir / "validation_complete.json", res)

        key = (res["success_rate"], res["mean_max_coverage"], -val_l1)
        if key > best_key:
            best_key = key
            best_step = step
            save_ckpt(step, "best.pt")
        save_ckpt(step)
        print(
            f"[VISION EVAL step={step}] sr={res['success_rate']:.3f} ({res['successes']}/{res['n']}) "
            f"cov={res['mean_max_coverage']:.4f} val_l1={val_l1:.4f} steps={res['mean_steps']:.1f}",
            flush=True,
        )
        model.train()

    prefetcher = AsyncPrefetcher(lambda: train_sampler.sample(cfg["micro_batch"]), maxsize=12)
    total = cfg["steps"]
    warm = min(cfg["warmup"], max(1, total // 4))
    started = time.monotonic()
    acc = []
    model.train()

    try:
        for step in range(start_step + 1, total + 1):
            if (ROOT / "STOP_REQUESTED").exists():
                save_ckpt(step - 1)
                save_json(run / "status.json", {"status": "stopped", "step": step - 1})
                return {}

            if step <= warm:
                lr = cfg["lr"] * (step / warm)
            else:
                ratio = (step - warm) / max(1, total - warm)
                lr = cfg["min_lr"] + (cfg["lr"] - cfg["min_lr"]) * 0.5 * (1.0 + math.cos(math.pi * min(ratio, 1.0)))
            for pg in opt.param_groups:
                pg["lr"] = lr

            if cfg["use_vae"]:
                kl_mult = min(1.0, step / max(1, cfg["kl_warmup"])) if cfg["kl_warmup"] > 0 else 1.0
                beta = cfg["kl_weight"] * kl_mult
            else:
                beta = 0.0

            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            step_recon = 0.0
            step_kl = 0.0
            step_active = 0.0
            for _ in range(cfg["accumulation"]):
                r_np, im_np, a_np, m_np = prefetcher.next()
                r_t = torch.from_numpy(r_np).pin_memory().to("cuda", non_blocking=True)
                im_t = torch.from_numpy(im_np).pin_memory().to("cuda", non_blocking=True)
                a_t = torch.from_numpy(a_np).pin_memory().to("cuda", non_blocking=True)
                m_t = torch.from_numpy(m_np).pin_memory().to("cuda", non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred, kl, diag = model(r_t, im_t, a_t, m_t, augment=True, free_bits=cfg["free_bits"])
                    recon = weighted_l1_loss(pred, a_t, m_t, time_decay=cfg["time_decay"])
                    loss = (recon + beta * kl) / cfg["accumulation"]
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite vision loss at step {step}")
                loss.backward()
                step_loss += float(loss.detach())
                step_recon += float(recon.detach()) / cfg["accumulation"]
                if "raw_kl" in diag:
                    step_kl += float(diag["raw_kl"]) / cfg["accumulation"]
                    step_active += float(diag["active_latents"]) / cfg["accumulation"]

            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad):
                raise RuntimeError(f"Nonfinite vision grad at step {step}")
            opt.step()
            update_ema(ema, model, cfg["ema_decay"])
            acc.append((step_loss, step_recon, step_kl, step_active, float(grad)))

            if step % 100 == 0 or step == start_step + 1:
                arr = np.mean(acc, axis=0)
                elapsed = time.monotonic() - started
                sps = (step - start_step) / max(elapsed, 1e-6)
                row = {
                    "status": "training",
                    "pid": os.getpid(),
                    "step": step,
                    "total_steps": total,
                    "loss": float(arr[0]),
                    "recon_l1": float(arr[1]),
                    "raw_kl": float(arr[2]),
                    "beta": float(beta),
                    "active_latents": float(arr[3]),
                    "grad_norm": float(arr[4]),
                    "lr": float(lr),
                    "steps_per_sec": float(sps),
                    "seconds": float(elapsed),
                    "peak_gpu_gb": float(torch.cuda.max_memory_allocated() / 1e9),
                }
                save_json(run / "status.json", row)
                with (run / "metrics.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                if step % 500 == 0 or step == start_step + 1:
                    print(
                        f"[vision {step:6d}/{total}] recon={arr[1]:.4f} kl={arr[2]:.3f} act_z={arr[3]:.1f} "
                        f"grad={arr[4]:.2f} lr={lr:.2e} {sps:.2f} it/s peak={row['peak_gpu_gb']:.2f}GB",
                        flush=True,
                    )
                acc = []

            if step % cfg["checkpoint_every"] == 0:
                save_ckpt(step)
            if step % cfg["eval_every"] == 0 or step == total:
                validate(step)
    finally:
        prefetcher.close()

    final = {
        "name": cfg["name"],
        "steps": total,
        "best_step": best_step,
        "best_validation_success": best_key[0],
        "best_validation_coverage": best_key[1],
        "best_validation_l1": -best_key[2],
        "seconds": time.monotonic() - started,
    }
    save_json(run / "complete.json", final)
    save_json(run / "status.json", {"status": "complete", **final})
    print(f"[VISION DONE] {json.dumps(final)}", flush=True)
    return final


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="vision_act_v2")
    p.add_argument("--token-mode", choices=["hybrid", "keypoints"], default="keypoints")
    p.add_argument("--n-obs-steps", type=int, default=2)
    p.add_argument("--chunk-size", type=int, default=16)
    p.add_argument("--n-action-steps", type=int, default=4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-encoder-layers", type=int, default=4)
    p.add_argument("--num-decoder-layers", type=int, default=4)
    p.add_argument("--dim-feedforward", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--use-vae", type=int, default=1)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--kl-weight", type=float, default=0.05)
    p.add_argument("--kl-warmup", type=int, default=5000)
    p.add_argument("--free-bits", type=float, default=0.0)
    p.add_argument("--time-decay", type=float, default=0.08)
    p.add_argument("--fine-prob", type=float, default=0.5)
    p.add_argument("--crop-size", type=int, default=116)
    p.add_argument("--micro-batch", type=int, default=32)
    p.add_argument("--accumulation", type=int, default=2)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--checkpoint-every", type=int, default=2500)
    p.add_argument("--eval-every", type=int, default=10000)
    p.add_argument("--eval-scenes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(build_config(parse_args()))
