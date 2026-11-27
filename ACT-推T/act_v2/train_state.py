"""Train improved State-ACT on RTX GPU with EMA, KL warmup, front-weighted L1, and closed-loop eval."""
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

from act_v2.data import (
    ACT_V2,
    StateSampler,
    fit_state_normalizer,
    load_precision_state_dataset,
    save_json,
)
from act_v2.inference import StateACTPolicy, evaluate_state
from act_v2.model import StateACT, weighted_l1_loss


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    if decay <= 0.0:
        ema.load_state_dict(model.state_dict())
        return
    for e, p in zip(ema.parameters(), model.parameters()):
        e.lerp_(p, 1.0 - decay)
    for e, p in zip(ema.buffers(), model.buffers()):
        e.copy_(p)


def build_config(args: argparse.Namespace) -> dict:
    return {
        "name": args.name,
        "algorithm": "act_v2_state",
        "feature_mode": args.feature_mode,
        "use_all_150": bool(args.use_all_150),
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
        "state_noise_std": args.state_noise_std,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup": args.warmup,
        "weight_decay": args.weight_decay,
        "ema_decay": args.ema_decay,
        "eval_every": args.eval_every,
        "eval_scenes": args.eval_scenes,
        "seed": args.seed,
    }


def train(cfg: dict) -> dict:
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    run = ACT_V2 / "runs" / cfg["name"]
    run.mkdir(parents=True, exist_ok=True)
    if (run / "complete.json").exists():
        return json.loads((run / "complete.json").read_text(encoding="utf-8"))

    save_json(run / "config.json", cfg)
    fmode = cfg.get("feature_mode", "relative")
    groups = load_precision_state_dataset(use_all_150=cfg["use_all_150"], feature_mode=fmode)
    stats = fit_state_normalizer(groups, feature_mode=fmode)
    save_json(run / "stats.json", stats)

    model = StateACT(
        n_obs_steps=cfg["n_obs_steps"],
        pusher_dim=6,
        object_dim=8 if fmode == "relative" else 10,
        chunk_size=cfg["chunk_size"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_encoder_layers=cfg["num_encoder_layers"],
        num_decoder_layers=cfg["num_decoder_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        use_vae=cfg["use_vae"],
        latent_dim=cfg["latent_dim"],
    ).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[StateACT] {cfg['name']}: params={n_params/1e6:.2f}M, train_eps={len(groups['train'])}, val_eps={len(groups['validation'])}", flush=True)

    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    train_sampler = StateSampler(
        groups["train"],
        stats,
        n_obs_steps=cfg["n_obs_steps"],
        chunk_size=cfg["chunk_size"],
        batch_size=cfg["batch_size"],
        fine_prob=cfg["fine_prob"],
        seed=cfg["seed"],
        validation=False,
        state_noise_std=cfg["state_noise_std"],
    )
    val_sampler = StateSampler(
        groups["validation"],
        stats,
        n_obs_steps=cfg["n_obs_steps"],
        chunk_size=cfg["chunk_size"],
        batch_size=cfg["batch_size"],
        fine_prob=0.0,
        seed=cfg["seed"] + 999,
        validation=True,
    )

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
        print(f"[StateACT] Resumed from step {start_step}, best={best_key}", flush=True)

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
            for _ in range(20):
                o_np, a_np, m_np = val_sampler.sample()
                o = torch.as_tensor(o_np, device="cuda")
                a = torch.as_tensor(a_np, device="cuda")
                m = torch.as_tensor(m_np, device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred, _, _ = ema(o)
                l1 = ((pred.float() - a).abs() * m[..., None]).sum() / (m.sum() * 2.0)
                val_l1s.append(float(l1))
        val_l1 = float(np.mean(val_l1s))
        save_ckpt(step)

        policy = StateACTPolicy(
            model=ema,
            stats=stats,
            cfg=cfg,
            n_action_steps=cfg["n_action_steps"],
        )
        seeds = range(9100000, 9100000 + cfg["eval_scenes"])
        eval_dir = run / f"eval_{step:06d}"
        res = evaluate_state(policy, seeds, out_dir=eval_dir)
        res.update(step=step, validation_l1=val_l1)
        save_json(eval_dir / "validation_complete.json", res)

        key = (res["success_rate"], res["mean_max_coverage"], -val_l1)
        if key > best_key:
            best_key = key
            best_step = step
            save_ckpt(step, "best.pt")
        save_ckpt(step)

        summary = {k: v for k, v in res.items() if k != "episodes"}
        print(f"[EVAL step={step}] sr={summary['success_rate']:.3f} ({summary['successes']}/{summary['n']}) cov={summary['mean_max_coverage']:.4f} val_l1={val_l1:.4f} steps={summary['mean_steps']:.1f}", flush=True)
        model.train()

    total = cfg["steps"]
    warm = min(cfg["warmup"], max(1, total // 4))
    started = time.monotonic()
    acc = []
    model.train()

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

        o_np, a_np, m_np = train_sampler.sample()
        o = torch.as_tensor(o_np, device="cuda")
        a = torch.as_tensor(a_np, device="cuda")
        m = torch.as_tensor(m_np, device="cuda")

        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, kl, diag = model(o, a, m, free_bits=cfg["free_bits"])
            recon = weighted_l1_loss(pred, a, m, time_decay=cfg["time_decay"])
            loss = recon + beta * kl

        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad):
            raise RuntimeError(f"Nonfinite grad at step {step}")
        opt.step()
        update_ema(ema, model, cfg["ema_decay"])

        raw_kl = float(diag["raw_kl"]) if "raw_kl" in diag else 0.0
        active = float(diag["active_latents"]) if "active_latents" in diag else 0.0
        acc.append((float(loss.detach()), float(recon.detach()), raw_kl, active, float(grad)))

        if step % 500 == 0 or step == start_step + 1:
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
            if step % 2000 == 0 or step == start_step + 1:
                print(
                    f"[step {step:6d}/{total}] recon={arr[1]:.4f} kl={arr[2]:.3f} act_z={arr[3]:.1f} "
                    f"grad={arr[4]:.2f} lr={lr:.2e} {sps:.1f} it/s",
                    flush=True,
                )
            acc = []

        if step % cfg["eval_every"] == 0 or step == total:
            validate(step)

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
    print(f"[DONE] {json.dumps(final)}", flush=True)
    return final


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="state_act_v2")
    p.add_argument("--feature-mode", choices=["relative", "legacy16"], default="relative")
    p.add_argument("--use-all-150", action="store_true")
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
    p.add_argument("--state-noise-std", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--eval-every", type=int, default=10000)
    p.add_argument("--eval-scenes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(build_config(parse_args()))
