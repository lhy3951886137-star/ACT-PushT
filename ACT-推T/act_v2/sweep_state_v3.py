"""Parallel pure-ACT inference & ensembling sweep on Full-Random 360-deg Dev seeds (30100000..49)."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from act_v2.full_random_state import (
    FullRandomStateACTEnsemblePolicy,
    FullRandomStateACTPolicy,
    evaluate_full_state,
)


def _eval_one(spec: dict) -> dict:
    torch.set_num_threads(1)
    tag = spec["tag"]
    seeds = list(range(spec["seed_start"], spec["seed_start"] + spec["n_seeds"]))
    if spec["kind"] == "single":
        pol = FullRandomStateACTPolicy(
            checkpoint_path=Path(spec["ckpt"]),
            n_action_steps=spec["k"],
            fine_action_steps=spec.get("fk"),
            temporal_ensemble=spec.get("te"),
            ensemble_window=spec.get("ew", 8),
            ensemble_order=spec.get("eo", "newest"),
            weight_kind=spec.get("wk", "ema"),
        )
    else:
        pol = FullRandomStateACTEnsemblePolicy(
            checkpoints=[Path(c) for c in spec["ckpts"]],
            n_action_steps=spec["k"],
            fine_action_steps=spec.get("fk"),
            weight_kind=spec.get("wk", "ema"),
        )
    t0 = time.monotonic()
    res = evaluate_full_state(pol, seeds)
    return {
        "tag": tag,
        "sr": res["success_rate"],
        "succ": res["successes"],
        "n": res["n"],
        "cov": res["mean_max_coverage"],
        "steps": res["mean_steps"],
        "buckets": {k: f"{v['successes']}/{v['n']}" for k, v in res["angle_buckets"].items()},
        "fails": [e["seed"] for e in res["episodes"] if not e["success"]],
        "sec": round(time.monotonic() - t0, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--mode", default="v3_sweep")
    args = p.parse_args()

    v3_latest = "act_v2/runs/full_random_state_act_v3/latest.pt"
    v3_best = "act_v2/runs/full_random_state_act_v3/best.pt"
    v2_best = "act_v2/runs/full_random_state_act_v2/best.pt"

    if args.mode == "v3_sweep":
        specs = [
            {"tag": "v3_100k_exec4", "kind": "single", "ckpt": v3_latest, "k": 4, "seed_start": 30100000, "n_seeds": 50},
            {"tag": "v3_100k_exec6", "kind": "single", "ckpt": v3_latest, "k": 6, "seed_start": 30100000, "n_seeds": 50},
            {"tag": "v3_100k_exec8", "kind": "single", "ckpt": v3_latest, "k": 8, "seed_start": 30100000, "n_seeds": 50},
            {"tag": "v3_100k_exec10", "kind": "single", "ckpt": v3_latest, "k": 10, "seed_start": 30100000, "n_seeds": 50},
            {"tag": "v3_100k_exec14", "kind": "single", "ckpt": v3_latest, "k": 14, "seed_start": 30100000, "n_seeds": 50},
            {"tag": "v3_100k_exec16", "kind": "single", "ckpt": v3_latest, "k": 16, "seed_start": 30100000, "n_seeds": 50},
            {"tag": "v3_100k_exec18", "kind": "single", "ckpt": v3_latest, "k": 18, "seed_start": 30100000, "n_seeds": 50},
            {"tag": "v3_100k_exec20", "kind": "single", "ckpt": v3_latest, "k": 20, "seed_start": 30100000, "n_seeds": 50},
            {"tag": "v3_100k_exec12_raw", "kind": "single", "ckpt": v3_latest, "k": 12, "wk": "model", "seed_start": 30100000, "n_seeds": 50},
            {"tag": "v3_100k_te03_w8", "kind": "single", "ckpt": v3_latest, "k": 1, "te": 0.3, "ew": 8, "seed_start": 30100000, "n_seeds": 50},
            {"tag": "v3_40k_exec16", "kind": "single", "ckpt": v3_best, "k": 16, "seed_start": 30100000, "n_seeds": 50},
            {"tag": "ens_v3latest_v2_exec12", "kind": "ens", "ckpts": [v3_latest, v2_best], "k": 12, "seed_start": 30100000, "n_seeds": 50},
        ]
    else:
        raise ValueError(args.mode)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_eval_one, s) for s in specs]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print(
                f"[{r['tag']:22s}] sr={r['sr']:.3f} ({r['succ']:2d}/{r['n']}) "
                f"cov={r['cov']:.4f} steps={r['steps']:6.1f} {r['buckets']} ({r['sec']}s)",
                flush=True,
            )
    results.sort(key=lambda r: (r["sr"], r["cov"]), reverse=True)
    print("\n=== LEADERBOARD ===", flush=True)
    for r in results:
        print(f"  {r['tag']:22s}: {r['sr']:.3f} ({r['succ']}/{r['n']}) cov={r['cov']:.4f} fails={r['fails']}", flush=True)


if __name__ == "__main__":
    main()
