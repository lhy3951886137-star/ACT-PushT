"""Parallel unseen blind confirmation (50-scene Conf-1 + 100-scene Virgin Conf-2) for State-ACT v3 & v4."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from act_v2.data import save_json
from act_v2.full_random_state import FullRandomStateACTPolicy, evaluate_full_state
from act_v2.inference import wilson_ci95


def _eval_slice(ckpt_str: str, k: int, seeds: list[int], out_dir_str: str) -> list[dict]:
    torch.set_num_threads(1)
    pol = FullRandomStateACTPolicy(checkpoint_path=Path(ckpt_str), n_action_steps=k, weight_kind="ema")
    res = evaluate_full_state(pol, seeds, out_dir=Path(out_dir_str), save_traces=True)
    return res["episodes"]


def run_parallel_eval(name: str, ckpt: Path, k: int, seeds: list[int], out_dir: Path, workers: int = 4) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    chunks = [seeds[i::workers] for i in range(workers) if seeds[i::workers]]
    rows = []
    with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        futs = [pool.submit(_eval_slice, str(ckpt), k, ch, str(out_dir)) for ch in chunks]
        for fut in as_completed(futs):
            rows.extend(fut.result())
    rows.sort(key=lambda r: r["seed"])

    n = len(rows)
    succ = sum(r["success"] for r in rows)
    failed = [r for r in rows if not r["success"]]
    buckets = {}
    for label, lo, hi in [("0-45deg", 0, 45), ("45-90deg", 45, 90), ("90-135deg", 90, 135), ("135-180deg", 135, 181)]:
        sub = [r for r in rows if lo <= abs(r["init_angle_deg"]) < hi]
        if sub:
            buckets[label] = {
                "n": len(sub),
                "successes": sum(r["success"] for r in sub),
                "sr": float(np.mean([r["success"] for r in sub])),
            }
    summary = {
        "name": name,
        "checkpoint": str(ckpt),
        "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "n_action_steps": k,
        "seed_range": [seeds[0], seeds[-1]],
        "success_rate": float(succ / n),
        "successes": int(succ),
        "n": int(n),
        "success_ci95": wilson_ci95(succ, n),
        "mean_max_coverage": float(np.mean([r["max_coverage"] for r in rows])),
        "mean_max_iou": float(np.mean([r["max_iou"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "failed_seeds": [r["seed"] for r in failed],
        "angle_buckets": buckets,
        "seconds": round(time.monotonic() - t0, 1),
        "episodes": rows,
    }
    save_json(out_dir / "summary.json", summary)
    bstr = " ".join(f"{bk}:{bv['successes']}/{bv['n']}({bv['sr']:.1%})" for bk, bv in buckets.items())
    print(
        f"[{name}] sr={summary['success_rate']:.3f} ({succ}/{n}) "
        f"ci95=[{summary['success_ci95'][0]:.3f}, {summary['success_ci95'][1]:.3f}] "
        f"cov={summary['mean_max_coverage']:.4f} steps={summary['mean_steps']:.1f} "
        f"| {bstr} ({summary['seconds']}s)",
        flush=True,
    )
    return summary


def main():
    v4_ckpt = Path("act_v2/runs/full_random_state_act_v4/latest.pt")
    v3_ckpt = Path("act_v2/runs/full_random_state_act_v3/latest.pt")

    conf1 = list(range(30200000, 30200050))
    conf2 = list(range(30300000, 30300100))

    r_v4_c1 = run_parallel_eval(
        "v4_50k_exec14_conf1_50", v4_ckpt, 14, conf1,
        Path("act_v2/runs/full_random_state_act_v4/final_eval/conf1_exec14"), workers=5,
    )
    r_v4_c2 = run_parallel_eval(
        "v4_50k_exec14_blind100", v4_ckpt, 14, conf2,
        Path("act_v2/runs/full_random_state_act_v4/final_eval/blind100_exec14"), workers=5,
    )
    r_v3_c2 = run_parallel_eval(
        "v3_100k_exec18_blind100", v3_ckpt, 18, conf2,
        Path("act_v2/runs/full_random_state_act_v3/final_eval/blind100_exec18"), workers=5,
    )

    gate = {
        "v4_50k_exec14_conf1_50": {k: v for k, v in r_v4_c1.items() if k != "episodes"},
        "v4_50k_exec14_blind100": {k: v for k, v in r_v4_c2.items() if k != "episodes"},
        "v3_100k_exec18_blind100": {k: v for k, v in r_v3_c2.items() if k != "episodes"},
    }
    save_json(Path("act_v2/runs/full_random_state_act_v4/final_eval/blind_gate_summary.json"), gate)
    print("[BLIND GATE COMPLETE]", flush=True)


if __name__ == "__main__":
    main()
