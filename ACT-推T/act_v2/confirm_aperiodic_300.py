"""Final 300-scene blind confirmation for Aperiodic Pure State-ACT v4 ([12, 16, 18, 14] + freeze-wake)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
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


def _eval_slice(ckpt_str: str, sched: list[int], freeze_wake: bool, seeds: list[int], out_dir_str: str) -> list[dict]:
    torch.set_num_threads(1)
    pol = FullRandomStateACTPolicy(
        checkpoint_path=Path(ckpt_str),
        n_action_steps=sched[0],
        action_schedule=sched,
        mc_dropout_on_freeze=freeze_wake,
        weight_kind="ema",
    )
    res = evaluate_full_state(pol, seeds, out_dir=Path(out_dir_str), save_traces=False)
    return res["episodes"]


def run_eval(name: str, ckpt: Path, sched: list[int], freeze_wake: bool, seeds: list[int], out_dir: Path, workers: int = 5) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    chunks = [seeds[i::workers] for i in range(workers) if seeds[i::workers]]
    rows = []
    with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        futs = [pool.submit(_eval_slice, str(ckpt), sched, freeze_wake, ch, str(out_dir)) for ch in chunks]
        for fut in as_completed(futs):
            rows.extend(fut.result())
    rows.sort(key=lambda r: r["seed"])
    n = len(rows)
    succ = sum(r["success"] for r in rows)
    buckets = {}
    for label, lo, hi in [("0-45deg", 0, 45), ("45-90deg", 45, 90), ("90-135deg", 90, 135), ("135-180deg", 135, 181)]:
        sub = [r for r in rows if lo <= abs(r["init_angle_deg"]) < hi]
        if sub:
            buckets[label] = {"n": len(sub), "successes": sum(r["success"] for r in sub), "sr": float(np.mean([r["success"] for r in sub]))}
    summary = {
        "name": name,
        "checkpoint": str(ckpt),
        "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "action_schedule": sched,
        "mc_dropout_on_freeze": freeze_wake,
        "seed_range": [seeds[0], seeds[-1]],
        "success_rate": float(succ / n),
        "successes": int(succ),
        "n": int(n),
        "success_ci95": wilson_ci95(succ, n),
        "mean_max_coverage": float(np.mean([r["max_coverage"] for r in rows])),
        "mean_max_iou": float(np.mean([r["max_iou"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "failed_seeds": [r["seed"] for r in rows if not r["success"]],
        "angle_buckets": buckets,
        "seconds": round(time.monotonic() - t0, 1),
        "episodes": rows,
    }
    save_json(out_dir / "summary.json", summary)
    bstr = " ".join(f"{bk}:{bv['successes']}/{bv['n']}({bv['sr']:.1%})" for bk, bv in buckets.items())
    print(
        f"[{name:30s}] sr={summary['success_rate']:.3f} ({succ:3d}/{n:3d}) "
        f"ci95=[{summary['success_ci95'][0]:.3f},{summary['success_ci95'][1]:.3f}] "
        f"cov={summary['mean_max_coverage']:.4f} steps={summary['mean_steps']:.1f} | {bstr} ({summary['seconds']}s)",
        flush=True,
    )
    return summary


def main():
    ckpt = Path("act_v2/runs/full_random_state_act_v4/latest.pt")
    sched = [12, 16, 18, 14]
    base_out = Path("act_v2/runs/full_random_state_act_v4/final_eval_aperiodic")

    # 1. Pure Aperiodic [12,16,18,14] + MC-dropout freeze-wake on all 300 scenes:
    r_dev = run_eval("v4_ap_fw_dev50", ckpt, sched, True, list(range(30100000, 30100050)), base_out / "dev50")
    r_c1 = run_eval("v4_ap_fw_conf1_50", ckpt, sched, True, list(range(30200000, 30200050)), base_out / "conf1_50")
    r_b2 = run_eval("v4_ap_fw_blind2_100", ckpt, sched, True, list(range(30300000, 30300100)), base_out / "blind2_100")
    r_v3 = run_eval("v4_ap_fw_virgin3_100", ckpt, sched, True, list(range(30400000, 30400100)), base_out / "virgin3_100")

    # Also test pure Aperiodic [12,16,18,14] WITHOUT freeze-wake on virgin3_100 for clean ablation!
    r_v3_nofw = run_eval("v4_ap_pure_virgin3_100", ckpt, sched, False, list(range(30400000, 30400100)), base_out / "virgin3_100_pure")

    total_succ = r_dev["successes"] + r_c1["successes"] + r_b2["successes"] + r_v3["successes"]
    total_n = r_dev["n"] + r_c1["n"] + r_b2["n"] + r_v3["n"]
    all_summary = {
        "total_300_sr": float(total_succ / total_n),
        "total_300_successes": int(total_succ),
        "total_300_n": int(total_n),
        "total_300_ci95": wilson_ci95(total_succ, total_n),
        "splits": {
            "dev50": {k: v for k, v in r_dev.items() if k != "episodes"},
            "conf1_50": {k: v for k, v in r_c1.items() if k != "episodes"},
            "blind2_100": {k: v for k, v in r_b2.items() if k != "episodes"},
            "virgin3_100": {k: v for k, v in r_v3.items() if k != "episodes"},
            "virgin3_100_pure": {k: v for k, v in r_v3_nofw.items() if k != "episodes"},
        },
    }
    save_json(base_out / "final_300_gate.json", all_summary)
    print(
        f"\n[FINAL 300-SCENE TOTAL] sr={all_summary['total_300_sr']:.3f} ({total_succ}/{total_n}) "
        f"ci95=[{all_summary['total_300_ci95'][0]:.3f}, {all_summary['total_300_ci95'][1]:.3f}]",
        flush=True,
    )


if __name__ == "__main__":
    main()
