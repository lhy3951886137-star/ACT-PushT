"""Evaluate ACT checkpoints across receding-horizon & temporal-ensembling settings + independent confirmation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from act_v2.data import save_json
from act_v2.inference import StateACTPolicy, VisionACTPolicy, evaluate_state, evaluate_vision


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["state", "vision"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sweep", action="store_true", help="Sweep n_action_steps & temporal ensembling on dev seeds")
    p.add_argument("--confirm", action="store_true", help="Run independent confirmation on unseen test seeds")
    p.add_argument("--zero-image-control", action="store_true", help="Run 20-seed zero-image diagnostic for vision")
    p.add_argument("--n-action-steps", type=int, default=4)
    p.add_argument("--temporal-ensemble", type=float, default=None)
    p.add_argument("--ensemble-window", type=int, default=8)
    p.add_argument("--ensemble-order", choices=["newest", "oldest"], default="newest")
    p.add_argument("--scenes", type=int, default=50)
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()

    if args.mode == "state":
        dev_seeds = list(range(9100000, 9100000 + args.scenes))
        conf_seeds = list(range(9200000, 9200000 + args.scenes))
        PolicyCls = StateACTPolicy
        eval_fn = evaluate_state
    else:
        dev_seeds = list(range(10100000, 10100000 + args.scenes))
        conf_seeds = list(range(10200000, 10200000 + args.scenes))
        PolicyCls = VisionACTPolicy
        eval_fn = evaluate_vision

    if args.sweep:
        candidates = [
            ("exec_4", dict(n_action_steps=4, temporal_ensemble=None)),
            ("exec_6", dict(n_action_steps=6, temporal_ensemble=None)),
            ("exec_8", dict(n_action_steps=8, temporal_ensemble=None)),
            ("exec_10", dict(n_action_steps=10, temporal_ensemble=None)),
            ("te_new_0.3_w8", dict(n_action_steps=1, temporal_ensemble=0.3, ensemble_window=8, ensemble_order="newest")),
            ("te_old_0.05_w8", dict(n_action_steps=1, temporal_ensemble=0.05, ensemble_window=8, ensemble_order="oldest")),
        ]
        board = []
        for label, kw in candidates:
            pol = PolicyCls(checkpoint_path=ckpt, **kw)
            res = eval_fn(pol, dev_seeds, out_dir=out / f"sweep_{label}")
            row = {"label": label, "params": kw, **{k: v for k, v in res.items() if k != "episodes"}}
            board.append(row)
            print(f"[SWEEP {label}] sr={row['success_rate']:.3f} ({row['successes']}/{row['n']}) cov={row['mean_max_coverage']:.4f} steps={row['mean_steps']:.1f}", flush=True)
        board.sort(key=lambda r: (r["success_rate"], r["mean_max_coverage"]), reverse=True)
        save_json(out / "sweep_summary.json", {"checkpoint": str(ckpt), "sha256": sha, "results": board})
        print(f"[SWEEP BEST] {board[0]['label']}: {board[0]['success_rate']:.3f}", flush=True)

    if args.confirm:
        pol = PolicyCls(
            checkpoint_path=ckpt,
            n_action_steps=args.n_action_steps,
            temporal_ensemble=args.temporal_ensemble,
            ensemble_window=args.ensemble_window,
            ensemble_order=args.ensemble_order,
        )
        res = eval_fn(pol, conf_seeds, out_dir=out / "confirmation", save_traces=True)
        summary = {
            "checkpoint": str(ckpt),
            "sha256": sha,
            "inference": {
                "n_action_steps": args.n_action_steps,
                "temporal_ensemble": args.temporal_ensemble,
                "ensemble_window": args.ensemble_window,
                "ensemble_order": args.ensemble_order,
            },
            **{k: v for k, v in res.items() if k != "episodes"},
        }
        save_json(out / "confirmation_summary.json", summary)
        print(f"[CONFIRMATION] sr={summary['success_rate']:.3f} ({summary['successes']}/{summary['n']}) ci95={summary['success_ci95']} cov={summary['mean_max_coverage']:.4f}", flush=True)

    if args.zero_image_control and args.mode == "vision":
        pol = PolicyCls(
            checkpoint_path=ckpt,
            n_action_steps=args.n_action_steps,
            temporal_ensemble=args.temporal_ensemble,
            ensemble_window=args.ensemble_window,
            ensemble_order=args.ensemble_order,
        )
        res = evaluate_vision(pol, list(range(10100000, 10100020)), out_dir=out / "zero_image_control", zero_images=True)
        summary = {k: v for k, v in res.items() if k != "episodes"}
        save_json(out / "zero_image_summary.json", summary)
        print(f"[ZERO IMAGE CONTROL] sr={summary['success_rate']:.3f} ({summary['successes']}/{summary['n']}) cov={summary['mean_max_coverage']:.4f}", flush=True)


if __name__ == "__main__":
    main()
