"""Dev horizon sweep (exec 4/6/8/10) + independent 50-scene confirmation for frozen State-ACT."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from act_v2.data import save_json
from act_v2.inference import StateACTPolicy, evaluate_state


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="act_v2/runs/state_act_v2_full/best.pt")
    p.add_argument("--out", default="act_v2/runs/state_act_v2_full/final_eval")
    p.add_argument("--horizons", type=int, nargs="+", default=[4, 6, 8, 10])
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()

    dev_seeds = list(range(9100000, 9100050))
    board = []
    for k in args.horizons:
        pol = StateACTPolicy(checkpoint_path=ckpt, n_action_steps=k)
        res = evaluate_state(pol, dev_seeds, out_dir=out / f"dev_exec_{k}")
        row = {"n_action_steps": k, **{kk: vv for kk, vv in res.items() if kk != "episodes"}}
        board.append(row)
        print(
            f"[DEV exec_{k}] sr={row['success_rate']:.3f} ({row['successes']}/{row['n']}) "
            f"cov={row['mean_max_coverage']:.4f} steps={row['mean_steps']:.1f}",
            flush=True,
        )

    board.sort(key=lambda r: (r["success_rate"], r["mean_max_coverage"]), reverse=True)
    best_k = board[0]["n_action_steps"]
    save_json(out / "dev_sweep.json", {"checkpoint": str(ckpt), "sha256": sha, "best_n_action_steps": best_k, "results": board})
    print(f"[SELECTED] n_action_steps={best_k} (dev sr={board[0]['success_rate']:.3f})", flush=True)

    conf_seeds = list(range(9200000, 9200050))
    pol = StateACTPolicy(checkpoint_path=ckpt, n_action_steps=best_k)
    conf = evaluate_state(pol, conf_seeds, out_dir=out / "confirmation", save_traces=True)
    c = {kk: vv for kk, vv in conf.items() if kk != "episodes"}
    passed = bool(board[0]["success_rate"] >= 0.85 and c["success_rate"] >= 0.85)
    summary = {
        "passed": passed,
        "required": 0.85,
        "checkpoint": str(ckpt),
        "sha256": sha,
        "n_action_steps": best_k,
        "dev_best": board[0],
        "confirmation": c,
    }
    save_json(out / "state_gate.json", summary)
    print(
        f"[CONFIRMATION] passed={passed} sr={c['success_rate']:.3f} ({c['successes']}/{c['n']}) "
        f"ci95={c['success_ci95']} cov={c['mean_max_coverage']:.4f} steps={c['mean_steps']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
