"""Diagnose ACT rollout failures (actions, movement, stalling, chunk commitment, z sampling)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from act_v2.inference import StateACTPolicy
from precision_env import make_env


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[9100000, 9100003, 9100004, 9100014, 9100023])
    p.add_argument("--n-action-steps", type=int, default=4)
    p.add_argument("--temporal-ensemble", type=float, default=None)
    args = p.parse_args()

    pol = StateACTPolicy(
        checkpoint_path=Path(args.checkpoint),
        n_action_steps=args.n_action_steps,
        temporal_ensemble=args.temporal_ensemble,
    )
    env = make_env(False)
    try:
        for seed in args.seeds:
            env.reset_task(seed)
            pol.reset(seed)
            acts = []
            covs = []
            ees = []
            for t in range(2000):
                raw = env.state_vector()
                a = pol.action(raw)
                acts.append(a)
                ees.append(raw[:2].copy())
                env.move(a)
                m = env.metrics()
                covs.append(m["coverage"])
                if m["coverage"] >= 0.95 or m["tilt"] > 0.2 or m["z"] < 0.79 or abs(env.xy()).max() > 0.35:
                    break
            acts = np.asarray(acts)
            ees = np.asarray(ees)
            covs = np.asarray(covs)
            norms = np.linalg.norm(acts, axis=1)
            # Check last 400 steps behavior
            tail_a = acts[-400:]
            tail_ee = ees[-400:]
            ee_span = np.ptp(tail_ee, axis=0) * 1000.0  # mm
            mean_norm_tail = float(np.linalg.norm(tail_a, axis=1).mean())
            mean_vec_tail = float(np.linalg.norm(tail_a.mean(axis=0)))
            print(
                f"seed={seed} steps={len(acts):4d} max_cov={covs.max():.4f} fin_cov={covs[-1]:.4f} "
                f"tail_act_norm={mean_norm_tail:.3f} tail_net_vec={mean_vec_tail:.3f} "
                f"tail_ee_span_mm=({ee_span[0]:.1f},{ee_span[1]:.1f})",
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
