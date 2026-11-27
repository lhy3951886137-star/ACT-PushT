"""DAgger-style recovery data collector for Full-Random 360-deg Push-T ACT.

Rolls out a frozen Full-Random State-ACT policy on strictly disjoint seeds,
detects genuine off-policy failure modes (oscillation, stalling, regression,
no progress, -X boundary drift, or scheduled mid-course handoff), hands over
control to FullRandomExpert from the exact policy-visited MuJoCo state, and
saves ONLY verified clean expert recovery trajectories (>=98% cov, <1.5mm, <1.5deg).
Supports parallel worker processes for 5x faster collection.
"""
from __future__ import annotations

import argparse
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

from act_v2.data import ACT_V2, save_json
from act_v2.full_random_collect import FullRandomExpert, qualified
from act_v2.full_random_env import make_full_env
from act_v2.full_random_state import FullRandomStateACTPolicy

DAGGER_DIR = ACT_V2 / "full_random_dagger"


def _dagger_worker(
    seeds: list[int],
    checkpoint_str: str,
    n_action_steps: int,
    out_dir_str: str,
    random_handoff_prob: float,
    failure_only: bool = False,
) -> list[dict]:
    torch.set_num_threads(1)
    out_dir = Path(out_dir_str)
    pol = FullRandomStateACTPolicy(checkpoint_path=Path(checkpoint_str), n_action_steps=n_action_steps)
    env = make_full_env(False)
    results = []
    try:
        for seed in seeds:
            env.reset_task(seed)
            pol.reset(seed)
            init_t_x = float(env.pose()[0][0])

            best_cov = float(env.metrics()["coverage"])
            last_improve_t = 0
            recent_t_pos = []
            recent_acts = []
            triggered = False
            trigger_reason = ""
            rng = np.random.default_rng(seed + 7777)
            random_handoff_t = (
                int(rng.integers(250, 750)) if (not failure_only and rng.random() < random_handoff_prob) else 9999
            )
            policy_solved = False
            trigger_t = -1
            max_pol_steps = 2200 if failure_only else 1800

            for t in range(max_pol_steps):
                raw = env.state_vector()
                a = pol.action(raw)
                env.move(a)
                m = env.metrics()
                if m["coverage"] >= 0.95 and m["tilt"] < 0.2:
                    policy_solved = True
                    break
                if m["tilt"] > 0.2 or m["z"] < 0.79 or abs(env.xy()).max() > 0.35:
                    break

                if m["coverage"] > best_cov + 0.015:
                    best_cov = m["coverage"]
                    last_improve_t = t
                t_pos = env.pose()[0][:2].copy()
                recent_t_pos.append(t_pos)
                recent_acts.append(a.copy())
                win = 160 if failure_only else 120
                if len(recent_t_pos) > win:
                    recent_t_pos.pop(0)
                    recent_acts.pop(0)

                if failure_only:
                    # Strictly target real v3 failures: late-stage stalling, 2-chunk oscillation,
                    # orbiting loops / severe regression near goal, or extended no-progress.
                    if t >= 700 and (t - last_improve_t) >= 420:
                        triggered = True
                        trigger_reason = "late_no_progress"
                        trigger_t = t
                        break
                    if t >= 350 and len(recent_t_pos) == win:
                        t_disp = float(np.linalg.norm(recent_t_pos[-1] - recent_t_pos[0]))
                        ra = np.asarray(recent_acts, dtype=np.float32)
                        act_mag = float(np.linalg.norm(ra, axis=-1).mean())
                        act_net = float(np.linalg.norm(ra.mean(axis=0)))
                        if t_disp < 0.004:
                            triggered = True
                            trigger_reason = "oscillating" if (act_mag > 0.15 and act_net < 0.08) else "stalled"
                            trigger_t = t
                            break
                    if t >= 300 and best_cov >= 0.65 and m["coverage"] < best_cov - 0.22:
                        triggered = True
                        trigger_reason = "orbit_regressed"
                        trigger_t = t
                        break
                    if t >= 200 and t_pos[0] < -0.205 and t_pos[0] < init_t_x - 0.03:
                        triggered = True
                        trigger_reason = "boundary_drift"
                        trigger_t = t
                        break
                else:
                    if t >= 220 and (t - last_improve_t) >= 220:
                        triggered = True
                        trigger_reason = "no_progress"
                        trigger_t = t
                        break
                    if t >= 220 and len(recent_t_pos) == win:
                        t_disp = float(np.linalg.norm(recent_t_pos[-1] - recent_t_pos[0]))
                        ra = np.asarray(recent_acts, dtype=np.float32)
                        act_mag = float(np.linalg.norm(ra, axis=-1).mean())
                        act_net = float(np.linalg.norm(ra.mean(axis=0)))
                        if t_disp < 0.003:
                            triggered = True
                            trigger_reason = "oscillating" if (act_mag > 0.12 and act_net < 0.06) else "stalled"
                            trigger_t = t
                            break
                    if t >= 150 and best_cov >= 0.40 and m["coverage"] < best_cov - 0.14:
                        triggered = True
                        trigger_reason = "regressed"
                        trigger_t = t
                        break
                    if t >= 100 and t_pos[0] < -0.195 and t_pos[0] < init_t_x - 0.02:
                        triggered = True
                        trigger_reason = "boundary_drift"
                        trigger_t = t
                        break
                    if t == random_handoff_t and m["coverage"] < 0.92:
                        triggered = True
                        trigger_reason = "mid_rollout"
                        trigger_t = t
                        break

            if not triggered:
                results.append({"seed": int(seed), "policy_solved": policy_solved, "triggered": False, "ok": False})
                continue

            # Handover to FullRandomExpert from the exact policy-visited MuJoCo state
            expert = FullRandomExpert()
            states = []
            actions = []
            coverage = []
            metrics = []
            holds = 0
            for _ in range(2200):
                m = env.metrics()
                if qualified(m):
                    if holds >= 5:
                        break
                    a = np.zeros(2, np.float32)
                    holds += 1
                else:
                    holds = 0
                    a = expert.action(env)
                states.append(env.state_vector())
                coverage.append(m["coverage"])
                actions.append(a.copy())
                env.move(a)
                m = env.metrics()
                metrics.append([m["coverage"], m["iou"], m["distance"], m["angle"], m["z"], m["tilt"]])
                if m["tilt"] > 0.2 or m["z"] < 0.79 or abs(env.xy()).max() > 0.35:
                    break

            ok = bool(holds >= 5 and qualified(env.metrics()) and len(actions) >= 20 and max(r[5] for r in metrics) < 0.2)
            if ok:
                raw_name = f"raw_dagger_{seed}.npz"
                fpath = out_dir / raw_name
                tmp = out_dir / (raw_name + ".tmp.npz")
                with tmp.open("wb") as f:
                    np.savez_compressed(
                        f,
                        seed=np.int64(seed),
                        states=np.asarray(states, np.float32),
                        actions=np.asarray(actions, np.float32),
                        coverage=np.asarray(coverage, np.float32),
                        metrics=np.asarray(metrics, np.float32),
                        terminal_state=env.state_vector(),
                    )
                tmp.replace(fpath)
                results.append(
                    {
                        "seed": int(seed),
                        "policy_solved": False,
                        "triggered": True,
                        "ok": True,
                        "raw_file": raw_name,
                        "frames": len(actions),
                        "trigger_reason": trigger_reason,
                        "trigger_step": trigger_t,
                    }
                )
            else:
                results.append({"seed": int(seed), "policy_solved": False, "triggered": True, "ok": False})
    finally:
        env.close()
    return results


def collect_dagger(
    checkpoint: Path,
    target: int = 160,
    max_seeds: int = 1200,
    n_action_steps: int = 12,
    seed_base: int = 30500000,
    out_dir: Path = DAGGER_DIR,
    random_handoff_prob: float = 0.20,
    workers: int = 5,
    failure_only: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "complete.json").exists():
        ex = json.loads((out_dir / "complete.json").read_text(encoding="utf-8"))
        if len(ex.get("episodes", [])) >= target:
            print(f"[DAGGER] Already have {len(ex['episodes'])} recovery episodes in {out_dir}", flush=True)
            return ex

    started = time.monotonic()
    accepted = []
    attempted = 0
    policy_solved = 0
    triggers = 0
    reason_counts: dict[str, int] = {}

    while len(accepted) < target and attempted < max_seeds:
        needed = target - len(accepted)
        mult = 6.0 if failure_only else 2.5
        wave_n = min(max_seeds - attempted, max(workers * 8, int(needed * mult)))
        wave_n = min(wave_n, workers * 30 if failure_only else workers * 16)
        seeds = [seed_base + attempted + i for i in range(wave_n)]
        attempted += wave_n
        chunks = [seeds[i::workers] for i in range(workers) if seeds[i::workers]]
        wave_res = []
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            futs = [
                pool.submit(
                    _dagger_worker, ch, str(checkpoint), n_action_steps, str(out_dir), random_handoff_prob, failure_only
                )
                for ch in chunks
            ]
            for fut in as_completed(futs):
                wave_res.extend(fut.result())
        wave_res.sort(key=lambda r: r["seed"])
        for r in wave_res:
            if r["policy_solved"]:
                policy_solved += 1
            if r["triggered"]:
                triggers += 1
            if r["ok"]:
                accepted.append(r)
                rc = r["trigger_reason"]
                reason_counts[rc] = reason_counts.get(rc, 0) + 1
        print(
            f"[DAGGER] collected={len(accepted)}/{target} attempted={attempted} "
            f"pol_solved={policy_solved} triggers={triggers} reasons={reason_counts} "
            f"elapsed={time.monotonic()-started:.1f}s",
            flush=True,
        )

    accepted = accepted[:target]
    episodes = []
    for idx, r in enumerate(accepted):
        src = out_dir / r["raw_file"]
        dst = out_dir / f"dagger_{idx:04d}.npz"
        if src.exists():
            src.replace(dst)
        episodes.append(
            {
                "index": idx,
                "file": dst.name,
                "seed": int(r["seed"]),
                "frames": int(r["frames"]),
                "partition": "train",
                "trigger_reason": r["trigger_reason"],
                "trigger_step": int(r["trigger_step"]),
                "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
            }
        )
    for leftover in out_dir.glob("raw_dagger_*.npz"):
        leftover.unlink(missing_ok=True)

    summary = {
        "source_checkpoint": str(checkpoint),
        "target": target,
        "collected": len(episodes),
        "attempted": attempted,
        "policy_solved_without_help": policy_solved,
        "triggers": triggers,
        "reason_counts": reason_counts,
        "total_frames": int(sum(e["frames"] for e in episodes)),
        "seconds": time.monotonic() - started,
        "episodes": episodes,
    }
    save_json(out_dir / "complete.json", summary)
    print(
        f"[DAGGER DONE] {len(episodes)} recovery episodes, {summary['total_frames']} frames, "
        f"reasons={reason_counts}, time={summary['seconds']:.1f}s",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="act_v2/runs/full_random_state_act/best.pt")
    p.add_argument("--target", type=int, default=160)
    p.add_argument("--max-seeds", type=int, default=1200)
    p.add_argument("--n-action-steps", type=int, default=12)
    p.add_argument("--seed-base", type=int, default=30600000)
    p.add_argument("--out-dir", default=str(DAGGER_DIR))
    p.add_argument("--random-handoff-prob", type=float, default=0.20)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--failure-only", action="store_true")
    args = p.parse_args()
    collect_dagger(
        Path(args.checkpoint),
        args.target,
        args.max_seeds,
        args.n_action_steps,
        args.seed_base,
        Path(args.out_dir),
        args.random_handoff_prob,
        args.workers,
        args.failure_only,
    )
