"""Direct RGB Vision-DAgger collector for Full-Random 360-deg Pure-Vision Vision-ACT.

Rolls out frozen Vision-ACT v2 (pure pixels + robot_xy) on fresh disjoint seeds (30800000+),
detects Vision-specific failure modes (+X boundary drift, 2-chunk oscillation, stalling,
or late-stage 0.85-0.94 near-miss stall), hands over directly to FullRandomExpert, and
renders the dual-camera uint8 RGB frames on-the-fly into act_v2/full_random_rgb_vis_dagger/.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from act_v2.full_random_env import CAMERAS, FULL_CAMERA_CONFIG, make_full_env, observe_full
from act_v2.full_random_vision import FullVisionACTPolicy

VIS_DAGGER_DIR = ACT_V2 / "full_random_rgb_vis_dagger"


def _vis_dagger_worker(seeds: list[int], ckpt_str: str, out_dir_str: str) -> list[dict]:
    torch.set_num_threads(1)
    out_dir = Path(out_dir_str)
    pol = FullVisionACTPolicy(
        checkpoint_path=Path(ckpt_str),
        n_action_steps=12,
        action_schedule=[12, 16, 18, 14],
        mc_dropout_on_freeze=True,
    )
    env = make_full_env(images=True)
    results = []
    try:
        for seed in seeds:
            env.reset_task(seed)
            pol.reset(seed)
            init_t_x = float(env.pose()[0][0])
            best_cov = float(env.metrics()["coverage"])
            last_improve_t = 0
            recent_t = []
            recent_a = []
            triggered = False
            reason = ""
            trig_t = -1
            pol_solved = False

            for t in range(2000):
                obs = observe_full(env)
                a = pol.action(obs)
                env.move(a)
                m = env.metrics()
                if m["coverage"] >= 0.95 and m["tilt"] < 0.2:
                    pol_solved = True
                    break
                if m["tilt"] > 0.2 or m["z"] < 0.79 or abs(env.xy()).max() > 0.35:
                    break
                if m["coverage"] > best_cov + 0.015:
                    best_cov = m["coverage"]
                    last_improve_t = t
                t_pos = env.pose()[0][:2].copy()
                recent_t.append(t_pos)
                recent_a.append(a.copy())
                if len(recent_t) > 140:
                    recent_t.pop(0)
                    recent_a.pop(0)

                # Target exact Vision-ACT v2 failure modes:
                # 1) +X boundary drift toward FOV edge (killed 4/9 blind scenes!)
                if t >= 60 and t_pos[0] > 0.065 and t_pos[0] > init_t_x + 0.02:
                    triggered, reason, trig_t = True, "plus_x_drift", t
                    break
                # 2) -X boundary drift
                if t >= 80 and t_pos[0] < -0.195 and t_pos[0] < init_t_x - 0.02:
                    triggered, reason, trig_t = True, "minus_x_drift", t
                    break
                # 3) Oscillation or stall (T-block barely moved over 140 steps after t>=260)
                if t >= 260 and len(recent_t) == 140:
                    t_disp = float(np.linalg.norm(recent_t[-1] - recent_t[0]))
                    ra = np.asarray(recent_a, dtype=np.float32)
                    mag = float(np.linalg.norm(ra, axis=-1).mean())
                    vec = float(np.linalg.norm(ra.mean(axis=0)))
                    if t_disp < 0.004:
                        triggered = True
                        reason = "oscillating" if (mag > 0.25 and vec < 0.12) else "stalled"
                        trig_t = t
                        break
                # 4) Near-miss fine alignment stall (cov stuck in 0.85..0.945 for 200 steps)
                if t >= 250 and best_cov >= 0.85 and (t - last_improve_t) >= 200:
                    triggered, reason, trig_t = True, "fine_near_miss", t
                    break
                # 5) Late no progress or severe regression
                if t >= 550 and (t - last_improve_t) >= 340:
                    triggered, reason, trig_t = True, "late_no_progress", t
                    break
                if t >= 250 and best_cov >= 0.55 and m["coverage"] < best_cov - 0.20:
                    triggered, reason, trig_t = True, "regressed", t
                    break

            if not triggered:
                results.append({"seed": int(seed), "pol_solved": pol_solved, "triggered": False, "ok": False})
                continue

            # Handover to FullRandomExpert and capture dual-camera RGB directly on the fly!
            expert = FullRandomExpert()
            states = []
            actions = []
            coverage = []
            frames_rgb = []
            holds = 0
            max_tilt = 0.0
            for _ in range(1800):
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
                frames_rgb.append(observe_full(env)["images"])
                actions.append(a.copy())
                env.move(a)
                m = env.metrics()
                max_tilt = max(max_tilt, float(m["tilt"]))
                if m["tilt"] > 0.2 or m["z"] < 0.79 or abs(env.xy()).max() > 0.35:
                    break

            ok = bool(holds >= 5 and qualified(env.metrics()) and len(actions) >= 20 and max_tilt < 0.2)
            if ok:
                uid = f"vis_dagger_{seed}"
                rgb_name = f"{uid}_rgb.npy"
                meta_name = f"{uid}.npz"
                rgb_path = out_dir / rgb_name
                tmp_rgb = out_dir / f"{uid}_rgb.tmp.npy"
                rgb_arr = np.asarray(frames_rgb, dtype=np.uint8)
                np.save(tmp_rgb, rgb_arr)
                tmp_rgb.replace(rgb_path)

                states_arr = np.asarray(states, dtype=np.float32)
                actions_arr = np.asarray(actions, dtype=np.float32)
                cov_arr = np.asarray(coverage, dtype=np.float32)
                meta_path = out_dir / meta_name
                tmp_meta = out_dir / f"{uid}.tmp.npz"
                with tmp_meta.open("wb") as f:
                    np.savez_compressed(
                        f,
                        robot_xy=states_arr[:, :2],
                        actions=actions_arr,
                        coverage=cov_arr,
                        training_only_aux_states=states_arr,
                    )
                tmp_meta.replace(meta_path)
                results.append(
                    {
                        "uid": uid,
                        "source_group": "vis_dagger",
                        "seed": int(seed),
                        "frames": len(actions),
                        "partition": "train",
                        "rgb_rel": f"full_random_rgb_vis_dagger/{rgb_name}",
                        "meta_rel": f"full_random_rgb_vis_dagger/{meta_name}",
                        "max_replay_error": 0.0,
                        "rgb_bytes": rgb_path.stat().st_size,
                        "trigger_reason": reason,
                        "trigger_step": trig_t,
                        "pol_solved": False,
                        "triggered": True,
                        "ok": True,
                    }
                )
            else:
                results.append({"seed": int(seed), "pol_solved": False, "triggered": True, "ok": False})
    finally:
        env.close()
    return results


def collect_vis_dagger(ckpt: Path, target: int = 90, max_seeds: int = 450, seed_base: int = 30800000, workers: int = 4) -> dict:
    VIS_DAGGER_DIR.mkdir(parents=True, exist_ok=True)
    if (VIS_DAGGER_DIR / "complete.json").exists():
        ex = json.loads((VIS_DAGGER_DIR / "complete.json").read_text(encoding="utf-8"))
        if len(ex.get("episodes", [])) >= target:
            print(f"[VIS DAGGER] Already have {len(ex['episodes'])} episodes", flush=True)
            return ex

    started = time.monotonic()
    accepted = []
    attempted = 0
    pol_solved = 0
    triggers = 0
    reasons: dict[str, int] = {}

    while len(accepted) < target and attempted < max_seeds:
        needed = target - len(accepted)
        wave_n = min(max_seeds - attempted, max(workers * 12, int(needed * 2.5)))
        wave_n = min(wave_n, workers * 36)
        seeds = [seed_base + attempted + i for i in range(wave_n)]
        attempted += wave_n
        chunks = [seeds[i::workers] for i in range(workers) if seeds[i::workers]]
        wave_res = []
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            futs = [pool.submit(_vis_dagger_worker, ch, str(ckpt), str(VIS_DAGGER_DIR)) for ch in chunks]
            for fut in as_completed(futs):
                wave_res.extend(fut.result())
        wave_res.sort(key=lambda r: r["seed"])
        for r in wave_res:
            if r["pol_solved"]:
                pol_solved += 1
            if r["triggered"]:
                triggers += 1
            if r["ok"]:
                rc = r["trigger_reason"]
                reasons[rc] = reasons.get(rc, 0) + 1
                accepted.append(r)
        print(
            f"[VIS DAGGER] collected={len(accepted)}/{target} attempted={attempted} "
            f"pol_solved={pol_solved} triggers={triggers} reasons={reasons} elapsed={(time.monotonic()-started)/60:.1f}m",
            flush=True,
        )

    accepted = accepted[:target]
    # Remove surplus files if wave collected slightly more than target
    keep_uids = {r["uid"] for r in accepted}
    for p in VIS_DAGGER_DIR.glob("vis_dagger_*"):
        uid = p.name.replace("_rgb.npy", "").replace(".npz", "")
        if uid not in keep_uids:
            p.unlink(missing_ok=True)

    episodes = [
        {k: r[k] for k in ["uid", "source_group", "seed", "frames", "partition", "rgb_rel", "meta_rel", "max_replay_error", "rgb_bytes", "trigger_reason", "trigger_step"]}
        for r in accepted
    ]
    summary = {
        "source_checkpoint": str(ckpt),
        "count": len(episodes),
        "attempted": attempted,
        "pol_solved": pol_solved,
        "triggers": triggers,
        "reasons": reasons,
        "total_frames": int(sum(e["frames"] for e in episodes)),
        "image_bytes": int(sum(e["rgb_bytes"] for e in episodes)),
        "cameras": list(CAMERAS),
        "camera_config": FULL_CAMERA_CONFIG,
        "seconds": time.monotonic() - started,
        "episodes": episodes,
    }
    save_json(VIS_DAGGER_DIR / "complete.json", summary)
    print(
        f"[VIS DAGGER DONE] {len(episodes)} eps, {summary['total_frames']} frames, "
        f"{summary['image_bytes']/1e9:.2f}GB, reasons={reasons}, time={summary['seconds']/60:.1f}m",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="act_v2/runs/full_random_vision_act_v2/best.pt")
    p.add_argument("--target", type=int, default=90)
    p.add_argument("--max-seeds", type=int, default=450)
    p.add_argument("--seed-base", type=int, default=30800000)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    collect_vis_dagger(Path(args.checkpoint), args.target, args.max_seeds, args.seed_base, args.workers)
