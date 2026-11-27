"""Parallel verified dual-camera RGB exporter for ALL clean Full-Random 360-deg trajectories.

Combines (all bit-exact replay verified with max_err <= 1e-5, filtering legacy singular spawns):
  1. act_v2/full_random_data (0..249): reuses existing act_v2/full_random_rgb/*.npy (0 extra disk/time)
  2. act_v2/full_random_data (250..399): exports remaining clean base episodes
  3. act_v2/full_random_data_extra (0..299): exports all 300 extra clean base episodes
  4. act_v2/full_random_dagger_r2 (0..159): replays v2 policy to exact trigger_step, exports 160 recovery eps
  5. act_v2/full_random_dagger_r3 (0..79): replays v3 policy to exact trigger_step, exports 80 pure-failure recovery eps

Saves pure-vision metadata (robot_xy, actions, coverage) PLUS training-only aux_states (10-d)
strictly separated for training-time auxiliary SE(2) supervision. Inference NEVER touches aux_states.
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
from act_v2.full_random_collect import FULL_DATA_DIR, qualified
from act_v2.full_random_env import CAMERAS, FULL_CAMERA_CONFIG, make_full_env, observe_full
from act_v2.full_random_state import FullRandomStateACTPolicy

OLD_RGB_DIR = ACT_V2 / "full_random_rgb"
ALL_RGB_DIR = ACT_V2 / "full_random_rgb_all"


def _export_worker(tasks: list[dict], out_dir_str: str) -> list[dict]:
    torch.set_num_threads(1)
    out_dir = Path(out_dir_str)
    env = make_full_env(images=True)
    policies: dict[tuple[str, int], FullRandomStateACTPolicy] = {}
    results = []
    try:
        for task in tasks:
            rec_path = out_dir / f"{task['uid']}.json"
            if rec_path.exists():
                results.append(json.loads(rec_path.read_text(encoding="utf-8")))
                continue

            src_npz = Path(task["src_npz"])
            with np.load(src_npz) as d:
                seed = int(d["seed"])
                states = d["states"].astype(np.float32)
                actions = d["actions"].astype(np.float32)
                coverage = d["coverage"].astype(np.float32)
                terminal_state = d["terminal_state"].astype(np.float32)

            if float(states[0, 0]) > 0.12:
                # Skip legacy pre-fix singular spawn episodes
                continue

            n = len(actions)
            meta_path = out_dir / f"{task['uid']}.npz"

            if task.get("reuse_rgb"):
                rgb_rel = task["reuse_rgb"]
                rgb_abs = (ACT_V2 / rgb_rel).resolve()
                rgb_bytes = rgb_abs.stat().st_size
                max_err = float(task.get("old_max_err", 0.0))
            else:
                rgb_name = f"{task['uid']}_rgb.npy"
                rgb_path = out_dir / rgb_name
                tmp_rgb = out_dir / f"{task['uid']}_rgb.tmp.npy"

                env.reset_task(seed)
                if task.get("dagger_ckpt"):
                    pkey = (task["dagger_ckpt"], int(task["dagger_k"]))
                    if pkey not in policies:
                        policies[pkey] = FullRandomStateACTPolicy(
                            checkpoint_path=Path(task["dagger_ckpt"]),
                            n_action_steps=int(task["dagger_k"]),
                            weight_kind="ema",
                        )
                    pol = policies[pkey]
                    pol.reset(seed)
                    for _ in range(int(task["trigger_step"]) + 1):
                        env.move(pol.action(env.state_vector()))

                err0 = float(np.abs(env.state_vector() - states[0]).max())
                if err0 > 1e-4:
                    # Legacy pre-fix staging collision trajectory; skip cleanly rather than corrupting RGB
                    print(f"[SKIP MISMATCH] {task['uid']} seed={seed} err0={err0:.2e}", flush=True)
                    continue

                max_err = err0
                mm = np.lib.format.open_memmap(tmp_rgb, mode="w+", dtype=np.uint8, shape=(n, 2, 128, 128, 3))
                ok_replay = True
                for t, a in enumerate(actions):
                    cur_err = float(np.abs(env.state_vector() - states[t]).max())
                    if cur_err > max_err:
                        max_err = cur_err
                    if cur_err > 1e-4:
                        ok_replay = False
                        break
                    mm[t] = observe_full(env)["images"]
                    env.move(a)
                mm.flush()
                del mm
                if not ok_replay:
                    tmp_rgb.unlink(missing_ok=True)
                    print(f"[SKIP MID MISMATCH] {task['uid']} seed={seed} max_err={max_err:.2e}", flush=True)
                    continue
                term_err = float(np.abs(env.state_vector() - terminal_state).max())
                max_err = max(max_err, term_err)
                if max_err > 1e-4 or not qualified(env.metrics()):
                    tmp_rgb.unlink(missing_ok=True)
                    print(f"[SKIP TERM MISMATCH] {task['uid']} seed={seed} max_err={max_err:.2e}", flush=True)
                    continue
                tmp_rgb.replace(rgb_path)
                rgb_rel = f"full_random_rgb_all/{rgb_name}"
                rgb_bytes = rgb_path.stat().st_size

            tmp_meta = out_dir / f"{task['uid']}.tmp.npz"
            with tmp_meta.open("wb") as f:
                np.savez_compressed(
                    f,
                    robot_xy=states[:, :2],
                    actions=actions,
                    coverage=coverage,
                    training_only_aux_states=states,
                )
            tmp_meta.replace(meta_path)

            rec = {
                "uid": task["uid"],
                "source_group": task["source_group"],
                "seed": seed,
                "frames": n,
                "partition": task["partition"],
                "rgb_rel": rgb_rel,
                "meta_rel": f"full_random_rgb_all/{meta_path.name}",
                "max_replay_error": max_err,
                "rgb_bytes": rgb_bytes,
            }
            save_json(rec_path, rec)
            results.append(rec)
            if len(results) % 15 == 0:
                print(f"  [worker] saved={len(results)}/{len(tasks)} last={task['uid']} err={max_err:.2e}", flush=True)
    finally:
        env.close()
    return results


def build_task_list() -> list[dict]:
    tasks: list[dict] = []

    # 1 & 2. full_random_data (400 eps: reuse 0..249 from old RGB dir, export 250..399)
    old_rgb_map = {}
    if (OLD_RGB_DIR / "complete.json").exists():
        om = json.loads((OLD_RGB_DIR / "complete.json").read_text(encoding="utf-8"))
        for r in om["episodes"]:
            old_rgb_map[int(r["seed"])] = r

    base_m = json.loads((FULL_DATA_DIR / "complete.json").read_text(encoding="utf-8"))
    for idx, row in enumerate(base_m["episodes"]):
        seed = int(row["seed"])
        t = {
            "uid": f"base_{idx:04d}",
            "source_group": "base",
            "src_npz": str(FULL_DATA_DIR / row["file"]),
            "partition": row["partition"],
        }
        if seed in old_rgb_map and (OLD_RGB_DIR / old_rgb_map[seed]["rgb"]).exists():
            t["reuse_rgb"] = f"full_random_rgb/{old_rgb_map[seed]['rgb']}"
            t["old_max_err"] = old_rgb_map[seed].get("max_replay_error", 0.0)
        tasks.append(t)

    # 3. full_random_data_extra (300 clean base eps)
    extra_dir = ACT_V2 / "full_random_data_extra"
    if (extra_dir / "complete.json").exists():
        em = json.loads((extra_dir / "complete.json").read_text(encoding="utf-8"))
        for idx, row in enumerate(em["episodes"]):
            tasks.append(
                {
                    "uid": f"extra_{idx:04d}",
                    "source_group": "extra",
                    "src_npz": str(extra_dir / row["file"]),
                    "partition": "train",
                }
            )

    # 4. full_random_dagger_r2 (160 recovery eps from v2/best.pt exec_12)
    r2_dir = ACT_V2 / "full_random_dagger_r2"
    if (r2_dir / "complete.json").exists():
        m2 = json.loads((r2_dir / "complete.json").read_text(encoding="utf-8"))
        for idx, row in enumerate(m2["episodes"]):
            tasks.append(
                {
                    "uid": f"dagger_r2_{idx:04d}",
                    "source_group": "dagger_r2",
                    "src_npz": str(r2_dir / row["file"]),
                    "partition": "train",
                    "dagger_ckpt": str(ROOT / "act_v2/runs/full_random_state_act_v2/best.pt"),
                    "dagger_k": 12,
                    "trigger_step": int(row["trigger_step"]),
                }
            )

    # 5. full_random_dagger_r3 (80 pure-failure recovery eps from v3/latest.pt exec_18)
    r3_dir = ACT_V2 / "full_random_dagger_r3"
    if (r3_dir / "complete.json").exists():
        m3 = json.loads((r3_dir / "complete.json").read_text(encoding="utf-8"))
        for idx, row in enumerate(m3["episodes"]):
            tasks.append(
                {
                    "uid": f"dagger_r3_{idx:04d}",
                    "source_group": "dagger_r3",
                    "src_npz": str(r3_dir / row["file"]),
                    "partition": "train",
                    "dagger_ckpt": str(ROOT / "act_v2/runs/full_random_state_act_v3/latest.pt"),
                    "dagger_k": 18,
                    "trigger_step": int(row["trigger_step"]),
                }
            )
    return tasks


def export_all(workers: int = 4, limit: int = 0) -> dict:
    ALL_RGB_DIR.mkdir(parents=True, exist_ok=True)
    tasks = build_task_list()
    if limit > 0:
        tasks = tasks[:limit]
    started = time.monotonic()

    # Fast-path: process reuse_rgb tasks first in main process (takes <2s)
    reuse_tasks = [t for t in tasks if t.get("reuse_rgb")]
    render_tasks = [t for t in tasks if not t.get("reuse_rgb")]
    print(f"[RGB ALL] Total tasks={len(tasks)} (reuse={len(reuse_tasks)}, render={len(render_tasks)}) with {workers} workers", flush=True)

    rows = []
    if reuse_tasks:
        rows.extend(_export_worker(reuse_tasks, str(ALL_RGB_DIR)))
        print(f"[RGB ALL] Reused {len(rows)} existing RGB episodes in {time.monotonic()-started:.1f}s", flush=True)

    if render_tasks:
        # Persistent worker pool (zero repeated process-spawn overhead)
        chunks = [render_tasks[i::workers] for i in range(workers) if render_tasks[i::workers]]
        # Sub-divide into ~4 batches per worker only for clean progress reporting? No: print inside worker every 15 eps!
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            futs = [pool.submit(_export_worker, ch, str(ALL_RGB_DIR)) for ch in chunks]
            for wi, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                rows.extend(res)
                print(
                    f"[RGB ALL] worker_done={wi}/{len(chunks)} total_saved={len(rows)} elapsed={(time.monotonic()-started)/60:.1f}m",
                    flush=True,
                )

    rows.sort(key=lambda r: r["uid"])
    train_seeds = {r["seed"] for r in rows if r["partition"] == "train"}
    val_seeds = {r["seed"] for r in rows if r["partition"] == "validation"}
    assert not (train_seeds & val_seeds), "Train/val seed overlap!"

    by_group: dict[str, int] = {}
    for r in rows:
        by_group[r["source_group"]] = by_group.get(r["source_group"], 0) + 1

    summary = {
        "environment": "full_random_v1",
        "episodes": rows,
        "count": len(rows),
        "train_episodes": sum(r["partition"] == "train" for r in rows),
        "val_episodes": sum(r["partition"] == "validation" for r in rows),
        "by_group": by_group,
        "total_frames": int(sum(r["frames"] for r in rows)),
        "image_bytes": int(sum(r["rgb_bytes"] for r in rows)),
        "max_replay_error": float(max(r["max_replay_error"] for r in rows)) if rows else 0.0,
        "cameras": list(CAMERAS),
        "camera_config": FULL_CAMERA_CONFIG,
        "policy_inputs": ["robot_xy[2]", "birdview", "robot0_eye_in_hand"],
        "privileged_inputs": [],
        "training_only_aux": ["SE(2) 14-d geometric targets (unused at inference)"],
        "seconds": time.monotonic() - started,
    }
    save_json(ALL_RGB_DIR / "complete.json", summary)
    print(
        f"[RGB ALL DONE] count={len(rows)} (train={summary['train_episodes']}, val={summary['val_episodes']}), "
        f"groups={by_group}, frames={summary['total_frames']}, bytes={summary['image_bytes']/1e9:.2f}GB, "
        f"max_err={summary['max_replay_error']:.2e}, time={summary['seconds']/60:.1f}m",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    export_all(workers=args.workers, limit=args.limit)
