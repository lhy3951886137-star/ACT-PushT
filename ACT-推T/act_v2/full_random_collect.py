"""Symmetry-consistent full-360-degree scripted expert & parallel demo collector for Full-Random Push-T."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from shapely.geometry import LineString, Point, Polygon

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from act_v2.data import ACT_V2, save_json
from act_v2.full_random_env import make_full_env
from precision_env import STEP, VERTICES, rotation

FULL_DATA_DIR = ACT_V2 / "full_random_data"


def qualified(m: dict) -> bool:
    return (
        m["coverage"] >= 0.98
        and m["distance"] < 0.0015
        and m["angle"] < np.deg2rad(1.5)
        and m["tilt"] < 0.2
    )


class FullRandomExpert:
    """Geometry-aware contact planner upgraded for full [-pi, pi] rotations & arbitrary positions.

    Key upgrades over precision_collect.Expert:
    1. Consistent 180-deg symmetry breaking: when |angle_err| > 2.6 rad, locks a consistent
       CCW (+) rotation direction until |angle_err| < 2.2 rad, eliminating +/-pi sign flips
       and preventing multimodal oscillation in the training demonstrations.
    2. Stronger angular torque priority when |angle_err| > 0.35 rad so large rotations
       resolve cleanly inside the workspace before final translation/micro-alignment.
    3. Workspace centering bias when the T-block drifts near the boundary.
    """

    def __init__(self):
        self.contact = None
        self.normal = None
        self.remaining = 0
        self.stage = "approach"
        self.route: deque = deque()
        self.planned_pose = None
        self.rot_lock: int = 0  # +1 CCW lock near +/-pi
        self.approach_steps = 0
        self.last_xy = None
        self.stall_steps = 0
        self.escape_steps = 0
        self.escape_dir = np.zeros(2, dtype=np.float64)

    def _signed_angle_err(self, goal_a: float, cur_a: float) -> float:
        err = float(np.arctan2(np.sin(goal_a - cur_a), np.cos(goal_a - cur_a)))
        if self.rot_lock != 0:
            if abs(err) < 2.1:
                self.rot_lock = 0
            else:
                return abs(err) * self.rot_lock
        elif abs(err) > 2.75:
            # Always break near-180-deg symmetry consistently CCW (+1) for unimodal BC/ACT
            self.rot_lock = 1
            return abs(err)
        return err

    def _desired_twist(self, env) -> np.ndarray:
        p, a = env.pose()
        delta = env.goal[:2] - p[:2]
        ang = self._signed_angle_err(env.goal[2], a)
        dist = float(np.linalg.norm(delta))
        # Workspace safety pull: prevent rotating T-block into Panda's -X / +X joint limits
        if p[0] < -0.175:
            delta = delta + np.array([(-0.175 - p[0]) * 2.5, 0.0], dtype=np.float64)
        elif p[0] > 0.035:
            delta = delta + np.array([(0.035 - p[0]) * 2.5, 0.0], dtype=np.float64)
        # Always pull strongly toward goal if far, while applying crisp torque for large angles
        if dist > 0.08:
            return np.r_[delta / 0.045, ang * 0.85]
        if abs(ang) > 0.45:
            return np.r_[delta / 0.065, ang * 1.35]
        return np.r_[delta / 0.055, ang * 0.95]

    def plan(self, env) -> None:
        p, a = env.pose()
        R = rotation(a)
        xy = env.xy()
        desired = self._desired_twist(env)
        dnorm = float(np.linalg.norm(desired)) + 1e-9
        com = env.sim.model.body_ipos[env.cube_body_id, :2].copy()
        clearance = Polygon(VERTICES @ R.T + p[:2]).buffer(0.0118, join_style=2)
        candidates = []
        for v1, v2 in zip(VERTICES, np.roll(VERTICES, -1, axis=0)):
            tangent = (v2 - v1) / np.linalg.norm(v2 - v1)
            normal = np.array([-tangent[1], tangent[0]])
            for f in np.linspace(0.08, 0.92, 17):
                local = v1 + (v2 - v1) * f
                world = p[:2] + R @ local
                n = R @ normal
                approach_pt = world - n * 0.014
                # Enforce Panda physical reachability (+X <= 0.105, -X >= -0.265, |Y| <= 0.25)
                if approach_pt[0] > 0.105 or approach_pt[0] < -0.265 or abs(approach_pt[1]) > 0.25:
                    continue
                if clearance.contains(Point(approach_pt)):
                    continue
                arm = R @ (local - com)
                twist = np.r_[n, np.cross(arm, n) / 0.045]
                cos_sim = float((desired @ twist) / (dnorm * (np.linalg.norm(twist) + 1e-9)))
                score = cos_sim - 0.30 * float(np.linalg.norm(world - xy))
                candidates.append((score, local, normal))
        if not candidates:
            # Fallback if near +X edge: pick best contact without X filter
            for v1, v2 in zip(VERTICES, np.roll(VERTICES, -1, axis=0)):
                tangent = (v2 - v1) / np.linalg.norm(v2 - v1)
                normal = np.array([-tangent[1], tangent[0]])
                local = 0.5 * (v1 + v2)
                world = p[:2] + R @ local
                n = R @ normal
                if clearance.contains(Point(world - n * 0.014)):
                    continue
                arm = R @ (local - com)
                twist = np.r_[n, np.cross(arm, n) / 0.045]
                cos_sim = float((desired @ twist) / (dnorm * (np.linalg.norm(twist) + 1e-9)))
                candidates.append((cos_sim, local, normal))
        _, self.contact, self.normal = max(candidates, key=lambda x: x[0])
        self.remaining = 65
        self.stage = "approach"
        self.route = deque()
        self.planned_pose = np.r_[p[:2], a]
        self.approach_steps = 0
        self.stall_steps = 0

    def route_target(self, xy, target, obstacle, center):
        if not self.route:
            safety = obstacle.buffer(-0.0018)
            if not LineString([xy, target]).intersects(safety):
                return target
            # Sample exterior vertices of obstacle (which lie 1.8mm strictly outside `safety`)
            ring_pts = [
                pt for pt in np.asarray(obstacle.exterior.coords)[:-1]
                if -0.275 <= pt[0] <= 0.108 and abs(pt[1]) <= 0.26
            ]
            nodes = [xy, target, *ring_pts]
            dist = [float("inf")] * len(nodes)
            dist[0] = 0.0
            previous = {}
            visited = set()
            for _ in nodes:
                u = min((i for i in range(len(nodes)) if i not in visited), key=lambda i: dist[i])
                if not np.isfinite(dist[u]) or u == 1:
                    break
                visited.add(u)
                for v in range(1, len(nodes)):
                    if v == u or v in visited:
                        continue
                    if LineString([nodes[u], nodes[v]]).intersects(safety):
                        continue
                    cost = dist[u] + float(np.linalg.norm(np.asarray(nodes[u]) - nodes[v]))
                    if cost < dist[v]:
                        dist[v] = cost
                        previous[v] = u
            if 1 in previous:
                route = [1]
                v = 1
                while previous[v] != 0:
                    v = previous[v]
                    route.append(v)
                self.route = deque(np.asarray(nodes[i]) for i in reversed(route))
            else:
                # Step cleanly outward away from obstacle for 5 steps to avoid boundary ping-pong
                from shapely.ops import nearest_points
                _, bpt = nearest_points(Point(xy), obstacle.exterior)
                out = np.array([bpt.x, bpt.y]) - xy
                nrm = float(np.linalg.norm(out))
                if nrm < 1e-6:
                    out = xy - center
                    nrm = float(np.linalg.norm(out)) + 1e-9
                self.escape_dir = out / nrm
                self.escape_steps = 5
                return xy + self.escape_dir * 0.02
        while len(self.route) > 1 and np.linalg.norm(xy - self.route[0]) < 0.006:
            self.route.popleft()
        return self.route[0]

    def action(self, env) -> np.ndarray:
        if self.escape_steps > 0:
            self.escape_steps -= 1
            return np.asarray(np.clip(self.escape_dir * 0.9, -1, 1), np.float32)

        if self.contact is None or self.remaining <= 0:
            self.plan(env)
        p, a = env.pose()
        R = rotation(a)
        xy = env.xy()
        if self.last_xy is not None and float(np.linalg.norm(xy - self.last_xy)) < 0.00025:
            self.stall_steps += 1
        else:
            self.stall_steps = 0
        self.last_xy = xy.copy()

        normal = R @ self.normal
        contact = p[:2] + R @ self.contact
        if self.stage == "approach" and (
            np.linalg.norm(p[:2] - self.planned_pose[:2]) > 0.004 or abs(a - self.planned_pose[2]) > 0.04
        ):
            self.plan(env)
            normal = R @ self.normal
            contact = p[:2] + R @ self.contact
        if self.stage == "approach":
            self.approach_steps += 1
            if self.stall_steps >= 15 or self.approach_steps > 150:
                radial = xy - p[:2]
                radial = radial / (float(np.linalg.norm(radial)) + 1e-9)
                self.escape_dir = radial
                self.escape_steps = 6
                self.contact = None
                self.stall_steps = 0
                return np.asarray(np.clip(radial * 0.9, -1, 1), np.float32)
            approach = contact - normal * 0.015
            if np.linalg.norm(xy - approach) < 0.004:
                self.stage = "push"
                self.route = deque()
            else:
                obstacle = Polygon(VERTICES @ R.T + p[:2]).buffer(0.0108, join_style=2)
                target = self.route_target(xy, approach, obstacle, p[:2])
                diff = target - xy
                gain = 0.65 if float(np.linalg.norm(diff)) < 0.012 else 0.95
                return np.asarray(np.clip(diff / STEP * gain, -1, 1), np.float32)

        gap = float((contact - xy) @ normal)
        if gap > 0.010:
            self.remaining = max(self.remaining, 20)
        else:
            self.remaining -= 1

        delta = env.goal[:2] - p[:2]
        signed_angle = self._signed_angle_err(env.goal[2], a)
        angle = abs(signed_angle)
        arm = R @ (self.contact - env.sim.model.body_ipos[env.cube_body_id, :2])
        twist = np.r_[normal, np.cross(arm, normal) / 0.045]
        desired = self._desired_twist(env)
        cos_sim = float((desired @ twist) / ((np.linalg.norm(desired) + 1e-9) * (np.linalg.norm(twist) + 1e-9)))
        if cos_sim < 0.15:
            self.remaining = 0
            return np.asarray(-normal * 0.5, np.float32)

        cov = env.metrics()["coverage"]
        max_inc = 0.0038 if cov < 0.70 else (0.0025 if cov < 0.90 else 0.0016)
        increment = float(np.clip(np.linalg.norm(delta) * 0.25 + angle * 0.006, 0.00008, max_inc))
        friction_compensation = 0.12 * 0.21 * 9.81 / 300.0
        target = contact - normal * 0.008 + normal * (friction_compensation + increment)
        displacement = target - xy
        along = float(displacement @ normal)
        command = (along * normal + 0.85 * (displacement - along * normal)) / STEP
        cap = 0.5 if cov > 0.90 else (0.85 if cov > 0.70 else 1.0)
        return np.asarray(np.clip(command, -cap, cap), np.float32)


def rollout_one(env, seed: int, horizon: int = 2400):
    env.reset_task(seed)
    expert = FullRandomExpert()
    states = []
    actions = []
    metrics = []
    coverage = []
    holds = 0
    for _ in range(horizon):
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
    success = bool(holds >= 5 and qualified(env.metrics()) and max(r[5] for r in metrics) < 0.2)
    data = dict(
        seed=np.int64(seed),
        states=np.asarray(states, np.float32),
        actions=np.asarray(actions, np.float32),
        coverage=np.asarray(coverage, np.float32),
        metrics=np.asarray(metrics, np.float32),
        terminal_state=env.state_vector(),
        terminal_metrics=np.asarray([env.metrics()[k] for k in ["coverage", "iou", "distance", "angle", "z", "tilt"]], np.float32),
    )
    row = dict(seed=int(seed), success=success, frames=len(actions), holds=holds, final=env.metrics())
    return data, row


def _worker_batch(seeds: list[int], out_dir_str: str) -> list[dict]:
    out_dir = Path(out_dir_str)
    env = make_full_env(images=False)
    results = []
    try:
        for seed in seeds:
            raw_path = out_dir / f"raw_seed_{seed}.npz"
            meta_path = out_dir / f"raw_seed_{seed}.json"
            if meta_path.exists():
                results.append(json.loads(meta_path.read_text(encoding="utf-8")))
                continue
            data, row = rollout_one(env, seed)
            if row["success"]:
                tmp = out_dir / f"raw_seed_{seed}.tmp.npz"
                with tmp.open("wb") as f:
                    np.savez_compressed(f, **data)
                tmp.replace(raw_path)
            save_json(meta_path, row)
            results.append(row)
    finally:
        env.close()
    return results


def verify_replays(out_dir: Path, episodes: list[dict], count: int = 5) -> list[dict]:
    env = make_full_env(images=False)
    checks = []
    try:
        for row in episodes[:count]:
            with np.load(out_dir / row["file"]) as d:
                seed = int(d["seed"])
                states = d["states"]
                actions = d["actions"]
            env.reset_task(seed)
            max_err = 0.0
            for raw, a in zip(states, actions):
                max_err = max(max_err, float(np.abs(env.state_vector() - raw).max()))
                env.move(a)
            m = env.metrics()
            passed = bool(max_err <= 1e-5 and qualified(m))
            checks.append({"seed": seed, "max_state_error": max_err, "passed": passed, "final": m})
            if not passed:
                raise RuntimeError(f"Full-random replay audit failed for seed {seed}: err={max_err}, final={m}")
    finally:
        env.close()
    return checks


def collect_parallel(
    target: int = 400,
    workers: int = 10,
    seed_base: int = 30000000,
    max_attempts: int = 1200,
    out_dir: Path = FULL_DATA_DIR,
    all_train: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "complete.json").exists():
        existing = json.loads((out_dir / "complete.json").read_text(encoding="utf-8"))
        if len(existing.get("episodes", [])) >= target:
            print(f"[COLLECT] Already have {len(existing['episodes'])} episodes in {out_dir}", flush=True)
            return existing

    started = time.monotonic()
    accepted_rows = []
    attempted = 0
    batch_size = workers * 4  # 40 seeds per parallel wave

    while len(accepted_rows) < target and attempted < max_attempts:
        needed = target - len(accepted_rows)
        wave_n = min(max_attempts - attempted, max(workers, int(needed * 1.25) + workers))
        wave_n = min(wave_n, batch_size * 2)
        seeds = [seed_base + attempted + i for i in range(wave_n)]
        attempted += wave_n

        chunks = [seeds[i::workers] for i in range(workers) if seeds[i::workers]]
        wave_results = []
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            futs = [pool.submit(_worker_batch, ch, str(out_dir)) for ch in chunks]
            for fut in as_completed(futs):
                wave_results.extend(fut.result())

        wave_results.sort(key=lambda r: r["seed"])
        wave_ok = [r for r in wave_results if r["success"]]
        accepted_rows.extend(wave_ok)
        elapsed = time.monotonic() - started
        print(
            f"[COLLECT] wave_ok={len(wave_ok)}/{len(wave_results)} total_accepted={len(accepted_rows)}/{target} "
            f"attempted={attempted} elapsed={elapsed:.1f}s",
            flush=True,
        )

    if len(accepted_rows) < target:
        raise RuntimeError(f"Only collected {len(accepted_rows)}/{target} within {attempted} attempts")

    accepted_rows = accepted_rows[:target]
    order = np.random.default_rng(20261126).permutation(target)
    n_val = 0 if all_train else max(20, target // 10)  # 40 val, 360 train for target=400
    val_set = set(order[:n_val].tolist())

    episodes = []
    for idx, row in enumerate(accepted_rows):
        seed = row["seed"]
        src = out_dir / f"raw_seed_{seed}.npz"
        dst = out_dir / f"episode_{idx:04d}.npz"
        if not dst.exists():
            src.replace(dst)
        elif src.exists():
            src.unlink()
        with np.load(dst) as d:
            frames = int(len(d["actions"]))
            terminal = d["terminal_metrics"].tolist()
            init_state = d["states"][0].tolist()
        episodes.append(
            {
                "index": idx,
                "file": dst.name,
                "seed": seed,
                "frames": frames,
                "partition": "validation" if idx in val_set else "train",
                "init_state": init_state,
                "terminal_metrics": terminal,
                "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
            }
        )

    # Clean up unaccepted raw_seed_*.json / surplus npz
    for p in out_dir.glob("raw_seed_*.npz"):
        p.unlink(missing_ok=True)

    checks = verify_replays(out_dir, episodes, count=5)
    summary = {
        "environment": "full_random_v1",
        "target": target,
        "attempted": attempted,
        "expert_acceptance_rate": float(len(accepted_rows) / attempted),
        "total_frames": int(sum(e["frames"] for e in episodes)),
        "mean_frames": float(np.mean([e["frames"] for e in episodes])),
        "train_episodes": sum(e["partition"] == "train" for e in episodes),
        "val_episodes": sum(e["partition"] == "validation" for e in episodes),
        "replay_checks": checks,
        "seconds": time.monotonic() - started,
        "episodes": episodes,
    }
    save_json(out_dir / "complete.json", summary)
    print(
        f"[COLLECT DONE] {target} episodes, total_frames={summary['total_frames']}, "
        f"mean_frames={summary['mean_frames']:.1f}, accept={summary['expert_acceptance_rate']:.3f}, "
        f"replay_max_err={max(c['max_state_error'] for c in checks):.2e}, time={summary['seconds']:.1f}s",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--probe", type=int, default=0, help="Run quick N-seed expert probe without saving dataset")
    p.add_argument("--target", type=int, default=400)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=30000000)
    p.add_argument("--max-attempts", type=int, default=1200)
    p.add_argument("--out-dir", default=str(FULL_DATA_DIR))
    p.add_argument("--all-train", action="store_true")
    args = p.parse_args()

    if args.probe > 0:
        t0 = time.monotonic()
        env = make_full_env(images=False)
        ok = 0
        frames = []
        angles = []
        try:
            for i in range(args.probe):
                d, r = rollout_one(env, 30900000 + i)
                s0 = d["states"][0]
                init_ang = np.degrees(np.arctan2(s0[4], s0[5]))
                init_dist = np.linalg.norm(s0[2:4] - s0[6:8]) * 100.0
                angles.append(init_ang)
                if r["success"]:
                    ok += 1
                    frames.append(r["frames"])
                print(
                    f"  seed={30900000+i} init_dist={init_dist:5.1f}cm init_ang={init_ang:+6.1f}deg "
                    f"ok={r['success']} frames={r['frames']:4d} cov={r['final']['coverage']:.4f}",
                    flush=True,
                )
        finally:
            env.close()
        print(
            f"[PROBE] {ok}/{args.probe} ({ok/args.probe:.1%}), mean_frames={np.mean(frames) if frames else 0:.1f}, "
            f"ang_range=[{min(angles):.1f}, {max(angles):.1f}]deg, elapsed={time.monotonic()-t0:.1f}s",
            flush=True,
        )
    else:
        collect_parallel(
            target=args.target,
            workers=args.workers,
            seed_base=args.seed_base,
            max_attempts=args.max_attempts,
            out_dir=Path(args.out_dir),
            all_train=args.all_train,
        )
