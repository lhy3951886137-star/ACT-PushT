"""Full-Random Position & Full-360-deg Angle Pure-Vision Dual-Camera ACT (export + train + confirm).

Upgraded v2 capabilities (100% pure ACT, 100% pure pixels + robot_xy at inference):
  - Loads unified multi-source RGB manifests (act_v2/full_random_rgb_all/complete.json)
  - Training-only dual-level SE(2) auxiliary geometric supervision:
      1) Pure-pixel per-frame head on crop-shift-invariant 8-d SE(2) geometry
      2) DETR encoder-memory head on full 14-d SE(2) geometry
      (NEVER executed or required at inference; Zero-Image Control verifies 0.0%)
  - Efficiency-balanced episode weighting (favors crisp, shorter trajectories to minimize rollout steps)
  - 4-stage angle+coverage stratified frame sampling + action-chunk smoothness loss
  - Aperiodic ACT chunk execution ([12, 16, 18, 14]) + pure-proprioception freeze-wake
  - Parallel multi-worker vision evaluation for fast dev & blind confirmation
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from act_v2.data import ACT_V2, AsyncPrefetcher, save_json
from act_v2.full_random_collect import FULL_DATA_DIR, qualified
from act_v2.full_random_env import CAMERAS, FULL_CAMERA_CONFIG, make_full_env, observe_full
from act_v2.full_random_state import chunk_smoothness_l1_loss
from act_v2.inference import wilson_ci95
from act_v2.model import VisionACT, weighted_l1_loss
from act_v2.train_state import update_ema

FULL_RGB_DIR = ACT_V2 / "full_random_rgb"
ALL_RGB_DIR = ACT_V2 / "full_random_rgb_all"


def aux_se2_targets(states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute training-only normalized SE(2) auxiliary targets from 10-d env states.

    Returns:
      aux_vis [..., 8]: crop-shift-invariant pure-pixel targets
        [cos_T, sin_T, g_dx/0.16, g_dy/0.16, g_loc_x/0.16, g_loc_y/0.16, cos_d, sin_d]
      aux_enc [..., 14]: full SE(2) encoder-memory targets
        [(T_x+0.08)/0.14, T_y/0.16, cos_T, sin_T, (g_x+0.07)/0.03, g_y/0.035,
         g_dx/0.16, g_dy/0.16, ee_loc_x/0.22, ee_loc_y/0.22, g_loc_x/0.16, g_loc_y/0.16, cos_d, sin_d]
    """
    r = np.asarray(states, dtype=np.float32)
    ee = r[..., 0:2]
    t_pos = r[..., 2:4]
    sin_t = r[..., 4:5]
    cos_t = r[..., 5:6]
    g_pos = r[..., 6:8]
    sin_g = r[..., 8:9]
    cos_g = r[..., 9:10]

    ee_rel = ee - t_pos
    ee_loc_x = cos_t * ee_rel[..., 0:1] + sin_t * ee_rel[..., 1:2]
    ee_loc_y = -sin_t * ee_rel[..., 0:1] + cos_t * ee_rel[..., 1:2]

    g_rel = g_pos - t_pos
    g_loc_x = cos_t * g_rel[..., 0:1] + sin_t * g_rel[..., 1:2]
    g_loc_y = -sin_t * g_rel[..., 0:1] + cos_t * g_rel[..., 1:2]

    cos_d = np.clip(cos_g * cos_t + sin_g * sin_t, -1.0, 1.0)
    sin_d = np.clip(sin_g * cos_t - cos_g * sin_t, -1.0, 1.0)

    aux_vis = np.concatenate(
        [cos_t, sin_t, g_rel / 0.16, g_loc_x / 0.16, g_loc_y / 0.16, cos_d, sin_d],
        axis=-1,
    ).astype(np.float32)
    aux_enc = np.concatenate(
        [
            (t_pos[..., 0:1] + 0.08) / 0.14,
            t_pos[..., 1:2] / 0.16,
            cos_t,
            sin_t,
            (g_pos[..., 0:1] + 0.07) / 0.03,
            g_pos[..., 1:2] / 0.035,
            g_rel / 0.16,
            ee_loc_x / 0.22,
            ee_loc_y / 0.22,
            g_loc_x / 0.16,
            g_loc_y / 0.16,
            cos_d,
            sin_d,
        ],
        axis=-1,
    ).astype(np.float32)
    return aux_vis, aux_enc


def export_rgb(limit: int = 250) -> dict:
    """Deterministic physical replay of verified full-random episodes -> uint8 dual-camera memmaps."""
    FULL_RGB_DIR.mkdir(parents=True, exist_ok=True)
    complete_path = FULL_RGB_DIR / "complete.json"
    if complete_path.exists():
        ex = json.loads(complete_path.read_text(encoding="utf-8"))
        if len(ex.get("episodes", [])) >= limit:
            print(f"[RGB EXPORT] Already exported {len(ex['episodes'])} episodes", flush=True)
            return ex

    src_manifest = json.loads((FULL_DATA_DIR / "complete.json").read_text(encoding="utf-8"))
    src_eps = src_manifest["episodes"][:limit]
    env = make_full_env(images=True)
    rows = []
    started = time.monotonic()
    try:
        for idx, row in enumerate(src_eps):
            rec_path = FULL_RGB_DIR / f"episode_{idx:04d}.json"
            if rec_path.exists():
                rows.append(json.loads(rec_path.read_text(encoding="utf-8")))
                continue
            with np.load(FULL_DATA_DIR / row["file"]) as d:
                seed = int(d["seed"])
                states = d["states"].astype(np.float32)
                actions = d["actions"].astype(np.float32)
                coverage = d["coverage"].astype(np.float32)
                terminal_state = d["terminal_state"].astype(np.float32)

            n = len(actions)
            rgb_path = FULL_RGB_DIR / f"episode_{idx:04d}_rgb.npy"
            tmp_rgb = FULL_RGB_DIR / f"episode_{idx:04d}_rgb.tmp.npy"
            meta_path = FULL_RGB_DIR / f"episode_{idx:04d}.npz"

            env.reset_task(seed)
            max_err = 0.0
            mm = np.lib.format.open_memmap(tmp_rgb, mode="w+", dtype=np.uint8, shape=(n, 2, 128, 128, 3))
            for t, a in enumerate(actions):
                max_err = max(max_err, float(np.abs(env.state_vector() - states[t]).max()))
                if max_err > 1e-5:
                    del mm
                    raise RuntimeError(f"Full-random RGB replay mismatch on ep {idx} t={t}: {max_err}")
                mm[t] = observe_full(env)["images"]
                if idx == 0 and t in (0, n // 2, n - 1):
                    pdir = FULL_RGB_DIR / "previews"
                    pdir.mkdir(exist_ok=True)
                    for ci, cname in enumerate(CAMERAS):
                        Image.fromarray(mm[t, ci]).save(pdir / f"{cname}_{t:04d}.png")
                env.move(a)
            mm.flush()
            del mm
            max_err = max(max_err, float(np.abs(env.state_vector() - terminal_state).max()))
            if max_err > 1e-5 or not qualified(env.metrics()):
                raise RuntimeError(f"Terminal RGB replay failed ep {idx}: err={max_err}")
            tmp_rgb.replace(rgb_path)

            tmp_meta = FULL_RGB_DIR / f"episode_{idx:04d}.tmp.npz"
            with tmp_meta.open("wb") as f:
                np.savez_compressed(f, robot_xy=states[:, :2], actions=actions, coverage=coverage)
            tmp_meta.replace(meta_path)

            rec = {
                "index": idx,
                "seed": seed,
                "frames": n,
                "partition": row["partition"],
                "rgb": rgb_path.name,
                "metadata": meta_path.name,
                "max_replay_error": max_err,
                "rgb_bytes": rgb_path.stat().st_size,
            }
            save_json(rec_path, rec)
            rows.append(rec)
    finally:
        env.close()

    summary = {
        "environment": "full_random_v1",
        "episodes": rows,
        "total_frames": int(sum(r["frames"] for r in rows)),
        "image_bytes": int(sum(r["rgb_bytes"] for r in rows)),
        "cameras": list(CAMERAS),
        "camera_config": FULL_CAMERA_CONFIG,
        "policy_inputs": ["robot_xy[2]", "birdview", "robot0_eye_in_hand"],
        "privileged_inputs": [],
        "seconds": time.monotonic() - started,
    }
    save_json(complete_path, summary)
    return summary


class FullVisionSampler:
    def __init__(
        self,
        rgb_dir: Path = FULL_RGB_DIR,
        extra_rgb_dirs: Optional[List[Path]] = None,
        stats: Optional[dict] = None,
        n_obs_steps: int = 3,
        chunk_size: int = 24,
        seed: int = 42,
        validation: bool = False,
        use_aux: bool = False,
        efficiency_weighting: bool = True,
        speed_power: float = 0.35,
        r3_boost: float = 2.5,
        r2_boost: float = 1.2,
        vis_dagger_boost: float = 3.0,
        balanced_angles: bool = False,
    ):
        self.rgb_dir = Path(rgb_dir)
        if os.name == "nt":
            try:
                import ctypes
                ctypes.CDLL("ucrtbase")._setmaxstdio(2048)
            except Exception:
                pass
        manifest = json.loads((self.rgb_dir / "complete.json").read_text(encoding="utf-8"))
        rows = [r for r in manifest["episodes"] if (r["partition"] == "validation") == validation]
        if extra_rgb_dirs and not validation:
            for edir in extra_rgb_dirs:
                mp = Path(edir) / "complete.json"
                if mp.exists():
                    em = json.loads(mp.read_text(encoding="utf-8"))
                    rows.extend(r for r in em["episodes"] if r.get("partition", "train") == "train")
        self.episodes = []
        self.rng = np.random.default_rng(seed)
        self.n_obs_steps = n_obs_steps
        self.chunk_size = chunk_size
        self.validation = validation
        self.use_aux = use_aux
        self.balanced_angles = balanced_angles

        for r in rows:
            meta_path = ACT_V2 / r["meta_rel"] if "meta_rel" in r else (self.rgb_dir / r["metadata"])
            rgb_path = ACT_V2 / r["rgb_rel"] if "rgb_rel" in r else (self.rgb_dir / r["rgb"])
            with np.load(meta_path) as d:
                robot_xy = d["robot_xy"].astype(np.float32).copy()
                actions = d["actions"].astype(np.float32).copy()
                coverage = d["coverage"].astype(np.float32).copy()
                aux_states = d["training_only_aux_states"].astype(np.float32).copy() if "training_only_aux_states" in d.files else None

            ep = {
                "robot_xy": robot_xy,
                "actions": actions,
                "coverage": coverage,
                "images": np.load(rgb_path, mmap_mode="r"),
                "mid": np.flatnonzero((coverage > 0.45) & (coverage <= 0.88)),
                "fine": np.flatnonzero(coverage > 0.88),
                "seed": int(r["seed"]),
                "source_group": r.get("source_group", "base"),
            }
            if aux_states is not None:
                av, ae = aux_se2_targets(aux_states)
                ep["aux_vis"] = av
                ep["aux_enc"] = ae
                cos_d = aux_states[:, 9] * aux_states[:, 5] + aux_states[:, 8] * aux_states[:, 4]
                ep["large_angle"] = np.flatnonzero(cos_d < 0.0)
            else:
                ep["large_angle"] = np.empty(0, dtype=np.int64)
            self.episodes.append(ep)

        lengths = np.asarray([len(e["actions"]) for e in self.episodes], dtype=np.float64)
        if efficiency_weighting and not validation:
            # Favor crisp, faster trajectories to shorten rollout steps while keeping high success
            w = (1150.0 / np.maximum(lengths, 650.0)) ** speed_power
            for i, e in enumerate(self.episodes):
                if e["source_group"] == "vis_dagger":
                    w[i] *= vis_dagger_boost
                elif e["source_group"] == "dagger_r3":
                    w[i] *= r3_boost
                elif e["source_group"] == "dagger_r2":
                    w[i] *= r2_boost
            self.weights = w / w.sum()
        else:
            self.weights = lengths / lengths.sum()

        if stats is None:
            xy = np.concatenate([e["robot_xy"] for e in self.episodes], axis=0)
            lo = xy.min(axis=0)
            hi = xy.max(axis=0)
            stats = {
                "robot_center": ((lo + hi) / 2.0).tolist(),
                "robot_scale": np.maximum((hi - lo) / 2.0, 0.01).tolist(),
                "action_center": [0.0, 0.0],
                "action_scale": [1.0, 1.0],
            }
        self.stats = stats
        self.robot_center = np.asarray(stats["robot_center"], dtype=np.float32)
        self.robot_scale = np.asarray(stats["robot_scale"], dtype=np.float32)

    def sample(self, batch_size: int):
        to = self.n_obs_steps
        k = self.chunk_size
        robot = np.empty((batch_size, to, 2), dtype=np.float32)
        images = np.empty((batch_size, to, 2, 128, 128, 3), dtype=np.uint8)
        actions = np.zeros((batch_size, k, 2), dtype=np.float32)
        valid = np.zeros((batch_size, k), dtype=bool)
        aux_vis = np.zeros((batch_size, to, 8), dtype=np.float32) if self.use_aux else None
        aux_enc = np.zeros((batch_size, 14), dtype=np.float32) if self.use_aux else None

        ep_indices = self.rng.choice(len(self.episodes), size=batch_size, p=self.weights)
        for bi, ei in enumerate(ep_indices):
            e = self.episodes[ei]
            n = len(e["actions"])
            if not self.validation:
                u = self.rng.random()
                if self.balanced_angles:
                    # 30% fine alignment (cures 0.92-0.94 near-misses), 20% mid, 10% large-angle, 40% uniform (sharp 0-45deg)
                    if u < 0.30 and len(e["fine"]) > 0:
                        t = int(self.rng.choice(e["fine"]))
                    elif u < 0.50 and len(e["mid"]) > 0:
                        t = int(self.rng.choice(e["mid"]))
                    elif u < 0.60 and len(e["large_angle"]) > 0:
                        t = int(self.rng.choice(e["large_angle"]))
                    else:
                        t = int(self.rng.integers(n))
                else:
                    if u < 0.25 and len(e["fine"]) > 0:
                        t = int(self.rng.choice(e["fine"]))
                    elif u < 0.50 and len(e["mid"]) > 0:
                        t = int(self.rng.choice(e["mid"]))
                    elif u < 0.70 and len(e["large_angle"]) > 0:
                        t = int(self.rng.choice(e["large_angle"]))
                    else:
                        t = int(self.rng.integers(n))
            else:
                t = int(self.rng.integers(n))

            hist = np.maximum(0, t - np.arange(to - 1, -1, -1))
            robot[bi] = e["robot_xy"][hist]
            images[bi] = e["images"][hist]
            count = min(k, n - t)
            actions[bi, :count] = e["actions"][t : t + count]
            valid[bi, :count] = True
            if self.use_aux:
                aux_vis[bi] = e["aux_vis"][hist]
                aux_enc[bi] = e["aux_enc"][t]

        robot = (robot - self.robot_center) / self.robot_scale
        if not self.validation:
            # Tiny temporally-correlated proprioception noise (preserves velocity)
            robot = robot + self.rng.normal(0.0, 0.003, size=(batch_size, 1, 2)).astype(np.float32)
        if self.use_aux:
            return robot, images, actions, valid, aux_vis, aux_enc
        return robot, images, actions, valid


class FullVisionACTPolicy:
    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        model: Optional[VisionACT] = None,
        stats: Optional[dict] = None,
        cfg: Optional[dict] = None,
        n_action_steps: int = 12,
        action_schedule: Optional[List[int]] = None,
        mc_dropout_on_freeze: bool = False,
        weight_kind: str = "ema",
    ):
        if model is None:
            ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            cfg = ck["config"]
            stats = ck["stats"]
            model = VisionACT(
                n_obs_steps=cfg["n_obs_steps"],
                chunk_size=cfg["chunk_size"],
                d_model=cfg["d_model"],
                nhead=cfg["nhead"],
                num_encoder_layers=cfg["num_encoder_layers"],
                num_decoder_layers=cfg["num_decoder_layers"],
                dim_feedforward=cfg["dim_feedforward"],
                dropout=cfg["dropout"],
                use_vae=False,
                crop_size=cfg.get("crop_size", 116),
                token_mode=cfg.get("token_mode", "keypoints"),
                keypoints_l3=cfg.get("keypoints_l3", 32),
                keypoints_l4=cfg.get("keypoints_l4", 32),
                aux_geom_dim=cfg.get("aux_geom_dim", 0),
            ).cuda().eval()
            model.load_state_dict(ck[weight_kind] if weight_kind in ck else ck["model"])
        self.model = model.eval()
        self.stats = stats
        self.cfg = cfg
        self.n_obs_steps = int(cfg["n_obs_steps"])
        self.n_action_steps = int(n_action_steps)
        self.action_schedule = [int(x) for x in action_schedule] if action_schedule else None
        self.mc_dropout_on_freeze = bool(mc_dropout_on_freeze)
        self.robot_center = np.asarray(stats["robot_center"], dtype=np.float32)
        self.robot_scale = np.asarray(stats["robot_scale"], dtype=np.float32)
        self.reset()

    def reset(self, seed: int = 0) -> None:
        self.history: deque = deque(maxlen=self.n_obs_steps)
        self.queue: deque = deque()
        self.replan_idx = 0
        self.ee_hist: deque = deque(maxlen=120)
        self.freeze_rng = np.random.default_rng(seed + 2026)

    @torch.inference_mode()
    def action(self, observation: dict) -> np.ndarray:
        assert set(observation) == {"robot_xy", "images"} and observation["robot_xy"].shape == (2,)
        self.history.append(observation)
        while len(self.history) < self.n_obs_steps:
            self.history.append(observation)
        self.ee_hist.append(observation["robot_xy"].copy())

        if not self.queue:
            robot = np.stack([x["robot_xy"] for x in self.history], axis=0)
            images = np.stack([x["images"] for x in self.history], axis=0)
            robot = (robot - self.robot_center) / self.robot_scale

            frozen = False
            if self.mc_dropout_on_freeze and len(self.ee_hist) == 120:
                eh = np.asarray(self.ee_hist, dtype=np.float32)
                ee_span = float(np.linalg.norm(eh.max(0) - eh.min(0)))
                if ee_span < 0.0025:
                    frozen = True

            if frozen:
                robot = robot + self.freeze_rng.normal(0.0, 0.012, size=(1, 2)).astype(np.float32)
                r_t = torch.as_tensor(robot, dtype=torch.float32, device="cuda")[None]
                i_t = torch.as_tensor(images, dtype=torch.uint8, device="cuda")[None]
                self.model.train()
                chunk = self.model(r_t, i_t, augment=False, return_aux=False)[0][0].float().cpu().numpy()
                self.model.eval()
            else:
                r_t = torch.as_tensor(robot, dtype=torch.float32, device="cuda")[None]
                i_t = torch.as_tensor(images, dtype=torch.uint8, device="cuda")[None]
                chunk = self.model(r_t, i_t, augment=False, return_aux=False)[0][0].float().cpu().numpy()

            if not np.isfinite(chunk).all():
                raise RuntimeError("Nonfinite Full Vision-ACT action")
            if self.action_schedule is not None:
                k = self.action_schedule[self.replan_idx % len(self.action_schedule)]
                self.replan_idx += 1
            else:
                k = self.n_action_steps
            self.queue.extend(chunk[:k])
        return np.clip(self.queue.popleft(), -1.0, 1.0).astype(np.float32)


def evaluate_full_vision(
    policy: FullVisionACTPolicy,
    seeds: Iterable[int],
    out_dir: Optional[Path] = None,
    horizon: int = 2400,
    zero_images: bool = False,
    save_traces: bool = False,
    write_summary: bool = True,
) -> dict:
    seeds = list(seeds)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    env = make_full_env(images=True)
    rows = []
    try:
        for seed in seeds:
            seed_file = out_dir / f"seed_{seed}.json" if out_dir is not None else None
            if seed_file is not None and seed_file.exists():
                row = json.loads(seed_file.read_text(encoding="utf-8"))
            else:
                init_raw = env.reset_task(seed)
                init_ang_deg = float(np.degrees(np.arctan2(init_raw[4], init_raw[5])))
                init_dist_cm = float(np.linalg.norm(init_raw[2:4] - init_raw[6:8]) * 100.0)
                policy.reset(seed)
                metrics = []
                actions = []
                states = []
                for _ in range(horizon):
                    if (ROOT / "STOP_REQUESTED").exists():
                        raise InterruptedError("Stop requested")
                    obs = observe_full(env)
                    if zero_images:
                        obs["images"][:] = 0
                    a = policy.action(obs)
                    states.append(env.state_vector())
                    actions.append(a)
                    env.move(a)
                    m = env.metrics()
                    metrics.append([m["coverage"], m["iou"], m["distance"], m["angle"], m["z"], m["tilt"]])
                    if m["coverage"] >= 0.95 or m["tilt"] > 0.2 or m["z"] < 0.79 or abs(env.xy()).max() > 0.35:
                        break
                mm = np.asarray(metrics, dtype=np.float32)
                idx = int(mm[:, 0].argmax())
                peak = mm[idx]
                success = bool(peak[0] >= 0.95 and mm[:, 5].max() < 0.2)
                row = {
                    "seed": int(seed),
                    "init_angle_deg": init_ang_deg,
                    "init_dist_cm": init_dist_cm,
                    "success": success,
                    "max_coverage": float(peak[0]),
                    "final_coverage": float(mm[-1, 0]),
                    "max_iou": float(mm[:, 1].max()),
                    "distance_at_max_mm": float(peak[2] * 1000.0),
                    "angle_at_max_deg": float(np.degrees(peak[3])),
                    "steps": len(actions),
                }
                if out_dir is not None:
                    if save_traces:
                        np.savez_compressed(
                            out_dir / f"trace_{seed}.npz",
                            states=np.asarray(states, dtype=np.float32),
                            actions=np.asarray(actions, dtype=np.float32),
                            metrics=mm,
                        )
                    save_json(seed_file, row)
            rows.append(row)
    finally:
        env.close()

    n = len(rows)
    succ = sum(r["success"] for r in rows)
    buckets = {}
    for label, lo, hi in [("0-45deg", 0, 45), ("45-90deg", 45, 90), ("90-135deg", 90, 135), ("135-180deg", 135, 181)]:
        sub = [r for r in rows if lo <= abs(r["init_angle_deg"]) < hi]
        if sub:
            buckets[label] = {"n": len(sub), "successes": sum(r["success"] for r in sub), "sr": float(np.mean([r["success"] for r in sub]))}
    succ_steps = [r["steps"] for r in rows if r["success"]]
    result = {
        "success_rate": float(succ / n),
        "successes": int(succ),
        "n": int(n),
        "success_ci95": wilson_ci95(succ, n),
        "mean_max_coverage": float(np.mean([r["max_coverage"] for r in rows])),
        "mean_max_iou": float(np.mean([r["max_iou"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "mean_success_steps": float(np.mean(succ_steps)) if succ_steps else None,
        "angle_buckets": buckets,
        "episodes": rows,
    }
    if out_dir is not None and write_summary:
        save_json(out_dir / "summary.json", result)
    return result


def _vis_eval_slice(
    ckpt_str: str,
    n_action_steps: int,
    action_schedule: Optional[List[int]],
    mc_dropout_on_freeze: bool,
    seeds: list[int],
    out_dir_str: Optional[str],
    zero_images: bool,
    save_traces: bool,
) -> list[dict]:
    torch.set_num_threads(1)
    pol = FullVisionACTPolicy(
        checkpoint_path=Path(ckpt_str),
        n_action_steps=n_action_steps,
        action_schedule=action_schedule,
        mc_dropout_on_freeze=mc_dropout_on_freeze,
    )
    res = evaluate_full_vision(
        pol,
        seeds,
        out_dir=Path(out_dir_str) if out_dir_str else None,
        zero_images=zero_images,
        save_traces=save_traces,
        write_summary=False,
    )
    return res["episodes"]


def evaluate_full_vision_parallel(
    ckpt: Path,
    seeds: Iterable[int],
    n_action_steps: int = 14,
    action_schedule: Optional[List[int]] = None,
    mc_dropout_on_freeze: bool = False,
    out_dir: Optional[Path] = None,
    zero_images: bool = False,
    save_traces: bool = False,
    workers: int = 4,
) -> dict:
    seeds = list(seeds)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Fast-path: if all per-seed JSONs already exist, skip spawning any OpenGL/MuJoCo process!
        if all((out_dir / f"seed_{s}.json").exists() for s in seeds):
            rows = [json.loads((out_dir / f"seed_{s}.json").read_text(encoding="utf-8")) for s in seeds]
        elif workers <= 1:
            pol = FullVisionACTPolicy(
                checkpoint_path=ckpt,
                n_action_steps=n_action_steps,
                action_schedule=action_schedule,
                mc_dropout_on_freeze=mc_dropout_on_freeze,
            )
            return evaluate_full_vision(pol, seeds, out_dir=out_dir, zero_images=zero_images, save_traces=save_traces)
        else:
            chunks = [seeds[i::workers] for i in range(workers) if seeds[i::workers]]
            rows = []
            with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
                futs = [
                    pool.submit(
                        _vis_eval_slice,
                        str(ckpt),
                        n_action_steps,
                        action_schedule,
                        mc_dropout_on_freeze,
                        ch,
                        str(out_dir),
                        zero_images,
                        save_traces,
                    )
                    for ch in chunks
                ]
                for fut in as_completed(futs):
                    rows.extend(fut.result())
    else:
        chunks = [seeds[i::workers] for i in range(workers) if seeds[i::workers]]
        rows = []
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            futs = [
                pool.submit(
                    _vis_eval_slice,
                    str(ckpt),
                    n_action_steps,
                    action_schedule,
                    mc_dropout_on_freeze,
                    ch,
                    None,
                    zero_images,
                    save_traces,
                )
                for ch in chunks
            ]
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
    succ_steps = [r["steps"] for r in rows if r["success"]]
    result = {
        "success_rate": float(succ / n),
        "successes": int(succ),
        "n": int(n),
        "success_ci95": wilson_ci95(succ, n),
        "mean_max_coverage": float(np.mean([r["max_coverage"] for r in rows])),
        "mean_max_iou": float(np.mean([r["max_iou"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "mean_success_steps": float(np.mean(succ_steps)) if succ_steps else None,
        "angle_buckets": buckets,
        "episodes": rows,
    }
    if out_dir is not None:
        save_json(out_dir / "summary.json", result)
    return result


def train_full_vision(args: argparse.Namespace) -> dict:
    use_aux = args.aux_weight > 0.0
    cfg = {
        "name": args.name,
        "algorithm": "act_v2_full_random_vision_aux" if use_aux else "act_v2_full_random_vision",
        "environment": "full_random_v1",
        "rgb_dir": str(args.rgb_dir),
        "token_mode": "keypoints",
        "keypoints_l3": args.keypoints_l3,
        "keypoints_l4": args.keypoints_l4,
        "aux_geom_dim": 14 if use_aux else 0,
        "aux_weight": float(args.aux_weight),
        "aux_min_weight": float(args.aux_min_weight),
        "smooth_weight": float(args.smooth_weight),
        "n_obs_steps": args.n_obs_steps,
        "chunk_size": args.chunk_size,
        "n_action_steps": args.n_action_steps,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "use_vae": False,
        "time_decay": args.time_decay,
        "crop_size": args.crop_size,
        "micro_batch": args.micro_batch,
        "accumulation": args.accumulation,
        "effective_batch": args.micro_batch * args.accumulation,
        "steps": args.steps,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup": args.warmup,
        "weight_decay": args.weight_decay,
        "ema_decay": args.ema_decay,
        "eval_every": args.eval_every,
        "eval_scenes": args.eval_scenes,
        "eval_workers": args.eval_workers,
        "seed": args.seed,
        "policy_inputs": ["robot_xy[2]", "birdview", "robot0_eye_in_hand"],
        "privileged_inputs": [],
        "training_only_aux": ["SE(2) 8-d pixel + 14-d encoder geometric supervision"] if use_aux else [],
    }
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    run = ACT_V2 / "runs" / cfg["name"]
    run.mkdir(parents=True, exist_ok=True)
    if (run / "complete.json").exists():
        return json.loads((run / "complete.json").read_text(encoding="utf-8"))

    save_json(run / "config.json", cfg)
    init_ck = None
    init_stats = None
    if args.init_from:
        init_ck = torch.load(Path(args.init_from), map_location="cpu", weights_only=False)
        init_stats = init_ck["stats"]
        print(f"[FullRandom VisionACT] Warm-starting from {args.init_from} (reusing exact stats)", flush=True)

    extra_rgb = [Path(p) for p in (args.extra_rgb_dirs or [])]
    train_sampler = FullVisionSampler(
        rgb_dir=Path(cfg["rgb_dir"]),
        extra_rgb_dirs=extra_rgb,
        stats=init_stats,
        n_obs_steps=cfg["n_obs_steps"],
        chunk_size=cfg["chunk_size"],
        seed=cfg["seed"],
        validation=False,
        use_aux=use_aux,
        efficiency_weighting=True,
        speed_power=args.speed_power,
        r3_boost=args.r3_boost,
        vis_dagger_boost=args.vis_dagger_boost,
        balanced_angles=args.balanced_angles,
    )
    stats = train_sampler.stats
    val_sampler = FullVisionSampler(
        rgb_dir=Path(cfg["rgb_dir"]),
        stats=stats,
        n_obs_steps=cfg["n_obs_steps"],
        chunk_size=cfg["chunk_size"],
        seed=cfg["seed"] + 999,
        validation=True,
        use_aux=use_aux,
        efficiency_weighting=False,
    )
    save_json(run / "stats.json", stats)

    model = VisionACT(
        n_obs_steps=cfg["n_obs_steps"],
        chunk_size=cfg["chunk_size"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_encoder_layers=cfg["num_encoder_layers"],
        num_decoder_layers=cfg["num_decoder_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        use_vae=False,
        crop_size=cfg["crop_size"],
        token_mode="keypoints",
        keypoints_l3=cfg["keypoints_l3"],
        keypoints_l4=cfg["keypoints_l4"],
        aux_geom_dim=cfg["aux_geom_dim"],
    ).cuda()
    if init_ck is not None:
        model.load_state_dict(init_ck["ema"] if "ema" in init_ck else init_ck["model"])
        del init_ck
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_frames = sum(len(e["actions"]) for e in train_sampler.episodes)
    print(
        f"[FullRandom VisionACT] {cfg['name']}: params={sum(p.numel() for p in model.parameters())/1e6:.2f}M, "
        f"train_eps={len(train_sampler.episodes)} ({total_frames} frames), val_eps={len(val_sampler.episodes)}, aux={use_aux}",
        flush=True,
    )

    start_step = 0
    best_key = (-1.0, -1.0, -float("inf"))
    best_step = 0
    if (run / "latest.pt").exists():
        ck = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["optimizer"])
        start_step = ck["step"]
        best_key = tuple(ck["best_key"])
        best_step = ck.get("best_step", start_step)
        train_sampler.rng.bit_generator.state = ck["train_rng"]
        val_sampler.rng.bit_generator.state = ck["val_rng"]
        torch.set_rng_state(ck["torch_rng"])
        torch.cuda.set_rng_state(ck["cuda_rng"])
        del ck
        print(f"[FullRandom VisionACT] Resumed from step {start_step}", flush=True)

    def save_ckpt(step: int, filename: str = "latest.pt") -> None:
        obj = {
            "config": cfg, "stats": stats, "model": model.state_dict(), "ema": ema.state_dict(),
            "optimizer": opt.state_dict(), "step": step, "best_key": best_key, "best_step": best_step,
            "train_rng": train_sampler.rng.bit_generator.state, "val_rng": val_sampler.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
        }
        tmp = run / (filename + ".tmp")
        torch.save(obj, tmp)
        tmp.replace(run / filename)

    def validate(step: int) -> None:
        nonlocal best_key, best_step
        ema.eval()
        val_l1s = []
        pos_errs_mm = []
        ang_errs_deg = []
        with torch.inference_mode():
            for _ in range(12):
                batch = val_sampler.sample(cfg["micro_batch"])
                if use_aux:
                    r_np, im_np, a_np, m_np, av_np, ae_np = batch
                else:
                    r_np, im_np, a_np, m_np = batch
                r_t = torch.from_numpy(r_np).to("cuda")
                im_t = torch.from_numpy(im_np).to("cuda")
                a_t = torch.from_numpy(a_np).to("cuda")
                m_t = torch.from_numpy(m_np).to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred, _, diag = ema(r_t, im_t, augment=False, return_aux=use_aux)
                val_l1s.append(float(((pred.float() - a_t).abs() * m_t[..., None]).sum() / (m_t.sum() * 2.0)))
                if use_aux:
                    pe = diag["aux_enc"].float().cpu().numpy()
                    dx = (pe[:, 0] - ae_np[:, 0]) * 0.14
                    dy = (pe[:, 1] - ae_np[:, 1]) * 0.16
                    pos_errs_mm.append(float(np.sqrt(dx * dx + dy * dy).mean() * 1000.0))
                    # Angle error from predicted (cos_T, sin_T)
                    pa = np.arctan2(pe[:, 3], pe[:, 2])
                    ta = np.arctan2(ae_np[:, 3], ae_np[:, 2])
                    da = np.abs(np.arctan2(np.sin(pa - ta), np.cos(pa - ta)))
                    ang_errs_deg.append(float(np.degrees(da).mean()))
        val_l1 = float(np.mean(val_l1s))
        val_pos_mm = float(np.mean(pos_errs_mm)) if pos_errs_mm else -1.0
        val_ang_deg = float(np.mean(ang_errs_deg)) if ang_errs_deg else -1.0
        save_ckpt(step)

        eval_dir = run / f"eval_{step:06d}"
        if cfg["eval_workers"] > 1:
            res = evaluate_full_vision_parallel(
                run / "latest.pt",
                range(30100000, 30100000 + cfg["eval_scenes"]),
                n_action_steps=cfg["n_action_steps"],
                out_dir=eval_dir,
                workers=cfg["eval_workers"],
            )
        else:
            pol = FullVisionACTPolicy(model=ema, stats=stats, cfg=cfg, n_action_steps=cfg["n_action_steps"])
            res = evaluate_full_vision(pol, range(30100000, 30100000 + cfg["eval_scenes"]), out_dir=eval_dir)
        res.update(step=step, validation_l1=val_l1, val_pos_mm=val_pos_mm, val_ang_deg=val_ang_deg)
        save_json(eval_dir / "validation_complete.json", res)

        key = (res["success_rate"], res["mean_max_coverage"], -val_l1)
        if key > best_key:
            best_key = key
            best_step = step
            save_ckpt(step, "best.pt")
        save_ckpt(step)
        bstr = " ".join(f"{k}:{v['successes']}/{v['n']}" for k, v in res["angle_buckets"].items())
        aux_str = f" pos={val_pos_mm:.1f}mm ang={val_ang_deg:.1f}deg" if use_aux else ""
        print(
            f"[FULL VISION EVAL step={step}] sr={res['success_rate']:.3f} ({res['successes']}/{res['n']}) "
            f"cov={res['mean_max_coverage']:.4f} val_l1={val_l1:.4f}{aux_str} steps={res['mean_steps']:.1f} | {bstr}",
            flush=True,
        )
        model.train()

    prefetcher = AsyncPrefetcher(lambda: train_sampler.sample(cfg["micro_batch"]), maxsize=10)
    total = cfg["steps"]
    warm = min(cfg["warmup"], max(1, total // 4))
    started = time.monotonic()
    acc = []
    model.train()
    try:
        for step in range(start_step + 1, total + 1):
            if (ROOT / "STOP_REQUESTED").exists():
                save_ckpt(step - 1)
                return {}
            if step <= warm:
                lr = cfg["lr"] * (step / warm)
            else:
                ratio = (step - warm) / max(1, total - warm)
                lr = cfg["min_lr"] + (cfg["lr"] - cfg["min_lr"]) * 0.5 * (1.0 + math.cos(math.pi * min(ratio, 1.0)))
            for pg in opt.param_groups:
                pg["lr"] = lr

            # Smoothly anneal auxiliary geometric weight so late training maximizes action precision
            progress = min(step / max(1, total), 1.0)
            cur_aux_w = cfg["aux_min_weight"] + (cfg["aux_weight"] - cfg["aux_min_weight"]) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

            opt.zero_grad(set_to_none=True)
            step_recon = 0.0
            step_aux = 0.0
            for _ in range(cfg["accumulation"]):
                batch = prefetcher.next()
                if use_aux:
                    r_np, im_np, a_np, m_np, av_np, ae_np = batch
                    av_t = torch.from_numpy(av_np).pin_memory().to("cuda", non_blocking=True)
                    ae_t = torch.from_numpy(ae_np).pin_memory().to("cuda", non_blocking=True)
                else:
                    r_np, im_np, a_np, m_np = batch
                r_t = torch.from_numpy(r_np).pin_memory().to("cuda", non_blocking=True)
                im_t = torch.from_numpy(im_np).pin_memory().to("cuda", non_blocking=True)
                a_t = torch.from_numpy(a_np).pin_memory().to("cuda", non_blocking=True)
                m_t = torch.from_numpy(m_np).pin_memory().to("cuda", non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred, _, diag = model(r_t, im_t, augment=True, return_aux=use_aux)
                    recon = weighted_l1_loss(pred, a_t, m_t, time_decay=cfg["time_decay"])
                    if cfg["smooth_weight"] > 0.0:
                        recon = recon + cfg["smooth_weight"] * chunk_smoothness_l1_loss(
                            pred, a_t, m_t, time_decay=cfg["time_decay"]
                        )
                    loss = recon
                    aux_val = 0.0
                    if use_aux:
                        l_vis = F.smooth_l1_loss(diag["aux_vis"].float(), av_t, beta=0.05)
                        l_enc = F.smooth_l1_loss(diag["aux_enc"].float(), ae_t, beta=0.05)
                        aux_loss = 0.5 * l_vis + 0.5 * l_enc
                        loss = loss + cur_aux_w * aux_loss
                        aux_val = float(aux_loss.detach())
                    loss = loss / cfg["accumulation"]
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite full vision loss at {step}")
                loss.backward()
                step_recon += float(recon.detach()) / cfg["accumulation"]
                step_aux += aux_val / cfg["accumulation"]
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad):
                raise RuntimeError(f"Nonfinite full vision grad at {step}")
            opt.step()
            update_ema(ema, model, cfg["ema_decay"])
            acc.append((step_recon, step_aux, float(grad)))

            if step % 500 == 0 or step == start_step + 1:
                arr = np.mean(acc, axis=0)
                elapsed = time.monotonic() - started
                sps = (step - start_step) / max(elapsed, 1e-6)
                row = {
                    "status": "training", "pid": os.getpid(), "step": step, "total_steps": total,
                    "recon_l1": float(arr[0]), "aux_loss": float(arr[1]), "grad_norm": float(arr[2]),
                    "lr": float(lr), "aux_w": float(cur_aux_w), "steps_per_sec": float(sps),
                    "seconds": float(elapsed), "peak_gpu_gb": float(torch.cuda.max_memory_allocated() / 1e9),
                }
                save_json(run / "status.json", row)
                with (run / "metrics.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(
                    f"[full_vis {step:6d}/{total}] recon={arr[0]:.4f} aux={arr[1]:.4f} grad={arr[2]:.2f} "
                    f"lr={lr:.2e} {sps:.2f} it/s peak={row['peak_gpu_gb']:.2f}GB",
                    flush=True,
                )
                acc = []
            if step % 5000 == 0:
                save_ckpt(step)
            if step % cfg["eval_every"] == 0 or step == total:
                validate(step)
    finally:
        prefetcher.close()

    final = {
        "name": cfg["name"], "steps": total, "best_step": best_step,
        "best_validation_success": best_key[0], "best_validation_coverage": best_key[1],
        "best_validation_l1": -best_key[2], "seconds": time.monotonic() - started,
    }
    save_json(run / "complete.json", final)
    save_json(run / "status.json", {"status": "complete", **final})
    print(f"[FULL VISION DONE] {json.dumps(final)}", flush=True)
    return final


def confirm_full_vision(
    ckpt: Path,
    out: Path,
    horizons: List[int],
    workers: int = 4,
    test_aperiodic: bool = True,
    blind2_scenes: int = 50,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    dev_seeds = list(range(30100000, 30100050))
    candidates: list[dict] = [{"tag": f"exec_{k}", "k": k, "sched": None, "fw": False} for k in horizons]
    if test_aperiodic:
        candidates.append({"tag": "ap_12_16_18_14", "k": 12, "sched": [12, 16, 18, 14], "fw": False})
        candidates.append({"tag": "ap_12_16_18_14_fw", "k": 12, "sched": [12, 16, 18, 14], "fw": True})
        candidates.append({"tag": "ap_14_18_16_20_fw", "k": 14, "sched": [14, 18, 16, 20], "fw": True})

    board = []
    for c in candidates:
        res = evaluate_full_vision_parallel(
            ckpt,
            dev_seeds,
            n_action_steps=c["k"],
            action_schedule=c["sched"],
            mc_dropout_on_freeze=c["fw"],
            out_dir=out / f"dev_{c['tag']}",
            workers=workers,
        )
        row = {**c, **{kk: vv for kk, vv in res.items() if kk != "episodes"}}
        board.append(row)
        print(
            f"[FULL VIS DEV {c['tag']:18s}] sr={row['success_rate']:.3f} ({row['successes']}/{row['n']}) "
            f"cov={row['mean_max_coverage']:.4f} steps={row['mean_steps']:.1f} succ_steps={row.get('mean_success_steps')} buckets={row['angle_buckets']}",
            flush=True,
        )
    # Pick highest SR, then highest coverage, then FEWEST steps!
    board.sort(key=lambda r: (r["success_rate"], r["mean_max_coverage"], -r["mean_steps"]), reverse=True)
    best = board[0]
    save_json(out / "dev_sweep.json", {"checkpoint": str(ckpt), "sha256": sha, "best": best, "results": board})

    conf = evaluate_full_vision_parallel(
        ckpt,
        list(range(30200000, 30200050)),
        n_action_steps=best["k"],
        action_schedule=best["sched"],
        mc_dropout_on_freeze=best["fw"],
        out_dir=out / "confirmation",
        save_traces=True,
        workers=workers,
    )
    c_out = {kk: vv for kk, vv in conf.items() if kk != "episodes"}

    b2_out = None
    if blind2_scenes > 0:
        b2 = evaluate_full_vision_parallel(
            ckpt,
            list(range(30300000, 30300000 + blind2_scenes)),
            n_action_steps=best["k"],
            action_schedule=best["sched"],
            mc_dropout_on_freeze=best["fw"],
            out_dir=out / "blind2",
            save_traces=False,
            workers=workers,
        )
        b2_out = {kk: vv for kk, vv in b2.items() if kk != "episodes"}

    zero = evaluate_full_vision_parallel(
        ckpt,
        list(range(30100000, 30100020)),
        n_action_steps=best["k"],
        action_schedule=best["sched"],
        mc_dropout_on_freeze=best["fw"],
        out_dir=out / "zero_image_control",
        zero_images=True,
        workers=min(workers, 4),
    )
    z_out = {kk: vv for kk, vv in zero.items() if kk != "episodes"}

    summary = {
        "passed_80": bool(best["success_rate"] >= 0.80 and c_out["success_rate"] >= 0.80 and z_out["success_rate"] == 0.0),
        "checkpoint": str(ckpt),
        "sha256": sha,
        "selected_inference": {"tag": best["tag"], "k": best["k"], "sched": best["sched"], "fw": best["fw"]},
        "dev_best": best,
        "confirmation": c_out,
        "blind2": b2_out,
        "zero_image_control": z_out,
    }
    save_json(out / "full_vision_gate.json", summary)
    b2_str = f" | blind2={b2_out['success_rate']:.3f} ({b2_out['successes']}/{b2_out['n']}) steps={b2_out['mean_steps']:.1f}" if b2_out else ""
    print(
        f"[FULL VIS CONFIRMATION] passed_80={summary['passed_80']} sr={c_out['success_rate']:.3f} "
        f"({c_out['successes']}/{c_out['n']}) ci95={c_out['success_ci95']} cov={c_out['mean_max_coverage']:.4f} "
        f"steps={c_out['mean_steps']:.1f} succ_steps={c_out.get('mean_success_steps')}{b2_str} | zero_img={z_out['success_rate']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--export", type=int, default=0, help="Export N full-random episodes to dual-camera uint8 RGB")
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--checkpoint", default="act_v2/runs/full_random_vision_act_v2/best.pt")
    p.add_argument("--out", default="act_v2/runs/full_random_vision_act_v2/final_eval")
    p.add_argument("--horizons", type=int, nargs="+", default=[10, 12, 14, 16, 18])
    p.add_argument("--rgb-dir", default=str(ALL_RGB_DIR))
    p.add_argument("--name", default="full_random_vision_act_v2")
    p.add_argument("--n-obs-steps", type=int, default=3)
    p.add_argument("--chunk-size", type=int, default=24)
    p.add_argument("--n-action-steps", type=int, default=14)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-encoder-layers", type=int, default=4)
    p.add_argument("--num-decoder-layers", type=int, default=4)
    p.add_argument("--dim-feedforward", type=int, default=1536)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--keypoints-l3", type=int, default=48)
    p.add_argument("--keypoints-l4", type=int, default=48)
    p.add_argument("--crop-size", type=int, default=124)
    p.add_argument("--time-decay", type=float, default=0.015)
    p.add_argument("--smooth-weight", type=float, default=0.15)
    p.add_argument("--aux-weight", type=float, default=0.60)
    p.add_argument("--aux-min-weight", type=float, default=0.25)
    p.add_argument("--micro-batch", type=int, default=64)
    p.add_argument("--accumulation", type=int, default=1)
    p.add_argument("--steps", type=int, default=80000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--weight-decay", type=float, default=3e-4)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--eval-every", type=int, default=20000)
    p.add_argument("--eval-scenes", type=int, default=50)
    p.add_argument("--eval-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init-from", default=None)
    p.add_argument("--extra-rgb-dirs", nargs="*", default=[])
    p.add_argument("--speed-power", type=float, default=0.35)
    p.add_argument("--r3-boost", type=float, default=2.5)
    p.add_argument("--vis-dagger-boost", type=float, default=3.0)
    p.add_argument("--balanced-angles", action="store_true")
    args = p.parse_args()

    if args.export > 0:
        export_rgb(args.export)
    elif args.confirm:
        confirm_full_vision(Path(args.checkpoint), Path(args.out), args.horizons, workers=args.eval_workers)
    else:
        train_full_vision(args)
