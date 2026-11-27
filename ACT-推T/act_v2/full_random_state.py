"""Full-Random Position & Full-360-deg Angle State-ACT training, horizon sweep, and blind confirmation."""
from __future__ import annotations

import argparse
from collections import deque
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from act_v2.data import ACT_V2, save_json
from act_v2.full_random_collect import FULL_DATA_DIR
from act_v2.full_random_env import make_full_env
from act_v2.inference import TemporalEnsembler, wilson_ci95
from act_v2.model import StateACT, weighted_l1_loss
from act_v2.train_state import update_ema

PUSHER_DIM = 9
OBJECT_DIM = 12
FEAT_DIM = PUSHER_DIM + OBJECT_DIM  # 21


def feature_dims(feature_mode: str = "full21") -> Tuple[int, int]:
    if feature_mode == "full28":
        return 12, 16
    return 9, 12


def full_se2_features(raw: np.ndarray) -> np.ndarray:
    """21-d exact SE(2) geometric features from 10-d env.state_vector()."""
    r = np.asarray(raw, dtype=np.float32)
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
    ee_to_g = ee - g_pos
    ee_dist = np.linalg.norm(ee_rel, axis=-1, keepdims=True)

    g_rel = g_pos - t_pos
    g_loc_x = cos_t * g_rel[..., 0:1] + sin_t * g_rel[..., 1:2]
    g_loc_y = -sin_t * g_rel[..., 0:1] + cos_t * g_rel[..., 1:2]
    g_dist = np.linalg.norm(g_rel, axis=-1, keepdims=True)

    cos_d = cos_g * cos_t + sin_g * sin_t
    sin_d = sin_g * cos_t - cos_g * sin_t
    ang_norm = np.arctan2(sin_d, cos_d) / np.pi

    return np.concatenate(
        [
            ee, ee_rel, ee_loc_x, ee_loc_y, ee_to_g, ee_dist,
            t_pos, cos_t, sin_t, g_rel, cos_d, sin_d, ang_norm, g_loc_x, g_loc_y, g_dist,
        ],
        axis=-1,
    ).astype(np.float32)


def full_se2_features_v2(raw: np.ndarray) -> np.ndarray:
    """28-d branch-cut-free, multi-scale high-resolution SE(2) features.

    Key upgrades over full21:
    1. Eliminates the +/-180deg branch-cut cliff of signed atan2(sin_d, cos_d)/pi by
       using continuous abs_ang = arccos(clip(cos_d, -1, 1)) / pi alongside smooth (cos_d, sin_d).
    2. Adds smooth T-local unit directions (ee_dir_local, goal_dir_local) and log-scale
       distances/angles so the Transformer retains crisp geometric resolution down to 1mm / 1deg.
    """
    r = np.asarray(raw, dtype=np.float32)
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
    ee_to_g = ee - g_pos
    ee_dist = np.linalg.norm(ee_rel, axis=-1, keepdims=True)
    ee_dir_denom = np.sqrt(ee_dist * ee_dist + (0.005 ** 2))
    ee_dir_lx = ee_loc_x / ee_dir_denom
    ee_dir_ly = ee_loc_y / ee_dir_denom
    ee_log_dist = np.log1p(ee_dist / 0.01)

    g_rel = g_pos - t_pos
    g_loc_x = cos_t * g_rel[..., 0:1] + sin_t * g_rel[..., 1:2]
    g_loc_y = -sin_t * g_rel[..., 0:1] + cos_t * g_rel[..., 1:2]
    g_dist = np.linalg.norm(g_rel, axis=-1, keepdims=True)
    g_dir_denom = np.sqrt(g_dist * g_dist + (0.003 ** 2))
    g_dir_lx = g_loc_x / g_dir_denom
    g_dir_ly = g_loc_y / g_dir_denom
    g_log_dist = np.log1p(g_dist / 0.005)

    cos_d = np.clip(cos_g * cos_t + sin_g * sin_t, -1.0, 1.0)
    sin_d = np.clip(sin_g * cos_t - cos_g * sin_t, -1.0, 1.0)
    abs_ang = np.arccos(cos_d)  # strictly continuous in [0, pi], ZERO +/-pi jump
    abs_ang_norm = abs_ang / np.pi
    log_ang = np.log1p(abs_ang / 0.05)

    return np.concatenate(
        [
            # Pusher group (12 dims: 0..11)
            ee, ee_rel, ee_loc_x, ee_loc_y, ee_to_g, ee_dist, ee_dir_lx, ee_dir_ly, ee_log_dist,
            # Object-Goal group (16 dims: 12..27)
            t_pos, cos_t, sin_t, g_rel, cos_d, sin_d, abs_ang_norm,
            g_loc_x, g_loc_y, g_dist, g_dir_lx, g_dir_ly, g_log_dist, log_ang,
        ],
        axis=-1,
    ).astype(np.float32)


def extract_full_features(raw: np.ndarray, feature_mode: str = "full21") -> np.ndarray:
    if feature_mode == "full28":
        return full_se2_features_v2(raw)
    return full_se2_features(raw)


def chunk_smoothness_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    time_decay: float = 0.025,
) -> torch.Tensor:
    """First-order action-chunk velocity matching L1 loss for crisp chunk trajectories."""
    dp = pred[:, 1:].float() - pred[:, :-1].float()
    dt = target[:, 1:].float() - target[:, :-1].float()
    dv = valid[:, 1:] & valid[:, :-1]
    k = dp.shape[1]
    if time_decay > 0.0:
        w = torch.exp(-time_decay * torch.arange(k, device=pred.device, dtype=torch.float32))
        w = w / w.mean()
    else:
        w = torch.ones(k, device=pred.device, dtype=torch.float32)
    mask = dv.float() * w.unsqueeze(0)
    l1 = (dp - dt).abs().mean(dim=-1)
    return (l1 * mask).sum() / mask.sum().clamp(min=1.0)


def load_full_random_dataset(
    extra_dirs: Optional[List[Path]] = None,
    feature_mode: str = "full21",
    filter_singularity: bool = False,
) -> Dict[str, List[dict]]:
    manifest = json.loads((FULL_DATA_DIR / "complete.json").read_text(encoding="utf-8"))
    groups: Dict[str, List[dict]] = {"train": [], "validation": []}
    filtered = 0

    def _add_ep(path: Path, seed: int, partition: str, source: str) -> None:
        nonlocal filtered
        with np.load(path) as d:
            states = d["states"].astype(np.float32).copy()
            if filter_singularity and float(states[0, 0]) > 0.12:
                filtered += 1
                return
            ep = {
                "states": states,
                "actions": d["actions"].astype(np.float32).copy(),
                "coverage": d["coverage"].astype(np.float32).copy(),
                "seed": int(seed),
                "source": source,
            }
        ep["features"] = extract_full_features(ep["states"], feature_mode=feature_mode)
        ep["mid_indices"] = np.flatnonzero((ep["coverage"] > 0.45) & (ep["coverage"] <= 0.88))
        ep["fine_indices"] = np.flatnonzero(ep["coverage"] > 0.88)
        cos_d = ep["states"][:, 9] * ep["states"][:, 5] + ep["states"][:, 8] * ep["states"][:, 4]
        ep["large_angle_indices"] = np.flatnonzero(cos_d < 0.0)  # |angle_err| > 90 deg
        groups[partition].append(ep)

    for row in manifest["episodes"]:
        _add_ep(FULL_DATA_DIR / row["file"], int(row["seed"]), row["partition"], row["file"])

    if extra_dirs:
        for edir in extra_dirs:
            mp = edir / "complete.json"
            if not mp.exists():
                continue
            em = json.loads(mp.read_text(encoding="utf-8"))
            for row in em["episodes"]:
                _add_ep(edir / row["file"], int(row["seed"]), row.get("partition", "train"), str(edir / row["file"]))

    train_seeds = {e["seed"] for e in groups["train"]}
    val_seeds = {e["seed"] for e in groups["validation"]}
    assert not (train_seeds & val_seeds), "Train/val seed overlap!"
    if filtered > 0:
        print(f"[DATASET] Filtered {filtered} legacy singular-spawn episodes (ee0_x > 0.12)", flush=True)
    return groups


def fit_full_normalizer(groups: Dict[str, List[dict]], feature_mode: str = "full21") -> dict:
    feats = np.concatenate([e["features"] for e in groups["train"]], axis=0)
    lo = feats.min(axis=0)
    hi = feats.max(axis=0)
    if feature_mode == "full28":
        for idx in (9, 10, 14, 15, 18, 19, 24, 25):
            lo[idx] = -1.0
            hi[idx] = 1.0
        lo[20] = 0.0
        hi[20] = 1.0
    else:
        for idx in (11, 12, 15, 16, 17):
            lo[idx] = -1.0
            hi[idx] = 1.0
    center = (hi + lo) / 2.0
    scale = np.maximum((hi - lo) / 2.0, 0.01)
    return {
        "mode": "MIN_MAX",
        "feature_mode": feature_mode,
        "feature_center": center.tolist(),
        "feature_scale": scale.tolist(),
        "action_center": [0.0, 0.0],
        "action_scale": [1.0, 1.0],
    }


class FullRandomStateSampler:
    def __init__(
        self,
        episodes: List[dict],
        stats: dict,
        n_obs_steps: int = 3,
        chunk_size: int = 16,
        batch_size: int = 128,
        seed: int = 42,
        validation: bool = False,
        state_noise_std: float = 0.004,
        boost_substr: Optional[str] = None,
        boost_factor: float = 1.0,
    ):
        self.episodes = episodes
        self.n_obs_steps = n_obs_steps
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.validation = validation
        self.state_noise_std = 0.0 if validation else state_noise_std
        self.rng = np.random.default_rng(seed)
        lengths = np.asarray([len(e["actions"]) for e in episodes], dtype=np.float64)
        if boost_substr and boost_factor > 1.0 and not validation:
            boosts = np.asarray(
                [boost_factor if boost_substr in str(e.get("source", "")) else 1.0 for e in episodes],
                dtype=np.float64,
            )
            lengths = lengths * boosts
        self.ep_weights = lengths / lengths.sum()
        self.center = np.asarray(stats["feature_center"], dtype=np.float32)
        self.scale = np.asarray(stats["feature_scale"], dtype=np.float32)
        self.feat_dim = len(self.center)

    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bs = self.batch_size
        to = self.n_obs_steps
        k = self.chunk_size
        obs = np.empty((bs, to, self.feat_dim), dtype=np.float32)
        act = np.zeros((bs, k, 2), dtype=np.float32)
        valid = np.zeros((bs, k), dtype=bool)

        ep_indices = self.rng.choice(len(self.episodes), size=bs, p=self.ep_weights)
        for bi, ei in enumerate(ep_indices):
            e = self.episodes[ei]
            n = len(e["actions"])
            if not self.validation:
                u = self.rng.random()
                if u < 0.25 and len(e["fine_indices"]) > 0:
                    t = int(self.rng.choice(e["fine_indices"]))
                elif u < 0.50 and len(e["mid_indices"]) > 0:
                    t = int(self.rng.choice(e["mid_indices"]))
                elif u < 0.70 and len(e["large_angle_indices"]) > 0:
                    t = int(self.rng.choice(e["large_angle_indices"]))
                else:
                    t = int(self.rng.integers(n))
            else:
                t = int(self.rng.integers(n))

            hist_idx = np.maximum(0, t - np.arange(to - 1, -1, -1))
            obs[bi] = e["features"][hist_idx]
            count = min(k, n - t)
            act[bi, :count] = e["actions"][t : t + count]
            valid[bi, :count] = True

        obs = (obs - self.center) / self.scale
        if self.state_noise_std > 0.0:
            obs = obs + self.rng.normal(0.0, self.state_noise_std, size=(bs, 1, self.feat_dim)).astype(np.float32)
        return obs, act, valid


class FullRandomStateACTPolicy:
    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        model: Optional[StateACT] = None,
        stats: Optional[dict] = None,
        cfg: Optional[dict] = None,
        n_action_steps: int = 8,
        action_schedule: Optional[List[int]] = None,
        mc_dropout_on_freeze: bool = False,
        fine_action_steps: Optional[int] = None,
        fine_dist_m: float = 0.018,
        fine_angle_deg: float = 12.0,
        temporal_ensemble: Optional[float] = None,
        ensemble_window: int = 8,
        ensemble_order: str = "newest",
        weight_kind: str = "ema",
    ):
        if model is None:
            ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            cfg = ck["config"]
            stats = ck["stats"]
            fmode = cfg.get("feature_mode", stats.get("feature_mode", "full21"))
            pdim, odim = feature_dims(fmode)
            model = StateACT(
                n_obs_steps=cfg["n_obs_steps"],
                pusher_dim=pdim,
                object_dim=odim,
                chunk_size=cfg["chunk_size"],
                d_model=cfg["d_model"],
                nhead=cfg["nhead"],
                num_encoder_layers=cfg["num_encoder_layers"],
                num_decoder_layers=cfg["num_decoder_layers"],
                dim_feedforward=cfg["dim_feedforward"],
                dropout=cfg["dropout"],
                use_vae=cfg["use_vae"],
                latent_dim=cfg.get("latent_dim", 32),
            ).cuda().eval()
            model.load_state_dict(ck[weight_kind] if weight_kind in ck else ck["model"])
        self.model = model.eval()
        self.stats = stats
        self.cfg = cfg
        self.feature_mode = cfg.get("feature_mode", stats.get("feature_mode", "full21"))
        self.n_obs_steps = int(cfg["n_obs_steps"])
        self.chunk_size = int(cfg["chunk_size"])
        self.n_action_steps = int(n_action_steps)
        self.action_schedule = [int(x) for x in action_schedule] if action_schedule else None
        self.mc_dropout_on_freeze = bool(mc_dropout_on_freeze)
        self.fine_action_steps = int(fine_action_steps) if fine_action_steps is not None else None
        self.fine_dist_m = float(fine_dist_m)
        self.fine_cos_thresh = float(math.cos(math.radians(fine_angle_deg)))
        self.center = np.asarray(stats["feature_center"], dtype=np.float32)
        self.scale = np.asarray(stats["feature_scale"], dtype=np.float32)
        self.ensembler = (
            TemporalEnsembler(self.chunk_size, temporal_ensemble, ensemble_window, ensemble_order)
            if temporal_ensemble is not None
            else None
        )
        self.reset()

    def reset(self, seed: int = 0) -> None:
        self.obs_history: deque = deque(maxlen=self.n_obs_steps)
        self.queue: deque = deque()
        self.replan_idx = 0
        self.freeze_hist: deque = deque(maxlen=120)
        self.freeze_rng = np.random.default_rng(seed + 2026)
        if self.ensembler is not None:
            self.ensembler.reset()

    @torch.inference_mode()
    def action(self, raw_state: np.ndarray) -> np.ndarray:
        feat = (extract_full_features(raw_state, feature_mode=self.feature_mode) - self.center) / self.scale
        self.obs_history.append(feat)
        while len(self.obs_history) < self.n_obs_steps:
            self.obs_history.append(feat)
        self.freeze_hist.append(raw_state[:6].copy())

        if self.ensembler is not None or not self.queue:
            obs_arr = np.stack(self.obs_history, axis=0)
            frozen = False
            if self.mc_dropout_on_freeze and len(self.freeze_hist) == 120:
                fh = np.asarray(self.freeze_hist, dtype=np.float32)
                ee_span = float(np.linalg.norm(fh[:, :2].max(0) - fh[:, :2].min(0)))
                t_span = float(np.linalg.norm(fh[:, 2:4].max(0) - fh[:, 2:4].min(0)))
                if ee_span < 0.0025 and t_span < 0.0020:
                    frozen = True
            if frozen:
                # Pure single-model ACT fixed-point wake: MC-dropout + tiny in-distribution jitter (0.01)
                obs_arr = obs_arr + self.freeze_rng.normal(0.0, 0.012, size=(1, obs_arr.shape[1])).astype(np.float32)
                obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device="cuda")[None]
                self.model.train()
                chunk = self.model(obs_t)[0][0].float().cpu().numpy()
                self.model.eval()
            else:
                obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device="cuda")[None]
                chunk = self.model(obs_t)[0][0].float().cpu().numpy()
            if not np.isfinite(chunk).all():
                raise RuntimeError("Nonfinite FullRandom State-ACT action")
            if self.ensembler is not None:
                return np.clip(self.ensembler.update(chunk), -1.0, 1.0).astype(np.float32)
            if self.action_schedule is not None:
                k = self.action_schedule[self.replan_idx % len(self.action_schedule)]
                self.replan_idx += 1
            else:
                k = self.n_action_steps
            if self.fine_action_steps is not None:
                g_dist = float(np.linalg.norm(raw_state[6:8] - raw_state[2:4]))
                cos_d = float(raw_state[9] * raw_state[5] + raw_state[8] * raw_state[4])
                if g_dist < self.fine_dist_m and cos_d > self.fine_cos_thresh:
                    k = self.fine_action_steps
            self.queue.extend(chunk[:k])
        return np.clip(self.queue.popleft(), -1.0, 1.0).astype(np.float32)


class FullRandomStateACTEnsemblePolicy:
    """Pure-ACT ensemble averaging predicted action chunks across multiple State-ACT checkpoints."""

    def __init__(
        self,
        checkpoints: List[Path],
        n_action_steps: int = 12,
        fine_action_steps: Optional[int] = None,
        fine_dist_m: float = 0.018,
        fine_angle_deg: float = 12.0,
        weight_kind: str = "ema",
    ):
        self.members = [
            FullRandomStateACTPolicy(checkpoint_path=Path(cp), n_action_steps=n_action_steps, weight_kind=weight_kind)
            for cp in checkpoints
        ]
        self.n_action_steps = int(n_action_steps)
        self.fine_action_steps = int(fine_action_steps) if fine_action_steps is not None else None
        self.fine_dist_m = float(fine_dist_m)
        self.fine_cos_thresh = float(math.cos(math.radians(fine_angle_deg)))
        self.reset()

    def reset(self, seed: int = 0) -> None:
        for m in self.members:
            m.reset(seed)
        self.queue: deque = deque()

    @torch.inference_mode()
    def action(self, raw_state: np.ndarray) -> np.ndarray:
        for m in self.members:
            feat = (extract_full_features(raw_state, feature_mode=m.feature_mode) - m.center) / m.scale
            m.obs_history.append(feat)
            while len(m.obs_history) < m.n_obs_steps:
                m.obs_history.append(feat)
        if not self.queue:
            chunks = []
            for m in self.members:
                obs_t = torch.as_tensor(np.stack(m.obs_history, axis=0), dtype=torch.float32, device="cuda")[None]
                chunks.append(m.model(obs_t)[0][0].float().cpu().numpy())
            min_k = min(c.shape[0] for c in chunks)
            avg_chunk = np.mean([c[:min_k] for c in chunks], axis=0)
            k = self.n_action_steps
            if self.fine_action_steps is not None:
                g_dist = float(np.linalg.norm(raw_state[6:8] - raw_state[2:4]))
                cos_d = float(raw_state[9] * raw_state[5] + raw_state[8] * raw_state[4])
                if g_dist < self.fine_dist_m and cos_d > self.fine_cos_thresh:
                    k = self.fine_action_steps
            self.queue.extend(avg_chunk[:k])
        return np.clip(self.queue.popleft(), -1.0, 1.0).astype(np.float32)


def evaluate_full_state(
    policy: FullRandomStateACTPolicy,
    seeds: Iterable[int],
    out_dir: Optional[Path] = None,
    horizon: int = 2400,
    save_traces: bool = False,
) -> dict:
    seeds = list(seeds)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    env = make_full_env(False)
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
                    raw = env.state_vector()
                    a = policy.action(raw)
                    states.append(raw)
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
    failed = [r for r in rows if not r["success"]]
    # Angle-bucket breakdown
    buckets = {}
    for label, lo, hi in [("0-45deg", 0, 45), ("45-90deg", 45, 90), ("90-135deg", 90, 135), ("135-180deg", 135, 181)]:
        sub = [r for r in rows if lo <= abs(r["init_angle_deg"]) < hi]
        if sub:
            buckets[label] = {"n": len(sub), "successes": sum(r["success"] for r in sub), "sr": float(np.mean([r["success"] for r in sub]))}
    result = {
        "success_rate": float(succ / n),
        "successes": int(succ),
        "n": int(n),
        "success_ci95": wilson_ci95(succ, n),
        "mean_max_coverage": float(np.mean([r["max_coverage"] for r in rows])),
        "mean_max_iou": float(np.mean([r["max_iou"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "failed_mean_distance_mm": float(np.mean([r["distance_at_max_mm"] for r in failed])) if failed else None,
        "failed_mean_angle_deg": float(np.mean([r["angle_at_max_deg"] for r in failed])) if failed else None,
        "angle_buckets": buckets,
        "episodes": rows,
    }
    if out_dir is not None:
        save_json(out_dir / "summary.json", result)
    return result


def train_full_state(args: argparse.Namespace) -> dict:
    cfg = {
        "name": args.name,
        "algorithm": "act_v2_full_random_state",
        "environment": "full_random_v1",
        "feature_mode": args.feature_mode,
        "filter_singularity": bool(args.filter_singularity),
        "smooth_weight": float(args.smooth_weight),
        "n_obs_steps": args.n_obs_steps,
        "chunk_size": args.chunk_size,
        "n_action_steps": args.n_action_steps,
        "fine_action_steps": args.fine_action_steps,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "use_vae": False,
        "time_decay": args.time_decay,
        "state_noise_std": args.state_noise_std,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup": args.warmup,
        "weight_decay": args.weight_decay,
        "ema_decay": args.ema_decay,
        "eval_every": args.eval_every,
        "eval_scenes": args.eval_scenes,
        "seed": args.seed,
        "extra_dirs": args.extra_dirs or [],
    }

    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True

    run = ACT_V2 / "runs" / cfg["name"]
    run.mkdir(parents=True, exist_ok=True)
    if (run / "complete.json").exists():
        return json.loads((run / "complete.json").read_text(encoding="utf-8"))

    save_json(run / "config.json", cfg)
    extra = [Path(p) for p in cfg["extra_dirs"]]
    groups = load_full_random_dataset(
        extra_dirs=extra,
        feature_mode=cfg["feature_mode"],
        filter_singularity=cfg["filter_singularity"],
    )
    init_ck = None
    if args.init_from:
        init_ck = torch.load(Path(args.init_from), map_location="cpu", weights_only=False)
        stats = init_ck["stats"]
        print(f"[FullRandom StateACT] Warm-starting from {args.init_from} (reusing exact stats)", flush=True)
    else:
        stats = fit_full_normalizer(groups, feature_mode=cfg["feature_mode"])
    save_json(run / "stats.json", stats)

    pdim, odim = feature_dims(cfg["feature_mode"])
    model = StateACT(
        n_obs_steps=cfg["n_obs_steps"],
        pusher_dim=pdim,
        object_dim=odim,
        chunk_size=cfg["chunk_size"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_encoder_layers=cfg["num_encoder_layers"],
        num_decoder_layers=cfg["num_decoder_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        use_vae=False,
    ).cuda()
    if init_ck is not None:
        src_sd = init_ck["ema"] if "ema" in init_ck else init_ck["model"]
        model.load_state_dict(src_sd)
    n_params = sum(p.numel() for p in model.parameters())
    total_frames = sum(len(e["actions"]) for e in groups["train"])
    print(
        f"[FullRandom StateACT] {cfg['name']}: params={n_params/1e6:.2f}M, "
        f"train_eps={len(groups['train'])} ({total_frames} frames), val_eps={len(groups['validation'])}",
        flush=True,
    )

    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    train_sampler = FullRandomStateSampler(
        groups["train"], stats, cfg["n_obs_steps"], cfg["chunk_size"], cfg["batch_size"],
        seed=cfg["seed"], validation=False, state_noise_std=cfg["state_noise_std"],
        boost_substr=args.boost_substr, boost_factor=args.boost_factor,
    )
    val_sampler = FullRandomStateSampler(
        groups["validation"], stats, cfg["n_obs_steps"], cfg["chunk_size"], cfg["batch_size"],
        seed=cfg["seed"] + 999, validation=True, state_noise_std=0.0,
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
        print(f"[FullRandom StateACT] Resumed from step {start_step}", flush=True)

    def save_ckpt(step: int, filename: str = "latest.pt") -> None:
        obj = {
            "config": cfg,
            "stats": stats,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "best_key": best_key,
            "best_step": best_step,
            "train_rng": train_sampler.rng.bit_generator.state,
            "val_rng": val_sampler.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        }
        tmp = run / (filename + ".tmp")
        torch.save(obj, tmp)
        tmp.replace(run / filename)

    def validate(step: int) -> None:
        nonlocal best_key, best_step
        ema.eval()
        val_l1s = []
        with torch.inference_mode():
            for _ in range(25):
                o_np, a_np, m_np = val_sampler.sample()
                o = torch.as_tensor(o_np, device="cuda")
                a = torch.as_tensor(a_np, device="cuda")
                m = torch.as_tensor(m_np, device="cuda")
                pred = ema(o)[0]
                l1 = ((pred.float() - a).abs() * m[..., None]).sum() / (m.sum() * 2.0)
                val_l1s.append(float(l1))
        val_l1 = float(np.mean(val_l1s))
        save_ckpt(step)

        pol = FullRandomStateACTPolicy(
            model=ema,
            stats=stats,
            cfg=cfg,
            n_action_steps=cfg["n_action_steps"],
            fine_action_steps=cfg.get("fine_action_steps"),
        )
        eval_dir = run / f"eval_{step:06d}"
        res = evaluate_full_state(pol, range(30100000, 30100000 + cfg["eval_scenes"]), out_dir=eval_dir)
        res.update(step=step, validation_l1=val_l1)
        save_json(eval_dir / "validation_complete.json", res)

        key = (res["success_rate"], res["mean_max_coverage"], -val_l1)
        if key > best_key:
            best_key = key
            best_step = step
            save_ckpt(step, "best.pt")
        save_ckpt(step)
        bstr = " ".join(f"{k}:{v['successes']}/{v['n']}" for k, v in res["angle_buckets"].items())
        print(
            f"[FULL EVAL step={step}] sr={res['success_rate']:.3f} ({res['successes']}/{res['n']}) "
            f"cov={res['mean_max_coverage']:.4f} val_l1={val_l1:.4f} steps={res['mean_steps']:.1f} | {bstr}",
            flush=True,
        )
        model.train()

    total = cfg["steps"]
    warm = min(cfg["warmup"], max(1, total // 4))
    started = time.monotonic()
    acc = []
    model.train()

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

        o_np, a_np, m_np = train_sampler.sample()
        o = torch.as_tensor(o_np, device="cuda")
        a = torch.as_tensor(a_np, device="cuda")
        m = torch.as_tensor(m_np, device="cuda")

        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(o)[0]
            loss = weighted_l1_loss(pred, a, m, time_decay=cfg["time_decay"])
            if cfg["smooth_weight"] > 0.0:
                loss = loss + cfg["smooth_weight"] * chunk_smoothness_l1_loss(pred, a, m, time_decay=cfg["time_decay"])
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad):
            raise RuntimeError(f"Nonfinite grad at step {step}")
        opt.step()
        update_ema(ema, model, cfg["ema_decay"])
        acc.append((float(loss.detach()), float(grad)))

        if step % 1000 == 0 or step == start_step + 1:
            arr = np.mean(acc, axis=0)
            elapsed = time.monotonic() - started
            sps = (step - start_step) / max(elapsed, 1e-6)
            row = {
                "status": "training",
                "pid": os.getpid(),
                "step": step,
                "total_steps": total,
                "recon_l1": float(arr[0]),
                "grad_norm": float(arr[1]),
                "lr": float(lr),
                "steps_per_sec": float(sps),
                "seconds": float(elapsed),
                "peak_gpu_gb": float(torch.cuda.max_memory_allocated() / 1e9),
            }
            save_json(run / "status.json", row)
            with (run / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if step % 2000 == 0 or step == start_step + 1:
                print(
                    f"[full {step:6d}/{total}] recon={arr[0]:.4f} grad={arr[1]:.2f} lr={lr:.2e} {sps:.1f} it/s peak={row['peak_gpu_gb']:.2f}GB",
                    flush=True,
                )
            acc = []

        if step % cfg["eval_every"] == 0 or step == total:
            validate(step)

    final = {
        "name": cfg["name"],
        "steps": total,
        "best_step": best_step,
        "best_validation_success": best_key[0],
        "best_validation_coverage": best_key[1],
        "best_validation_l1": -best_key[2],
        "seconds": time.monotonic() - started,
    }
    save_json(run / "complete.json", final)
    save_json(run / "status.json", {"status": "complete", **final})
    print(f"[FULL STATE DONE] {json.dumps(final)}", flush=True)
    return final


def confirm_full_state(
    ckpt: Path,
    out: Path,
    horizons: List[int],
    fine_options: Optional[List[Optional[int]]] = None,
    required: float = 0.90,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    dev_seeds = list(range(30100000, 30100050))
    if not fine_options:
        fine_options = [None]
    board = []
    for k in horizons:
        for fk in fine_options:
            if fk is not None and fk >= k:
                continue
            tag = f"exec_{k}" if fk is None else f"exec_{k}_fine_{fk}"
            pol = FullRandomStateACTPolicy(checkpoint_path=ckpt, n_action_steps=k, fine_action_steps=fk)
            res = evaluate_full_state(pol, dev_seeds, out_dir=out / f"dev_{tag}")
            row = {"tag": tag, "n_action_steps": k, "fine_action_steps": fk, **{kk: vv for kk, vv in res.items() if kk != "episodes"}}
            board.append(row)
            print(
                f"[FULL DEV {tag}] sr={row['success_rate']:.3f} ({row['successes']}/{row['n']}) "
                f"cov={row['mean_max_coverage']:.4f} steps={row['mean_steps']:.1f} buckets={row['angle_buckets']}",
                flush=True,
            )
    board.sort(key=lambda r: (r["success_rate"], r["mean_max_coverage"]), reverse=True)
    best_k = board[0]["n_action_steps"]
    best_fk = board[0]["fine_action_steps"]
    save_json(
        out / "dev_sweep.json",
        {
            "checkpoint": str(ckpt),
            "sha256": sha,
            "best_tag": board[0]["tag"],
            "best_n_action_steps": best_k,
            "best_fine_action_steps": best_fk,
            "results": board,
        },
    )

    conf_seeds = list(range(30200000, 30200050))
    pol = FullRandomStateACTPolicy(checkpoint_path=ckpt, n_action_steps=best_k, fine_action_steps=best_fk)
    conf = evaluate_full_state(pol, conf_seeds, out_dir=out / "confirmation", save_traces=True)
    c = {kk: vv for kk, vv in conf.items() if kk != "episodes"}
    passed = bool(board[0]["success_rate"] >= required and c["success_rate"] >= required)
    summary = {
        "passed": passed,
        "required": required,
        "checkpoint": str(ckpt),
        "sha256": sha,
        "n_action_steps": best_k,
        "fine_action_steps": best_fk,
        "dev_best": board[0],
        "confirmation": c,
    }
    save_json(out / "full_state_gate.json", summary)
    print(
        f"[FULL CONFIRMATION] passed={passed} sr={c['success_rate']:.3f} ({c['successes']}/{c['n']}) "
        f"ci95={c['success_ci95']} cov={c['mean_max_coverage']:.4f} buckets={c['angle_buckets']}",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--checkpoint", default="act_v2/runs/full_random_state_act/best.pt")
    p.add_argument("--out", default="act_v2/runs/full_random_state_act/final_eval")
    p.add_argument("--horizons", type=int, nargs="+", default=[8, 10, 12, 14])
    p.add_argument("--fine-steps", type=int, nargs="*", default=[])
    p.add_argument("--required", type=float, default=0.90)
    p.add_argument("--name", default="full_random_state_act")
    p.add_argument("--feature-mode", default="full21", choices=["full21", "full28"])
    p.add_argument("--filter-singularity", action="store_true")
    p.add_argument("--smooth-weight", type=float, default=0.0)
    p.add_argument("--n-obs-steps", type=int, default=3)
    p.add_argument("--chunk-size", type=int, default=16)
    p.add_argument("--n-action-steps", type=int, default=8)
    p.add_argument("--fine-action-steps", type=int, default=None)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-encoder-layers", type=int, default=4)
    p.add_argument("--num-decoder-layers", type=int, default=4)
    p.add_argument("--dim-feedforward", type=int, default=1536)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--time-decay", type=float, default=0.04)
    p.add_argument("--state-noise-std", type=float, default=0.004)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--steps", type=int, default=80000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=1500)
    p.add_argument("--weight-decay", type=float, default=3e-4)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--eval-every", type=int, default=20000)
    p.add_argument("--eval-scenes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--extra-dirs", nargs="*", default=[])
    p.add_argument("--init-from", default=None)
    p.add_argument("--boost-substr", default=None)
    p.add_argument("--boost-factor", type=float, default=1.0)
    args = p.parse_args()

    if args.confirm:
        fine_opts: List[Optional[int]] = [None] + [int(x) for x in args.fine_steps]
        confirm_full_state(Path(args.checkpoint), Path(args.out), args.horizons, fine_opts, required=args.required)
    else:
        train_full_state(args)
