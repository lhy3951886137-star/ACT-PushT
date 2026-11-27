"""Closed-loop ACT inference (receding horizon + temporal ensembling) & parallel/serial evaluation."""
from __future__ import annotations

from collections import deque
import math
import os
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch

from act_v2.data import ROOT, save_json, state_features
from act_v2.model import StateACT, VisionACT


def wilson_ci95(successes: int, n: int) -> List[float]:
    if n == 0:
        return [0.0, 0.0]
    p = successes / n
    z = 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - m), float(c + m)]


class TemporalEnsembler:
    """Temporal ensembling over overlapping ACT action chunks.

    order='newest': weight(age) = exp(-coeff * age), prioritizing freshest closed-loop feedback.
    order='oldest': weight(idx) = exp(-coeff * idx), original Tony Zhao ALOHA convention.
    """

    def __init__(self, chunk_size: int, coeff: float = 0.3, max_history: int = 8, order: str = "newest"):
        self.chunk_size = chunk_size
        self.coeff = coeff
        self.max_history = min(chunk_size, max_history)
        self.order = order
        self.reset()

    def reset(self) -> None:
        # Store (predicted_at_step, chunk[K, 2])
        self.history: deque = deque(maxlen=self.max_history)
        self.step = 0

    def update(self, chunk: np.ndarray) -> np.ndarray:
        self.history.append((self.step, chunk))
        preds = []
        ages = []
        for pred_step, pred_chunk in self.history:
            age = self.step - pred_step
            if 0 <= age < self.chunk_size:
                preds.append(pred_chunk[age])
                ages.append(age)
        preds_arr = np.stack(preds, axis=0)  # oldest -> newest
        if self.order == "oldest":
            weights = np.exp(-self.coeff * np.arange(len(preds_arr), dtype=np.float64))
        else:
            weights = np.exp(-self.coeff * np.asarray(ages, dtype=np.float64))
        weights /= weights.sum()
        action = (preds_arr * weights[:, None]).sum(axis=0)
        self.step += 1
        return action.astype(np.float32)


class StateACTPolicy:
    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        model: Optional[StateACT] = None,
        stats: Optional[dict] = None,
        cfg: Optional[dict] = None,
        n_action_steps: int = 4,
        temporal_ensemble: Optional[float] = None,
        ensemble_window: int = 8,
        ensemble_order: str = "newest",
        weight_kind: str = "ema",
    ):
        if model is None:
            ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            cfg = ck["config"]
            stats = ck["stats"]
            fmode = cfg.get("feature_mode", "legacy16")
            obj_dim = 8 if fmode == "relative" else 10
            model = StateACT(
                n_obs_steps=cfg["n_obs_steps"],
                pusher_dim=6,
                object_dim=obj_dim,
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
        self.feature_mode = cfg.get("feature_mode", stats.get("feature_mode", "legacy16"))
        self.n_obs_steps = int(cfg["n_obs_steps"])
        self.chunk_size = int(cfg["chunk_size"])
        self.n_action_steps = int(n_action_steps)
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
        if self.ensembler is not None:
            self.ensembler.reset()

    @torch.inference_mode()
    def action(self, raw_state: np.ndarray) -> np.ndarray:
        feat = (state_features(raw_state, mode=self.feature_mode) - self.center) / self.scale
        self.obs_history.append(feat)
        while len(self.obs_history) < self.n_obs_steps:
            self.obs_history.append(feat)

        if self.ensembler is not None:
            obs_t = torch.as_tensor(np.stack(self.obs_history, axis=0), dtype=torch.float32, device="cuda")[None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                chunk = self.model(obs_t)[0][0].float().cpu().numpy()
            if not np.isfinite(chunk).all():
                raise RuntimeError("Nonfinite State-ACT action")
            act = self.ensembler.update(chunk)
            return np.clip(act, -1.0, 1.0).astype(np.float32)

        if not self.queue:
            obs_t = torch.as_tensor(np.stack(self.obs_history, axis=0), dtype=torch.float32, device="cuda")[None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                chunk = self.model(obs_t)[0][0].float().cpu().numpy()
            if not np.isfinite(chunk).all():
                raise RuntimeError("Nonfinite State-ACT action")
            self.queue.extend(chunk[: self.n_action_steps])
        return np.clip(self.queue.popleft(), -1.0, 1.0).astype(np.float32)


class VisionACTPolicy:
    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        model: Optional[VisionACT] = None,
        stats: Optional[dict] = None,
        cfg: Optional[dict] = None,
        n_action_steps: int = 4,
        temporal_ensemble: Optional[float] = None,
        ensemble_window: int = 8,
        ensemble_order: str = "newest",
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
                use_vae=cfg["use_vae"],
                latent_dim=cfg.get("latent_dim", 32),
                crop_size=cfg.get("crop_size", 116),
                token_mode=cfg.get("token_mode", "hybrid"),
            ).cuda().eval()
            model.load_state_dict(ck[weight_kind] if weight_kind in ck else ck["model"])
        self.model = model.eval()
        self.stats = stats
        self.cfg = cfg
        self.n_obs_steps = int(cfg["n_obs_steps"])
        self.chunk_size = int(cfg["chunk_size"])
        self.n_action_steps = int(n_action_steps)
        self.robot_center = np.asarray(stats["robot_center"], dtype=np.float32)
        self.robot_scale = np.asarray(stats["robot_scale"], dtype=np.float32)
        self.ensembler = (
            TemporalEnsembler(self.chunk_size, temporal_ensemble, ensemble_window, ensemble_order)
            if temporal_ensemble is not None
            else None
        )
        self.reset()

    def reset(self, seed: int = 0) -> None:
        self.history: deque = deque(maxlen=self.n_obs_steps)
        self.queue: deque = deque()
        if self.ensembler is not None:
            self.ensembler.reset()

    @torch.inference_mode()
    def action(self, observation: dict) -> np.ndarray:
        assert set(observation) == {"robot_xy", "images"} and observation["robot_xy"].shape == (2,)
        self.history.append(observation)
        while len(self.history) < self.n_obs_steps:
            self.history.append(observation)

        if self.ensembler is not None or not self.queue:
            robot = np.stack([x["robot_xy"] for x in self.history], axis=0)
            images = np.stack([x["images"] for x in self.history], axis=0)
            robot = (robot - self.robot_center) / self.robot_scale
            r_t = torch.as_tensor(robot, dtype=torch.float32, device="cuda")[None]
            i_t = torch.as_tensor(images, dtype=torch.uint8, device="cuda")[None]
            chunk = self.model(r_t, i_t, augment=False)[0][0].float().cpu().numpy()
            if not np.isfinite(chunk).all():
                raise RuntimeError("Nonfinite Vision-ACT action")
            if self.ensembler is not None:
                return np.clip(self.ensembler.update(chunk), -1.0, 1.0).astype(np.float32)
            self.queue.extend(chunk[: self.n_action_steps])
        return np.clip(self.queue.popleft(), -1.0, 1.0).astype(np.float32)


def evaluate_state(
    policy: StateACTPolicy,
    seeds: Iterable[int],
    out_dir: Optional[Path] = None,
    horizon: int = 2000,
    save_traces: bool = False,
) -> dict:
    from precision_env import make_env

    seeds = list(seeds)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(False)
    rows = []
    try:
        for seed in seeds:
            seed_file = out_dir / f"seed_{seed}.json" if out_dir is not None else None
            if seed_file is not None and seed_file.exists():
                row = __import__("json").loads(seed_file.read_text(encoding="utf-8"))
            else:
                env.reset_task(seed)
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
        "episodes": rows,
    }
    if out_dir is not None:
        save_json(out_dir / "summary.json", result)
    return result


def evaluate_vision(
    policy: VisionACTPolicy,
    seeds: Iterable[int],
    out_dir: Optional[Path] = None,
    horizon: int = 2000,
    zero_images: bool = False,
    save_traces: bool = False,
) -> dict:
    from vision_env import make_env, observe

    seeds = list(seeds)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    env = make_env()
    rows = []
    try:
        for seed in seeds:
            seed_file = out_dir / f"seed_{seed}.json" if out_dir is not None else None
            if seed_file is not None and seed_file.exists():
                row = __import__("json").loads(seed_file.read_text(encoding="utf-8"))
            else:
                env.reset_task(seed)
                policy.reset(seed)
                metrics = []
                actions = []
                states = []
                for _ in range(horizon):
                    if (ROOT / "STOP_REQUESTED").exists():
                        raise InterruptedError("Stop requested")
                    obs = observe(env)
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
            if out_dir is not None:
                save_json(out_dir / "progress.json", {"pid": os.getpid(), "completed": len(rows), "target": len(seeds), "last": row})
    finally:
        env.close()

    n = len(rows)
    succ = sum(r["success"] for r in rows)
    failed = [r for r in rows if not r["success"]]
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
        "episodes": rows,
    }
    if out_dir is not None:
        save_json(out_dir / "summary.json", result)
    return result
