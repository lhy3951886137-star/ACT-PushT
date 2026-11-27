"""Dataset loaders, exact geometric features, MIN_MAX normalizers, and fast prefetchers."""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R4_DATA = ROOT / "round4" / "precision_data"
S2_DATA = ROOT / "stage2" / "data"
ACT_V2 = ROOT / "act_v2"


def save_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def state_features(raw: np.ndarray, mode: str = "relative") -> np.ndarray:
    """Compute exact, noise-free geometric features from 10-d env.state_vector().

    raw layout: [ee_x, ee_y, T_x, T_y, sin_T, cos_T, g_x, g_y, sin_g, cos_g]
    mode='relative' (14 dims, 100% translation-invariant in table XY):
      Pusher-centric (0..5):
        0: ee_x - T_x
        1: ee_y - T_y
        2: ee_local_x  (R(theta_T)^T @ (ee - T))_x
        3: ee_local_y  (R(theta_T)^T @ (ee - T))_y
        4: ee_x - g_x
        5: ee_y - g_y
      Object & Goal (6..13):
        6: g_x - T_x
        7: g_y - T_y
        8: cos_T
        9: sin_T
       10: cos(g_theta - T_theta)
       11: sin(g_theta - T_theta)
       12: goal_local_x (R(theta_T)^T @ (goal - T))_x
       13: goal_local_y (R(theta_T)^T @ (goal - T))_y
    mode='legacy16' (16 dims, includes absolute ee_xy and T_xy).
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

    g_rel = g_pos - t_pos
    g_loc_x = cos_t * g_rel[..., 0:1] + sin_t * g_rel[..., 1:2]
    g_loc_y = -sin_t * g_rel[..., 0:1] + cos_t * g_rel[..., 1:2]

    cos_d = cos_g * cos_t + sin_g * sin_t
    sin_d = sin_g * cos_t - cos_g * sin_t

    if mode == "legacy16":
        return np.concatenate(
            [ee, ee_rel, ee_loc_x, ee_loc_y, t_pos, cos_t, sin_t, g_rel, cos_d, sin_d, g_loc_x, g_loc_y],
            axis=-1,
        ).astype(np.float32)
    return np.concatenate(
        [ee_rel, ee_loc_x, ee_loc_y, ee_to_g, g_rel, cos_t, sin_t, cos_d, sin_d, g_loc_x, g_loc_y],
        axis=-1,
    ).astype(np.float32)


def load_precision_state_dataset(use_all_150: bool = False, feature_mode: str = "relative") -> Dict[str, List[dict]]:
    """Load verified precision_env state episodes with fixed disjoint train/val splits.

    When use_all_150=False, uses the exact 100 episodes (86 train / 14 val) from round4/precision_data
    for strict apple-to-apple comparison with Round 4 State-Diffusion.
    When use_all_150=True, also includes the 50 extra expert trajectories from stage2/data
    (128 train / 22 val total).
    """
    groups: Dict[str, List[dict]] = {"train": [], "validation": []}
    if not use_all_150:
        manifest = json.loads((R4_DATA / "complete.json").read_text(encoding="utf-8"))
        for row in manifest["episodes"]:
            with np.load(R4_DATA / row["file"]) as d:
                ep = {
                    "states": d["states"].astype(np.float32).copy(),
                    "actions": d["actions"].astype(np.float32).copy(),
                    "coverage": d["coverage"].astype(np.float32).copy(),
                    "seed": int(row["seed"]),
                    "source": row["file"],
                }
            ep["features"] = state_features(ep["states"], mode=feature_mode)
            ep["fine_indices"] = np.flatnonzero(ep["coverage"] > 0.70)
            ep["very_fine_indices"] = np.flatnonzero(ep["coverage"] > 0.88)
            groups[row["partition"]].append(ep)
    else:
        manifest = json.loads((S2_DATA / "complete.json").read_text(encoding="utf-8"))
        for row in manifest["episodes"]:
            src_text = row["source"].replace("\\", "/")
            marker = "/self/"
            src = ROOT / (src_text.split(marker, 1)[1] if marker in src_text else src_text)
            with np.load(src) as d:
                ep = {
                    "states": d["states"].astype(np.float32).copy(),
                    "actions": d["actions"].astype(np.float32).copy(),
                    "coverage": d["coverage"].astype(np.float32).copy(),
                    "seed": int(row["seed"]),
                    "source": str(src),
                }
            ep["features"] = state_features(ep["states"], mode=feature_mode)
            ep["fine_indices"] = np.flatnonzero(ep["coverage"] > 0.70)
            ep["very_fine_indices"] = np.flatnonzero(ep["coverage"] > 0.88)
            groups[row["partition"]].append(ep)

    train_seeds = {e["seed"] for e in groups["train"]}
    val_seeds = {e["seed"] for e in groups["validation"]}
    assert not (train_seeds & val_seeds), "Train/val seed overlap detected!"
    return groups


def fit_state_normalizer(groups: Dict[str, List[dict]], feature_mode: str = "relative") -> dict:
    """Fit MIN_MAX normalizer strictly on training split; sin/cos and actions keep [-1, 1]."""
    feats = np.concatenate([e["features"] for e in groups["train"]], axis=0)
    lo = feats.min(axis=0)
    hi = feats.max(axis=0)
    trig_idx = (8, 9, 12, 13) if feature_mode == "legacy16" else (8, 9, 10, 11)
    for idx in trig_idx:
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


class StateSampler:
    def __init__(
        self,
        episodes: List[dict],
        stats: dict,
        n_obs_steps: int = 2,
        chunk_size: int = 16,
        batch_size: int = 64,
        fine_prob: float = 0.5,
        seed: int = 42,
        validation: bool = False,
        state_noise_std: float = 0.0,
    ):
        self.episodes = episodes
        self.stats = stats
        self.n_obs_steps = n_obs_steps
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.fine_prob = 0.0 if validation else fine_prob
        self.validation = validation
        self.state_noise_std = 0.0 if validation else state_noise_std
        self.rng = np.random.default_rng(seed)

        lengths = np.asarray([len(e["actions"]) for e in episodes], dtype=np.float64)
        self.ep_weights = lengths / lengths.sum()
        self.center = np.asarray(stats["feature_center"], dtype=np.float32)
        self.scale = np.asarray(stats["feature_scale"], dtype=np.float32)

    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bs = self.batch_size
        to = self.n_obs_steps
        k = self.chunk_size
        fdim = len(self.center)
        obs = np.empty((bs, to, fdim), dtype=np.float32)
        act = np.zeros((bs, k, 2), dtype=np.float32)
        valid = np.zeros((bs, k), dtype=bool)

        ep_indices = self.rng.choice(len(self.episodes), size=bs, p=self.ep_weights)
        for bi, ei in enumerate(ep_indices):
            e = self.episodes[ei]
            n = len(e["actions"])
            if not self.validation and self.fine_prob > 0.0 and len(e["fine_indices"]) > 0 and self.rng.random() < self.fine_prob:
                if len(e["very_fine_indices"]) > 0 and self.rng.random() < 0.4:
                    t = int(self.rng.choice(e["very_fine_indices"]))
                else:
                    t = int(self.rng.choice(e["fine_indices"]))
            else:
                t = int(self.rng.integers(n))

            hist_idx = np.maximum(0, t - np.arange(to - 1, -1, -1))
            obs[bi] = e["features"][hist_idx]
            count = min(k, n - t)
            act[bi, :count] = e["actions"][t : t + count]
            valid[bi, :count] = True

        obs = (obs - self.center) / self.scale
        if self.state_noise_std > 0.0:
            # Broadcast noise across n_obs_steps so frame-to-frame velocity is preserved
            obs = obs + self.rng.normal(0.0, self.state_noise_std, size=(bs, 1, fdim)).astype(np.float32)
        return obs, act, valid


class VisionSampler:
    """Stage 2 pure-vision dataset sampler (robot_xy + dual 128x128 uint8 cameras)."""

    def __init__(
        self,
        stats: Optional[dict] = None,
        n_obs_steps: int = 2,
        chunk_size: int = 16,
        fine_prob: float = 0.5,
        seed: int = 42,
        validation: bool = False,
    ):
        manifest = json.loads((S2_DATA / "complete.json").read_text(encoding="utf-8"))
        rows = [r for r in manifest["episodes"] if (r["partition"] == "validation") == validation]
        if not rows:
            raise ValueError("No episodes in vision partition")
        self.episodes = []
        self.rng = np.random.default_rng(seed)
        self.n_obs_steps = n_obs_steps
        self.chunk_size = chunk_size
        self.fine_prob = 0.0 if validation else fine_prob
        self.validation = validation

        for r in rows:
            with np.load(S2_DATA / r["metadata"]) as d:
                meta = {k: d[k].copy() for k in d.files}
            assert set(meta) == {"robot_xy", "actions", "coverage"} and meta["robot_xy"].shape[1] == 2
            self.episodes.append(
                {
                    "robot_xy": meta["robot_xy"].astype(np.float32),
                    "actions": meta["actions"].astype(np.float32),
                    "coverage": meta["coverage"].astype(np.float32),
                    "images": np.load(S2_DATA / r["rgb"], mmap_mode="r"),
                    "fine": np.flatnonzero(meta["coverage"] > 0.85),
                    "very_fine": np.flatnonzero(meta["coverage"] > 0.92),
                    "seed": int(r["seed"]),
                }
            )
        lengths = np.asarray([len(e["actions"]) for e in self.episodes], dtype=np.float64)
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

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        to = self.n_obs_steps
        k = self.chunk_size
        robot = np.empty((batch_size, to, 2), dtype=np.float32)
        images = np.empty((batch_size, to, 2, 128, 128, 3), dtype=np.uint8)
        actions = np.zeros((batch_size, k, 2), dtype=np.float32)
        valid = np.zeros((batch_size, k), dtype=bool)

        ep_indices = self.rng.choice(len(self.episodes), size=batch_size, p=self.weights)
        for bi, ei in enumerate(ep_indices):
            e = self.episodes[ei]
            n = len(e["actions"])
            if not self.validation and len(e["fine"]) > 0 and self.rng.random() < self.fine_prob:
                if len(e["very_fine"]) > 0 and self.rng.random() < 0.45:
                    t = int(self.rng.choice(e["very_fine"]))
                else:
                    t = int(self.rng.choice(e["fine"]))
            else:
                t = int(self.rng.integers(n))

            hist = np.maximum(0, t - np.arange(to - 1, -1, -1))
            robot[bi] = e["robot_xy"][hist]
            images[bi] = e["images"][hist]
            count = min(k, n - t)
            actions[bi, :count] = e["actions"][t : t + count]
            valid[bi, :count] = True

        robot = (robot - self.robot_center) / self.robot_scale
        return robot, images, actions, valid


class AsyncPrefetcher:
    """Background thread prefetcher to overlap memmap disk reads with GPU compute."""

    def __init__(self, fn, maxsize: int = 8):
        self.fn = fn
        self.q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while not self.stop_event.is_set():
            item = self.fn()
            while not self.stop_event.is_set():
                try:
                    self.q.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def next(self):
        return self.q.get()

    def close(self):
        self.stop_event.set()
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        self.thread.join(timeout=1.0)
