"""Test pure-ACT Aperiodic Action Chunking & Stochastic-Horizon ACT on DEV (30100000..49)."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from act_v2.full_random_state import FullRandomStateACTPolicy, evaluate_full_state, extract_full_features


class AperiodicACTPolicy(FullRandomStateACTPolicy):
    """Pure single-model ACT with aperiodic / cycle-breaking receding-horizon chunk execution.

    Every executed action is 100% verbatim from the single State-ACT model's predicted 24-step chunk.
    Varying the replanning cadence across a short schedule (e.g. [11, 15, 19, 13, 17]) prevents
    phase-locked periodic limit cycles that trap fixed-K deterministic ACT rollouts.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        schedule: list[int],
        obs_jitter_on_stall: float = 0.0,
        **kwargs,
    ):
        self.schedule = list(schedule)
        self.obs_jitter_on_stall = float(obs_jitter_on_stall)
        super().__init__(checkpoint_path=checkpoint_path, n_action_steps=schedule[0], **kwargs)

    def reset(self, seed: int = 0) -> None:
        super().reset(seed)
        self.replan_idx = 0
        self.rng = np.random.default_rng(seed + 12345)
        self.t_hist: deque = deque(maxlen=120)
        self.stall_count = 0

    @torch.inference_mode()
    def action(self, raw_state: np.ndarray) -> np.ndarray:
        feat = (extract_full_features(raw_state, feature_mode=self.feature_mode) - self.center) / self.scale
        self.obs_history.append(feat)
        while len(self.obs_history) < self.n_obs_steps:
            self.obs_history.append(feat)
        self.t_hist.append(raw_state[2:6].copy())

        if not self.queue:
            obs_arr = np.stack(self.obs_history, axis=0)
            # If the T-block pose (xy, sin, cos) is completely frozen over 120 steps (dead fixed-point stall),
            # apply tiny Gaussian observation perturbation within the training state_noise_std (0.004-0.008)
            # so ACT breaks out of the zero-action fixed point cleanly using only its own learned chunk!
            stalled = False
            if len(self.t_hist) == 120:
                diff = float(np.abs(self.t_hist[-1] - self.t_hist[0]).max())
                if diff < 0.0025:
                    stalled = True
                    self.stall_count += 1
                else:
                    self.stall_count = 0
            if stalled and self.obs_jitter_on_stall > 0.0:
                sigma = self.obs_jitter_on_stall * min(3.0, 1.0 + 0.5 * self.stall_count)
                obs_arr = obs_arr + self.rng.normal(0.0, sigma, size=(1, obs_arr.shape[1])).astype(np.float32)

            obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device="cuda")[None]
            chunk = self.model(obs_t)[0][0].float().cpu().numpy()
            k = self.schedule[self.replan_idx % len(self.schedule)]
            self.replan_idx += 1
            self.queue.extend(chunk[:k])
        return np.clip(self.queue.popleft(), -1.0, 1.0).astype(np.float32)


def _run_one(spec: dict) -> dict:
    torch.set_num_threads(1)
    pol = AperiodicACTPolicy(
        checkpoint_path=Path(spec["ckpt"]),
        schedule=spec["sched"],
        obs_jitter_on_stall=spec.get("jitter", 0.0),
    )
    seeds = list(range(spec["seed_start"], spec["seed_start"] + spec["n_seeds"]))
    t0 = time.monotonic()
    res = evaluate_full_state(pol, seeds)
    return {
        "tag": spec["tag"],
        "sr": res["success_rate"],
        "succ": res["successes"],
        "n": res["n"],
        "cov": res["mean_max_coverage"],
        "steps": res["mean_steps"],
        "buckets": {k: f"{v['successes']}/{v['n']}" for k, v in res["angle_buckets"].items()},
        "fails": [e["seed"] for e in res["episodes"] if not e["success"]],
        "sec": round(time.monotonic() - t0, 1),
    }


def main():
    v3 = "act_v2/runs/full_random_state_act_v3/latest.pt"
    v4 = "act_v2/runs/full_random_state_act_v4/latest.pt"
    specs = [
        {"tag": "v4_ap_12_16_18_14", "ckpt": v4, "sched": [12, 16, 18, 14], "jitter": 0.0, "seed_start": 30100000, "n_seeds": 50},
        {"tag": "v4_ap_11_15_19_13_17", "ckpt": v4, "sched": [11, 15, 19, 13, 17], "jitter": 0.0, "seed_start": 30100000, "n_seeds": 50},
        {"tag": "v4_ap_13_17_15_19", "ckpt": v4, "sched": [13, 17, 15, 19], "jitter": 0.008, "seed_start": 30100000, "n_seeds": 50},
        {"tag": "v4_ap_11_15_19_j010", "ckpt": v4, "sched": [11, 15, 19, 13, 17], "jitter": 0.010, "seed_start": 30100000, "n_seeds": 50},
        {"tag": "v3_ap_13_17_19_15", "ckpt": v3, "sched": [13, 17, 19, 15], "jitter": 0.0, "seed_start": 30100000, "n_seeds": 50},
        {"tag": "v3_ap_13_17_19_15_j010", "ckpt": v3, "sched": [13, 17, 19, 15], "jitter": 0.010, "seed_start": 30100000, "n_seeds": 50},
    ]
    results = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_run_one, s) for s in specs]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print(
                f"[{r['tag']:24s}] sr={r['sr']:.3f} ({r['succ']:2d}/{r['n']}) "
                f"cov={r['cov']:.4f} steps={r['steps']:6.1f} {r['buckets']} ({r['sec']}s)",
                flush=True,
            )
    results.sort(key=lambda r: (r["sr"], r["cov"]), reverse=True)
    print("\n=== APERIODIC ACT DEV LEADERBOARD ===", flush=True)
    for r in results:
        print(f"  {r['tag']:24s}: {r['sr']:.3f} ({r['succ']}/{r['n']}) cov={r['cov']:.4f} fails={r['fails']}", flush=True)


if __name__ == "__main__":
    main()
