"""Create single-model SWA/Soup checkpoints from same-basin v3/v4 State-ACT models and sweep DEV (30100000..49)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from act_v2.full_random_state import FullRandomStateACTPolicy, evaluate_full_state


def make_soup(paths: list[Path], weights: list[float], out_path: Path) -> None:
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    base = cks[-1]
    new_ema = {}
    for k in base["ema"].keys():
        acc = None
        for ck, w in zip(cks, weights):
            t = ck["ema"][k].float() * w
            acc = t if acc is None else acc + t
        new_ema[k] = acc.to(dtype=base["ema"][k].dtype)
    out = {
        "config": base["config"],
        "stats": base["stats"],
        "ema": new_ema,
        "model": new_ema,
        "soup_sources": [str(p) for p in paths],
        "soup_weights": weights,
    }
    torch.save(out, out_path)


def _eval_one(spec: dict) -> dict:
    torch.set_num_threads(1)
    seeds = list(range(spec["seed_start"], spec["seed_start"] + spec["n_seeds"]))
    pol = FullRandomStateACTPolicy(
        checkpoint_path=Path(spec["ckpt"]),
        n_action_steps=spec["k"],
        fine_action_steps=spec.get("fk"),
        weight_kind="ema",
    )
    t0 = time.monotonic()
    res = evaluate_full_state(pol, seeds)
    return {
        "tag": spec["tag"],
        "ckpt": spec["ckpt"],
        "k": spec["k"],
        "fk": spec.get("fk"),
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
    v3_latest = Path("act_v2/runs/full_random_state_act_v3/latest.pt")
    v4_best = Path("act_v2/runs/full_random_state_act_v4/best.pt")
    v4_latest = Path("act_v2/runs/full_random_state_act_v4/latest.pt")
    soup_30_50 = Path("act_v2/runs/full_random_state_act_v4/soup_30_50.pt")
    soup_v3_v4 = Path("act_v2/runs/full_random_state_act_v4/soup_v3_v4.pt")

    make_soup([v4_best, v4_latest], [0.5, 0.5], soup_30_50)
    make_soup([v3_latest, v4_best, v4_latest], [0.4, 0.3, 0.3], soup_v3_v4)
    print("Created SWA/Soup single State-ACT checkpoints.", flush=True)

    specs = [
        {"tag": "v4_50k_exec12", "ckpt": str(v4_latest), "k": 12, "seed_start": 30100000, "n_seeds": 50},
        {"tag": "v4_50k_exec14", "ckpt": str(v4_latest), "k": 14, "seed_start": 30100000, "n_seeds": 50},
        {"tag": "v4_50k_exec16", "ckpt": str(v4_latest), "k": 16, "seed_start": 30100000, "n_seeds": 50},
        {"tag": "v4_50k_exec20", "ckpt": str(v4_latest), "k": 20, "seed_start": 30100000, "n_seeds": 50},
        {"tag": "v4_50k_exec22", "ckpt": str(v4_latest), "k": 22, "seed_start": 30100000, "n_seeds": 50},
        {"tag": "v4_30k_exec14", "ckpt": str(v4_best), "k": 14, "seed_start": 30100000, "n_seeds": 50},
        {"tag": "v4_30k_exec16", "ckpt": str(v4_best), "k": 16, "seed_start": 30100000, "n_seeds": 50},
        {"tag": "v4_30k_exec20", "ckpt": str(v4_best), "k": 20, "seed_start": 30100000, "n_seeds": 50},
        {"tag": "soup3050_exec14", "ckpt": str(soup_30_50), "k": 14, "seed_start": 30100000, "n_seeds": 50},
        {"tag": "soup3050_exec16", "ckpt": str(soup_30_50), "k": 16, "seed_start": 30100000, "n_seeds": 50},
        {"tag": "soup3050_exec18", "ckpt": str(soup_30_50), "k": 18, "seed_start": 30100000, "n_seeds": 50},
        {"tag": "soup3050_exec20", "ckpt": str(soup_30_50), "k": 20, "seed_start": 30100000, "n_seeds": 50},
        {"tag": "soup_v3v4_exec14", "ckpt": str(soup_v3_v4), "k": 14, "seed_start": 30100000, "n_seeds": 50},
        {"tag": "soup_v3v4_exec16", "ckpt": str(soup_v3_v4), "k": 16, "seed_start": 30100000, "n_seeds": 50},
        {"tag": "soup_v3v4_exec18", "ckpt": str(soup_v3_v4), "k": 18, "seed_start": 30100000, "n_seeds": 50},
        {"tag": "soup_v3v4_exec20", "ckpt": str(soup_v3_v4), "k": 20, "seed_start": 30100000, "n_seeds": 50},
        {"tag": "soup_v3v4_e18_f10", "ckpt": str(soup_v3_v4), "k": 18, "fk": 10, "seed_start": 30100000, "n_seeds": 50},
        {"tag": "v4_50k_e18_f10", "ckpt": str(v4_latest), "k": 18, "fk": 10, "seed_start": 30100000, "n_seeds": 50},
    ]

    results = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_eval_one, s) for s in specs]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print(
                f"[{r['tag']:18s}] sr={r['sr']:.3f} ({r['succ']:2d}/{r['n']}) "
                f"cov={r['cov']:.4f} steps={r['steps']:6.1f} {r['buckets']} ({r['sec']}s)",
                flush=True,
            )
    results.sort(key=lambda r: (r["sr"], r["cov"]), reverse=True)
    print("\n=== V4 / SOUP DEV LEADERBOARD ===", flush=True)
    for r in results:
        print(f"  {r['tag']:18s}: {r['sr']:.3f} ({r['succ']}/{r['n']}) cov={r['cov']:.4f} fails={r['fails']}", flush=True)
    out_json = Path("act_v2/runs/full_random_state_act_v4/dev_sweep_all.json")
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
