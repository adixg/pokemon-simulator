"""Phase 3 CPU baseline: environment steps per second.

Measures the reference engine (one battle at a time, Python objects) and the
batched NumPy engine at several batch sizes, with damage from the
precomputed table and computed arithmetically. Battles play uniformly random
legal actions and finished battles are replaced immediately, so every
counted step advances a live battle (normal turns and forced-switch phases
both count as one step).

Reported per configuration, median of --trials runs:
  env_steps_per_s      battle steps / time spent in step() alone
  end_to_end_per_s     battle steps / time including action sampling and resets

Usage: uv run python tools/bench_cpu.py [--seconds 2] [--trials 3] [--out results/phase3_cpu.json]
"""

import argparse
import json
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np

from pokearena import batched, engine
from pokearena.data import load_tables

BATCH_SIZES = [1, 16, 256, 4096, 16384, 65536]
POOL = 4096


def team_pool(rng: np.random.Generator) -> np.ndarray:
    return np.stack([np.stack([rng.permutation(30)[:3] for _ in range(2)]) for _ in range(POOL)]).astype(np.int32)


def bench_reference(seconds: float) -> dict:
    t = load_tables()
    rng = random.Random(0)
    new = lambda: engine.new_battle([rng.sample(range(30), 3), rng.sample(range(30), 3)], rng.randrange(2**32), t)  # noqa: E731
    b = new()
    steps, step_time = 0, 0.0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        actions = []
        for side in (0, 1):
            legal = np.flatnonzero(engine.action_mask(b, side))
            actions.append(int(rng.choice(legal)) if len(legal) else 0)
        t0 = time.perf_counter()
        b, _ = engine.step(b, tuple(actions), t)
        step_time += time.perf_counter() - t0
        steps += 1
        if b.phase == engine.DONE:
            b = new()
    total = time.perf_counter() - start
    return {"env_steps_per_s": steps / step_time, "end_to_end_per_s": steps / total}


def bench_batched(n: int, seconds: float, table_damage: bool, pre: batched.Precomputed) -> dict:
    rng = np.random.default_rng(n)
    pool = team_pool(rng)
    pick = lambda: (pool[rng.integers(0, POOL, n)], rng.integers(0, 2**32, n))  # noqa: E731
    s = batched.new_batch(*pick(), pre)
    for _ in range(3):                                           # warm up
        s = batched.step(s, batched.random_actions(s, rng), pre, check=False, table_damage=table_damage)
    iterations, step_time = 0, 0.0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds or iterations < 3:
        actions = batched.random_actions(s, rng)
        t0 = time.perf_counter()
        s = batched.step(s, actions, pre, check=False, table_damage=table_damage)
        step_time += time.perf_counter() - t0
        s = batched.reset_done(s, *pick(), pre)
        iterations += 1
    total = time.perf_counter() - start
    return {"env_steps_per_s": n * iterations / step_time, "end_to_end_per_s": n * iterations / total}


def median_of(trials: int, fn) -> dict:
    runs = [fn() for _ in range(trials)]
    return {k: statistics.median(r[k] for r in runs) for k in runs[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--out", default="results/phase3_cpu.json")
    args = parser.parse_args()

    pre = batched.precompute()
    rows = [{"engine": "reference", "batch": 1, "damage": "computed",
             **median_of(args.trials, lambda: bench_reference(args.seconds))}]
    print(f"reference            {rows[-1]['env_steps_per_s']:>14,.0f}")
    for table_damage in (True, False):
        for n in BATCH_SIZES:
            result = median_of(args.trials, lambda: bench_batched(n, args.seconds, table_damage, pre))
            rows.append({"engine": "batched_numpy", "batch": n,
                         "damage": "table" if table_damage else "computed", **result})
            print(f"numpy {rows[-1]['damage']:<8} N={n:<6} {result['env_steps_per_s']:>14,.0f}"
                  f"  end-to-end {result['end_to_end_per_s']:>12,.0f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "machine": {"cpu": platform.processor() or platform.machine(), "python": platform.python_version(),
                    "numpy": np.__version__, "platform": platform.platform()},
        "settings": {"seconds": args.seconds, "trials": args.trials, "threads": "single (NumPy default)"},
        "rows": rows,
    }, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
