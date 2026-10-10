"""Phase 4 GPU throughput: environment steps per second for the PyTorch engine.

Configurations: eager PyTorch, torch.compile (default), and torch.compile with
mode "reduce-overhead" (CUDA graphs), each at several batch sizes. As in
bench_cpu.py, battles play uniformly random legal actions and finished battles
are replaced every step from a team pool on the device, so every counted step
advances a live battle.

  env_steps_per_s     battle steps / GPU time of step() alone (CUDA events)
  end_to_end_per_s    battle steps / wall time including action sampling and resets

The step has no data-dependent branches, so its cost does not depend on how
far along the battles are.

Usage: uv run python tools/bench_gpu.py [--seconds 2] [--trials 3] [--out results/phase4_gpu.json]
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from pokearena import batched, gpu

BATCH_SIZES = [1024, 4096, 16384, 65536, 262144, 1048576]
CONFIGS = {"eager": None, "compile": "default", "cudagraph": "reduce-overhead"}
POOL = 4096


def bench(n: int, step_fn, seconds: float, t: gpu.TorchTables, cudagraph: bool) -> dict:
    device = "cuda"
    rng = np.random.default_rng(n)
    pool = torch.as_tensor(np.stack([np.stack([rng.permutation(30)[:3] for _ in range(2)]) for _ in range(POOL)]),
                           dtype=torch.int32, device=device)
    gen = torch.Generator(device=device).manual_seed(n)

    def fresh():
        index = torch.randint(0, POOL, (n,), device=device, generator=gen)
        seeds = torch.randint(0, 2**32, (n,), device=device, generator=gen, dtype=torch.int64)
        return pool[index], seeds

    s = gpu.new_batch(*fresh(), t)

    def one(s):
        actions = gpu.random_actions(s, gen)
        if cudagraph:
            torch.compiler.cudagraph_mark_step_begin()
        start.record()
        s = gpu.step(s, actions, t, check=False, fn=step_fn)
        if cudagraph:                       # graph outputs are reused by the next replay
            s = gpu.TorchState(**{k: v.clone() for k, v in vars(s).items()})
        end.record()
        return gpu.reset_done(s, *fresh(), t)

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(10):                     # warm up, including compilation
        s = one(s)
    torch.cuda.synchronize()

    step_ms, iterations = 0.0, 0
    wall = time.perf_counter()
    while time.perf_counter() - wall < seconds or iterations < 5:
        s = one(s)
        torch.cuda.synchronize()            # so the event pair can be read; counted in end-to-end
        step_ms += start.elapsed_time(end)
        iterations += 1
    wall = time.perf_counter() - wall
    return {"env_steps_per_s": n * iterations / (step_ms / 1000), "end_to_end_per_s": n * iterations / wall}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--sizes", type=int, nargs="*", default=BATCH_SIZES)
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--out", default="results/phase4_gpu.json")
    args = parser.parse_args()

    t = gpu.tables_on("cuda", batched.precompute())
    rows = []
    for name in args.configs:
        mode = CONFIGS[name]
        for n in args.sizes:
            # A fresh compiled function per batch size, each specialized to its shapes.
            step_fn = gpu.step_tensors if mode is None else gpu.compiled_step(mode)
            runs = [bench(n, step_fn, args.seconds, t, cudagraph=name == "cudagraph") for _ in range(args.trials)]
            result = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
            rows.append({"engine": f"torch_{name}", "batch": n, **result})
            print(f"{name:<10} N={n:<8} {result['env_steps_per_s']:>16,.0f}"
                  f"  end-to-end {result['end_to_end_per_s']:>14,.0f}", flush=True)

    props = torch.cuda.get_device_properties(0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "machine": {"gpu": props.name, "gpu_memory_gb": round(props.total_memory / 2**30, 1),
                    "torch": torch.__version__, "cuda": torch.version.cuda, "platform": platform.platform()},
        "settings": {"seconds": args.seconds, "trials": args.trials},
        "rows": rows,
    }, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
