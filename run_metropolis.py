"""Run all Metropolis simulations. Safe to restart: finished runs are skipped."""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import config
import ising

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results" / "runs"
DATA = ROOT / "data"


def run_path(L: int, tag: str) -> Path:
    return RUNS / f"metropolis_L{L}_T{tag}.json"


def jobs():
    out = []
    for L in config.L_LIST:
        for i, T in enumerate(config.T_GRID):
            out.append((L, T, i, f"{T:.2f}"))
        out.append((L, ising.tc_exact(), config.TC_RUN_INDEX, "Tc"))
    out.sort(key=lambda j: -j[0])  # largest lattices first
    return out


def do_job(job):
    L, T, run_index, tag = job
    t0 = time.time()
    rng = ising.make_rng(config.ALGO_METROPOLIS, L, run_index)
    out = ising.run_metropolis(L, T, config.N_THERM, config.N_MEAS, rng)
    res = ising.summarize_series(out["E"], out["M"], L, T)
    res.update({"algorithm": "metropolis", "run_index": run_index, "tag": tag,
                "acceptance": out["acceptance"], "wall_time": time.time() - t0})
    if tag == "Tc":
        DATA.mkdir(exist_ok=True)
        np.savez_compressed(DATA / f"tc_series_metropolis_L{L}.npz",
                            E=out["E"].astype(np.int32), M=out["M"].astype(np.int32))
    tmp = run_path(L, tag).with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    os.replace(tmp, run_path(L, tag))
    return L, tag, res["wall_time"]


def time_estimate():
    rng = ising.make_rng(config.ALGO_METROPOLIS, 64, 0)
    t0 = time.time()
    ising.run_metropolis(64, 2.27, 0, 2000, rng)
    per_sweep = (time.time() - t0) / 2000
    return per_sweep


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs() if not run_path(j[0], j[3]).exists()]
    print(f"{len(todo)} runs to do")
    if not todo:
        return
    per_sweep64 = time_estimate()
    workers = min(config.MAX_WORKERS, os.cpu_count() or 1)
    sweeps = config.N_THERM + config.N_MEAS
    proj = 0.0
    for L, _, _, _ in todo:
        proj += per_sweep64 * sweeps * (0.45 + 0.55 * (L / 64) ** 2)
    print(f"timing at L=64: {per_sweep64 * 1e6:.0f} us per sweep; projected total "
          f"{proj / workers / 3600:.2f} h with {workers} workers (rough)")
    with open(ROOT / "RUN_LOG.md", "a") as f:
        f.write(f"- {time.strftime('%F %T')} Metropolis timing: 2000 sweeps at L=64 took "
                f"{per_sweep64 * 2000:.2f} s ({per_sweep64 * 1e6:.0f} us per sweep). Projected total "
                f"about {proj / workers / 3600:.2f} h with {workers} workers (rough estimate, "
                f"assumes cost per sweep = 45 percent fixed overhead + 55 percent proportional to L^2). "
                f"Running regardless.\n")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(do_job, j) for j in todo]
        for k, fu in enumerate(as_completed(futs), 1):
            L, tag, wt = fu.result()
            print(f"[{k}/{len(todo)}] L={L} T={tag} {wt:.0f}s elapsed {time.time() - t0:.0f}s", flush=True)
    with open(ROOT / "RUN_LOG.md", "a") as f:
        f.write(f"- {time.strftime('%F %T')} Metropolis runs finished, wall time of this invocation "
                f"{(time.time() - t0) / 3600:.2f} h with {workers} workers.\n")


if __name__ == "__main__":
    main()
