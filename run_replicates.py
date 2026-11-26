"""Metropolis fine grid with new seeds (replicates 1, 2, 3). Added after the first version
(see UPGRADE_PLAN.md). Safe to restart: finished runs are skipped.

    python run_replicates.py            full run, writes results/replicates/
    python run_replicates.py --quick    L = 8 and 16, replicate 1, 50000 sweeps, writes results_quick/replicates/
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import config
import ising

ROOT = Path(__file__).resolve().parent
REPLICATES = [1, 2, 3]


def run_index_for(rep: int, T: float) -> int:
    """Run index of a replicate run. The first version used 0 to 24 and 999."""
    return 10000 * rep + config.T_GRID.index(T)


def run_path(out_dir: Path, L: int, T: float, rep: int) -> Path:
    return out_dir / f"metropolis_L{L}_T{T:.2f}_r{rep}.json"


def make_jobs(replicates, sizes):
    jobs = []
    for rep in replicates:
        for L in sizes:
            for T in config.T_FINE:
                jobs.append((rep, L, T))
    jobs.sort(key=lambda j: -j[1])  # largest lattices first
    return jobs


def do_job(args):
    rep, L, T, out_dir, n_meas = args
    t0 = time.time()
    idx = run_index_for(rep, T)
    rng = ising.make_rng(config.ALGO_METROPOLIS, L, idx)
    out = ising.run_metropolis(L, T, config.N_THERM, n_meas, rng)
    res = ising.summarize_series(out["E"], out["M"], L, T)
    res.update({"algorithm": "metropolis", "replicate": rep, "run_index": idx, "tag": f"{T:.2f}",
                "acceptance": out["acceptance"], "wall_time": time.time() - t0})
    p = run_path(Path(out_dir), L, T, rep)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    os.replace(tmp, p)
    return L, T, rep, res["wall_time"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        out_dir, reps, sizes, n_meas = ROOT / "results_quick" / "replicates", [1], [8, 16], 50000
    else:
        out_dir, reps, sizes, n_meas = ROOT / "results" / "replicates", REPLICATES, config.L_LIST, config.N_MEAS
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [j for j in make_jobs(reps, sizes) if not run_path(out_dir, j[1], j[2], j[0]).exists()]
    print(f"{len(todo)} runs to do", flush=True)
    if not todo:
        return
    t_est = time.time()
    ising.run_metropolis(64, 2.27, 0, 2000, ising.make_rng(config.ALGO_METROPOLIS, 64, 0))
    per_sweep64 = (time.time() - t_est) / 2000
    workers = min(config.MAX_WORKERS, os.cpu_count() or 1)
    proj = sum(per_sweep64 * (config.N_THERM + n_meas) * (0.45 + 0.55 * (j[1] / 64) ** 2) for j in todo)
    print(f"timing at L=64: {per_sweep64 * 1e6:.0f} us per sweep; projected total "
          f"{proj / workers / 60:.0f} min with {workers} workers (rough)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(do_job, (rep, L, T, str(out_dir), n_meas)) for rep, L, T in todo]
        for k, fu in enumerate(as_completed(futs), 1):
            L, T, rep, wt = fu.result()
            print(f"[{k}/{len(todo)}] L={L} T={T:.2f} r{rep} {wt:.0f}s elapsed {time.time() - t0:.0f}s", flush=True)
    print(f"total time {(time.time() - t0) / 60:.1f} min with {workers} workers")


if __name__ == "__main__":
    main()
