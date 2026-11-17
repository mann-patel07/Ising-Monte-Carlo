"""Run the Wolff simulations at Tc for every L. Safe to restart."""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import config
import ising

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results" / "runs"
DATA = ROOT / "data"


def run_path(L: int) -> Path:
    return RUNS / f"wolff_L{L}_TTc.json"


def do_job(L):
    t0 = time.time()
    T = ising.tc_exact()
    rng = ising.make_rng(config.ALGO_WOLFF, L, config.TC_RUN_INDEX)
    out = ising.run_wolff(L, T, config.WOLFF_THERM, config.WOLFF_MEAS, rng)
    res = ising.summarize_series(out["E"], out["M"], L, T)
    res.update({"algorithm": "wolff", "run_index": config.TC_RUN_INDEX, "tag": "Tc",
                "mean_cluster_size": float(np.mean(out["sizes"])),
                "mean_cluster_fraction": float(np.mean(out["sizes"]) / (L * L)),
                "wall_time": time.time() - t0})
    DATA.mkdir(exist_ok=True)
    np.savez_compressed(DATA / f"tc_series_wolff_L{L}.npz", E=out["E"].astype(np.int32),
                        M=out["M"].astype(np.int32), sizes=out["sizes"].astype(np.int32))
    tmp = run_path(L).with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    os.replace(tmp, run_path(L))
    return L, res["wall_time"]


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    todo = sorted([L for L in config.L_LIST if not run_path(L).exists()], reverse=True)
    t0 = time.time()
    workers = min(config.MAX_WORKERS, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(do_job, L) for L in todo]
        for fu in as_completed(futs):
            L, wt = fu.result()
            print(f"Wolff L={L} done in {wt:.0f}s", flush=True)
    with open(ROOT / "RUN_LOG.md", "a") as f:
        f.write(f"- {time.strftime('%F %T')} Wolff runs finished ({len(todo)} new), "
                f"wall time of this invocation {(time.time() - t0) / 60:.1f} min.\n")


if __name__ == "__main__":
    main()
