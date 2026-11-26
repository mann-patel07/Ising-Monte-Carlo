"""Analysis of the seed replicates. Added after the first version (see UPGRADE_PLAN.md).

Replicate 0 is the first version's run. Replicates 1 to 3 are in results/replicates/. For each pair of sizes the
Binder crossing is computed for every replicate with the same rule as the first version (no bootstrap), and the
spread over replicates is compared with the first version's bootstrap standard error.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import analyze
import config
import ising

ROOT = Path(__file__).resolve().parent
REPLICATES = [0, 1, 2, 3]
TC = config.TC_EXACT
OUT_CSV = ROOT / "results" / "crossing_replicates.csv"
OUT_SUMMARY = ROOT / "results" / "crossing_replicates_summary.csv"


def load_blocks(rep):
    """Block means for replicate rep: {(L, T): array (blocks, 5)} for the fine grid."""
    blocks = {}
    for L in config.L_LIST:
        for T in config.T_FINE:
            if rep == 0:
                p = ROOT / "results" / "runs" / f"metropolis_L{L}_T{T:.2f}.json"
            else:
                p = ROOT / "results" / "replicates" / f"metropolis_L{L}_T{T:.2f}_r{rep}.json"
            blocks[(L, T)] = np.array(json.loads(p.read_text())["block_means"], dtype=np.float64)
    return blocks


def crossing(means_by_LT, pair):
    """Binder crossing of a pair from mean block means, computed as in analyze.crossing_analysis."""
    La, Lb = pair
    Ua = [analyze.u_from_means(means_by_LT[(La, T)], La, T) for T in config.T_FINE]
    Ub = [analyze.u_from_means(means_by_LT[(Lb, T)], Lb, T) for T in config.T_FINE]
    return ising.binder_crossing(config.T_FINE, Ua, Ub)


def verdict(crossings, pooled, tc=TC):
    """(sem, diff_in_sem, same_side, verdict) for the crossings of the replicates.

    sem = sample SD / sqrt(number of replicates). The verdict is "finite-size effect" if the pooled crossing differs
    from tc by more than 3 sem and every replicate is on the same side of tc, and "consistent with noise" otherwise.
    With sem == 0 the difference in sem is nan and the verdict is "consistent with noise".
    """
    c = np.asarray(crossings, dtype=float)
    sd = float(np.std(c, ddof=1))
    sem = sd / math.sqrt(len(c))
    same_side = bool(np.all(c > tc) or np.all(c < tc))
    if sem == 0:
        return sem, float("nan"), same_side, "consistent with noise"
    diff_in_sem = (float(c.mean()) - tc) / sem
    real = abs(pooled - tc) > 3 * sem and same_side
    return sem, diff_in_sem, same_side, "finite-size effect" if real else "consistent with noise"


def plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summ = pd.read_csv(OUT_SUMMARY)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (_, r) in enumerate(summ.iterrows()):
        pts = df[df.pair == r.pair].crossing
        ax.plot(np.full(len(pts), i) - 0.08, pts, "o", color="gray", label="one replicate" if i == 0 else None)
        ax.errorbar([i + 0.08], [r["mean"]], yerr=[r["sem"]], fmt="s", color="tab:blue", capsize=4,
                    label="mean and standard error of the mean" if i == 0 else None)
    ax.axhline(TC, ls="--", color="black", label="exact Tc")
    ax.set_xticks(range(len(summ)))
    ax.set_xticklabels([f"L = {p}" for p in summ.pair])
    ax.set_xlabel("Pair of sizes")
    ax.set_ylabel("Binder crossing temperature (J / k_B)")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    t0 = time.time()
    blocks = {r: load_blocks(r) for r in REPLICATES}
    first = pd.read_csv(ROOT / "results" / "crossings.csv").set_index("pair")
    rows, summ = [], []
    for pair in config.CROSSING_PAIRS:
        name = f"{pair[0]},{pair[1]}"
        cs = []
        for r in REPLICATES:
            means = {k: v.mean(axis=0) for k, v in blocks[r].items()}
            c = crossing(means, pair)
            cs.append(c)
            rows.append({"pair": name, "replicate": r, "crossing": c})
        pooled_means = {k: np.mean([blocks[r][k] for r in REPLICATES], axis=0).mean(axis=0) for k in blocks[0]}
        pooled = crossing(pooled_means, pair)
        sem, dsem, same, verd = verdict(cs, pooled)
        sd = float(np.std(cs, ddof=1))
        mean = float(np.mean(cs))
        bse = float(first.loc[name, "se"])
        summ.append({"pair": name, "mean": mean, "sd": sd, "sem": sem, "diff": mean - TC, "diff_in_sem": dsem,
                     "same_side": same, "pooled_crossing": pooled, "bootstrap_se_first_version": bse,
                     "ratio": bse / sd if sd else float("nan"), "verdict": verd})
    df = pd.DataFrame(rows, columns=["pair", "replicate", "crossing"])
    df.to_csv(OUT_CSV, index=False)
    pd.DataFrame(summ).to_csv(OUT_SUMMARY, index=False)
    plot(df, ROOT / "Images" / "crossing_replicates.png")
    print(pd.DataFrame(summ).to_string(index=False))
    print("run time %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
