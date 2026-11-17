"""Analysis of all runs. Reads results/runs, writes the CSV files in results/."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
import ising

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results" / "runs"
OUT = ROOT / "results"
TC = ising.tc_exact()


def load_runs():
    runs = {}
    for p in sorted(RUNS.glob("*.json")):
        r = json.loads(p.read_text())
        r["block_means"] = np.array(r["block_means"], dtype=np.float64)
        for k in ("tau_absm", "tau_E"):
            if r[k] is None:
                r[k] = float("nan")
        runs[(r["algorithm"], r["L"], r["tag"])] = r
    return runs


def row_for(r):
    row = {"algorithm": r["algorithm"], "L": r["L"], "T": r["T"], "tag": r["tag"],
           "run_index": r["run_index"], "n_meas": r["n_meas"], "block_length": r["block_length"]}
    for k in ising.ESTIMATOR_KEYS:
        row[k] = r["estimators"][k]
        row[k + "_err"] = r["errors"][k]
    row.update({"tau_absm": r["tau_absm"], "tau_E": r["tau_E"],
                "tau_absm_converged": r["tau_absm_converged"], "tau_E_converged": r["tau_E_converged"],
                "blocks_short": r["blocks_short"],
                "block_length_over_tau_absm": r["block_length"] / r["tau_absm"]
                if np.isfinite(r["tau_absm"]) else float("nan"),
                "acceptance": r.get("acceptance", float("nan")),
                "mean_cluster_size": r.get("mean_cluster_size", float("nan")),
                "wall_time_s": r["wall_time"]})
    return row


def u_from_means(m, L, T):
    return float(ising.estimators(m, L * L, T)["U"])


def crossing_analysis(runs, pair):
    La, Lb = pair
    Tf = config.T_FINE
    keys = [("metropolis", L, f"{T:.2f}") for L in pair for T in Tf]
    sets = [runs[k]["block_means"] for k in keys]
    n = len(Tf)

    def combine(means):
        Ua = [u_from_means(means[i], La, Tf[i]) for i in range(n)]
        Ub = [u_from_means(means[n + i], Lb, Tf[i]) for i in range(n)]
        return ising.binder_crossing(Tf, Ua, Ub)

    point = combine([s.mean(axis=0) for s in sets])
    bs = ising.bootstrap(sets, combine)
    diff = point - TC
    return {"pair": f"{La},{Lb}", "Tc_estimate": point, "se": bs["se"], "lo": bs["lo"],
            "hi": bs["hi"], "n_nan": bs["n_nan"], "Tc_exact": TC, "diff": diff,
            "diff_in_se": diff / bs["se"] if bs["se"] and np.isfinite(bs["se"]) else float("nan"),
            "interval_covers_exact": bool(bs["lo"] <= TC <= bs["hi"])}


def exponent_analysis(runs, name, sizes, exact):
    if name == "gamma/nu":
        keys = [("metropolis", L, "Tc") for L in sizes]

        def combine(means):
            y = [ising.estimators(means[i], L * L, TC)["chi"] for i, L in enumerate(sizes)]
            return ising.loglog_slope(sizes, y)
    elif name == "beta/nu":
        keys = [("metropolis", L, "Tc") for L in sizes]

        def combine(means):
            y = [ising.estimators(means[i], L * L, TC)["absm_mean"] for i, L in enumerate(sizes)]
            return -ising.loglog_slope(sizes, y)
    else:
        keys = []
        for L in sizes:
            keys.append(("metropolis", L, f"{config.SLOPE_T_LOW:.2f}"))
            keys.append(("metropolis", L, f"{config.SLOPE_T_HIGH:.2f}"))

        def combine(means):
            y = []
            for i, L in enumerate(sizes):
                ulo = u_from_means(means[2 * i], L, config.SLOPE_T_LOW)
                uhi = u_from_means(means[2 * i + 1], L, config.SLOPE_T_HIGH)
                y.append((ulo - uhi) / (config.SLOPE_T_HIGH - config.SLOPE_T_LOW))
            return ising.loglog_slope(sizes, y)
    sets = [runs[k]["block_means"] for k in keys]
    point = combine([s.mean(axis=0) for s in sets])
    bs = ising.bootstrap(sets, combine)
    return {"quantity": name, "sizes": " ".join(str(s) for s in sizes), "estimate": point,
            "se": bs["se"], "lo": bs["lo"], "hi": bs["hi"], "n_nan": bs["n_nan"], "exact": exact,
            "diff": point - exact,
            "diff_in_se": (point - exact) / bs["se"] if np.isfinite(bs["se"]) and bs["se"] > 0 else float("nan")}


def exact_check_4x4():
    """Repeat the 4 x 4 exact-enumeration comparison of tests 3 and 4 and store the pulls."""
    rows = []
    for T in [2.0, TC, 3.0]:
        exact = ising.exact_enumeration(4, T)
        sims = [("metropolis", ising.run_metropolis(4, T, 20000, 400000, ising.make_rng(config.ALGO_METROPOLIS, 4, 0)),
                 ["e_mean", "absm_mean", "C", "chi_c", "U"]),
                ("wolff", ising.run_wolff(4, T, 2000, 200000, ising.make_rng(config.ALGO_WOLFF, 4, 0)),
                 ["e_mean", "absm_mean"])]
        for alg, out, keys in sims:
            est, err = ising.estimators_with_errors(ising.block_means(out["E"], out["M"]), 16, T)
            for k in keys:
                rows.append({"algorithm": alg, "T": T, "quantity": k, "simulated": est[k],
                             "jackknife_se": err[k], "exact": exact[k], "pull": (est[k] - exact[k]) / err[k]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exact_check_4x4.csv", index=False)
    assert df["pull"].abs().max() < 5, "4 x 4 exact check disagrees by more than 5 standard errors"


def main():
    runs = load_runs()
    # ---- observables
    rows = [row_for(r) for _, r in sorted(runs.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[1]["T"]))]
    obs = pd.DataFrame(rows)
    obs.to_csv(OUT / "observables.csv", index=False)

    # ---- sanity assertions
    met_grid = obs[(obs.algorithm == "metropolis") & (obs.tag != "Tc")]
    for L in config.L_LIST:
        for T in config.T_GRID:
            n = int(((met_grid.L == L) & (np.isclose(met_grid["T"], T))).sum())
            assert n == 1, f"design cell L={L} T={T} has {n} rows"
    assert len(met_grid) == len(config.L_LIST) * len(config.T_GRID)
    for L in config.L_LIST:
        assert ((obs.algorithm == "metropolis") & (obs.tag == "Tc") & (obs.L == L)).sum() == 1
        assert ((obs.algorithm == "wolff") & (obs.tag == "Tc") & (obs.L == L)).sum() == 1
    assert not met_grid[ising.ESTIMATOR_KEYS].isna().any().any(), "NaN estimator in grid runs"
    assert met_grid["U"].between(-0.1, 0.7).all(), "U outside (-0.1, 0.7)"

    # ---- crossings
    cross = pd.DataFrame([crossing_analysis(runs, p) for p in config.CROSSING_PAIRS])
    cross.to_csv(OUT / "crossings.csv", index=False)

    # ---- exponents
    exact = {"gamma/nu": config.EXACT_GAMMA_OVER_NU, "beta/nu": config.EXACT_BETA_OVER_NU,
             "1/nu": config.EXACT_ONE_OVER_NU}
    erows = []
    for name in ["gamma/nu", "beta/nu", "1/nu"]:
        for fit, sizes in [("primary", config.EXPONENT_SIZES_PRIMARY), ("all", config.EXPONENT_SIZES_ALL)]:
            d = exponent_analysis(runs, name, sizes, exact[name])
            d["fit"] = fit
            erows.append(d)
    expo = pd.DataFrame(erows)[["quantity", "fit", "sizes", "estimate", "se", "lo", "hi", "n_nan",
                                "exact", "diff", "diff_in_se"]]
    expo.to_csv(OUT / "exponents.csv", index=False)

    # ---- dynamics
    drows = []
    taus = {"metropolis": [], "wolff": []}
    for L in config.L_LIST:
        m = runs[("metropolis", L, "Tc")]
        w = runs[("wolff", L, "Tc")]
        tau_sw = w["tau_absm"] * w["mean_cluster_size"] / (L * L)
        drows.append({"row_type": "per_L", "algorithm": "metropolis", "L": L,
                      "tau_absm": m["tau_absm"], "tau_E": m["tau_E"], "mean_cluster_fraction": float("nan"),
                      "tau_absm_sweeps": m["tau_absm"], "z": float("nan"), "z_se": float("nan")})
        drows.append({"row_type": "per_L", "algorithm": "wolff", "L": L,
                      "tau_absm": w["tau_absm"], "tau_E": w["tau_E"],
                      "mean_cluster_fraction": w["mean_cluster_fraction"],
                      "tau_absm_sweeps": tau_sw, "z": float("nan"), "z_se": float("nan")})
        taus["metropolis"].append(m["tau_absm"])
        taus["wolff"].append(tau_sw)
    for alg in ["metropolis", "wolff"]:
        z, se = ising.loglog_slope_se(config.L_LIST, taus[alg])
        drows.append({"row_type": "z_fit", "algorithm": alg, "L": float("nan"), "tau_absm": float("nan"),
                      "tau_E": float("nan"), "mean_cluster_fraction": float("nan"),
                      "tau_absm_sweeps": float("nan"), "z": z, "z_se": se})
    dyn = pd.DataFrame(drows)
    dyn.to_csv(OUT / "dynamics.csv", index=False)

    # ---- exact comparison
    erows = []
    for T in config.T_GRID:
        r = runs[("metropolis", 64, f"{T:.2f}")]
        erows.append({"T": T, "L": 64, "e_mean": r["estimators"]["e_mean"], "e_mean_err": r["errors"]["e_mean"],
                      "u_exact": ising.u_exact(T), "absm_mean": r["estimators"]["absm_mean"],
                      "absm_mean_err": r["errors"]["absm_mean"], "m0_exact": ising.m0_exact(T)})
    pd.DataFrame(erows).to_csv(OUT / "exact_comparison.csv", index=False)

    # ---- summary
    s = []
    prim = cross[cross.pair == f"{config.PRIMARY_PAIR[0]},{config.PRIMARY_PAIR[1]}"].iloc[0]
    for k in ["Tc_estimate", "se", "lo", "hi", "n_nan", "Tc_exact", "diff", "diff_in_se"]:
        s.append((f"primary_{k}", prim[k]))
    s.append(("primary_interval_covers_exact", bool(prim["interval_covers_exact"])))
    for _, c in cross.iterrows():
        s.append((f"crossing_{c['pair']}_estimate", c["Tc_estimate"]))
        s.append((f"crossing_{c['pair']}_se", c["se"]))
    for _, e in expo.iterrows():
        tag = f"{e['quantity']}_{e['fit']}"
        for k in ["estimate", "se", "lo", "hi", "exact", "diff_in_se"]:
            s.append((f"exponent_{tag}_{k}", e[k]))
    for _, d in dyn[dyn.row_type == "z_fit"].iterrows():
        s.append((f"z_{d['algorithm']}", d["z"]))
        s.append((f"z_{d['algorithm']}_se", d["z_se"]))
    for L in config.L_LIST:
        mm = dyn[(dyn.row_type == "per_L") & (dyn.L == L) & (dyn.algorithm == "metropolis")].iloc[0]
        ww = dyn[(dyn.row_type == "per_L") & (dyn.L == L) & (dyn.algorithm == "wolff")].iloc[0]
        s.append((f"tau_metropolis_L{L}", mm["tau_absm"]))
        s.append((f"tau_wolff_sweeps_L{L}", ww["tau_absm_sweeps"]))
        s.append((f"speedup_L{L}", mm["tau_absm"] / ww["tau_absm_sweeps"]))
    s.append(("n_blocks_short_runs", int(obs[(obs.algorithm == "metropolis")].blocks_short.sum())))
    s.append(("n_metropolis_runs", int((obs.algorithm == "metropolis").sum())))
    pd.DataFrame(s, columns=["key", "value"]).to_csv(OUT / "summary.csv", index=False)
    exact_check_4x4()
    print("analysis done")


if __name__ == "__main__":
    main()
