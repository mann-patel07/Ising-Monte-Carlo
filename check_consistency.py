"""Check that the numbers in README.md equal the values in results/ after the same rounding, and that paths exist."""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
README = (ROOT / "README.md").read_text()
RES = ROOT / "results"
failures = []
checked = 0
NUM = r"([+-]?\d+(?:\.\d+)?)"


def first_number_after(pattern):
    m = re.search(re.escape(pattern) + r"[^0-9+\-]*" + NUM, README)
    return float(m.group(1)) if m else None


def check(label, pattern, expected, decimals):
    global checked
    checked += 1
    got = first_number_after(pattern)
    if got is None:
        failures.append(f"{label}: pattern '{pattern}' not found")
        return
    if abs(got - round(float(expected), decimals)) > 1e-9:
        failures.append(f"{label}: README has {got}, results give {round(float(expected), decimals)}")


summ = pd.read_csv(RES / "summary.csv").set_index("key")["value"]
cross = pd.read_csv(RES / "crossings.csv")
expo = pd.read_csv(RES / "exponents.csv")
dyn = pd.read_csv(RES / "dynamics.csv")
s = lambda k: float(summ[k])

# headline
check("primary Tc", "Tc from the Binder crossing of L = 32 and L = 64: **", s("primary_Tc_estimate"), 4)
check("primary se", "bootstrap standard error", s("primary_se"), 4)
m = re.search(r"indicative interval \[([\d.]+), ([\d.]+)\]", README)
checked += 2
if not m:
    failures.append("primary interval not found")
else:
    if abs(float(m.group(1)) - round(s("primary_lo"), 4)) > 1e-9:
        failures.append("primary interval low end")
    if abs(float(m.group(2)) - round(s("primary_hi"), 4)) > 1e-9:
        failures.append("primary interval high end")
check("exact Tc", "Exact Tc:", s("primary_Tc_exact"), 4)
check("difference", "Difference (estimate minus exact):", s("primary_diff"), 4)
m = re.search(r"Difference \(estimate minus exact\): [+-][\d.]+, which is ([+-][\d.]+) standard errors", README)
checked += 1
if not m or abs(float(m.group(1)) - round(s("primary_diff_in_se"), 2)) > 1e-9:
    failures.append("difference in standard errors")
covers = str(summ["primary_interval_covers_exact"]).lower() == "true"
checked += 1
if covers and "does not cover" in README.split("## ")[1] or (not covers and "covers the exact value" in README.split("## ")[1] and "does not cover" not in README.split("## ")[1]):
    failures.append("coverage sentence does not match the numbers")

# all crossing pairs
for _, c in cross.iterrows():
    row = re.search(r"\| " + re.escape(c["pair"].replace(",", " and ")) + r" \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", README)
    checked += 4
    if not row:
        failures.append(f"crossing row missing: {c['pair']}")
        continue
    for val, key, d in [(row.group(1), "Tc_estimate", 4), (row.group(2), "se", 4), (row.group(3), "lo", 4), (row.group(4), "hi", 4)]:
        if abs(float(val) - round(float(c[key]), d)) > 1e-9:
            failures.append(f"crossing {c['pair']} {key}: README {val}, results {round(float(c[key]), d)}")

# exponent table
for _, e in expo.iterrows():
    row = re.search(r"\| " + re.escape(e["quantity"]) + r" \| " + re.escape(e["sizes"].replace(" ", ", ")) +
                    r" \| ([+-]?[\d.]+) \| ([\d.]+) \| \[([\d.]+), ([\d.]+)\] \| ([\d.]+) \| ([+-][\d.]+) \|", README)
    checked += 6
    if not row:
        failures.append(f"exponent row missing: {e['quantity']} sizes {e['sizes']}")
        continue
    want = [round(e["estimate"], 3), round(e["se"], 3), round(e["lo"], 3), round(e["hi"], 3), round(e["exact"], 3), round(e["diff_in_se"], 2)]
    for i, (g_, w_) in enumerate(zip(row.groups(), want)):
        if abs(float(g_) - w_) > 1e-9:
            failures.append(f"exponent {e['quantity']} sizes {e['sizes']} column {i}: README {g_}, results {w_}")

# dynamic exponents
check("z metropolis", "Metropolis z =", s("z_metropolis"), 2)
check("z metropolis se", "Metropolis z = " + f"{s('z_metropolis'):.2f}" + " +/-", s("z_metropolis_se"), 2)
check("z wolff", "Wolff z =", s("z_wolff"), 2)
check("z wolff se", "Wolff z = " + f"{s('z_wolff'):.2f}" + " +/-", s("z_wolff_se"), 2)
check("speedup L=64", "At L = 64 the Metropolis autocorrelation time is", s("tau_metropolis_L64"), 1)
check("wolff tau L=64", "sweeps and the Wolff time is", s("tau_wolff_sweeps_L64"), 2)
check("speedup ratio", "a ratio of", s("speedup_L64"), 1)
check("blocks_short count", "Runs flagged blocks_short:", s("n_blocks_short_runs"), 0)
check("metropolis run count", "Metropolis runs in total:", s("n_metropolis_runs"), 0)

# paths
paths = set(re.findall(r"`([\w./-]+\.(?:py|md|csv|json|png|ipynb|txt|npz))`", README))
paths |= set(re.findall(r"\]\(([\w./-]+\.(?:png|md|csv|json))\)", README))
paths |= {p for p in re.findall(r"`([\w./-]+/)`", README)}
if "`LICENSE`" in README:
    paths.add("LICENSE")
for p in sorted(paths):
    checked += 1
    if "{" in p or "*" in p:
        continue
    if not (ROOT / p).exists():
        failures.append(f"path named in README does not exist: {p}")

if failures:
    print("CONSISTENCY CHECK FAILED")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print(f"consistency check passed: {checked} items checked, README matches results/")
