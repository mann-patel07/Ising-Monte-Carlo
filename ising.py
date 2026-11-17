"""Library for the 2D Ising model: samplers, estimators, error bars, exact formulas."""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.special import ellipkm1

import config

# Columns of a block-means array
COL_E, COL_E2, COL_ABSM, COL_M2, COL_M4 = 0, 1, 2, 3, 4
ESTIMATOR_KEYS = ["e_mean", "absm_mean", "m2_mean", "m4_mean", "C", "chi_c", "chi", "U"]


# ----------------------------------------------------------------------------
# Exact results
# ----------------------------------------------------------------------------
def tc_exact() -> float:
    """Exact critical temperature of the square-lattice Ising model (J = 1)."""
    return 2.0 / float(np.log(1.0 + np.sqrt(2.0)))


def m0_exact(T: float) -> float:
    """Infinite-lattice spontaneous magnetization per site (Yang 1952)."""
    if T >= tc_exact():
        return 0.0
    return float((1.0 - np.sinh(2.0 / T) ** (-4.0)) ** 0.125)


def u_exact(T: float) -> float:
    """Infinite-lattice energy per site (Onsager 1944)."""
    if T == tc_exact():
        return -float(np.sqrt(2.0))
    b = 1.0 / T
    kp = 2.0 * np.tanh(2.0 * b) ** 2 - 1.0
    if kp == 0.0:
        return -float(np.sqrt(2.0))
    # 1 - k**2 equals kp**2 exactly, so ellipkm1(kp**2) is the same K(k) without rounding error near Tc
    return float(-(1.0 / np.tanh(2.0 * b)) * (1.0 + (2.0 / np.pi) * kp * ellipkm1(kp ** 2)))


# ----------------------------------------------------------------------------
# Lattice basics
# ----------------------------------------------------------------------------
def make_rng(algorithm_id: int, L: int, run_index: int) -> np.random.Generator:
    """Generator for one run, seeded from the master seed and the run identity."""
    seq = np.random.SeedSequence([config.MASTER_SEED, algorithm_id, L, run_index])
    return np.random.default_rng(seq)


def energy(s: np.ndarray, J: float = config.J) -> float:
    """Total energy: -J * sum of s_i s_j over nearest-neighbour pairs, each once."""
    bonds = s * (np.roll(s, -1, axis=1) + np.roll(s, -1, axis=0))
    return float(-J * np.sum(bonds, dtype=np.int64))


def magnetization(s: np.ndarray) -> float:
    """Total magnetization: sum of spins."""
    return float(np.sum(s, dtype=np.int64))


def colour_masks(L: int) -> List[np.ndarray]:
    """Boolean masks of the two checkerboard colours, colour = (i + j) % 2."""
    ii, jj = np.indices((L, L))
    c = (ii + jj) % 2
    return [c == 0, c == 1]


def neighbour_sum(s: np.ndarray) -> np.ndarray:
    """Sum of the four nearest neighbours of every site (periodic)."""
    return (np.roll(s, 1, axis=0) + np.roll(s, -1, axis=0)
            + np.roll(s, 1, axis=1) + np.roll(s, -1, axis=1))


def metropolis_half_step(s: np.ndarray, T: float, mask: np.ndarray,
                         rng: np.random.Generator, J: float = config.J) -> int:
    """Update all sites of one colour in place. Returns the number of flipped spins."""
    nn = neighbour_sum(s)
    dE = 2.0 * J * s * nn
    r = rng.random(s.shape)
    accept = (dE <= 0) | (r < np.exp(-dE / T))
    flip = accept & mask
    s[flip] = -s[flip]
    return int(np.count_nonzero(flip))


def metropolis_sweep(s: np.ndarray, T: float, rng: np.random.Generator,
                     masks: Sequence[np.ndarray], J: float = config.J) -> int:
    """One full checkerboard sweep. Returns the number of flipped spins."""
    n = 0
    for mask in masks:
        n += metropolis_half_step(s, T, mask, rng, J)
    return n


def run_metropolis(L: int, T: float, n_therm: int, n_meas: int,
                   rng: np.random.Generator, J: float = config.J) -> Dict[str, object]:
    """Metropolis run from all spins up. Returns int32 series E, M and the acceptance rate."""
    s = np.ones((L, L), dtype=np.int8)
    masks = colour_masks(L)
    for _ in range(n_therm):
        metropolis_sweep(s, T, rng, masks, J)
    E = np.empty(n_meas, dtype=np.int32)
    M = np.empty(n_meas, dtype=np.int32)
    flips = 0
    for t in range(n_meas):
        flips += metropolis_sweep(s, T, rng, masks, J)
        E[t] = energy(s, J)
        M[t] = magnetization(s)
    acceptance = flips / (n_meas * L * L)
    return {"E": E, "M": M, "acceptance": acceptance, "spins": s}


class _UniformBuffer:
    """Hands out uniform numbers from a generator in chunks (fast scalar access)."""

    def __init__(self, rng: np.random.Generator, size: int = 1 << 16) -> None:
        self.rng = rng
        self.size = size
        self.buf = rng.random(size).tolist()
        self.pos = 0

    def refill(self) -> None:
        self.buf = self.rng.random(self.size).tolist()
        self.pos = 0


def run_wolff(L: int, T: float, n_therm: int, n_meas: int,
              rng: np.random.Generator, J: float = config.J) -> Dict[str, object]:
    """Wolff cluster run from all spins up. Returns int32 series E, M and cluster sizes."""
    N = L * L
    p_add = 1.0 - float(np.exp(-2.0 * J / T))
    nbrs = []
    for i in range(L):
        for j in range(L):
            nbrs.append((((i - 1) % L) * L + j, ((i + 1) % L) * L + j,
                         i * L + (j - 1) % L, i * L + (j + 1) % L))
    s = [1] * N
    Ecur = -2 * N * J
    Mcur = N
    ub = _UniformBuffer(rng)
    total = n_therm + n_meas
    E = np.empty(n_meas, dtype=np.int32)
    M = np.empty(n_meas, dtype=np.int32)
    sizes = np.empty(n_meas, dtype=np.int32)
    for step in range(total):
        if ub.pos + 1 >= ub.size:
            ub.refill()
        seed = int(ub.buf[ub.pos] * N)
        ub.pos += 1
        spin = s[seed]
        inc = bytearray(N)
        inc[seed] = 1
        stack = [seed]
        cluster = [seed]
        while stack:
            site = stack.pop()
            for nb in nbrs[site]:
                if s[nb] == spin and not inc[nb]:
                    if ub.pos >= ub.size:
                        ub.refill()
                    u = ub.buf[ub.pos]
                    ub.pos += 1
                    if u < p_add:
                        inc[nb] = 1
                        stack.append(nb)
                        cluster.append(nb)
        # energy change from boundary bonds, computed before the flip
        boundary = 0
        for site in cluster:
            for nb in nbrs[site]:
                if not inc[nb]:
                    boundary += spin * s[nb]
        Ecur += 2 * J * boundary
        for site in cluster:
            s[site] = -spin
        Mcur += -2 * spin * len(cluster)
        if step >= n_therm:
            k = step - n_therm
            E[k] = int(round(Ecur))
            M[k] = Mcur
            sizes[k] = len(cluster)
    return {"E": E, "M": M, "sizes": sizes, "spins": np.array(s, dtype=np.int8).reshape(L, L)}


# ----------------------------------------------------------------------------
# Block means, estimators, jackknife
# ----------------------------------------------------------------------------
def block_means(E: np.ndarray, M: np.ndarray, n_blocks: int = config.N_BLOCKS) -> np.ndarray:
    """Means of E, E^2, |M|, M^2, M^4 in equal consecutive blocks, shape (n_blocks, 5)."""
    n = len(E)
    bl = n // n_blocks
    e = np.asarray(E[: bl * n_blocks], dtype=np.float64).reshape(n_blocks, bl)
    m = np.asarray(M[: bl * n_blocks], dtype=np.float64).reshape(n_blocks, bl)
    out = np.empty((n_blocks, 5))
    out[:, COL_E] = e.mean(axis=1)
    out[:, COL_E2] = (e ** 2).mean(axis=1)
    out[:, COL_ABSM] = np.abs(m).mean(axis=1)
    out[:, COL_M2] = (m ** 2).mean(axis=1)
    out[:, COL_M4] = (m ** 4).mean(axis=1)
    return out


def estimators(means: np.ndarray, N: int, T: float) -> Dict[str, np.ndarray]:
    """All estimators from averaged block means (last axis has length 5)."""
    means = np.asarray(means, dtype=np.float64)
    E1 = means[..., COL_E]
    E2 = means[..., COL_E2]
    A = means[..., COL_ABSM]
    M2 = means[..., COL_M2]
    M4 = means[..., COL_M4]
    m2 = M2 / N ** 2
    m4 = M4 / N ** 4
    with np.errstate(divide="ignore", invalid="ignore"):
        U = 1.0 - m4 / (3.0 * m2 ** 2)
    return {
        "e_mean": E1 / N,
        "absm_mean": A / N,
        "m2_mean": m2,
        "m4_mean": m4,
        "C": (E2 - E1 ** 2) / (N * T ** 2),
        "chi_c": (M2 - A ** 2) / (N * T),
        "chi": M2 / (N * T),
        "U": U,
    }


def jackknife_se(blocks: np.ndarray, stat: Callable[[np.ndarray], np.ndarray]) -> float:
    """Delete-one-block jackknife standard error of stat(average of blocks).

    blocks has shape (B, k). stat maps an array of averaged means (..., k) to values (...).
    """
    blocks = np.asarray(blocks, dtype=np.float64)
    B = blocks.shape[0]
    loo = (blocks.sum(axis=0) - blocks) / (B - 1)
    vals = np.asarray(stat(loo), dtype=np.float64)
    return float(np.sqrt((B - 1) / B * np.sum((vals - vals.mean()) ** 2)))


def estimators_with_errors(blocks: np.ndarray, N: int, T: float
                           ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Estimators from all blocks and their jackknife standard errors."""
    est = {k: float(v) for k, v in estimators(blocks.mean(axis=0), N, T).items()}
    B = blocks.shape[0]
    loo = (blocks.sum(axis=0) - blocks) / (B - 1)
    loo_est = estimators(loo, N, T)
    err = {}
    for k, v in loo_est.items():
        err[k] = float(np.sqrt((B - 1) / B * np.sum((v - v.mean()) ** 2)))
    return est, err


# ----------------------------------------------------------------------------
# Autocorrelation
# ----------------------------------------------------------------------------
def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Normalized autocorrelation function by FFT (rho(0) = 1)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    x = x - x.mean()
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, nfft)
    acov = np.fft.irfft(f * np.conj(f), nfft)[:n]
    if acov[0] == 0:
        return np.ones(1)
    return acov / acov[0]


def integrated_autocorr_time(x: np.ndarray, c: float = config.WINDOW_C
                             ) -> Tuple[float, int, bool]:
    """Integrated autocorrelation time with the automatic window W >= c * tau(W).

    Returns (tau, W, converged). If no window below n/10 satisfies the rule,
    returns (nan, -1, False).
    """
    rho = autocorrelation(x)
    n = len(x)
    wmax = min(len(rho) - 1, n // config.TAU_MAX_FRACTION)
    if wmax < 1:
        return float("nan"), -1, False
    tau_w = 0.5 + np.cumsum(rho[1: wmax + 1])
    W = np.arange(1, wmax + 1)
    ok = np.nonzero(W >= c * tau_w)[0]
    if len(ok) == 0:
        return float("nan"), -1, False
    i = int(ok[0])
    return float(tau_w[i]), int(W[i]), True


# ----------------------------------------------------------------------------
# Crossing, fits, bootstrap
# ----------------------------------------------------------------------------
def binder_crossing(T: Sequence[float], Ua: Sequence[float], Ub: Sequence[float]) -> float:
    """Temperature where U_a - U_b changes from <= 0 to > 0 (last such index), linear interpolation."""
    T = np.asarray(T, dtype=np.float64)
    D = np.asarray(Ua, dtype=np.float64) - np.asarray(Ub, dtype=np.float64)
    idx = [i for i in range(len(T) - 1) if D[i] <= 0 and D[i + 1] > 0]
    if not idx:
        return float("nan")
    i = idx[-1]
    return float(T[i] + (0.0 - D[i]) * (T[i + 1] - T[i]) / (D[i + 1] - D[i]))


def loglog_slope(L: Sequence[float], y: Sequence[float]) -> float:
    """Ordinary least squares slope of ln(y) on ln(L)."""
    x = np.log(np.asarray(L, dtype=np.float64))
    z = np.log(np.asarray(y, dtype=np.float64))
    return float(np.polyfit(x, z, 1)[0])


def loglog_slope_se(L: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """Slope of ln(y) on ln(L) and its least-squares standard error."""
    x = np.log(np.asarray(L, dtype=np.float64))
    z = np.log(np.asarray(y, dtype=np.float64))
    n = len(x)
    slope, icpt = np.polyfit(x, z, 1)
    resid = z - (slope * x + icpt)
    s2 = np.sum(resid ** 2) / (n - 2)
    se = float(np.sqrt(s2 / np.sum((x - x.mean()) ** 2)))
    return float(slope), se


def bootstrap(block_sets: Sequence[np.ndarray],
              combine: Callable[[List[np.ndarray]], float],
              n_boot: int = config.N_BOOT) -> Dict[str, float]:
    """Block bootstrap over several runs.

    block_sets: list of (B, 5) block-mean arrays, one per run involved.
    combine: maps the list of resampled averaged means (one (5,) vector per run) to a number.
    """
    vals = np.empty(n_boot)
    for b in range(n_boot):
        rng = np.random.default_rng(np.random.SeedSequence([config.MASTER_SEED,
                                                            config.BOOTSTRAP_STREAM, b]))
        means = []
        for blocks in block_sets:
            B = blocks.shape[0]
            idx = rng.integers(0, B, size=B)
            means.append(blocks[idx].mean(axis=0))
        with np.errstate(all="ignore"):
            vals[b] = combine(means)
    good = vals[np.isfinite(vals)]
    n_nan = int(n_boot - len(good))
    if len(good) < 2:
        return {"se": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_nan": n_nan}
    return {"se": float(np.std(good, ddof=1)),
            "lo": float(np.percentile(good, 2.5)),
            "hi": float(np.percentile(good, 97.5)),
            "n_nan": n_nan}


# ----------------------------------------------------------------------------
# Exact enumeration (used by tests)
# ----------------------------------------------------------------------------
def exact_enumeration(L: int, T: float, J: float = config.J) -> Dict[str, float]:
    """Exact per-site estimators on a small L x L lattice by summing over all 2^(L*L) states."""
    N = L * L
    states = np.arange(2 ** N, dtype=np.int64)
    bits = ((states[:, None] >> np.arange(N)) & 1).astype(np.int8) * 2 - 1
    grid = bits.reshape(-1, L, L)
    bonds = grid * (np.roll(grid, -1, axis=2) + np.roll(grid, -1, axis=1))
    E = -J * bonds.sum(axis=(1, 2)).astype(np.float64)
    M = grid.sum(axis=(1, 2)).astype(np.float64)
    w = np.exp(-(E - E.min()) / T)
    w /= w.sum()
    avg = lambda x: float(np.sum(w * x))
    means = np.array([[avg(E), avg(E ** 2), avg(np.abs(M)), avg(M ** 2), avg(M ** 4)]])
    est = estimators(means, N, T)
    return {k: float(v[0]) for k, v in est.items()}


# ----------------------------------------------------------------------------
# Per-run summary
# ----------------------------------------------------------------------------
def summarize_series(E: np.ndarray, M: np.ndarray, L: int, T: float,
                     n_blocks: int = config.N_BLOCKS) -> Dict[str, object]:
    """Estimators, jackknife errors, block means, autocorrelation times and the block quality flag."""
    N = L * L
    blocks = block_means(E, M, n_blocks)
    est, err = estimators_with_errors(blocks, N, T)
    tau_m, w_m, ok_m = integrated_autocorr_time(np.abs(np.asarray(M, dtype=np.float64)))
    tau_e, w_e, ok_e = integrated_autocorr_time(np.asarray(E, dtype=np.float64))
    block_length = len(E) // n_blocks
    short = (not ok_m) or (block_length / tau_m < config.BLOCKS_SHORT_RATIO)
    return {
        "L": L, "T": T, "N": N, "n_meas": int(len(E)), "n_blocks": n_blocks,
        "block_length": int(block_length),
        "estimators": est, "errors": err,
        "block_means": blocks.tolist(),
        "tau_absm": tau_m if ok_m else None, "tau_absm_window": w_m, "tau_absm_converged": ok_m,
        "tau_E": tau_e if ok_e else None, "tau_E_window": w_e, "tau_E_converged": ok_e,
        "blocks_short": bool(short),
    }
