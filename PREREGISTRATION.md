# Pre-registration: two-dimensional Ising model

Written before any simulation was run.

## Questions

1. Simulating the two-dimensional Ising model with the Metropolis algorithm on small lattices, how close do I get to the exactly known critical temperature and critical exponents, and do my error bars cover the difference?
2. How much faster does the Wolff cluster algorithm decorrelate than Metropolis at the critical point?

## Model

Square lattice, L x L spins, each +1 or -1, periodic boundaries in both directions. Energy `E = -J * sum over nearest-neighbour pairs of s_i * s_j`, each pair counted once, `J = 1`, Boltzmann constant 1, no external field. `N = L * L`. Magnetization `M` is the sum of spins. Per site: `e = E / N`, `m = M / N`. Every run starts with all spins +1.

## Exact values used for comparison

- `Tc = 2 / ln(1 + sqrt(2))`, about 2.269185.
- `beta = 1/8`, `gamma = 7/4`, `nu = 1`, so `beta/nu = 0.125`, `gamma/nu = 1.75`, `1/nu = 1`.
- Infinite-lattice magnetization `m0(T) = (1 - sinh(2/T)^(-4))^(1/8)` below Tc, and 0 at and above Tc.
- Infinite-lattice energy per site with `b = 1/T`: `k = 2 sinh(2b) / cosh(2b)^2`, `kp = 2 tanh(2b)^2 - 1`, `u(T) = -(1/tanh(2b)) * (1 + (2/pi) * kp * K(k))`, with `K` the complete elliptic integral of the first kind (modulus k). At Tc, `u = -sqrt(2)`.

## Algorithms

- Metropolis with a checkerboard update. Colour a site by `(i + j) % 2`. One sweep updates all colour-0 sites, then all colour-1 sites. For one colour: `nn` is the sum of the four neighbours, `dE = 2 J s nn`, draw `r = rng.random((L, L))`, flip the sites of that colour where `dE <= 0` or `r < exp(-dE / T)`.
- Wolff cluster update with `p_add = 1 - exp(-2 J / T)`. Pick a site at random, grow a cluster with a stack, add each same-spin neighbour not yet in the cluster with probability `p_add`, flip the whole cluster.
- Random numbers: every run has its own generator `numpy.random.default_rng(SeedSequence([MASTER_SEED, algorithm_id, L, run_index]))`. `algorithm_id` is 0 for Metropolis and 1 for Wolff. `run_index` is the position of the temperature in the sorted grid. The Tc run uses `run_index = 999`.

## Constants (the same as in `config.py`)

```
J = 1.0
L_LIST = [8, 16, 32, 64]
T_COARSE = 2.00 to 2.60 inclusive in steps of 0.05          # 13 values
T_FINE = 2.20 to 2.34 inclusive in steps of 0.01            # 15 values
T grid = sorted union of T_COARSE and T_FINE, rounded to 2 decimals, duplicates removed (25 values)
Every L also gets one run at exactly Tc (the "Tc run").
N_THERM = 50000
N_MEAS = 500000
N_BLOCKS = 50
WINDOW_C = 6.0
N_BOOT = 2000
MASTER_SEED = 42
WOLFF_THERM = 5000
WOLFF_MEAS = 100000
CROSSING_PAIRS = [(8, 16), (16, 32), (32, 64)]
PRIMARY_PAIR = (32, 64)
EXPONENT_SIZES_PRIMARY = [16, 32, 64]
EXPONENT_SIZES_ALL = [8, 16, 32, 64]
SLOPE_T_LOW = 2.26
SLOPE_T_HIGH = 2.28
MAX_WORKERS = 4
```

## Estimators

All estimators and all error bars are computed from block means. The measured series (length n) is split into `N_BLOCKS` equal consecutive blocks. For each block I store the means of `E`, `E^2`, `|M|`, `M^2`, `M^4`. An estimator on a set of blocks is the formula applied to the average of those blocks' means.

- `e_mean = mean(E) / N`
- `absm_mean = mean(|M|) / N`
- `m2_mean = mean(M^2) / N^2`, `m4_mean = mean(M^4) / N^4`
- specific heat per site `C = (mean(E^2) - mean(E)^2) / (N T^2)`
- connected susceptibility `chi_c = (mean(M^2) - mean(|M|)^2) / (N T)`
- susceptibility `chi = mean(M^2) / (N T)`
- Binder cumulant `U = 1 - m4_mean / (3 m2_mean^2)`

Error bars: delete-one-block jackknife over the 50 blocks.

Integrated autocorrelation time: autocovariance by FFT, normalized so `rho(0) = 1`. `tau(W) = 0.5 + sum_{t=1..W} rho(t)`. The window is the smallest `W` with `W >= WINDOW_C * tau(W)`. If no such W exists below `n / 10`, tau is reported as not converged. It is computed for `|M|` and for `E` in every run.

Block quality flag: `block_length / tau_absm < 10` marks the run `blocks_short`. Such runs are still used, and their error bars are probably too small.

Binder crossing for a pair (La, Lb) with La < Lb: on the fine grid only, `D(T) = U_La(T) - U_Lb(T)`. Take the last index i with `D(T_i) <= 0` and `D(T_{i+1}) > 0`. The crossing is the linear interpolation between those two points. If no such index exists, the crossing is NaN.

Bootstrap for anything that combines runs: for every run involved, resample its 50 blocks with replacement (generator `SeedSequence([MASTER_SEED, 7, replicate_index])`), recompute the estimators, recompute the combined quantity. 2000 replicates. I report the 2.5 and 97.5 percentiles as an indicative interval, the standard deviation as the standard error, and the number of NaN replicates.

Finite-size scaling fits are ordinary least squares of `ln(y)` on `ln(L)`:

- `gamma/nu` is the slope for `y = chi` at the Tc runs.
- `beta/nu` is minus the slope for `y = absm_mean` at the Tc runs.
- `1/nu` is the slope for `y = (U(2.26) - U(2.28)) / (2.28 - 2.26)` from the grid runs.

Each is fitted with sizes 16, 32, 64 (primary) and with all four sizes.

Dynamic exponent `z`: slope of `ln(tau_absm)` on `ln(L)` at Tc for Metropolis (in sweeps) and for Wolff. For Wolff, tau is first converted to sweep equivalents: `tau_sweeps = tau_in_updates * mean(cluster size) / N`. All four sizes are used. The error on `z` is the standard error of the least-squares slope. Four points make it a rough estimate.

## Primary and secondary results

Primary result: the critical temperature from the Binder cumulant crossing of L = 32 and L = 64, compared with the exact `Tc`, with its bootstrap standard error and indicative interval, and the difference in standard errors.

Secondary results: `gamma/nu`, `beta/nu`, `1/nu` (each in two fits), and the two dynamic exponents.

## Checks that must pass before any result is used

Unit tests including an exact enumeration of the 4 x 4 lattice at T = 2.0, Tc and 3.0 against Metropolis (and a Wolff check on `e_mean` and `absm_mean`), each within 5 jackknife standard errors.

These choices were fixed before any simulation was run and were not changed afterward.
