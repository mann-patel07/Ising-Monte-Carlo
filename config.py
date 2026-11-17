"""All constants for the Ising Monte Carlo project. No magic numbers elsewhere."""
import numpy as np

J = 1.0
L_LIST = [8, 16, 32, 64]
T_COARSE = [round(float(t), 2) for t in np.arange(0, 13) * 0.05 + 2.00]  # 2.00 to 2.60, 13 values
T_FINE = [round(float(t), 2) for t in np.arange(0, 15) * 0.01 + 2.20]    # 2.20 to 2.34, 15 values
# Temperature grid: sorted union of T_COARSE and T_FINE, rounded to 2 decimals, duplicates removed (25 values).
T_GRID = sorted(set(T_COARSE) | set(T_FINE))
# In addition every L gets one run at exactly Tc (the "Tc run").
TC_RUN_INDEX = 999
N_THERM = 50000            # Metropolis sweeps discarded before measuring
N_MEAS = 500000            # Metropolis sweeps measured, one measurement per sweep
N_BLOCKS = 50              # blocks of 10000 sweeps
WINDOW_C = 6.0             # automatic window constant for the integrated autocorrelation time
N_BOOT = 2000
MASTER_SEED = 42
WOLFF_THERM = 5000         # cluster updates discarded
WOLFF_MEAS = 100000        # cluster updates measured
CROSSING_PAIRS = [(8, 16), (16, 32), (32, 64)]
PRIMARY_PAIR = (32, 64)
EXPONENT_SIZES_PRIMARY = [16, 32, 64]
EXPONENT_SIZES_ALL = [8, 16, 32, 64]
SLOPE_T_LOW = 2.26
SLOPE_T_HIGH = 2.28
MAX_WORKERS = 4

# Quantities that are part of the design but are not free parameters
ALGO_METROPOLIS = 0
ALGO_WOLFF = 1
BOOTSTRAP_STREAM = 7
BLOCKS_SHORT_RATIO = 10.0   # block_length / tau below this flags a run
TAU_MAX_FRACTION = 10       # window must stay below n / 10
EXACT_BETA_OVER_NU = 0.125
EXACT_GAMMA_OVER_NU = 1.75
EXACT_ONE_OVER_NU = 1.0
TC_EXACT = 2.0 / float(np.log(1.0 + np.sqrt(2.0)))
