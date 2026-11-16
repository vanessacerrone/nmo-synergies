"""Convergence of the DeepCore tabulated response in its fine grid of true cells (App. A.4).

The intercepts that the paper quotes from the tabulated response, on the fine grid of Eq. (A.3), 300 x 180 cells in ln E
and cos theta_z, and on a grid five times finer in both, 1500 x 900 cells: nu_mu disappearance and all channels with
theta23 and both phases fixed at x0 (fig_deepcore_bin_weights and the line of Sec. 4), and all channels with theta23
fitted at x = 2.5e-3 eV^2 and delta_CP = 177 deg in both orderings (fig_full_fit_validation, App. E.2). Output:
results/grid.npz. Run from the repository root: python -m compute.grid (about ten minutes)
"""
import gc

import numpy as np

from compute.deepcore import fixed, theta_fitted
from nmosyn import fit
from nmosyn.experiments.deepcore import FINE_CELLS, DeepCore
from nmosyn.params import DCP_NO, S23SQ, X0, intercept
from nmosyn.paths import RESULTS

FACTOR = 5
X_FITTED = 2.5e-3


def intercepts(fine_cells):
    """b of nu_mu and of all channels with theta23 fixed at x0, and of all channels with theta23 fitted."""
    numu = DeepCore('numu', fine_cells=fine_cells)
    b_numu = intercept(fixed(numu, X0), X0)
    del numu
    gc.collect()
    allch = DeepCore('all', fine_cells=fine_cells)
    b_all = intercept(fixed(allch, X0), X0)
    _, y, _ = theta_fitted(fit.Harmonics(allch), allch.counts(X_FITTED, S23SQ, DCP_NO), X_FITTED, DCP_NO,
                           intervals=((0.5, 0.7),))
    return b_numu, b_all, intercept(y, X_FITTED)


def main():
    cells = np.array([FINE_CELLS, (FACTOR * FINE_CELLS[0], FACTOR * FINE_CELLS[1])])
    b = np.array([intercepts(tuple(int(n) for n in c)) for c in cells])
    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(RESULTS / 'grid.npz', fine_cells=cells, x_fitted=X_FITTED,
                        numu_fixed=b[:, 0], all_fixed=b[:, 1], all_fitted=b[:, 2])


if __name__ == '__main__':
    main()
