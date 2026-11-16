"""Direction resolution of IceCube DeepCore (Sec. 4): b with the zenith width of the Gaussian response scaled.

The Gaussian response with the energy and zenith widths of the release, with the variance of the analysis bins
removed, and its analysis bins (App. A.4), the
zenith width multiplied by a common factor; nu_mu disappearance and all channels, theta23 and both phases fixed,
at x0. Output: results/direction.npz. Run from the repository root: python -m compute.direction
"""
import numpy as np

from nmosyn.average import gauss_newton
from nmosyn.experiments.deepcore import DeepCore
from nmosyn.params import DCP_IO, DCP_NO, S23SQ, X0, b_lbl, intercept
from nmosyn.paths import RESULTS

COSZ_SCALE = np.array([0.5, 0.7, 1.0, 1.4, 2.0])


def fixed(model):
    """b at x0 with theta23 and both phases fixed at the NuFIT values (Gauss-Newton, App. B.3)."""
    y = gauss_newton(lambda v: model.counts(v, S23SQ, DCP_IO), model.counts(X0, S23SQ, DCP_NO),
                     -X0 + b_lbl(DCP_NO, DCP_IO))
    return intercept(y, X0)


def main():
    out = dict(cosz_scale=COSZ_SCALE)
    for channels in ('numu', 'all'):
        model = DeepCore(channels)
        out[channels] = np.array([fixed(model.gaussian(cosz_scale=f)) for f in COSZ_SCALE])
    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(RESULTS / 'direction.npz', **out)


if __name__ == '__main__':
    main()
