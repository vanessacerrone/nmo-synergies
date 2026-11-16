"""Inverted-ordering splitting of the IceCube DeepCore 9.3-year sample, from its released likelihood map.

The release (doi:10.7910/DVN/U20MMB) gives -2 Delta lnL on a grid of sin^2 theta23 x Dm32 for each ordering,
relative to the best fit of that ordering, and the best fit itself. The inverted-ordering map is profiled over
sin^2 theta23 (minimum over the grid values at each Dm32), and the 1 sigma interval of Dm32 is where the
profile, interpolated by a cubic spline in Dm32, equals 1. Output: results/measured.npz.
Run from the repository root: python -m compute.measured
"""
import pickle
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from nmosyn.paths import EXTERNAL, RESULTS

IO_MAP = EXTERNAL / 'icecube_u20mmb' / 'saved_io_delta_LLH_map.npy'
# the map is a pickled dict of arrays: unpickling may only build numpy arrays, with the names the file uses
NUMPY_CLASSES = {('numpy.core.multiarray', '_reconstruct'), ('numpy', 'ndarray'), ('numpy', 'dtype')}


class _ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in NUMPY_CLASSES:
            raise pickle.UnpicklingError(f'{module}.{name} is not allowed in a likelihood map')
        return super().find_class(module, name)


def read_map(path: Path) -> dict[str, np.ndarray]:
    """The dict of the release: sin2_theta23, deltam232 [eV^2], deltallh, bfp_sin2th23, bfp_dm232."""
    with open(path, 'rb') as f:
        np.lib.format.read_magic(f)
        np.lib.format.read_array_header_1_0(f)     # the release files are .npy version 1.0
        return _ArrayUnpickler(f).load().item()


def profile_interval(m: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Dm32 nodes, the profile over sin^2 theta23, and the Dm32 range where the profile is at most 1."""
    s23, dm32 = np.unique(m['sin2_theta23']), np.unique(m['deltam232'])
    llh = np.full((dm32.size, s23.size), np.nan)
    llh[np.searchsorted(dm32, m['deltam232']), np.searchsorted(s23, m['sin2_theta23'])] = m['deltallh']
    prof = llh.min(axis=1)
    roots = CubicSpline(dm32, prof).solve(1.0, extrapolate=False)
    return dm32, prof, (roots.min(), roots.max())


def main() -> None:
    m = read_map(IO_MAP)
    dm32, prof, (lo, hi) = profile_interval(m)
    best = float(m['bfp_dm232'])
    err = np.array([best - lo, hi - best])
    np.savez(RESULTS / 'measured.npz', ic93_io_dm32=best, ic93_io_dm32_err=err, ic93_io_dm32_nodes=dm32,
             ic93_io_profile=prof)
    print(f'IceCube 9.3 yr, IO: Dm32 = {best * 1e3:.4f} -{err[0] * 1e3:.4f} +{err[1] * 1e3:.4f} [1e-3 eV^2]')


if __name__ == '__main__':
    main()
