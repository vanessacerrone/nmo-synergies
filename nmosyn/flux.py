"""Atmospheric neutrino flux: the Honda et al. 2014 tables, azimuth averaged (PRD 92, 023004 (2015))."""
import re

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .paths import EXTERNAL

SOUTH_POLE = EXTERNAL / 'honda' / 'spl-ally-20-01-solmin.d'      # IceCube, solar minimum, no mountain
FREJUS = EXTERNAL / 'honda' / 'frj-ally-20-01-solmin.d'          # KM3NeT/ORCA, as prescribed by the ORCA6 release
SPECIES = ('numu', 'numubar', 'nue', 'nuebar')                    # the columns of the tables after the energy
_BLOCK = re.compile(r'average flux in \[cosZ =\s*(-?[\d.]+)\s*--\s*(-?[\d.]+)')


class Honda:
    """Flux [(m^2 s sr GeV)^-1] of one site, bilinear in (cos theta_z, log10 E) on the logarithm of the table.

    Each table block covers a bin of cos theta_z of width 0.1, taken at its center; outside the outer centers,
    and outside the energy range, the interpolation is continued linearly.
    """

    def __init__(self, path=SOUTH_POLE):
        text = open(path).read()
        heads = list(_BLOCK.finditer(text))
        ends = [m.start() for m in heads[1:]] + [len(text)]
        centers, tables = [], []
        for head, end in zip(heads, ends):
            lines = text[head.end():end].splitlines()[2:]          # rest of the header line, column names
            tables.append(np.array([line.split() for line in lines if line.strip()], dtype=float))
            centers.append(0.5 * (float(head.group(1)) + float(head.group(2))))
        order = np.argsort(centers)
        log_e = np.log10(tables[0][:, 0])
        log_flux = np.log10(np.clip(np.array(tables)[order, :, 1:5], 1e-300, None))     # (cos, E, species)
        self._grid = {s: RegularGridInterpolator((np.array(centers)[order], log_e), log_flux[:, :, k],
                                                 bounds_error=False, fill_value=None)
                      for k, s in enumerate(SPECIES)}

    def __call__(self, species, e_gev, cosz):
        """Flux of species ('numu', 'numubar', 'nue', 'nuebar') at energies e_gev and directions cosz."""
        e, c = np.broadcast_arrays(np.asarray(e_gev, dtype=float), np.asarray(cosz, dtype=float))
        return 10.0 ** self._grid[species](np.stack([c, np.log10(e)], axis=-1))
