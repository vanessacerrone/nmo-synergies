"""KM3NeT/ORCA6 (433 kt yr): expected counts from the binned response of the public release (App. A.5).

The release gives, for every neutrino flavor, charge, and interaction, the weight W per unit flux of each true
cell (E, cos theta_z) in each reconstructed bin (E_reco, cos theta_z,reco, class), to be multiplied by the
Honda flux at the Frejus site (solar minimum, no mountain) at the cell center, sqrt(E_lo E_hi) and the middle
cos theta_z. The count of a bin is sum W (phi_e P_e->beta + phi_mu P_mu->beta) for nu_beta CC and
sum W (phi_e + phi_mu) for NC, plus the atmospheric-muon background of the release. In each true cell below
1 TeV, P_alpha->beta is averaged over the cell with the weight phi_alpha, on Gauss-Legendre nodes in log E
and equal steps in cos theta_z, so the flux normalization stays that of the release; the cells above 1 TeV
are unoscillated. The detector sits 2.45 km below the sea surface; the paths cross the atmosphere, the sea
water, and PREM (earth.layers). Reconstructed space: 15 energy bins x 10 up-going cos theta_z bins per
class, ordered (class, E_reco, cos theta_z,reco).
"""
from pathlib import Path

import numpy as np
import uproot
from scipy.sparse import csr_array

from ..earth import MAX_KM, WATER_YE, layers
from ..flux import FREJUS, Honda
from ..osc import path_amplitudes, probabilities
from ..params import NUFIT, Params
from ..paths import EXTERNAL

RELEASE = EXTERNAL / 'km3net_orca6' / 'ORCA6_433kt-y_opendata_v0.5.root'
N_CLASSES = 3                       # 1 high-purity tracks, 2 showers, 3 low-purity tracks
DEPTH_KM = 2.45                     # detector depth below the sea surface
OVERBURDEN = 1.03 * WATER_YE        # rho*Ye of the sea water above it [g/cm^3]
E_OSC_MAX = 1000.0                  # GeV; above it the phase Dm31 L / 4E across the Earth is below 0.05 rad
N_SUB_E, N_SUB_COSZ = 8, 8          # sub-grid points per true cell in log E and in cos theta_z
N_E_RECO, N_COSZ_RECO = 15, 10      # reconstructed energy bins and up-going cos theta_z bins of each class
N_CHANNELS = 4                      # nu_e CC, nu_mu CC, nu_tau CC, NC; for neutrinos and antineutrinos
MAX_CACHED = 4


class Orca6:
    """Expected ORCA6 counts per reconstructed bin for any Dm31, theta23, and delta_CP (App. A.5).

    classes: analysis classes of the release, 1 high-purity tracks, 2 showers, 3 low-purity tracks.
    density_scale multiplies every density (0 gives vacuum); max_km is the longest layer of earth.layers;
    n_sub_e and n_sub_cosz are the points of the sub-grid of a true cell in log E and in cos theta_z.
    Attributes: n_bins, observed (the counts of the release), muons (its atmospheric-muon background, included
    in counts).
    """

    def __init__(self, classes: tuple[int, ...] = (1, 2, 3), p: Params = NUFIT, density_scale: float = 1.0,
                 max_km: float = MAX_KM, n_sub_e: int = N_SUB_E, n_sub_cosz: int = N_SUB_COSZ,
                 path: Path = RELEASE):
        self.classes = tuple(classes)
        self.p = p
        self.n_bins = N_E_RECO * N_COSZ_RECO * len(self.classes)
        with uproot.open(path) as release:
            nu = release['binned_nu_response'].arrays(library='np')
            mu = release['binned_muon'].arrays(library='np')
            data = release['binned_data'].arrays(library='np')
            e_edges = release['E_true_axis'].edges()
            c_edges = release['Ct_true_axis'].edges()
        n_cosz = c_edges.size - 1
        n_cells = (e_edges.size - 1) * n_cosz

        position = np.full(N_CLASSES + 1, -1)
        position[list(self.classes)] = np.arange(len(self.classes))

        def reco_bin(tree):
            keep = (position[tree['AnaClass']] >= 0) & (tree['Ct_reco_bin'] <= N_COSZ_RECO)
            index = ((position[tree['AnaClass']] * N_E_RECO + tree['E_reco_bin'] - 1) * N_COSZ_RECO
                     + tree['Ct_reco_bin'] - 1)
            return keep, index

        keep, index = reco_bin(mu)
        self.muons = np.bincount(index[keep], mu['W'][keep], minlength=self.n_bins)
        keep, index = reco_bin(data)
        self.observed = np.bincount(index[keep], data['W'][keep], minlength=self.n_bins)

        keep, index = reco_bin(nu)
        flavor = np.abs(nu['Pdg'][keep]) // 2 - 6                                   # 12, 14, 16 -> 0, 1, 2
        channel = np.where(nu['IsCC'][keep], flavor, 3) + N_CHANNELS * (nu['Pdg'][keep] < 0)
        cell = (nu['E_true_bin'][keep] - 1) * n_cosz + nu['Ct_true_bin'][keep] - 1
        # response (reco bin, channel x true cell); counts = response @ rate
        self._response = csr_array((nu['W'][keep], (index[keep], channel * n_cells + cell)),
                                   shape=(self.n_bins, 2 * N_CHANNELS * n_cells))

        flux = Honda(FREJUS)
        species = (('nue', 'numu'), ('nuebar', 'numubar'))
        cell_e, cell_c = np.divmod(np.arange(n_cells), n_cosz)
        e_lo, e_hi, c_lo, c_hi = e_edges[cell_e], e_edges[cell_e + 1], c_edges[cell_c], c_edges[cell_c + 1]
        e_center, c_center = np.sqrt(e_lo * e_hi), 0.5 * (c_lo + c_hi)
        # flux at the cell centers, (antineutrino, initial flavor e or mu, cell)
        phi = np.array([[flux(s, e_center, c_center) for s in pair] for pair in species])
        # rates per true cell without oscillations, (antineutrino, channel, cell); the NC rate is flavor blind
        self._rate0 = np.zeros((2, N_CHANNELS, n_cells))
        self._rate0[:, :2] = phi
        self._rate0[:, 3] = phi.sum(axis=1)

        osc = np.flatnonzero(e_center < E_OSC_MAX)
        nodes, w_e = np.polynomial.legendre.leggauss(n_sub_e)
        u = 0.5 * (nodes[:, None] + 1)
        v = (np.arange(n_sub_cosz) + 0.5) / n_sub_cosz
        e_sub = e_lo[osc, None, None] * (e_hi / e_lo)[osc, None, None] ** u
        cosz_sub = c_lo[osc, None, None] + (c_hi - c_lo)[osc, None, None] * v
        # points ordered (cell, E node, cos theta_z step)
        self._e_sub, cosz_sub = (a.ravel() for a in np.broadcast_arrays(e_sub, cosz_sub))
        self._layers = layers(cosz_sub, DEPTH_KM, OVERBURDEN, max_km, density_scale)
        # weights w phi_alpha(point) / sum_cell w phi_alpha x phi_alpha(center), w the quadrature weights,
        # (antineutrino, cell, point, alpha): each cell keeps the release rate, with the flux-weighted mean
        # probability
        phi_sub = np.array([[flux(s, self._e_sub, cosz_sub) for s in pair] for pair in species])
        phi_sub = phi_sub.reshape(2, 2, osc.size, n_sub_e * n_sub_cosz) * np.repeat(w_e, n_sub_cosz)
        weight = phi_sub / phi_sub.sum(axis=-1, keepdims=True) * phi[:, :, osc, None]
        self._weight = weight.transpose(0, 2, 3, 1)
        self._osc_cells = osc
        self._cache = {}

    def _amplitudes(self, y):
        key = float(y)
        if key not in self._cache:
            if len(self._cache) >= MAX_CACHED:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = [path_amplitudes(self._e_sub, *self._layers, key, self.p, anti)
                                for anti in (False, True)]
        return self._cache[key]

    def counts(self, y: float, s23sq: float, dcp: float) -> np.ndarray:
        """Expected counts per reconstructed bin, muons included (App. A.5); y is the signed Dm31 [eV^2]."""
        rate = self._rate0.copy()
        for anti, amp in enumerate(self._amplitudes(y)):
            prob = probabilities(amp, s23sq, dcp, antineutrino=bool(anti))[..., :2]
            prob = prob.reshape(self._weight.shape[1:3] + (3, 2))
            rate[anti][:3, self._osc_cells] = np.einsum('csba,csa->bc', prob, self._weight[anti])
        return self._response @ rate.ravel() + self.muons
