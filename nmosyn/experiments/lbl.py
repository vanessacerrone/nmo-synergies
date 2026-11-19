"""Accelerator experiments: the generic beam and the NOvA and T2K far-detector spectra (App. A.3).

All probabilities are the exact three-flavor ones at constant crust density. The generic beam has nu_mu disappearance
only, with a Gaussian flux, a cross section rising like E, and a Gaussian energy resolution. It serves the closed-form
checks of Sec. 3 (fig_accelerator_closed_form, Eq. (3.6)) and the antineutrino share of App. C.2
(fig_accelerator_nubar_share). NOvA and T2K have nu_mu disappearance and nu_e appearance in both beam modes, with the
spectra of lbl_spectra.
"""
import numpy as np
from scipy.special import erf

from .. import osc
from ..params import NUFIT, Params
from . import lbl_spectra

CRUST_RHOYE = 1.35                  # rho*Ye of the Earth's crust [g/cm^3]
RESOLUTION = 0.10                   # sigma_E/E of the generic beam and of T2K
N_CACHE = 4                         # values of Dm31 whose amplitudes are kept
# baseline [km], flux peak and width [GeV] of the NOvA-like beam of fig_accelerator_nubar_share
NOVA_LIKE = (810.0, 2.0, 0.30)
T2K_LIKE = (295.0, 0.6, 0.10)       # same for the T2K-like beam


class _Amplitudes:
    """Amplitudes without theta23 and delta_CP, neutrinos and antineutrinos, cached for the last few Dm31."""

    def __init__(self, e_gev, l_km, rhoye, p):
        self.e_gev, self.l_km, self.rhoye, self.p = e_gev, l_km, rhoye, p
        self._cache = {}

    def __call__(self, y):
        key = float(y)
        if key not in self._cache:
            if len(self._cache) >= N_CACHE:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = tuple(osc.constant_amplitudes(self.e_gev, self.l_km, self.rhoye, key, self.p,
                                                             anti) for anti in (False, True))
        return self._cache[key]


def _migration(e_gev, edges, resolution):
    """K[b, j]: fraction of the events at true energy E_j reconstructed in bin b, sigma = resolution E."""
    lo, hi = edges[:-1, None], edges[1:, None]
    width = np.sqrt(2.0) * resolution * e_gev[None, :]
    return 0.5 * (erf((hi - e_gev[None, :]) / width) - erf((lo - e_gev[None, :]) / width))


class Beam:
    """Generic accelerator nu_mu disappearance, neutrino and antineutrino runs (Sec. 3, App. A.3, App. C.2).

    The flux is a Gaussian at e_peak [GeV] of the given width, times a cross section rising like E; e_gev
    is the grid of true energies, edges [GeV] the bins of reconstructed energy, nubar_share the antineutrino
    share of the unoscillated events. Bins with less than 1e-6 of the largest unoscillated content are
    dropped. Counts are in arbitrary normalization, neutrino run first.
    """

    def __init__(self, l_km: float, e_peak: float, width: float, e_gev: np.ndarray, edges: np.ndarray,
                 resolution: float = RESOLUTION, nubar_share: float = 0.5, rhoye: float = CRUST_RHOYE,
                 p: Params = NUFIT):
        e_gev = np.asarray(e_gev, dtype=float)
        flux_xsec = np.exp(-0.5 * ((e_gev - e_peak) / width) ** 2) * e_gev
        response = flux_xsec[None, :] * _migration(e_gev, np.asarray(edges, dtype=float), resolution)
        unoscillated = response.sum(axis=1)
        self.response = response[unoscillated > unoscillated.max() * 1e-6]
        self.share = (1.0 - nubar_share, nubar_share)
        self.n_bins = 2 * self.response.shape[0]
        self.disappearance = np.ones(self.n_bins, dtype=bool)
        self._amplitudes = _Amplitudes(e_gev, l_km, rhoye, p)

    def counts(self, y: float, s23sq: float, dcp: float) -> np.ndarray:
        """Expected nu_mu events per bin, neutrino run then antineutrino run."""
        out = []
        for share, amp, anti in zip(self.share, self._amplitudes(y), (False, True)):
            out.append(share * (self.response @ osc.probabilities(amp, s23sq, dcp, anti)[:, 1, 1]))
        return np.concatenate(out)


def generic_beam(nubar_share: float = 0.5, rhoye: float = CRUST_RHOYE, p: Params = NUFIT) -> Beam:
    """The 810 km beam of fig_accelerator_closed_form (App. A.3): flux at 2 GeV, 0.1 GeV bins in 0.5-5 GeV, sigma_E/E
    10%."""
    return Beam(810.0, 2.0, 0.30, np.arange(0.5, 5.0, 0.002), np.arange(0.5, 5.0 + 1e-9, 0.10),
                nubar_share=nubar_share, rhoye=rhoye, p=p)


def peaked_beam(l_km: float, e_peak: float, width: float, nubar_share: float, rhoye: float = CRUST_RHOYE,
                p: Params = NUFIT) -> Beam:
    """NOvA-like or T2K-like beam of App. C.2 (fig_accelerator_nubar_share), sigma_E/E = 10%; use with NOVA_LIKE or
    T2K_LIKE.

    True energies from e_peak - 4 width (at least 0.1 GeV) to e_peak + 6 width in steps of e_peak/500, bins of
    width e_peak/20.
    """
    e_gev = np.arange(max(0.1, e_peak - 4 * width), e_peak + 6 * width, e_peak / 500)
    edges = np.arange(e_gev[0], e_gev[-1] + 1e-9, e_peak / 20)
    return Beam(l_km, e_peak, width, e_gev, edges, nubar_share=nubar_share, rhoye=rhoye, p=p)


class FarDetector:
    """NOvA or T2K: mu-like counts in FHC and RHC, then e-like counts in FHC and RHC (App. A.3).

    The probabilities are averaged over each bin with the response: a top-hat for spectra in reconstructed
    energy (NOvA), a Gaussian of relative width resolution for spectra in true energy (T2K). The backgrounds
    are not oscillated. disappearance marks the mu-like bins. Counts are expected events of the real
    exposures.
    """

    def __init__(self, spectra: lbl_spectra.Spectra, p: Params = NUFIT, resolution: float = RESOLUTION):
        edges = spectra.edges
        e_lo = max(edges[0], 0.05)                    # the lowest NOvA bin starts at zero energy
        if spectra.true_energy:
            e_gev = np.linspace(e_lo, (1 + 3 * resolution) * edges[-1], 1500)
            kernel = _migration(e_gev, edges, resolution)
        else:
            e_gev = np.linspace(e_lo, edges[-1], 4000)
            kernel = ((e_gev[None, :] >= edges[:-1, None]) & (e_gev[None, :] < edges[1:, None])).astype(float)
        self.average = kernel / kernel.sum(axis=1, keepdims=True)
        self.name = spectra.name
        self.events = spectra.events
        n = len(edges) - 1
        self.n_bins = 4 * n
        self.disappearance = np.arange(self.n_bins) < 2 * n
        self._amplitudes = _Amplitudes(e_gev, spectra.l_km, spectra.rhoye, p)

    def _bin_average(self, prob):
        """Bin averages of P_mumu, P_mue, P_ee."""
        return {k: self.average @ v for k, v in lbl_spectra.channels(prob).items()}

    def counts(self, y: float, s23sq: float, dcp: float) -> np.ndarray:
        """Expected events per bin: mu-like FHC, mu-like RHC, e-like FHC, e-like RHC."""
        amp_nu, amp_nubar = self._amplitudes(y)
        nu = self._bin_average(osc.probabilities(amp_nu, s23sq, dcp))
        nubar = self._bin_average(osc.probabilities(amp_nubar, s23sq, dcp, antineutrino=True))
        mu_fhc, e_fhc = lbl_spectra.samples(self.events, 'FHC', nu, nubar)
        mu_rhc, e_rhc = lbl_spectra.samples(self.events, 'RHC', nu, nubar)
        return np.concatenate([mu_fhc, mu_rhc, e_fhc, e_rhc])


def nova(p: Params = NUFIT) -> FarDetector:
    """NOvA with its far-detector spectra from the public releases (App. A.3)."""
    return FarDetector(lbl_spectra.nova(), p)


def t2k(p: Params = NUFIT) -> FarDetector:
    """T2K with its far-detector spectra from the public releases (App. A.3)."""
    return FarDetector(lbl_spectra.t2k(), p)
