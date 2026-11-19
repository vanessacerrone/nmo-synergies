"""JUNO: eight reactor cores, the reactor spectrum, and the energy resolution (App. A.2).

Expected inverse-beta-decay counts in bins of reconstructed neutrino energy, 20 keV wide from 1.81 MeV
(just above threshold) to 8.99 MeV, for any Dm31. Each bin is the reactor spectrum times the survival
probability, averaged over the eight cores with weights proportional to thermal power over L^2 and folded
with a Gaussian energy resolution. The survival probability is the exact three-flavor one at the constant
average density that JUNO estimates for the crust and sediment along its baselines.

The rotation in theta23 and delta_CP acts only on the mu-tau block, so it leaves the ee element of the
amplitude unchanged (App. D.2): P_ee and the counts depend on neither, and the JUNO solution depends on the
true splitting only. The counts are per unit exposure, in arbitrary units: the solution does not depend on
the normalization.
"""
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq, minimize_scalar
from scipy.special import erf

from ..average import bin_solutions, gauss_newton
from ..osc import path_amplitudes
from ..params import NUFIT, Params, b_sbr, dm2_phi_ee
from .reactor import ibd_spectrum, prompt_energy

# JUNO collaboration, Chin. Phys. C 46 (2022) 123001, arXiv:2204.13249, Table 2:
# core, thermal power [GW_th], baseline [km]
CORES = (
    ('Yangjiang 1', 2.9, 52.74),
    ('Yangjiang 2', 2.9, 52.82),
    ('Yangjiang 3', 2.9, 52.41),
    ('Yangjiang 4', 2.9, 52.49),
    ('Yangjiang 5', 2.9, 52.11),
    ('Yangjiang 6', 2.9, 52.19),
    ('Taishan 1', 4.6, 52.77),
    ('Taishan 2', 4.6, 52.64),
)
L_NOMINAL_KM = 52.5         # nominal baseline, used where Eq. (3.10) needs a single one
# rho * Y_e [g/cm^3] along the baselines: Y_e = 0.5, rho = 2.45 g/cm^3 for crust and sediment, arXiv:2204.13249 Sec. 2.2
CRUST_RHOYE = 0.5 * 2.45
E_MIN_MEV, E_MAX_MEV = 1.81, 9.0
DE_TRUE_MEV = 1e-3          # integration step in true energy
A_RES, B_RES, C_RES = 0.0261, 0.0064, 0.0120      # full simulation, arXiv:2405.17860, used in 2405.18008
R1_RES = float(np.sqrt(A_RES ** 2 + B_RES ** 2 + C_RES ** 2))   # sigma_E / E at a prompt energy of 1 MeV, 2.94%
MAX_CACHED = 4              # values of Dm31 whose counts are kept


def core_shares() -> np.ndarray:
    """Share of each core in the rate without oscillations, P_th / L^2 over its sum (App. A.2)."""
    w = np.array([power / l_km ** 2 for _, power, l_km in CORES])
    return w / w.sum()


def resolution(e_mev: ArrayLike, a_res: float = A_RES, b_res: float = B_RES,
               c_res: float = C_RES) -> np.ndarray:
    """sigma_E [MeV] for true neutrino energy e_mev, evaluated at the prompt energy E (App. A.2).

    sigma_E / E = sqrt((a_res / sqrt(E / MeV))^2 + b_res^2 + (c_res MeV / E)^2).
    """
    e = prompt_energy(e_mev)
    return np.sqrt(a_res ** 2 * e + b_res ** 2 * e ** 2 + c_res ** 2)


def migration(e_true: np.ndarray, edges: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Probability (n_bins, n_true) that an event at e_true is reconstructed in each bin (App. A.2).

    Gaussian of width sigma (n_true,) integrated over the bin; without resolution each event stays in its bin.
    """
    lo, hi = edges[:-1, None], edges[1:, None]
    if not np.any(sigma):
        return ((e_true >= lo) & (e_true < hi)).astype(float)
    r = np.sqrt(2.0) * sigma
    return 0.5 * (erf((hi - e_true) / r) - erf((lo - e_true) / r))


class Juno:
    """Expected JUNO counts in bins of reconstructed neutrino energy, centered at e_mev (App. A.2).

    a_res, b_res, c_res: stochastic, constant, and noise terms of the resolution; rhoye = 0 gives vacuum;
    bin_mev: width of the analysis bins. The three terms are set independently, so Juno(a_res=0.0) keeps
    b_res and c_res; the scan of the paper scales the three terms together, Juno.scaled(a_res), and labels it by
    the resolution at 1 MeV, Juno.at_1mev(r1_res).
    """

    def __init__(self, a_res: float = A_RES, b_res: float = B_RES, c_res: float = C_RES, p: Params = NUFIT,
                 rhoye: float = CRUST_RHOYE, bin_mev: float = 0.020):
        self.p = p
        self.rhoye = rhoye
        n_true = round((E_MAX_MEV - E_MIN_MEV) / DE_TRUE_MEV)
        self.e_true = E_MIN_MEV + DE_TRUE_MEV * (np.arange(n_true) + 0.5)
        self.n_bins = int((E_MAX_MEV - E_MIN_MEV) / bin_mev + 1e-9)
        self.edges = E_MIN_MEV + bin_mev * np.arange(self.n_bins + 1)
        self.e_mev = 0.5 * (self.edges[:-1] + self.edges[1:])
        sigma = resolution(self.e_true, a_res, b_res, c_res)
        self._response = migration(self.e_true, self.edges, sigma) * (ibd_spectrum(self.e_true) * DE_TRUE_MEV)
        self._core_w = core_shares()
        self._e_gev = np.tile(self.e_true * 1e-3, len(CORES))
        self._lengths = np.repeat([l_km for _, _, l_km in CORES], n_true)[:, None]
        self._cache = {}

    @classmethod
    def scaled(cls, a_res: float, **kwargs) -> 'Juno':
        """Model with b_res and c_res scaled by the same factor as a_res, 0.64% and 1.20% at a_res = 2.61%, so
        that the resolution is perfect at a_res = 0 (Sec. 3, App. A.2, App. C.3)."""
        return cls(a_res, B_RES * a_res / A_RES, C_RES * a_res / A_RES, **kwargs)

    @classmethod
    def at_1mev(cls, r1_res: float, **kwargs) -> 'Juno':
        """Juno.scaled labeled by the resolution r1_res = sigma_E / E at a prompt energy of 1 MeV: the three terms
        scaled by r1_res / R1_RES, so R1_RES = 2.94% is the model of the full simulation (Sec. 3, App. C.3)."""
        return cls.scaled(A_RES * r1_res / R1_RES, **kwargs)

    def survival(self, y: float) -> np.ndarray:
        """P_ee of antineutrinos at the true energies e_true, averaged over the cores by rate (App. A.2)."""
        amp = path_amplitudes(self._e_gev, self._lengths, self.rhoye, y, self.p, antineutrino=True)
        return self._core_w @ (np.abs(amp[:, 0, 0]) ** 2).reshape(len(CORES), -1)

    def counts(self, y: float, s23sq: float, dcp: float) -> np.ndarray:
        """Expected counts per bin for the signed splitting y (App. A.2); s23sq and dcp do not enter P_ee."""
        return self._counts(y)

    def _counts(self, y: float) -> np.ndarray:
        """Counts per bin as a function of y alone, read-only, cached for the last few values of y."""
        key = float(y)
        if key not in self._cache:
            if len(self._cache) >= MAX_CACHED:
                del self._cache[next(iter(self._cache))]
            n = self._response @ self.survival(key)
            n.flags.writeable = False
            self._cache[key] = n
        return self._cache[key]

    def solution(self, x: float) -> float:
        """Wrong-ordering solution y for NO Asimov data at x: Eq. (2.3), solved by the Gauss-Newton iteration of
        App. B.3.

        The iteration starts at the short-baseline closed form, y = -x + 2 s12^2 Dm21 (Eq. 3.9).
        """
        return gauss_newton(self._counts, self._counts(x), -x + b_sbr(self.p))

    def omega(self, x: float, y: float) -> np.ndarray:
        """Weights omega_i = W_i s_i^2 of the bins at the solution y (Eq. 2.3), normalized to unit sum."""
        _, omega = bin_solutions(self._counts, self._counts(x), y)
        return omega / omega.sum()

    def solar_phase_average(self, x: float, y: float, l_km: float = L_NOMINAL_KM) -> float:
        """<Dm2_phi,ee>_omega of Eq. (3.10): Dm2_phi,ee at the bin centers, weights omega_i at y [eV^2]."""
        return float(np.sum(self.omega(x, y) * dm2_phi_ee(self.e_mev, l_km, self.p)))


def effective_energy(avg: float, p: Params = NUFIT, l_km: float = L_NOMINAL_KM) -> float:
    """Energy [MeV] above the maximum of Dm2_phi,ee(E) at which it equals avg [eV^2] (App. C.3)."""
    peak = minimize_scalar(lambda e: -dm2_phi_ee(e, l_km, p), bounds=(E_MIN_MEV, E_MAX_MEV),
                           method='bounded').x
    return brentq(lambda e: dm2_phi_ee(e, l_km, p) - avg, peak, E_MAX_MEV, xtol=1e-9)


def b_mbr(e_mev: float = 4.0, p: Params = NUFIT, l_km: float = L_NOMINAL_KM) -> float:
    """Eq. (3.10) with the solar phase at one energy, 2 s12^2 Dm21 - Dm2_phi,ee(E) (App. C.3).

    Returned in eV^2, like params.b_sbr; times 1e6 it is the intercept b in 1e-6 eV^2.
    """
    return b_sbr(p) - float(dm2_phi_ee(e_mev, l_km, p))
