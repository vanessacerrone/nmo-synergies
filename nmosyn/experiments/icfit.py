"""Full fit of the IceCube DeepCore public release (App. E), used to validate the weighted bin average.

The release (doi:10.7910/DVN/B4RITM) contains the observed counts of the golden sample of PRD 108, 012014 in
10 x 10 x 2 bins of reconstructed energy, cos theta_z, and particle identification, the simulated neutrino
events with their weights, the atmospheric-muon background per bin, and the hypersurfaces of the detector
parameters.

Expected count of bin i (App. E.1):
    N_i = (1 + a_nu) sum_k f_k,i(detector, |Dm31|) sum_{j in channel k, bin i} c_j + (1 + a_mu) N_mu,i,
    c_j = w_j T [phi_e P(e -> beta_j) + phi_mu P(mu -> beta_j)] (E_j / 24 GeV)^dgamma x_j,
with w_j the event weight [GeV cm^2 sr], T the livetime, phi the Honda South Pole flux of the charge of the
event, P the three-flavor probability through the layers of constant density of earth.layers (atmosphere,
ice, and PREM cut at its discontinuities into pieces of at most 500 km), beta_j the flavor of the event,
x_j the axial-mass reweighting of the release times the neutral-current normalization, and f_k,i the
hypersurface factor of channel k (NC and nu_e CC, nu_mu CC, nu_tau CC), linear in the detector parameters and
interpolated linearly in |Dm31|. The chi^2 is the modified Pearson form
sum_i (N_i - n_i)^2 / (N_i + sigma^2_MC,i) plus the Gaussian pull terms.
"""
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
from iminuit import Minuit
from contourpy import contour_generator
from scipy.interpolate import RegularGridInterpolator

from ..average import profiled_step
from ..earth import MAX_KM, layers
from ..flux import SOUTH_POLE, Honda
from ..osc import path_amplitudes, probabilities
from ..params import S23SQ, Params
from .deepcore import COSZ_EDGES, DEPTH_KM, E_EDGES, OVERBURDEN, RELEASE, _readonly

CLASS_EDGES = _readonly([0.55, 0.75, 1.0])
N_BINS = 200
MC_FILES = ('mc_nu_nc', 'mc_nue_cc', 'mc_numu_cc', 'mc_nutau_cc')
HS_FILES = ('hs_nu_nc_nue_cc', 'hs_numu_cc', 'hs_nutau_cc')        # channels k = 0, 1, 2
DETECTOR = ('dom_eff', 'hole_ice_p0', 'hole_ice_p1', 'bulk_ice_abs', 'bulk_ice_scatter')

# Oscillation parameters held fixed in the fits of the observed data, as in the release (App. E.4)
RELEASE_OSC = Params(s12sq=0.3098, s13sq=float(np.sin(np.radians(8.61)) ** 2), dm21=7.39e-5)
# Sum of the livetimes of the eight data-taking periods, PRD 108 Table III (the text rounds it to 7.5 years).
# The effective-area normalization has no prior, so the livetime sets only the value of that normalization.
LIVETIME_S = 7.45 * 365.25 * 86400
E_PIVOT = 24.0                      # GeV, pivot of the spectral index

# The 13 fit parameters; the three normalizations are relative changes, zero at the nominal expectation
NAMES = ('s23sq', 'dm31', *DETECTOR, 'aeff_norm', 'muon_norm', 'delta_gamma', 'ma_qe', 'ma_res', 'nc_norm')
NOMINAL = MappingProxyType({'dom_eff': 1.0, 'hole_ice_p0': 0.10, 'hole_ice_p1': -0.05, 'bulk_ice_abs': 1.0,
                            'bulk_ice_scatter': 1.0, 'aeff_norm': 0.0, 'muon_norm': 0.0, 'delta_gamma': 0.0,
                            'ma_qe': 0.0, 'ma_res': 0.0, 'nc_norm': 0.0})
PRIORS = MappingProxyType({'dom_eff': 0.10, 'delta_gamma': 0.10, 'ma_qe': 1.0, 'ma_res': 1.0,
                           'nc_norm': 0.20})
# Bounds: |dm31| within the hypersurface tables, the bulk-ice parameters within the ranges of PRD 108, the
# parameters with a prior within five prior widths, the normalizations above -1 (no negative rate), the others
# wide; none is reached at any minimum of the paper.
LIMITS = MappingProxyType({'s23sq': (0.01, 0.99), 'dm31': (1.5e-3, 3.5e-3), 'dom_eff': (0.5, 1.5),
                           'hole_ice_p0': (-3.0, 3.0), 'hole_ice_p1': (-0.5, 0.5),
                           'bulk_ice_abs': (0.90, 1.10), 'bulk_ice_scatter': (0.95, 1.15),
                           'aeff_norm': (-0.99, 10.0), 'muon_norm': (-0.99, 10.0), 'delta_gamma': (-0.5, 0.5),
                           'ma_qe': (-5.0, 5.0), 'ma_res': (-5.0, 5.0), 'nc_norm': (-0.99, 1.0)})
# Initial MIGRAD steps: about the uncertainties of the fit to the data
STEPS = MappingProxyType({'s23sq': 0.06, 'dm31': 8e-5, 'dom_eff': 0.03, 'hole_ice_p0': 0.13, 'hole_ice_p1': 0.03,
                          'bulk_ice_abs': 0.02, 'bulk_ice_scatter': 0.02, 'aeff_norm': 0.05, 'muon_norm': 0.4,
                          'delta_gamma': 0.03, 'ma_qe': 1.0, 'ma_res': 1.0, 'nc_norm': 0.2})
S23SQ_STARTS = (S23SQ, 1.0 - S23SQ)  # one start in each octant: the NuFIT value and its mirror
DM31_START = 2.45e-3                # eV^2, |Dm31| at the start of the fits
B_START = 103.8                     # 1e-6 eV^2, starting intercept: the vacuum value 2 c12^2 Dm21 (App. B.3)
MIN_COUNT = 1e-9                    # smallest expected count of a bin in the chi^2
TOL = 1e-5                          # MIGRAD: EDM < 2e-8, intercepts converged to 1e-3 x 1e-6 eV^2
H_DERIVATIVE = MappingProxyType({'dm31': 1e-7})     # steps of the numerical derivatives; 1e-4 for the others
# Parameters profiled in the three weighted bin averages on the fitter's expectation (App. E.2)
WBA_CASES = MappingProxyType({'y': (), 't23': ('s23sq',), 'all': tuple(k for k in NAMES if k != 'dm31')})

# Grid of the profiled contour of fig_deepcore_reproduction
S23SQ_GRID = _readonly(np.linspace(0.38, 0.64, 12))
DM32_GRID = _readonly(np.linspace(2.15e-3, 2.65e-3, 12))       # eV^2

MAX_AMPLITUDES = 4
MAX_RATES = 8


@dataclass(frozen=True)
class FitResult:
    """Minimum of a fit: chi^2, values, HESSE errors (None without HESSE; zero for fixed parameters),
    validity, calls and seconds of all starts."""

    chi2: float
    values: dict
    errors: dict | None
    valid: bool
    n_calls: int
    seconds: float


def _bin_index(table: pd.DataFrame) -> np.ndarray:
    """Analysis bin (i_E, i_cos, i_pid) of each row, flattened in that order."""
    i_e = np.digitize(table['reco_energy'].to_numpy(), E_EDGES) - 1
    i_c = np.digitize(table['reco_coszen'].to_numpy(), COSZ_EDGES) - 1
    i_p = np.digitize(table['pid'].to_numpy(), CLASS_EDGES) - 1
    return (i_e * (COSZ_EDGES.size - 1) + i_c) * (CLASS_EDGES.size - 1) + i_p


def truth(x: float, nuisances: Mapping[str, float], s23sq: float = S23SQ) -> dict:
    """Parameters of the normal-ordering truth at the splitting x [eV^2], nuisance parameters from nuisances
    (App. E.1)."""
    return {k: nuisances[k] for k in NOMINAL} | {'s23sq': s23sq, 'dm31': x}


class DeepCoreFit:
    """Expected counts, chi^2, and fits of the DeepCore public release with 13 parameters (App. E).

    p, dcp: oscillation parameters held fixed, RELEASE_OSC and 0 for the observed data (fig_deepcore_reproduction, App.
    E.3, App. E.4), NUFIT and DCP_NO in both orderings for the Asimov checks (fig_full_fit_validation, App. E.1-E.2);
    max_km, depth_km: largest length of the layers of constant density along each path (earth.layers) and detector depth
    [km]. Parameters travel as a dict with the keys NAMES; dm31 is signed (negative for the inverted ordering). The
    amplitudes are cached per Dm31 and rotated for theta23 and delta_CP (App. D.2). Attributes: observed, muons,
    muon_variance (per bin), hs_grid.

    hs_reading sets where the hypersurfaces, tabulated against |Dm31| for normal-ordering oscillations, are
    read: 'matched' reads a normal-ordering hypothesis at |Dm31| and an inverted-ordering one at
    |Dm31,IO| + 2 c12^2 Dm21, the splitting with the closest muon-disappearance pattern (App. E.1);
    'splitting' reads both at |Dm31|; a number reads every hypothesis at that splitting [eV^2].

    counts(y, s23sq, dcp) is the interface of the experiment modules: the expected counts with the nuisance
    parameters held at nuisances and the hypersurfaces read as hs_reading, both given to the constructor.
    """

    n_bins = N_BINS

    def __init__(self, p: Params = RELEASE_OSC, dcp: float = 0.0, max_km: float = MAX_KM,
                 depth_km: float = DEPTH_KM, livetime_s: float = LIVETIME_S,
                 nuisances: Mapping[str, float] = NOMINAL, hs_reading: float | str = 'matched',
                 path: Path = RELEASE):
        self.p = p
        self.dcp = dcp
        self.nuisances = MappingProxyType({k: float(nuisances[k]) for k in NOMINAL})
        self.hs_reading = hs_reading

        data = pd.read_csv(path / 'data.tab', sep='\t')
        self.observed = _readonly(np.bincount(_bin_index(data), data['count'].to_numpy(), N_BINS))
        muons = pd.read_csv(path / 'mc_mu.tab', sep='\t')
        self.muons = _readonly(np.bincount(_bin_index(muons), muons['count'].to_numpy(), N_BINS))
        sigma_mu = muons['abs_uncertainty'].to_numpy()
        self.muon_variance = _readonly(np.bincount(_bin_index(muons), sigma_mu ** 2, N_BINS))

        hs = [pd.read_csv(path / f'{name}.tab', sep='\t') for name in HS_FILES]
        self.hs_grid = _readonly(np.unique(hs[0]['deltam31'].to_numpy()))
        # hypersurface coefficients (Dm31 node, channel, bin, [intercept, slopes of DETECTOR])
        self._hs = np.zeros((self.hs_grid.size, len(HS_FILES), N_BINS, 1 + len(DETECTOR)))
        for k, table in enumerate(hs):
            node = np.searchsorted(self.hs_grid, table['deltam31'].to_numpy())
            self._hs[node, k, _bin_index(table)] = table[['intercept', *DETECTOR]].to_numpy()

        mc = pd.concat([pd.read_csv(path / f'{name}.tab', sep='\t') for name in MC_FILES], ignore_index=True)
        mc = mc.iloc[np.argsort(mc['pdg'].to_numpy() < 0, kind='stable')].reset_index(drop=True)
        pdg = mc['pdg'].to_numpy()
        n_nu = int(np.sum(pdg > 0))
        self._parts = (slice(None, n_nu), slice(n_nu, None))              # neutrinos, antineutrinos
        self._e = mc['true_energy'].to_numpy()
        self._cosz = mc['true_coszen'].to_numpy()
        self._layers = [layers(self._cosz[part], depth_km, OVERBURDEN, max_km) for part in self._parts]
        self._flavor = np.abs(pdg) // 2 - 6                                # 12, 14, 16 -> 0, 1, 2
        nc = mc['type'].to_numpy() == 0
        channel = np.where(nc, 0, self._flavor)                            # NC of all flavors: table 0
        self._cell = channel * N_BINS + _bin_index(mc)
        self._nc = np.flatnonzero(nc)
        self._log_e = np.log(self._e / E_PIVOT)
        axial = mc[['MaCCQE_linear', 'MaCCQE_quad', 'MaCCRES_linear', 'MaCCRES_quad']].to_numpy()
        self._axial_rows = np.flatnonzero(np.any(axial != 0, axis=1))     # CC QE and RES events
        self._axial = axial[self._axial_rows].T

        # events per unit probability from the nu_e and nu_mu flux (columns) of the charge of each event
        flux = Honda(SOUTH_POLE)
        anti = pdg < 0
        exposure = mc['weight'].to_numpy() * livetime_s * 1e-4            # flux in m^-2 -> cm^-2
        self._phi = np.stack([exposure * np.where(anti, flux(species + 'bar', self._e, self._cosz),
                                                  flux(species, self._e, self._cosz))
                              for species in ('nue', 'numu')], axis=1)
        self._amplitude_cache = {}
        self._rate_cache = {}

    def _amplitudes(self, dm31):
        key = float(dm31)
        if key not in self._amplitude_cache:
            if len(self._amplitude_cache) >= MAX_AMPLITUDES:
                self._amplitude_cache.pop(next(iter(self._amplitude_cache)))
            amp = [path_amplitudes(self._e[part], *chord, key, self.p, anti)
                   for anti, part, chord in zip((False, True), self._parts, self._layers)]
            self._amplitude_cache[key] = amp
        return self._amplitude_cache[key]

    def _rates(self, dm31, s23sq, dcp):
        """Oscillated rate of every event, w T [phi_e P(e -> beta) + phi_mu P(mu -> beta)]."""
        key = (float(dm31), float(s23sq), float(dcp))
        if key not in self._rate_cache:
            if len(self._rate_cache) >= MAX_RATES:
                self._rate_cache.pop(next(iter(self._rate_cache)))
            rate = np.empty(self._flavor.size)
            for anti, part, amp in zip((False, True), self._parts, self._amplitudes(dm31)):
                prob = probabilities(amp, s23sq, dcp, anti)
                rows, beta = np.arange(prob.shape[0]), self._flavor[part]
                phi = self._phi[part]
                rate[part] = phi[:, 0] * prob[rows, beta, 0] + phi[:, 1] * prob[rows, beta, 1]
            self._rate_cache[key] = rate
        return self._rate_cache[key]

    def hs_splitting(self, dm31: float, hs_reading: float | str = 'matched') -> float:
        """Splitting |Dm31| [eV^2] at which the hypersurfaces are read for the hypothesis dm31 (App. E.1)."""
        if not isinstance(hs_reading, str):
            return float(hs_reading)
        if hs_reading not in ('matched', 'splitting'):
            raise ValueError(f'unknown hypersurface reading {hs_reading!r}')
        if dm31 > 0 or hs_reading == 'splitting':
            return abs(dm31)
        return -dm31 + 2 * (1 - self.p.s12sq) * self.p.dm21

    def _hs_factors(self, splitting, params):
        """Hypersurface factors (channel, bin) at the splitting, linear between the nodes of the table."""
        grid = self.hs_grid
        s = float(np.clip(splitting, grid[0], grid[-1]))
        k = int(np.clip(np.searchsorted(grid, s), 1, grid.size - 1))
        t = (s - grid[k - 1]) / (grid[k] - grid[k - 1])
        delta = np.array([1.0] + [params[n] - NOMINAL[n] for n in DETECTOR])
        return ((1 - t) * self._hs[k - 1] + t * self._hs[k]) @ delta

    def _expected(self, params, dcp, splitting):
        """Expected counts and MC variance per bin at delta_CP dcp, hypersurfaces at the splitting [eV^2]."""
        c = self._rates(params['dm31'], params['s23sq'], dcp) * np.exp(params['delta_gamma'] * self._log_e)
        qe, res = params['ma_qe'], params['ma_res']
        lin_qe, quad_qe, lin_res, quad_res = self._axial
        c[self._axial_rows] *= ((1 + lin_qe * qe + quad_qe * qe ** 2)
                                * (1 + lin_res * res + quad_res * res ** 2))
        c[self._nc] *= 1 + params['nc_norm']
        h = np.bincount(self._cell, c, len(HS_FILES) * N_BINS).reshape(len(HS_FILES), N_BINS)
        h2 = np.bincount(self._cell, c * c, len(HS_FILES) * N_BINS).reshape(len(HS_FILES), N_BINS)
        f = self._hs_factors(splitting, params)
        a_nu, a_mu = 1 + params['aeff_norm'], 1 + params['muon_norm']
        counts = a_nu * np.sum(f * h, axis=0) + a_mu * self.muons
        variance = a_nu ** 2 * np.sum(f * f * h2, axis=0) + a_mu ** 2 * self.muon_variance
        return counts, variance

    def predict(self, params: Mapping[str, float],
                hs_reading: float | str = 'matched') -> tuple[np.ndarray, np.ndarray]:
        """Expected counts and Monte Carlo variance sigma^2_MC per analysis bin (App. E.1)."""
        return self._expected(params, self.dcp, self.hs_splitting(params['dm31'], hs_reading))

    def counts(self, y: float, s23sq: float, dcp: float) -> np.ndarray:
        """Expected counts per analysis bin at the signed Dm31 y [eV^2], with the nuisance parameters and the
        hypersurface reading of the constructor (App. E.1)."""
        params = self.nuisances | {'dm31': y, 's23sq': s23sq}
        return self._expected(params, dcp, self.hs_splitting(y, self.hs_reading))[0]

    def chi2(self, params: Mapping[str, float], data: np.ndarray | None = None,
             center: Mapping[str, float] | None = None, hs_reading: float | str = 'matched') -> float:
        """Modified Pearson chi^2 of App. E.1 plus the pull terms of PRIORS, centered at center (default: the
        nominal values). The hypersurfaces are linear in the detector parameters and can drive an expected count
        below zero far from the data; the count is then held at MIN_COUNT, which keeps the chi^2 positive."""
        counts, variance = self.predict(params, hs_reading)
        counts = np.maximum(counts, MIN_COUNT)
        data = self.observed if data is None else data
        center = NOMINAL | (center or {})
        pulls = sum(((params[k] - center[k]) / width) ** 2 for k, width in PRIORS.items())
        return float(np.sum((counts - data) ** 2 / (counts + variance)) + pulls)

    def fit(self, ordering: str, s23sq_starts: tuple[float, ...] = S23SQ_STARTS,
            fixed: Mapping[str, float] | None = None, init: Mapping[str, float] | None = None,
            data: np.ndarray | None = None, center: Mapping[str, float] | None = None,
            hs_reading: float | str = 'matched', hesse: bool = False, tol: float = TOL) -> FitResult:
        """Minimum of the chi^2 of App. E.1 in the ordering 'NO' or 'IO' by MIGRAD, started at each
        sin^2 theta23 of s23sq_starts.

        fixed: parameters held at the given values; init: initial values of the other parameters (default:
        the nominal nuisance parameters and |dm31| = DM31_START); data, center, and hs_reading as in chi2.
        hesse adds the HESSE errors.
        """
        sign = {'NO': 1.0, 'IO': -1.0}[ordering]
        fixed = fixed or {}
        if 's23sq' in fixed:
            s23sq_starts = (fixed['s23sq'],)
        best, n_calls, t0 = None, 0, time.perf_counter()
        for s0 in s23sq_starts:
            point = NOMINAL | {'dm31': sign * DM31_START} | (init or {}) | {'s23sq': s0} | fixed
            m = Minuit(lambda *q: self.chi2(dict(zip(NAMES, q)), data, center, hs_reading),
                       *[point[k] for k in NAMES], name=NAMES)
            m.errordef = Minuit.LEAST_SQUARES
            m.tol = tol
            for k in NAMES:
                m.errors[k] = STEPS[k]
                m.limits[k] = LIMITS[k]
            if sign < 0:
                m.limits['dm31'] = (-LIMITS['dm31'][1], -LIMITS['dm31'][0])
            for k in fixed:
                m.fixed[k] = True
            m.migrad()
            n_calls += m.nfcn
            if best is None or m.fval < best.fval:
                best = m
        errors = None
        if hesse:
            n_migrad = best.nfcn
            best.hesse()
            n_calls += best.nfcn - n_migrad
            errors = {k: 0.0 if k in fixed else float(best.errors[k]) for k in NAMES}
        return FitResult(chi2=float(best.fval), values={k: float(best.values[k]) for k in NAMES},
                         errors=errors, valid=bool(best.valid), n_calls=n_calls,
                         seconds=time.perf_counter() - t0)

    def asimov(self, x: float, nuisances: Mapping[str, float],
               s23sq: float = S23SQ) -> tuple[np.ndarray, np.ndarray]:
        """Normal-ordering Asimov counts and MC variance, hypersurfaces at the true splitting (App. E.1)."""
        return self.predict(truth(x, nuisances, s23sq), hs_reading=x)

    def asimov_io_fit(self, x: float, nuisances: Mapping[str, float], s23sq: float = S23SQ,
                      hs_reading: float | str | None = None, tol: float = TOL) -> FitResult:
        """Inverted-ordering fit of all 13 parameters to the Asimov data at x (App. E.1, fig_full_fit_validation).

        The pull terms are centered at the true values, the fit starts from them at the intercept B_START,
        and the hypersurfaces are read at the true splitting unless hs_reading says otherwise. The intercept
        is params.intercept(values['dm31'], x).
        """
        center = truth(x, nuisances, s23sq)
        data, _ = self.asimov(x, nuisances, s23sq)
        return self.fit('IO', init=center | {'dm31': B_START * 1e-6 - x}, data=data, center=center,
                        hs_reading=x if hs_reading is None else hs_reading, tol=tol)

    def _derivative(self, point, name, hs_reading):
        h = H_DERIVATIVE.get(name, 1e-4)
        up = self.predict(point | {name: point[name] + h}, hs_reading)[0]
        down = self.predict(point | {name: point[name] - h}, hs_reading)[0]
        return (up - down) / (2 * h)

    def weighted_bin_average(self, x: float, nuisances: Mapping[str, float], profiled: tuple[str, ...] = (),
                             s23sq: float = S23SQ, tol: float = 1e-11,
                             max_iter: int = 30) -> tuple[dict, int]:
        """Weighted bin average on the fitter's expectation for the Asimov data at x (App. E.2).

        The Gauss-Newton iteration of App. B.3 with the parameters named in profiled projected out through
        Eqs. (B.5)-(B.6), with the pull terms of the fit and the modified Pearson weights
        1 / (N_i + sigma^2_MC,i) at the truth; the other parameters stay at the truth, and the hypersurfaces
        are read at the true splitting. WBA_CASES lists the three evaluations of App. E.2. Returns the
        solution and the number of iterations; raises RuntimeError if the step in Dm31 does not fall below
        tol [eV^2] within max_iter iterations.
        """
        center = truth(x, nuisances, s23sq)
        data, variance = self.asimov(x, nuisances, s23sq)
        weights = 1.0 / (data + variance)
        prior_inv = np.diag([PRIORS[k] ** -2 if k in PRIORS else 0.0 for k in profiled])
        theta0 = np.array([center[k] for k in profiled])
        point = center | {'dm31': B_START * 1e-6 - x}
        for n_iter in range(1, max_iter + 1):
            resid = data - self.predict(point, hs_reading=x)[0]
            s = self._derivative(point, 'dm31', x)
            jac = np.array([self._derivative(point, k, x) for k in profiled]).reshape(len(profiled), N_BINS).T
            theta = np.array([point[k] for k in profiled])
            dy, dtheta = profiled_step(resid, s, jac, weights, prior_inv, prior_inv @ (theta - theta0))
            point = point | {'dm31': point['dm31'] + dy} | {k: point[k] + d for k, d in zip(profiled, dtheta)}
            if abs(dy) < tol:
                return point, n_iter
        raise RuntimeError(f'weighted bin average at x = {x} not converged in {max_iter} iterations')

    def profiled_grid(self, s23sq_grid: np.ndarray = S23SQ_GRID, dm32_grid: np.ndarray = DM32_GRID,
                      tol: float = TOL) -> np.ndarray:
        """chi^2 of the normal ordering on the observed data with sin^2 theta23 and Dm32 fixed at each grid point and
        the 11 nuisance parameters fitted (fig_deepcore_reproduction); shape (len(s23sq_grid), len(dm32_grid))."""
        return np.array([[self.fit('NO', fixed={'s23sq': s, 'dm31': m + self.p.dm21}, tol=tol).chi2
                          for m in dm32_grid] for s in s23sq_grid])


def contour(s23sq_grid: np.ndarray, dm32_grid: np.ndarray, chi2: np.ndarray, chi2_min: float,
            level: float = 4.61) -> np.ndarray:
    """The line Delta chi^2 = level (90% CL for two parameters) of a chi^2 known on a grid, as (n, 2) points (sin^2
    theta23, Dm32 [eV^2]) (fig_deepcore_reproduction, App. E.3).

    The chi^2 is interpolated by piecewise cubic polynomials on a 0.1% mesh of the grid and the level line is
    traced through it; the longest closed line is returned.
    """
    s = np.linspace(s23sq_grid[0], s23sq_grid[-1], 1001)
    m = np.linspace(dm32_grid[0], dm32_grid[-1], 1001)
    surface = RegularGridInterpolator((s23sq_grid, dm32_grid), chi2 - chi2_min, method='cubic')
    values = surface(np.stack(np.meshgrid(s, m, indexing='ij'), axis=-1))
    lines = contour_generator(x=m, y=s, z=values).lines(level)
    return max(lines, key=len)[:, ::-1]
