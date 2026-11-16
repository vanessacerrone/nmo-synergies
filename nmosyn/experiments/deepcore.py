"""IceCube DeepCore: expected counts with the detector response of the public release in tables (App. A.4).

The release of the 9-year golden sample (PRD 108, 012014 (2023), doi:10.7910/DVN/B4RITM) gives the
simulated events of the analysis with their flavor and charge, true and reconstructed energy and
direction, and a weight in GeV cm^2 sr. The reconstructed variables are given as the centers of the
published analysis bins. From the events reconstructed in the 10 x 10 analysis bins (event classes
merged) we build, for each channel (nu_e, nu_mu, and nu_tau CC, and NC by flavor) and separately for
neutrinos and antineutrinos,
  - the acceptance R: summed weight per unit ln E and cos theta_z in 120 x 60 true cells over
    1-1000 GeV and the full sky;
  - the energy migration K_E: fraction of the events of each of 30 true-energy cells in each
    analysis energy bin;
  - the zenith migration K_C: the same for 30 x 20 true (E, cos theta_z) cells and the analysis
    zenith bins.
The migrations are filled with the weight times the Honda flux of the flavor of each event (nu_mu for
nu_tau), so that inside a cell they follow the spectrum of the observed events. Every acceptance cell
lies inside one migration cell, so the three tables define a response that is constant on each
acceptance cell. Eq. (A.3) sums it over a fine grid of 300 x 180 true cells, with the Honda flux at the
South Pole and the three-flavor probabilities along the path of each cell center (earth.layers:
atmosphere, ice, and PREM; Sec. 4). A fine cell can straddle up to 2 x 2 acceptance cells: its
R_c Delta ln E_c Delta cos theta_z,c K_E K_C is the integral of the tables over the cell, in which each
acceptance cell enters with the area it shares with the fine cell. On 1500 x 900 cells the intercepts change
by up to 0.034e-6 eV^2 (compute/grid.py), almost all from cos theta_z: near the horizon the path length,
atmosphere included, changes fast with the direction. Coarser grids, or grids whose cos theta_z cells do not
divide the acceptance cells, sample the oscillation too sparsely and move the intercepts by up to 1e-6 eV^2.

counts() includes the normalization to the 21402 neutrinos of the sample (21914 observed events
minus 512 expected atmospheric muons), with all channels at the NuFIT 6.0 truth. It is the
exposure of the detector, the same for every model of this module, and sets only the scale of the
chi^2 of Eq. (2.1).

`DeepCore.gaussian` keeps the acceptance and replaces the tabulated migration with Gaussian kernels in ln E and cos
theta_z (Sec. 4, fig_deepcore_resolution and fig_deepcore_energy_window, App. A.4), with the resolution extracted from
the simulation of the release by `resolution` and the variance of the analysis bins, which it includes, removed in
quadrature.
"""
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from ..flux import SOUTH_POLE, Honda
from ..earth import MAX_KM, WATER_YE, layers
from ..osc import path_amplitudes, probabilities
from ..params import DCP_NO, NUFIT, S23SQ, X0, Params
from ..paths import EXTERNAL


def _readonly(values) -> np.ndarray:
    """Float copy of values that cannot be modified in place (constants and arrays shared between models)."""
    out = np.array(values, dtype=float)
    out.flags.writeable = False
    return out


RELEASE = EXTERNAL / 'icecube_b4ritm'
FILES = (('cc', 'mc_nue_cc.tab'), ('cc', 'mc_numu_cc.tab'), ('cc', 'mc_nutau_cc.tab'),
         ('nc', 'mc_nu_nc.tab'))
COLUMNS = ('pdg', 'true_energy', 'true_coszen', 'reco_energy', 'reco_coszen', 'weight')

# published analysis bins: uniform in ln E over 6.31-158.49 GeV, and cos theta_z from -1 to 0.1
E_EDGES = _readonly([6.31, 8.45862141, 11.33887101, 15.19987592, 20.37559363, 27.3136977, 36.61429921,
                     49.08185342, 65.79474104, 88.19854278, 158.49])
COSZ_EDGES = _readonly(np.linspace(-1.0, 0.1, 11))

E_TRUE = (1.0, 1000.0)              # GeV, range of the true cells
FINE_CELLS = (300, 180)             # fine grid of Eq. (A.3), aligned with the acceptance cells
ACCEPTANCE_CELLS = (120, 60)
MIGRATION_CELLS = (30, 20)          # each holds 4 x 3 acceptance cells
RESOLUTION_E_EDGES = _readonly([5.0, 8.0, 12.0, 18.0, 27.0, 40.0, 60.0, 90.0, 130.0, 200.0])
DEPTH_KM = 2.0                      # detector depth below the surface of the ice
OVERBURDEN = 0.92 * WATER_YE        # rho*Ye of the ice above it [g/cm^3]
N_EVENTS = 21402                    # 21914 observed events minus 512 expected atmospheric muons
KINDS = ('nu', 'nubar')
SOURCES = ('nue', 'numu', 'numu')   # Honda flux behind each flavor (e, mu, tau); nu_tau appears from nu_mu
MAX_CACHED = 4

# scans of fig_deepcore_resolution and fig_deepcore_energy_window (Sec. 4, App. A.4)
SIGMA_SCAN = _readonly(np.arange(0.05, 0.6001, 0.025))     # sigma_lnE of the resolution scan
E_MAX_SCAN = _readonly(np.geomspace(20.0, 158.49, 19))     # upper edge of the window [GeV]
E_MIN_SCAN = _readonly(np.geomspace(6.31, 25.0, 13))       # lower edge of the window [GeV]


def read_release(name: str, path: Path = RELEASE) -> pd.DataFrame:
    """Simulated events of one file of the release, e.g. 'mc_numu_cc.tab' (the columns COLUMNS)."""
    return pd.read_csv(path / name, sep='\t', usecols=list(COLUMNS))


def _species(flavor: int, kind: str) -> str:
    """Honda species that feeds flavor 0, 1, 2 (e, mu, tau), for kind 'nu' or 'nubar'."""
    return SOURCES[flavor] + ('bar' if kind == 'nubar' else '')


def _cell(values, edges):
    """Index of the cell of edges that contains each value; values outside go to the outer cells."""
    return np.clip(np.digitize(values, edges) - 1, 0, len(edges) - 2)


def _overlap(n_fine: int, n_coarse: int) -> np.ndarray:
    """Fraction (n_fine, n_coarse) of each of n_fine equal cells that lies in each of n_coarse equal cells.

    Both grids divide the same interval. The fractions are computed in units of the fine cells, where
    the coarse edges of this module are exact binary fractions, so they are exact (0, 1/4, 1/2, 3/4, 1).
    """
    fine = np.arange(n_fine + 1.0)
    coarse = np.arange(n_coarse + 1.0) * (n_fine / n_coarse)
    shared = np.minimum(fine[1:, None], coarse[None, 1:]) - np.maximum(fine[:-1, None], coarse[None, :-1])
    return np.clip(shared, 0.0, None)


def _quantiles(values: np.ndarray, weights: np.ndarray, q) -> np.ndarray:
    """Weighted quantiles q: the smallest value whose normalized cumulative weight reaches q."""
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    index = np.minimum(np.searchsorted(cumulative, q, side='left'), len(values) - 1)
    return values[order][index]


@dataclass(frozen=True)
class Resolution:
    """Widths in ln E and cos theta_z against the true energy, given at the node energies e [GeV].

    lne, cosz are the widths of the release, measured on the centers of the analysis bins, so they include the
    bins; bins_lne, bins_cosz are the variances that the bins add, the mean of w^2 / 12 over the events, with w the
    width of the analysis bin of each event. With binned=False the methods return the widths without the bins,
    sqrt(width^2 - bins), the widths of the Gaussian kernels, which add the bins back by integrating over them.
    Between the nodes the widths are interpolated linearly in ln E; outside they keep the outer values.
    """

    e: np.ndarray
    lne: np.ndarray
    cosz: np.ndarray
    bins_lne: np.ndarray
    bins_cosz: np.ndarray

    def sigma_lne(self, e_gev: np.ndarray, binned: bool = True) -> np.ndarray:
        """Width of ln(E_reco / E) at the true energies e_gev [GeV], with or without the bins."""
        lne = self.lne if binned else np.sqrt(self.lne**2 - self.bins_lne)
        return np.interp(np.log(e_gev), np.log(self.e), lne)

    def sigma_cosz(self, e_gev: np.ndarray, binned: bool = True) -> np.ndarray:
        """Width of cos theta_z,reco - cos theta_z at the true energies e_gev [GeV], with or without the bins."""
        cosz = self.cosz if binned else np.sqrt(self.cosz**2 - self.bins_cosz)
        return np.interp(np.log(e_gev), np.log(self.e), cosz)


def resolution(events: pd.DataFrame) -> Resolution:
    """Energy and zenith resolution of DeepCore from the nu_mu CC events of the release (App. A.4).

    In the true-energy bins RESOLUTION_E_EDGES, half the central 68% interval (16th to 84th
    percentile, weighted with the event weight) of ln(E_reco / E) and of cos theta_z,reco - cos theta_z,
    at the geometric bin centers; neutrinos and antineutrinos together. The reconstructed values of the
    release are the centers of the analysis bins, so the widths include the size of the bins.
    sigma_lnE is 0.30-0.36 (Sec. 4); sigma_cosz is 0.24 at 6 GeV and 0.08 at 160 GeV. Without the bins
    (Resolution, binned=False) sigma_lnE is 0.29-0.34 and sigma_cosz 0.24 at 6 GeV and 0.08 at 160 GeV.
    """
    e, w = events['true_energy'].to_numpy(), events['weight'].to_numpy()
    d_lne = np.log(events['reco_energy'].to_numpy() / e)
    d_cosz = events['reco_coszen'].to_numpy() - events['true_coszen'].to_numpy()
    # variance of the uniform distribution over the analysis bin of each event (the last energy bin is twice as wide)
    var_lne = np.diff(np.log(E_EDGES))[_cell(events['reco_energy'].to_numpy(), E_EDGES)] ** 2 / 12
    var_cosz = np.diff(COSZ_EDGES)[_cell(events['reco_coszen'].to_numpy(), COSZ_EDGES)] ** 2 / 12

    def half_width(d, m):
        lo, hi = _quantiles(d[m], w[m], [0.16, 0.84])
        return 0.5 * (hi - lo)

    bins = [(e >= lo) & (e < hi) for lo, hi in zip(RESOLUTION_E_EDGES[:-1], RESOLUTION_E_EDGES[1:])]
    return Resolution(e=_readonly(np.sqrt(RESOLUTION_E_EDGES[:-1] * RESOLUTION_E_EDGES[1:])),
                      lne=_readonly([half_width(d_lne, m) for m in bins]),
                      cosz=_readonly([half_width(d_cosz, m) for m in bins]),
                      bins_lne=_readonly([np.average(var_lne[m], weights=w[m]) for m in bins]),
                      bins_cosz=_readonly([np.average(var_cosz[m], weights=w[m]) for m in bins]))


def gaussian_migration(e: np.ndarray, cosz: np.ndarray, sigma_lne: np.ndarray, sigma_cosz: np.ndarray,
                       e_edges: np.ndarray, cosz_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities that a true cell (e [GeV], cosz) is reconstructed in each energy and zenith bin.

    Gaussians of widths sigma_lne in ln E and sigma_cosz in cos theta_z (per cell), integrated over the
    bin edges e_edges [GeV] and cosz_edges (App. A.4); the probability outside the edges is lost.
    Returns k_e (n_cells, n_E) and k_c (n_cells, n_C); the probability of bin (i_E, i_C) is their product.
    """
    def integral(x, sigma, edges):
        return np.diff(ndtr((edges[None, :] - x[:, None]) / sigma[:, None]), axis=1)

    return (integral(np.log(e), sigma_lne, np.log(np.asarray(e_edges, dtype=float))),
            integral(cosz, sigma_cosz, np.asarray(cosz_edges, dtype=float)))


def edges_one_sigma(sigma_lne: float) -> np.ndarray:
    """Energy edges of the resolution scan: bins about sigma_lne wide in ln E, at least two (fig_deepcore_resolution,
    fig_deepcore_energy_window)."""
    n = max(int(round(np.log(E_EDGES[-1] / E_EDGES[0]) / sigma_lne)), 2)
    return np.geomspace(E_EDGES[0], E_EDGES[-1], n + 1)


def window_edges(e_min: float = E_EDGES[0], e_max: float = E_EDGES[-1]) -> np.ndarray:
    """Published energy edges cut to the window e_min-e_max [GeV] (Sec. 4, fig_deepcore_resolution and
    fig_deepcore_energy_window, App. A.4).

    The ends of the window become the outer edges. A published edge within 0.1% of e_min or e_max is
    dropped, so that no bin is a sliver.
    """
    inner = E_EDGES[(E_EDGES > 1.001 * e_min) & (E_EDGES < 0.999 * e_max)]
    return np.concatenate([[e_min], inner, [e_max]])


def _tables(events: pd.DataFrame, interaction: str, honda: Honda) -> dict:
    """Acceptance and migrations of one file of the release, per (interaction, kind, flavor) (App. A.4).

    Returns {key: (density, k_e, k_c)} on the acceptance cells, ordered (E, cos theta_z) with cos theta_z
    fastest: density (n_true,) the summed weight per unit ln E and cos theta_z [GeV cm^2 sr]; k_e and k_c
    (n_true, 10) the fractions of the events in the analysis energy and zenith bins, those of the
    migration cell that holds the acceptance cell.
    """
    e, cosz = events['true_energy'].to_numpy(), events['true_coszen'].to_numpy()
    w, pdg = events['weight'].to_numpy(), events['pdg'].to_numpy()
    flavor = events['pdg'].abs().map({12: 0, 14: 1, 16: 2}).to_numpy()
    n_e, n_c = len(E_EDGES) - 1, len(COSZ_EDGES) - 1
    i_e = np.digitize(events['reco_energy'].to_numpy(), E_EDGES) - 1
    i_c = np.digitize(events['reco_coszen'].to_numpy(), COSZ_EDGES) - 1
    inside = (i_e >= 0) & (i_e < n_e) & (i_c >= 0) & (i_c < n_c) & (e >= E_TRUE[0]) & (e < E_TRUE[1])

    (n_te, n_tc), (n_me, n_mc) = ACCEPTANCE_CELLS, MIGRATION_CELLS
    edges_e = np.geomspace(*E_TRUE, n_te + 1)
    edges_c = np.linspace(-1.0, 1.0, n_tc + 1)
    area = (np.diff(np.log(edges_e))[:, None] * np.diff(edges_c)[None, :]).ravel()

    def migration_cells(t_e, t_c):
        """Energy and (energy, zenith) migration cells that hold the acceptance cells (t_e, t_c)."""
        m_e = t_e // (n_te // n_me)
        return m_e, m_e * n_mc + t_c // (n_tc // n_mc)

    def fractions(cells, bins, n_cells, n_bins, rate):
        rows = np.bincount(cells * n_bins + bins, rate, n_cells * n_bins).reshape(n_cells, n_bins)
        total = rows.sum(axis=1, keepdims=True)
        return rows / np.where(total > 0, total, 1.0)

    row_e, row_c = migration_cells(*np.divmod(np.arange(n_te * n_tc), n_tc))
    out = {}
    for kind in KINDS:
        for fl in np.unique(flavor):
            m = inside & ((pdg < 0) == (kind == 'nubar')) & (flavor == fl)
            t_e, t_c = _cell(e[m], edges_e), _cell(cosz[m], edges_c)
            m_e, m_c = migration_cells(t_e, t_c)
            rate = w[m] * honda(_species(fl, kind), e[m], cosz[m])
            density = np.bincount(t_e * n_tc + t_c, w[m], minlength=area.size) / area
            k_e = fractions(m_e, i_e[m], n_me, n_e, rate)
            k_c = fractions(m_c, i_c[m], n_me * n_mc, n_c, rate)
            out[(interaction, kind, int(fl))] = (density, k_e[row_e], k_c[row_c])
    return out


def _earth(e, path, y, p):
    """Amplitudes along the paths of the fine cells, (neutrinos, antineutrinos), for the signed Dm31 y.

    path: (lengths, rhoye) of the layers of each cell, from earth.layers.
    """
    return tuple(path_amplitudes(e, *path, y, p, kind == 'nubar') for kind in KINDS)


def _detector_flux(amplitudes, s23sq, dcp, flux, sources):
    """Flux of each flavor at the detector per fine cell, {(kind, flavor): (n_cells,)}, from the sources."""
    out = {}
    for kind, amp in zip(KINDS, amplitudes):
        prob = probabilities(amp, s23sq, dcp, antineutrino=kind == 'nubar')
        for beta in range(3):
            out[(kind, beta)] = sum(flux[_species(a, kind)] * prob[:, beta, a] for a in sources)
    return out


def _fold(rates: dict, kernels: dict, overlap: tuple) -> np.ndarray:
    """Events per analysis bin without the normalization, Eq. (A.3).

    rates: flux at the detector per fine cell, {(kind, flavor): (n_cells,)}. kernels: {(interaction, kind,
    flavor): (density, k_e, k_c)} on the true cells of the response. overlap: (o_e, o_c), the lengths in
    ln E and cos theta_z that each fine cell (rows) shares with each true cell (columns).
    """
    o_e, o_c = overlap
    shape = (o_e.shape[0], o_c.shape[0])
    per_cell = {s: (o_e.T @ rates[s].reshape(shape) @ o_c).ravel() for s in {key[1:] for key in kernels}}
    return sum((k_e * (density * per_cell[key[1:]])[:, None]).T @ k_c
               for key, (density, k_e, k_c) in kernels.items()).ravel()


class DeepCore:
    """Expected DeepCore neutrino events per analysis bin for any Dm31, theta23, and delta_CP (Eq. A.3).

    channels: 'numu' for nu_mu CC disappearance alone (nu_mu flux times P_mumu), 'all' for every channel
    with its appearance. density_scale multiplies every density (0 gives vacuum); max_km is the largest
    length of the layers of constant density along each path (earth.layers); fine_cells the numbers of fine
    cells in ln E and in cos theta_z (Eq. A.3).

    Attributes:
      n_bins, e_edges, cosz_edges  analysis bins, ordered (i_E, i_C) with i_C fastest
      e, cosz, dlne, dcosz         fine cells (n_cells,): center energy [GeV] and cos theta_z, and the
                                   sizes in ln E and cos theta_z; ordered (E, cos theta_z), cos theta_z
                                   fastest
      flux                         Honda flux per fine cell, keys 'nue', 'numu', 'nuebar', 'numubar'
                                   [(m^2 s sr GeV)^-1]
      acceptance                   {(interaction, kind, flavor): acceptance R integrated over each fine
                                   cell (n_cells,)} [GeV cm^2 sr]; interaction 'cc' or 'nc', kind 'nu' or
                                   'nubar', flavor 0, 1, 2 (e, mu, tau) of the neutrino at the detector
      norm                         neutrino events per unit of response x flux x probability, the same
                                   for every model; included in counts()
      resolution                   `Resolution` of the release, used by `gaussian`
    The response matrix of a channel on the fine cells is `response(key)`.
    """

    def __init__(self, channels: str = 'numu', p: Params = NUFIT, density_scale: float = 1.0,
                 max_km: float = MAX_KM, fine_cells: tuple[int, int] = FINE_CELLS):
        if channels not in ('numu', 'all'):
            raise ValueError(f"channels must be 'numu' or 'all', not {channels!r}")
        self.channels = channels
        self.p = p
        self.density_scale = density_scale
        self.max_km = max_km
        self.e_edges, self.cosz_edges = E_EDGES, COSZ_EDGES
        self.n_bins = (len(E_EDGES) - 1) * (len(COSZ_EDGES) - 1)

        self.fine_cells = n_e, n_c = fine_cells
        e_grid = np.geomspace(*E_TRUE, n_e + 1)
        c_grid = np.linspace(-1.0, 1.0, n_c + 1)
        dlne, dcosz = np.diff(np.log(e_grid)), np.diff(c_grid)
        self.e = _readonly(np.repeat(np.sqrt(e_grid[:-1] * e_grid[1:]), n_c))
        self.cosz = _readonly(np.tile(0.5 * (c_grid[:-1] + c_grid[1:]), n_e))
        self.dlne = _readonly(np.repeat(dlne, n_c))
        self.dcosz = _readonly(np.tile(dcosz, n_e))
        honda = Honda(SOUTH_POLE)
        self.flux = {s: _readonly(honda(s, self.e, self.cosz)) for s in ('nue', 'numu', 'nuebar', 'numubar')}

        tables = {}
        for interaction, name in FILES:
            events = read_release(name)
            tables.update(_tables(events, interaction, honda))
            if name == 'mc_numu_cc.tab':
                self.resolution = resolution(events)
        self._overlap = (_overlap(n_e, ACCEPTANCE_CELLS[0]) * dlne[:, None],
                         _overlap(n_c, ACCEPTANCE_CELLS[1]) * dcosz[:, None])

        self._path = layers(self.cosz, DEPTH_KM, OVERBURDEN, max_km, density_scale)
        truth = _earth(self.e, layers(self.cosz, DEPTH_KM, OVERBURDEN), X0, NUFIT)
        rates = _detector_flux(truth, S23SQ, DCP_NO, self.flux, (0, 1))
        self.norm = N_EVENTS / _fold(rates, tables, self._overlap).sum()

        self._kernels = {k: v for k, v in tables.items() if channels == 'all' or (k[0] == 'cc' and k[2] == 1)}
        o_e, o_c = self._overlap
        self.acceptance = {k: _readonly((o_e @ density.reshape(ACCEPTANCE_CELLS) @ o_c.T).ravel())
                           for k, (density, _, _) in self._kernels.items()}
        self._sources = (0, 1) if channels == 'all' else (1,)
        self._cache = {}

    def _amplitudes(self, y):
        key = float(y)
        if key not in self._cache:
            if len(self._cache) >= MAX_CACHED:
                del self._cache[next(iter(self._cache))]
            self._cache[key] = _earth(self.e, self._path, key, self.p)
        return self._cache[key]

    def counts(self, y: float, s23sq: float, dcp: float) -> np.ndarray:
        """Expected neutrino events per analysis bin, Eq. (A.3); y is the signed Dm31 [eV^2]."""
        rates = _detector_flux(self._amplitudes(y), s23sq, dcp, self.flux, self._sources)
        return self.norm * _fold(rates, self._kernels, self._overlap)

    def response(self, key: tuple) -> np.ndarray:
        """Response matrix (n_cells, n_bins) of one channel on the fine cells [GeV cm^2 sr] (Eq. A.3).

        counts() is norm times the sum over the channels of the flux at the detector @ response(key).
        """
        density, k_e, k_c = self._kernels[key]
        o_e, o_c = self._overlap
        per_cell = density[:, None, None] * k_e[:, :, None] * k_c[:, None, :]
        per_cell = per_cell.reshape(o_e.shape[1], o_c.shape[1], self.n_bins)
        return np.einsum('ea,cb,abi->eci', o_e, o_c, per_cell, optimize=True).reshape(-1, self.n_bins)

    def gaussian(self, e_edges: np.ndarray = E_EDGES, sigma_lne: float | None = None,
                 cosz_edges: np.ndarray = COSZ_EDGES, cosz_scale: float = 1.0) -> 'DeepCore':
        """New model with Gaussian kernels in place of the tabulated migration (Sec. 4, App. A.4).

        sigma_lne: width of the kernel in ln E, or None for that of the release (`resolution`) without the
        analysis bins; the zenith width is that of the release without the bins, times cosz_scale. The widths of
        the release are measured on bin centers, and the kernels add the bins again by the integral over the edges,
        so the bins are removed from them first (`Resolution`, binned=False). The kernels are evaluated at the
        centers of the fine cells, integrated over the reconstructed edges e_edges [GeV] and cosz_edges, and are the
        same for every channel. Acceptance, fluxes, normalization, and oscillation settings are those of this model.
        """
        if sigma_lne is None:
            sigma = self.resolution.sigma_lne(self.e, binned=False)
        else:
            sigma = np.full(self.e.size, float(sigma_lne))
        sigma_cosz = cosz_scale * self.resolution.sigma_cosz(self.e, binned=False)
        k_e, k_c = gaussian_migration(self.e, self.cosz, sigma, sigma_cosz, e_edges, cosz_edges)
        area = self.dlne * self.dcosz
        model = copy.copy(self)
        model.e_edges, model.cosz_edges = _readonly(e_edges), _readonly(cosz_edges)
        model.n_bins = k_e.shape[1] * k_c.shape[1]
        model.flux, model.acceptance = dict(self.flux), dict(self.acceptance)
        # the true cells of the Gaussian response are the fine cells themselves
        model._overlap = (np.diag(self.dlne[::self.fine_cells[1]]), np.diag(self.dcosz[:self.fine_cells[1]]))
        model._kernels = {k: (acc / area, k_e, k_c) for k, acc in self.acceptance.items()}
        model._cache = dict(self._cache)
        return model
