"""Every number of the paper that the code computes, in paper order, against the value the paper quotes.

One line per number: its id; its location, the section and the line of the TeX source (main:
nmo_sumrule_atmo_short.tex, app: appendices_body.tex); the quantity; the value quoted in the paper; the value
computed here, at full precision; and MOVED where the computed value no longer rounds to the quoted one, no
longer satisfies a quoted bound, or no longer lies inside a quoted band. Statements without a number get no
mark. Numbers that need a value no result file provides are listed at the end with the reason. Intercepts b
are in 1e-6 eV^2, splittings in 1e-3 eV^2, shares in %. The same table goes to results/numbers.json.

The script reads results/*.npz and data/published_best_fits.json and evaluates the closed forms of
nmosyn.params and nmosyn.synergy, so it runs in seconds. It is not called numbers.py: that name would shadow
the module of the standard library that numpy imports. Run from the repository root: python paper_numbers.py
"""
import json
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from nmosyn import earth, fit, synergy
from nmosyn.earth import SHELLS, prem_rhoye
from nmosyn.experiments import juno, lbl_spectra
from nmosyn.experiments.deepcore import COSZ_EDGES, DEPTH_KM, E_EDGES
from nmosyn.experiments.juno import L_NOMINAL_KM
from nmosyn.osc import A_PER_RHOYE_GEV
from nmosyn.params import (DCP_IO, DCP_NO, NUFIT, PHASE, S23SQ, S23SQ_RANGE, X0, X_GRID, b_lbl, b_sbr, dm2_phi_mumu,
                           intercept)
from nmosyn.paths import DATA, RESULTS

PRODUCERS = {'icfit': 'icecube_fit'}     # result files whose compute script has another name
S23SQ_BAR = 0.56                         # true sin^2 theta23 of the delta_CP bars of fig_intercept_ranges
LEAD = 2 * (1 - NUFIT.s12sq) * NUFIT.dm21 * 1e6      # 2 c12^2 Dm21, the common part of Eq. (4.1)


class Missing(Exception):
    """A value that no result file provides."""


class Results:
    """The cached results/<name>.npz, loaded on first use; a missing file raises Missing."""

    def __init__(self) -> None:
        self._files: dict[str, dict[str, np.ndarray]] = {}

    def __getitem__(self, name: str) -> dict[str, np.ndarray]:
        if name not in self._files:
            path = RESULTS / f'{name}.npz'
            if not path.exists():
                raise Missing(f'results/{name}.npz not found (python -m compute.{PRODUCERS.get(name, name)})')
            with np.load(path) as f:
                self._files[name] = {k: f[k] for k in f.files}
        return self._files[name]


@dataclass(frozen=True)
class Spec:
    """A number of the paper and the function of the results that computes it.

    quote: the value as printed; its numbers are compared with the computed ones, each to half a unit of its
    last digit unless paper and tol are given. check: 'round' (the default when the quote has numbers),
    'bound' (no computed absolute value may exceed the quoted number), 'inside' (the value is (b, b_lo, b_hi)
    for one or more points), or '' (a statement without a number).
    """

    id: str
    line: int
    what: str
    quote: str
    value: Callable[['Results'], object]
    check: str | None = None
    paper: tuple[float, ...] | None = None
    tol: float | None = None


@dataclass(frozen=True)
class Number:
    """A number of the paper with its computed value."""

    id: str
    where: str
    what: str
    quote: str
    new: tuple[float, ...]
    paper: tuple[float, ...]
    tol: tuple[float, ...]
    check: str

    @property
    def moved(self) -> bool | None:
        """True where the computed value no longer agrees with the quoted one; None for statements."""
        if not self.check:
            return None
        new = np.array(self.new)
        if self.check == 'inside':
            b, lo, hi = new.reshape(-1, 3).T
            return bool(np.any((b < lo) | (b > hi)))
        if self.check == 'bound':
            return bool(np.max(np.abs(new)) > max(self.paper))
        return bool(np.any(np.abs(new - np.array(self.paper)) > np.array(self.tol) + 1e-9))


def evaluate(s: Spec, r: Results, where: str) -> Number:
    new = tuple(float(v) for v in np.atleast_1d(np.asarray(s.value(r), dtype=float)))
    if s.check == '':
        return Number(s.id, where, s.what, s.quote, new, (), (), '')
    found = re.findall(r'[-+]?\d+(?:\.\d+)?', s.quote.replace('--', ' ').replace('+-', ' '))
    if s.paper is None:
        paper = tuple(float(v) for v in found)
        tol = tuple(0.5 * 10.0 ** -len(v.partition('.')[2]) if s.tol is None else s.tol for v in found)
    else:
        paper, tol = s.paper, (s.tol,) * len(s.paper)
    check = s.check if s.check is not None else ('round' if paper else '')
    return Number(s.id, where, s.what, s.quote, new, paper, tol, check)


# ---------------------------------------------------------------- quantities used by the sections

def b_weights(r: Results) -> float:
    """b of DeepCore nu_mu disappearance at the NuFIT point, theta23 and the phases fixed (fig_deepcore_bin_weights)."""
    return float(intercept(r['deepcore']['fig1_y'], X0))


def bins(r: Results, e_lo: float, e_hi: float, cz_max: float) -> np.ndarray:
    """The reconstructed DeepCore bins within [e_lo, e_hi] GeV with cos theta_z,reco up to cz_max, as a mask on the
    (energy, cos theta_z) bins of fig_deepcore_bin_weights."""
    e, cz = r['deepcore']['e_edges'], r['deepcore']['cosz_edges']
    return (e[:-1, None] >= e_lo) & (e[1:, None] <= e_hi) & (cz[None, 1:] <= cz_max + 1e-9)


def weight_share(r: Results, e_lo: float = 0.0, e_hi: float = np.inf, cz_max: float = 0.0) -> float:
    """Share [%] of the DeepCore weight omega_i of fig_deepcore_bin_weights in the bins of bins(); up-going only by
    default."""
    d = r['deepcore']
    w = d['fig1_omega'].reshape(d['e_edges'].size - 1, d['cosz_edges'].size - 1)       # cos theta_z runs fastest
    return 100 * w[bins(r, e_lo, e_hi, cz_max)].sum() / w.sum()


def preferred_range(r: Results, e_lo: float, e_hi: float, gray: float = 5e-4) -> tuple[float, float]:
    """Smallest and largest |y_i| [1e-3 eV^2] of the up-going bins within [e_lo, e_hi] GeV that are not gray in
    fig_deepcore_bin_weights (b), that is, with at least the share gray of the weight."""
    d = r['deepcore']
    w = d['fig1_omega'].reshape(d['e_edges'].size - 1, d['cosz_edges'].size - 1)
    y = -d['fig1_y_i'].reshape(w.shape)[bins(r, e_lo, e_hi, 0.0) & (w / w.sum() >= gray)] * 1e3
    return float(y.min()), float(y.max())


def solar_phase_mumu(r: Results) -> tuple[float, float]:
    """|Dm2_phi,mumu| [1e-6 eV^2] at the center of each DeepCore analysis bin (energy, and the path length at its cos
    theta_z): largest over the bins not gray in fig_deepcore_bin_weights, and mean weighted with its omega_i."""
    d = r['deepcore']
    e, cz = d['e_edges'], d['cosz_edges']
    share = d['fig1_omega'].reshape(e.size - 1, cz.size - 1) / d['fig1_omega'].sum()
    energy = np.sqrt(e[:-1] * e[1:])[:, None]
    length = earth.path_length(0.5 * (cz[:-1] + cz[1:]), DEPTH_KM)[None, :]
    phi = np.abs(dm2_phi_mumu(energy, length)) * 1e6
    return float(phi[share >= 5e-4].max()), float((share * phi).sum())


def slopes(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """dy/dx of the wrong-ordering solution y = b - x on the nodes x."""
    return np.gradient(np.asarray(b) * 1e-6 - x, x)


def slope_range(x: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    s = slopes(x, b)
    return float(s.min()), float(s.max())


def slope_deviation(x: np.ndarray, b: np.ndarray) -> float:
    """Largest |1 + dy/dx| [%] over the nodes x."""
    return 100 * float(np.max(np.abs(1 + slopes(x, b))))


def juno_dc_slopes(r: Results) -> tuple[float, float]:
    """Largest |1 + dy/dx| [%] of JUNO (a_res = 2.61%) and of DeepCore nu_mu with theta23 fixed (Sec. 2)."""
    v, d = r['vacuum'], r['deepcore']
    juno = slope_deviation(v['x'], intercept(v['juno_y'], v['x']))
    return juno, slope_deviation(d['x'], d['numu_fixed_line'])


def orca_slopes(r: Results) -> tuple[float, float]:
    return slope_range(r['orca']['x'], r['orca']['orca_all_b'])


def orca_x0(r: Results, key: str) -> float:
    """A line of fig_deepcore_vs_orca6 (global minimum) at x0."""
    o = r['orca']
    return float(o[key][np.argmin(np.abs(o['x'] - X0))])


def juno_b(r: Results, a_res: float) -> float:
    """JUNO b at the stochastic term a_res, with b_res and c_res scaled with it (fig_juno_intercept_vs_resolution)."""
    v = r['vacuum']
    return float(v['juno_b'][np.flatnonzero(np.isclose(v['a_res'], a_res))[0]])


def juno_line(r: Results, x: float) -> float:
    """b of JUNO at a_res = 2.61% at the true splitting x, from its line (fig_solutions_plane, fig_published_best_fits,
    fig_synergy_vs_precision, and fig_synergy_rings)."""
    v = r['vacuum']
    return float(np.interp(x, v['x'], intercept(v['juno_y'], v['x'])))


def e_eff(r: Results, a_res: float) -> float:
    """Effective energy [MeV] of the solar-phase term of JUNO (Eq. 3.8, fig_juno_effective_energy)."""
    v = r['vacuum']
    return float(v['eeff_energy'][np.flatnonzero(np.isclose(v['a_eeff'], a_res))[0]])


def dphi_peak(r: Results) -> float:
    """Energy [MeV] of the maximum of Dm2_phi,ee(E) at 52.5 km (App. C.3)."""
    v = r['vacuum']
    return float(v['eeff_bins'][np.argmax(v['dm2_phi_bins'])])


def beam_shift(r: Results, beam: str, share: int, tag: str) -> float:
    """b of a peaked beam minus the closed form of Eq. (3.4) (App. C.2, fig_accelerator_nubar_share); share 0: nu only,
    -1: equal nu and nubar; tag 'nufit' (true delta_CP = 177 deg) or 'zero'."""
    v = r['vacuum']
    return float(v[f'{beam}_{tag}'][share] - v[f'closed_{tag}'])


def ix0(r: Results) -> int:
    return int(np.argmin(np.abs(r['bands']['x'] - X0)))


def ranges(r: Results, sample: str, part: str = 'grid', ix: int | None = None) -> np.ndarray:
    """(b_lo, b_hi) per true point under the rule of App. B.3, on the continuous profile of the fit.

    A minimum is a range of b when more than three values of delta_CP^IO in 10-deg steps lie within 0.01 of
    it, in the better octant; otherwise it is the global minimum. JUNO has one value, from its line.
    """
    if sample == 'juno':
        b = juno_line(r, X0)
        return np.array([[b, b]])
    return r['bands'][f'{sample}/{part}/range_paper/{ix0(r) if ix is None else ix}']


def with_range(r: Results, sample: str) -> int:
    """Number of the 396 true points at x0 whose solution is a range of b (Sec. 5)."""
    rng = ranges(r, sample)
    return int(np.sum(rng[:, 1] > rng[:, 0]))


def scan(r: Results, sample: str, which: str, key: str = 'b') -> np.ndarray:
    """fig_intercept_vs_truth at x0: true delta_CP (which='dcp', true sin^2 theta23 = 0.561) or true theta23 ('s23',
    true delta_CP = 177 deg); key 'b' is the intercept at the global minimum, 's23sq' and 'dcp' the IO values."""
    return r['bands'][f'{sample}/scan_{which}/{key}/{ix0(r)}']


def app_nova_t2k(r: Results) -> float:
    """Largest |b of NOvA - b of T2K| with appearance at the same true point (grid of fig_solutions_plane and
    fig_intercept_ranges, x0)."""
    b = r['bands']
    return float(np.max(np.abs(b[f'nova_all/grid/b/{ix0(r)}'] - b[f't2k_all/grid/b/{ix0(r)}'])))


def dc_valley_width(r: Results) -> float:
    """Widest range of b over delta_CP^IO within 0.01 of the minimum, DeepCore, scans of fig_intercept_vs_truth (x0)."""
    b = r['bands']
    return float(max(np.max(np.ptp(b[f'{s}/scan_{w}/range_tol/{ix0(r)}'], axis=1))
                     for s in ('dcn', 'dca') for w in ('dcp', 's23')))




def bars(r: Results, sample: str) -> dict[str, tuple[float, float]]:
    """(b_lo, b_hi) of the bars of fig_intercept_ranges at x0: delta_CP unknown (true sin^2 theta23 = 0.56), theta23
    unknown (true delta_CP = 177 deg), both unknown."""
    s0 = r['bands']['s0']
    grid = ranges(r, sample)
    dcp = grid.reshape(s0.size, -1, 2)[np.argmin(np.abs(s0 - S23SQ_BAR))]
    return {k: (float(v[:, 0].min()), float(v[:, 1].max()))
            for k, v in (('dcp', dcp), ('t23', ranges(r, sample, 't23')), ('both', grid))}


def band_at(r: Results, sample: str, x: float) -> tuple[float, float]:
    """Edges of the band of fig_solutions_plane and fig_published_best_fits (all true points) at the true splitting x,
    linear in x."""
    nodes = r['bands']['x']
    edges = np.array([(ranges(r, sample, ix=i)[:, 0].min(), ranges(r, sample, ix=i)[:, 1].max())
                      for i in range(nodes.size)])
    return float(np.interp(x, nodes, edges[:, 0])), float(np.interp(x, nodes, edges[:, 1]))


def gaps(r: Results, a: str, b: str) -> tuple[float, float]:
    """Smallest and largest closest gap |b_A - b_B| of the joint fit over the true points (fig_synergy_vs_precision)."""
    g = synergy.gap(ranges(r, a), ranges(r, b))
    return float(g.min()), float(g.max())


def nova_dc_edge(r: Results, s23sq: float = 0.53) -> tuple[float, float]:
    """Largest closest gap of NOvA and DeepCore in disappearance over the true points with sin^2 theta23 below
    s23sq and with s23sq or more (App. F)."""
    s0 = r['bands']['s0']
    g = synergy.gap(ranges(r, 'nova_dis'), ranges(r, 'dcn')).reshape(s0.size, -1)
    return float(g[s0 < s23sq - 1e-9].max()), float(g[s0 >= s23sq - 1e-9].max())


def joint_larger(r: Results) -> tuple[float, float]:
    """Share [%] of the true points where the joint NOvA-DeepCore IO fit gives a larger synergy than Eq. (1.1), both
    at 1.5% of |Dm31| (compute/joint.py): disappearance, all channels."""
    return tuple(100 * float(np.mean(r['joint'][f'{p}/joint'] > r['joint'][f'{p}/eq11'])) for p in ('dis', 'app'))


def joint_max(r: Results) -> tuple[float, ...]:
    """Largest joint NOvA-DeepCore synergy at 1.5%, per panel (disappearance, all channels): its true
    sin^2 theta23, Eq. (1.1) at that true point, and the joint value."""
    out = []
    for p in ('dis', 'app'):
        j = r['joint'][f'{p}/joint']
        i = j.argmax()
        out += [r['joint'][f'{p}/s23sq'][i], r['joint'][f'{p}/eq11'][i], j[i]]
    return tuple(float(v) for v in out)


def ring(r: Results, partner: str, juno_pct: float, partner_pct: float) -> tuple[float, float]:
    """Inner and outer synergy term of a ring of fig_synergy_rings (Eq. 1.1), from the closest gap of
    fig_synergy_vs_precision."""
    def sigma(pct: float) -> float:
        return pct / 100 * X0 * 1e6
    return synergy.synergy_range(ranges(r, partner), ranges(r, 'juno'), sigma(partner_pct), sigma(juno_pct))


def ring_ratio(r: Results, partner: str) -> float:
    """Outer over inner synergy term of the rings of fig_synergy_rings, the same at every pair of precisions."""
    lo, hi = gaps(r, 'juno', partner)
    return (hi / lo) ** 2


def circular_span(deg: np.ndarray) -> float:
    """Shortest arc [deg] that contains all the angles."""
    a = np.sort(np.mod(deg, 360.0))
    return float(360.0 - np.max(np.diff(np.append(a, a[0] + 360.0))))


def t2k_dcp_io(r: Results) -> np.ndarray:
    """delta_CP^IO [rad] of T2K with appearance at x0 over the true delta_CP of the bands, at true sin^2 theta23 = 0.56
    (T2K has no scan in fig_intercept_vs_truth)."""
    b = r['bands']
    return b[f't2k_all/grid/dcp/{ix0(r)}'].reshape(b['s0'].size, -1)[np.argmin(np.abs(b['s0'] - S23SQ_BAR))]


def measured(r: Results) -> dict[str, tuple[float, float]]:
    """(Dm31_NO, Dm31_IO) [eV^2] of the published best fits of fig_published_best_fits, Dm32 + Dm21 for the experiments
    that quote Dm32."""
    with open(DATA / 'published_best_fits.json') as f:
        entries = json.load(f)['measurements']
    out = {}
    for e in entries:
        dm21 = e.get('dm21', NUFIT.dm21 * 1e3)
        quantity = [q.strip() for q in e['quantity'].split(',')]
        io = float(r['measured']['ic93_io_dm32']) * 1e3 if isinstance(e['io'], str) else e['io'][0]
        no = e['no'][0] + (dm21 if quantity[0].startswith('dm32') else 0.0)
        io += dm21 if quantity[-1].startswith('dm32') else 0.0
        out[e['label']] = (no * 1e-3, io * 1e-3)
    return out


def b_measured(r: Results, label: str) -> float:
    x, y = measured(r)[label]
    return float(intercept(y, x))


def in_band(r: Results, sample: str, label: str) -> tuple[float, float, float]:
    """(b, b_lo, b_hi) of a published best fit and of the band of fig_published_best_fits at its Dm31_NO."""
    x, y = data_fit(r) if label == 'this work' else measured(r)[label]
    return (float(intercept(y, x)), *band_at(r, sample, x))


def data_fit(r: Results) -> tuple[float, float]:
    """(Dm31_NO, Dm31_IO) [eV^2] of our fit of the DeepCore public data (App. E.4)."""
    ic = r['icfit']
    return float(ic['data_no_dm31']), float(ic['data_io_dm31'])


def full_fit(r: Results, x: float | np.ndarray) -> np.ndarray:
    """b of the full inverted-ordering fit to Asimov data (App. E.2), on its nodes x."""
    ic = r['icfit']
    return np.interp(x, ic['asimov_x'], ic['asimov_b'])


def wba(r: Results, case: str) -> np.ndarray:
    """Weighted bin average on the fitter's expectation (App. E.2) at its nodes: case 'y' (splitting only),
    't23' (theta23 also profiled), or 'all' (every nuisance parameter, Eq. B.6)."""
    return r['icfit'][f'wba_{case}']


def above_full_fit(r: Results, case: str) -> tuple[float, float]:
    """Smallest and largest b minus the full fit over the Asimov x of App. E.2: case 'tab' is the tabulated response
    with all channels and theta23 fitted (the fig_full_fit_validation line), the others are wba() cases."""
    if case == 'tab':
        x, b = r['deepcore']['x_line'], r['deepcore']['fig6_line']
    else:
        x, b = r['icfit']['wba_x'], wba(r, case)
    d = b - full_fit(r, x)
    return float(d.min()), float(d.max())


def nuisance_effect(r: Results) -> tuple[float, float]:
    """How much profiling the nuisance parameters lowers the WBA on the fitter's expectation, theta23 profiled
    in both (min, max over x; Sec. 2, App. E.2)."""
    d = wba(r, 't23') - wba(r, 'all')
    return float(d.min()), float(d.max())


def fitted(r: Results, ordering: str, name: str, scale: float = 1.0) -> tuple[float, float]:
    """Value and error of a parameter of our fit of the public data (App. E.3, E.4), times scale."""
    ic = r['icfit']
    i = list(ic['names']).index(name)
    return float(ic[f'data_{ordering}_values'][i] * scale), float(ic[f'data_{ordering}_errors'][i] * scale)


def splitting(r: Results, ordering: str, shift: float = 0.0) -> tuple[float, float]:
    """Dm31 - shift and its error [1e-3 eV^2] of our fit of the public data (App. E.3, E.4)."""
    ic = r['icfit']
    return float(ic[f'data_{ordering}_dm31'] - shift) * 1e3, float(ic[f'data_{ordering}_dm31_err']) * 1e3


def vacuum_limit(r: Results, theta23: str) -> float:
    """Largest |b without matter - Eq. (3.4)| over the open points of fig_deepcore_dcp (delta_CP^IO = 285 deg), theta23
    'fixed' in both orderings or 'fitted' (Eq. 3.4 at the fitted sin^2 theta23_IO) (App. C.1)."""
    d = r['deepcore']
    dcp = d['dcp_fig3']
    if theta23 == 'fixed':
        return float(np.max(np.abs(d['fig3_fixed_b_vacuum'] - b_lbl(dcp, DCP_IO) * 1e6)))
    return float(np.max(np.abs(d['fig3_b_vacuum'] - b_lbl(dcp, DCP_IO, S23SQ, d['fig3_s23sq_vacuum']) * 1e6)))


def eq38_gap(r: Results, a_res: float) -> float:
    """|Eq. (3.8) - weighted bin average| of JUNO at the stochastic term a_res (App. C.3, fig_juno_effective_energy)."""
    v = r['vacuum']
    i = np.flatnonzero(np.isclose(v['a_eeff'], a_res))[0]
    return float(abs(b_sbr() * 1e6 - v['eeff_average'][i] - v['eeff_b'][i]))


def direction(r: Results) -> tuple[float, ...]:
    """DeepCore b with the zenith width x0.5 and x2: nu_mu, then all channels (Sec. 4)."""
    d = r['direction']
    i = [int(np.argmin(np.abs(d['cosz_scale'] - f))) for f in (0.5, 2.0)]
    return (*d['numu'][i], *d['all'][i])


def nova_band(r: Results, matter: bool) -> tuple[np.ndarray, np.ndarray]:
    """NOvA disappearance band of fig_intercept_vs_truth (left) at x0 over the true delta_CP and the closed form over
    its valley, each (n, 2), with or without matter (Sec. 5)."""
    if matter:
        b = r['bands']
        return b[f'nova_dis/scan_dcp/range_paper/{ix0(r)}'], b[f'nova_dis/scan_dcp/cf/{ix0(r)}']
    v = r['vacuum']
    return v['nova_dis_vacuum_range'], v['nova_dis_vacuum_cf']


def nova_narrower(r: Results, matter: bool = True) -> tuple[float, float]:
    """How much narrower [%] the NOvA disappearance band is than the closed form (min, max over the true dCP)."""
    band, cf = nova_band(r, matter)
    share = 100 * (1 - np.ptp(band, axis=1) / np.ptp(cf, axis=1))
    return float(share.min()), float(share.max())


def nova_center(r: Results) -> float:
    """Largest |center of the NOvA disappearance band - center of the closed form| with matter (Sec. 5)."""
    band, cf = nova_band(r, True)
    return float(np.max(np.abs(band.mean(axis=1) - cf.mean(axis=1))))


def solar(r: Results, parameter: str) -> dict[str, np.ndarray]:
    """Changes of b for the JUNO precision of a solar parameter (Sec. 6, footnote): JUNO, the NOvA closed form on
    the 30-deg grid of both phases, DeepCore all channels."""
    s = r['solar']
    i = list(s['parameters']).index(parameter)
    return {'juno': s['juno'][i], 'nova': s['nova'][i].ravel(), 'deepcore': s['deepcore'][i]}


def solar_separations(r: Results, parameter: str) -> tuple[float, float]:
    """Smallest and largest |change of the separation from JUNO| of NOvA and DeepCore."""
    d = solar(r, parameter)
    sep = np.abs(np.append(d['nova'], d['deepcore']) - d['juno'])
    return float(sep.min()), float(sep.max())


def cp_cells(r: Results, i: int = 0, ahat_min: float = 0.0, ahat_max: float = np.inf, tag: str = 'matter') -> float:
    """First-order contribution to the CP width of DeepCore of the true cells with A-hat in [ahat_min, ahat_max)
    (App. D.2); i = 0, 1, 2 for sin^2 theta23 = 0.561, 0.5, 0.439."""
    a = r['analytic']
    edges = a['cp_ahat_edges'][:-1]
    keep = (edges >= ahat_min) & (edges < ahat_max)
    return float(a[f'cp_cells_{tag}'][i][keep].sum())


def analytic_share(r: Results, key: str) -> tuple[float, float]:
    """Smallest and largest share [%] over x of delta_atm = b - 2 c12^2 Dm21 at delta_CP = 90 deg reproduced by an
    analytic form (App. D.1, fig_deepcore_analytic)."""
    a = r['analytic']
    share = 100 * (a[key] - LEAD) / (a['exact_90'] - LEAD)
    return float(share.min()), float(share.max())


def matter_parts(r: Results) -> tuple[float, float, float]:
    """delta_atm at x0 and the NuFIT phases (Sec. 4.1): share [%] of the frequency shift; then, of the rest, the
    shares of the amplitude (one mass scale minus frequency shift, delta_CP = 90 deg) and of the CP amplitude term."""
    a = {k: float(np.interp(X0, r['analytic']['x'], r['analytic'][k]))
         for k in ('exact_nufit', 'exact_90', 'frequency_nufit', 'frequency_90', 'one_mass_scale')}
    rest = a['exact_nufit'] - a['frequency_nufit']
    return (100 * (a['frequency_nufit'] - LEAD) / (a['exact_nufit'] - LEAD),
            (a['one_mass_scale'] - a['frequency_90']) / rest, (a['exact_nufit'] - a['exact_90']) / rest)


def cp_term(r: Results) -> tuple[float, float]:
    """CP amplitude term at the NuFIT phases: exact b at the NuFIT phases minus exact b at 90 deg, over x."""
    a = r['analytic']
    d = a['exact_nufit'] - a['exact_90']
    return float(d.min()), float(d.max())


def mantle_resonance(r: Results) -> tuple[float, float]:
    """E [GeV] of the 1-3 MSW resonance, x0 cos 2theta13 = A(E), in the densest and the lightest mantle."""
    rhoye = [prem_rhoye(float(v)) for v in np.linspace(SHELLS[1], SHELLS[6] - 1e-6, 2000)]
    return tuple(X0 * (1 - 2 * NUFIT.s13sq) / (A_PER_RHOYE_GEV * rho) for rho in (max(rhoye), min(rhoye)))


def precision(label: str) -> float:
    """Relative 1 sigma of a published Dm31 or Dm32 under the normal ordering [%], the mean of the two errors
    (data/published_best_fits.json; Sec. 6)."""
    with open(DATA / 'published_best_fits.json') as f:
        no = next(e['no'] for e in json.load(f)['measurements'] if e['label'] == label)
    return 100 * (no[1] + no[2]) / 2 / no[0]


def juno_resolution(e_mev: float) -> float:
    """sigma_E / E [%] of JUNO at the prompt energy e_mev (App. A.2)."""
    return 100 * float(np.sqrt(juno.A_RES ** 2 / e_mev + juno.B_RES ** 2 + (juno.C_RES / e_mev) ** 2))


# ---------------------------------------------------------------- the numbers, by section in paper order

def introduction() -> list[Spec]:
    """Abstract: the synergy with JUNO."""
    return [
        Spec('abs.dc_independent', 83, 'JUNO - DeepCore all closest gap (min, max)', '-',
             lambda r: gaps(r, 'juno', 'dca')),
    ]


def sec2() -> list[Spec]:
    """Sec. 2 and fig_deepcore_bin_weights: the weighted bin average of DeepCore, bin by bin (Eq. 2.3), and the
    slopes."""
    return [
        # the closed forms of Eqs. (3.4) and (3.7) do not depend on x, so their solutions have slope -1
        Spec('sec2.slope_sbr_lbl', 177, 'dy/dx of the closed forms, reactor and accelerator ("slope close to -1")',
             '-1',
             lambda r: [slopes(X_GRID, np.full(X_GRID.size, b * 1e6)).mean()
                        for b in (b_sbr(), b_lbl(DCP_NO, DCP_IO))]),
        Spec('sec2.weight_20_50', 209, 'weight share [%] of up-going bins at 20-50 GeV', 'two thirds',
             lambda r: weight_share(r, 20.0, 50.0), paper=(66.7,), tol=3.0),
        Spec('sec2.weight_core', 209, 'weight share [%] below 20 GeV, cos theta_z < -0.56', 'about a sixth',
             lambda r: weight_share(r, 0.0, 20.5, -0.56), paper=(16.7,), tol=2.0),
    ]


def sec3() -> list[Spec]:
    """Sec. 3 and fig_accelerator_closed_form and fig_juno_intercept_vs_resolution: accelerators and reactors near the
    vacuum limit (Eqs. 3.1-3.8)."""
    return [
        Spec('sec3.lbl_agree', 270, 'generic beam minus Eq. (3.4), max over dCP', '0.1',
             lambda r: np.max(np.abs(r['vacuum']['beam_b_line'] - r['vacuum']['beam_closed_line'])),
             check='bound'),
        Spec('sec3.eq36_103p8', 259, '2 c12^2 Dm21 of Eq. (3.6)', '103.8', lambda r: LEAD),
        Spec('sec3.eq36_11p6', 259, 's13 sin 2theta12 tan theta23 Dm21 at 0.561', '11.6',
             lambda r: np.sqrt(NUFIT.s13sq * 4 * NUFIT.s12sq * (1 - NUFIT.s12sq) * S23SQ / (1 - S23SQ))
             * NUFIT.dm21 * 1e6),
        Spec('sec3.lbl_unequal', 283, 'NOvA-like nu only minus Eq. (3.4), dCP 177, 0 deg', 'a few',
             lambda r: (beam_shift(r, 'nova_like', 0, 'nufit'), beam_shift(r, 'nova_like', 0, 'zero'))),
        Spec('sec3.dphi_above_2p5', 307, 'energy [MeV] of the maximum of Dm2_phi,ee', '2.5 MeV', dphi_peak),
        Spec('sec3.juno_resolution', 307, 'sigma_E / E [%] of JUNO at 1 MeV from a, b, c (the paper quotes JUNO)',
             '2.95%', lambda r: juno_resolution(1.0), check=''),
        Spec('sec3.juno_b', 307, 'JUNO b at a_res = 2.61%', '25.7', lambda r: juno_b(r, juno.A_RES)),
        Spec('sec3.juno_2to4', 307, 'JUNO b(a_res = 4%) - b(2%)', 'about 10',
             lambda r: juno_b(r, 0.04) - juno_b(r, 0.02), tol=1.0),
        Spec('sec3.eeff_juno', 312, 'E_eff [MeV] at a_res = 2.61%', 'about 4 MeV', lambda r: e_eff(r, juno.A_RES),
             tol=0.5),
    ]


def sec4() -> list[Spec]:
    """Sec. 4 (fig_deepcore_dcp, fig_deepcore_resolution, fig_deepcore_energy_window, fig_deepcore_vs_orca6): the
    atmospheric solution of DeepCore and ORCA6 (Eqs. 4.1-4.2)."""
    def d(r):
        return r['deepcore']
    return [
        Spec('sec4.msw_energy', 325, '1-3 resonance [GeV], densest and lightest mantle', '6--10 GeV',
             mantle_resonance),
        Spec('sec4.b_reference', 329, 'b of the reference configuration (fig_deepcore_bin_weights)', '119.7',
             b_weights),
        Spec('sec4.weight_above_10', 335, 'weight share [%] of the bins above 10 GeV', 'above about 10 GeV',
             lambda r: weight_share(r, 10.0, cz_max=np.inf), check=''),
        Spec('sec4.freq_share', 358, 'frequency shift: share [%] of delta_atm at x0, NuFIT phases', 'about 70%',
             lambda r: matter_parts(r)[0], paper=(70.0,), tol=3.0),
        Spec('sec4.amp_rest', 358, 'amplitude effect: share of the rest', 'most of the rest',
             lambda r: matter_parts(r)[1], check=''),
        Spec('sec4.datm_x_dependence', 360, 'max |1 + dy/dx| [%], DeepCore nu_mu', 'about 1%',
             lambda r: slope_deviation(d(r)['x'], d(r)['numu_fixed_line']), tol=0.5),
        Spec('fig_deepcore_resolution.sigma_lnE', 399,
             'sigma_lnE of the kernels, bin variance removed (min, max over true energy)', '0.29--0.34',
             lambda r: (r['release']['kernel_lne'].min(), r['release']['kernel_lne'].max())),
        Spec('sec4.valid_tab', 406, 'tabulated response minus full fit (max over x)', 'within 2',
             lambda r: above_full_fit(r, 'tab')[1], check='bound'),
        Spec('sec4.res_scan', 408, 'range of b over the Gaussian sigma_lnE scan', 'up to about 11',
             lambda r: np.ptp(d(r)['fig4_b_sigma']), tol=1.0),
        Spec('sec4.emax_raise', 410, 'rise at E_max 20 GeV; NOvA-DC nu_mu: centers, max gap', '-',
             lambda r: (d(r)['fig4_b_e_max'][0] - d(r)['fig4_b_release_resolution'],
                        np.mean(bars(r, 'dcn')['both']) - np.mean(bars(r, 'nova_dis')['both']),
                        gaps(r, 'nova_dis', 'dcn')[1])),
        Spec('sec4.direction', 411, 'b(zenith width x0.5) - b(x2): nu_mu, all channels; range of the sigma_lnE scan',
             'by less than the energy resolution moves it',
             lambda r: (direction(r)[0] - direction(r)[1], direction(r)[2] - direction(r)[3],
                        np.ptp(d(r)['fig4_b_sigma'])), check=''),
        Spec('sec4.orca_vs_dc', 430, 'DeepCore all minus ORCA6 all at x0 (fig_deepcore_vs_orca6)', 'about 11',
             lambda r: orca_x0(r, 'dc_all_b') - orca_x0(r, 'orca_all_b'), tol=1.0),
        Spec('sec4.orca_trk_close', 430, 'ORCA6 tracks minus DeepCore all at x0', '-',
             lambda r: orca_x0(r, 'orca_tracks_b') - orca_x0(r, 'dc_all_b')),
        Spec('sec4.orca_shower_weight', 430, 'shower share [%] of the ORCA6 weight, global minimum', 'about 10%',
             lambda r: 100 * r['orca']['orca_global_class_weight'][1]),
        Spec('sec4.orca_slope', 430, 'dy/dx of ORCA6 all classes (min, max)', 'a few percent', orca_slopes),
    ]


BARS = {    # id of the sample: sample, printed widths of the bars with delta_CP, theta23, both unknown
    'nova_dis': ('nova_dis', (40, 22, 41)), 'nova': ('nova_all', (25, 6, 29)),
    't2k_dis': ('t2k_dis', (41, 24, 42)), 't2k': ('t2k_all', (28, 3, 31)),
    'dcn': ('dcn', (2, 21, 25)), 'dca': ('dca', (3, 5, 7)),
}


def sec5() -> list[Spec]:
    """Sec. 5 (fig_solutions_plane, fig_intercept_ranges, fig_intercept_vs_truth, fig_published_best_fits): the unknown
    delta_CP and theta23 (Eq. 5.1, App. B.3)."""
    bar_specs = [Spec(f'fig_intercept_ranges.width.{name}.{kind}', 468,
                      f'fig_intercept_ranges bar width, {sample}, {kind} unknown', str(w),
                 lambda r, s=sample, k=kind: np.diff(bars(r, s)[k])[0])
            for name, (sample, widths) in BARS.items() for kind, w in zip(('dcp', 't23', 'both'), widths)]

    def edge_differences(r):
        return [bars(r, a)['both'][i] - bars(r, b)['both'][i]
                for a, b in (('nova_dis', 't2k_dis'), ('nova_all', 't2k_all')) for i in (0, 1)]

    def t2k_minus_nova(r):
        diff = ranges(r, 't2k_dis') - ranges(r, 'nova_dis')
        return diff[:, 0].min(), diff[:, 0].max(), diff[:, 1].min(), diff[:, 1].max()

    return [
        Spec('fig_solutions_plane.s23_range', 445, '3 sigma range of the true sin^2 theta23', '0.430--0.596',
             lambda r: S23SQ_RANGE),
        Spec('sec5.valley_acc_dis', 454, 'true points with a range of b (of 396): NOvA, T2K dis', '-',
             lambda r: (with_range(r, 'nova_dis'), with_range(r, 't2k_dis'))),
        Spec('sec5.valley_dc_app', 454, 'with a range: DC nu_mu, DC all, NOvA all, T2K all', '-',
             lambda r: [with_range(r, s) for s in ('dcn', 'dca', 'nova_all', 't2k_all')]),
        *bar_specs,
        Spec('fig_intercept_vs_truth.valley_tol', 482, 'chi^2 tolerance of the valley of dCP_IO', '0.01',
             lambda r: fit.ValleyRule().tol),
        Spec('fig_intercept_vs_truth.dc_range', 483, 'DeepCore: widest range of b within 0.01 over dCP_IO (scans)',
             'changes little', dc_valley_width, check=''),
        Spec('fig_intercept_vs_truth.t2k_same_dis', 483, 'T2K - NOvA dis per true point: lo, hi (min, max each)', '-',
             t2k_minus_nova),
        Spec('sec5.nova_t2k_few', 494, 'NOvA - T2K, both-unknown bars: dis lo, hi; app lo, hi', 'a few',
             edge_differences),
        Spec('sec5.app_nova_t2k', 495, 'NOvA minus T2K with appearance at the same true point (max)', 'about 20',
             app_nova_t2k, paper=(20.0,), tol=1.0),
        Spec('sec5.app_t2k', 495, 'arc [deg] spanned by T2K dCP_IO with app over true dCP', 'changes',
             lambda r: circular_span(np.degrees(t2k_dcp_io(r)))),
        Spec('sec5.app_3pi2', 495, 'dCP_IO [deg] of NOvA with appearance over true dCP', 'close to 270 deg',
             lambda r: np.sort(np.degrees(scan(r, 'nova_all', 'dcp', 'dcp')) % 360)[[0, -1]], check=''),
        Spec('sec5.nova_vacuum', 508, 'NOvA dis band narrower than the closed form without matter [%] (min, max)',
             'is slightly narrower', lambda r: nova_narrower(r, matter=False), check=''),
        Spec('sec5.dca_constant', 508, 'DeepCore all: range of b over true dCP, true theta23', 'constant',
             lambda r: (np.ptp(scan(r, 'dca', 'dcp')), np.ptp(scan(r, 'dca', 's23')))),
        Spec('sec5.dyb_on_line', 525, 'b of Daya Bay; the line 2 s12^2 Dm21', 'on its line',
             lambda r: (b_measured(r, 'Daya Bay 2022'), b_sbr() * 1e6)),
        Spec('sec5.juno_measured_b', 525, 'b of JUNO (207 days)', '27', lambda r: b_measured(r, 'JUNO 2026')),
        Spec('sec5.juno_on_line', 525, 'JUNO line (a_res = 2.61%) at the measured x', 'consistent with the line',
             lambda r: juno_line(r, measured(r)['JUNO 2026'][0]), check=''),
        Spec('sec5.nova_in_app_band', 527, 'NOvA, T2K, T2K+NOvA in NOvA all band: (b, lo, hi) x 3', 'inside',
             lambda r: [v for label in ('NOvA 2025', 'T2K 2024', 'T2K + NOvA 2025')
                        for v in in_band(r, 'nova_all', label)], check='inside'),
        Spec('sec5.nufit_in_app_band', 527, 'NuFIT 6.0 in the NOvA all band: (b, lo, hi)', 'inside',
             lambda r: in_band(r, 'nova_all', 'NuFIT 6.0 (global)'), check='inside'),
        Spec('sec5.dcdata_in_dca_band', 530, 'our DeepCore data fit in DC all band: (b, lo, hi)', 'inside',
             lambda r: in_band(r, 'dca', 'this work'), check='inside'),
        Spec('sec5.dc93_above_dca', 532, 'DeepCore 9.3 yr and the DC all band at its x: (b, lo, hi)',
             'lies higher than our all-channel band', lambda r: in_band(r, 'dca', 'IceCube DeepCore 9.3 yr'),
             check=''),
    ]


GAPS = {'dis.JUNO_NOvA': ('juno', 'nova_dis'), 'dis.JUNO_DeepCore': ('juno', 'dcn'),
        'dis.NOvA_DeepCore': ('nova_dis', 'dcn'), 'all.JUNO_NOvA': ('juno', 'nova_all'),
        'all.JUNO_DeepCore': ('juno', 'dca'), 'all.NOvA_DeepCore': ('nova_all', 'dca')}
RINGS = {    # printed rings [inner, outer]: partner, then (sigma_JUNO, sigma of the partner) [%]
    'dca': {(0.5, 1.0): '[9.8, 11.3]', (0.5, 1.5): '[4.9, 5.7]', (0.5, 2.0): '[2.9, 3.3]',
            (0.8, 1.0): '[7.5, 8.6]', (0.8, 1.5): '[4.2, 4.9]', (0.8, 2.0): '[2.6, 3.0]',
            (1.0, 1.0): '[6.1, 7.1]', (1.0, 1.5): '[3.8, 4.4]', (1.0, 2.0): '[2.5, 2.8]'},
    'nova_all': {(0.5, 1.0): '[5.1, 10.7]', (0.5, 1.5): '[2.5, 5.3]', (0.5, 2.0): '[1.5, 3.1]',
                 (0.8, 1.0): '[3.9, 8.1]', (0.8, 1.5): '[2.2, 4.6]', (0.8, 2.0): '[1.4, 2.9]',
                 (1.0, 1.0): '[3.2, 6.7]', (1.0, 1.5): '[1.9, 4.1]', (1.0, 2.0): '[1.3, 2.7]'},
}


def sec6() -> list[Spec]:
    """Sec. 6 (fig_synergy_vs_precision, fig_synergy_rings): the synergy term (Eq. 1.1). The joint fit takes the closest
    points of the two ranges of b (nmosyn.synergy.gap), in fig_synergy_vs_precision and in the rings of
    fig_synergy_rings."""
    gap_specs = [Spec(f'fig_synergy_vs_precision.gap.{name}', 561,
                      f'fig_synergy_vs_precision closest gap {a} - {b} (min, max)', '-',
                  lambda r, a=a, b=b: gaps(r, a, b)) for name, (a, b) in GAPS.items()]
    ring_specs = [Spec(f'fig_synergy_rings.ring.{partner}.j{j:.1f}.p{p:.1f}', 578,
                  f'fig_synergy_rings ring, JUNO {j}% + {partner} {p}% (inner, outer)', quote,
                  lambda r, partner=partner, j=j, p=p: ring(r, partner, j, p))
             for partner, rings in RINGS.items() for (j, p), quote in rings.items()]
    return [
        Spec('sec6.precisions', 576, 'published 1 sigma of |Dm31| [%]: JUNO, NOvA, DeepCore 9.3 yr',
             '1.0%, 1.4%, 1.9%',
             lambda r: [precision(k) for k in ('JUNO 2026', 'NOvA 2025', 'IceCube DeepCore 9.3 yr')]),
        *gap_specs,
        Spec('sec6.solar_sep', 564, 'largest |change of the separation from JUNO| for the JUNO precision of Dm21 '
             'and sin^2 theta12', 'at most 2',
             lambda r: max(solar_separations(r, 'dm21')[1], solar_separations(r, 's12sq')[1])),
        Spec('sec6.joint_larger', 566, 'share of the true points where the joint NOvA-DC IO fit exceeds Eq. (1.1) '
             'at 1.5% [%]: dis, all', 'almost all', joint_larger, check=''),
        *ring_specs,
        Spec('sec6.ring_ratio', 579, 'outer / inner synergy with JUNO: DC all (narrow), NOvA all', 'factor of two',
             lambda r: (ring_ratio(r, 'dca'), ring_ratio(r, 'nova_all')), check=''),
        Spec('sec6.equal_prec', 579, 'rings at 1% / 1%: DC all, NOvA all (inner, outer)', '-',
             lambda r: (*ring(r, 'dca', 1.0, 1.0), *ring(r, 'nova_all', 1.0, 1.0))),
    ]


def conclusions() -> list[Spec]:
    """Sec. 7: the slopes, the validation, and the synergy of JUNO with DeepCore."""
    return [
        Spec('concl.slope_juno', 594, 'max |1 + dy/dx| [%] of JUNO', 'within about 1%', lambda r: juno_dc_slopes(r)[0]),
        Spec('concl.slope_dc', 594, 'max |1 + dy/dx| [%] of DeepCore nu_mu', 'within about 1%',
             lambda r: juno_dc_slopes(r)[1]),
        Spec('concl.orca_slope', 594, 'dy/dx of ORCA6 all classes (min, max)', 'a few percent', orca_slopes),
        Spec('concl.dc_nearly_fixed', 602, 'outer / inner synergy with JUNO: DC all, NOvA all', '-',
             lambda r: (ring_ratio(r, 'dca'), ring_ratio(r, 'nova_all'))),
    ]


def app_a() -> list[Spec]:
    """App. A: inputs (Eq. A.1) and detector models (Eq. A.3)."""
    return [
        Spec('appA.b_sbr', 23, '2 s12^2 Dm21, Eq. (3.7)', '46.0', lambda r: b_sbr() * 1e6),
        Spec('appA.b_lead', 24, '2 c12^2 Dm21', '103.8', lambda r: LEAD),
        Spec('appA.juno_abc', 39, 'a_res, b_res, c_res [%] of JUNO', '2.61%, 0.64%, 1.20%',
             lambda r: (100 * juno.A_RES, 100 * juno.B_RES, 100 * juno.C_RES)),
        Spec('appA.juno_rhoye', 42, 'rho Y_e of the JUNO crust [g/cm^3]', '1.225', lambda r: juno.CRUST_RHOYE),
        Spec('appA.lbl_density', 59, 'crust density of NOvA and T2K [g/cm^3]', '2.84, 2.6',
             lambda r: (lbl_spectra.inputs()['nova']['density'], lbl_spectra.inputs()['t2k']['density'])),
        Spec('appA.dc.events', 81, 'DeepCore neutrino events', '21402', lambda r: r['release']['neutrinos']),
        Spec('appA.dc.bins', 85, 'analysis energy bins: number, first edge, last two edges [GeV]',
             '10, 6.31, 88.2, 158.5',
             lambda r: (E_EDGES.size - 1, E_EDGES[0], E_EDGES[-2], E_EDGES[-1])),
        Spec('appA.dc.cosz_bins', 86, 'analysis zenith bins: first and last edge of cos theta_z', '-1, 0.1',
             lambda r: (COSZ_EDGES[0], COSZ_EDGES[-1])),
        # the intercepts are quoted to 0.1, so unchanged means below half of it
        Spec('appA.dc.grid_converged', 101, 'largest |b change| on a five times finer grid (1500 x 900)',
             'unchanged at the precision quoted in this paper',
             lambda r: max(abs(np.diff(r['grid'][k])[0]) for k in ('numu_fixed', 'all_fixed', 'all_fitted')),
             paper=(0.05,), check='bound'),
        Spec('appA.dc.direction', 103, 'b(zenith width x0.5) - b(x2): nu_mu, all channels', '6.5, 5.0',
             lambda r: (direction(r)[0] - direction(r)[1], direction(r)[2] - direction(r)[3])),
        Spec('appA.dc.gauss_vs_tab', 103,
             'b Gaussian response minus tabulated (fig_deepcore_resolution and fig_deepcore_energy_window)', '3',
             lambda r: r['deepcore']['fig4_b_release_resolution'] - b_weights(r)),
        Spec('appA.orca.total', 113, '|ORCA6 predicted / observed total - 1| [%]', '0.7%',
             lambda r: 100 * abs(r['orca']['orca_predicted_total'] / r['orca']['orca_observed_total'] - 1)),
        Spec('appA.orca.weights', 113, 'weight shares [%]: high-purity tracks, low-purity tracks, showers',
             '65%, 25%, 10%', lambda r: 100 * r['orca']['orca_global_class_weight'][[0, 2, 1]]),
        Spec('appA.orca.lp_add', 113, 'adding the low-purity tracks: |change of b|, fixed and profiled',
             'less than 2', lambda r: (abs(r['orca']['orca_tracks_lp_fixed_b'] - r['orca']['orca_tracks_fixed_b']),
                                       abs(r['orca']['orca_tracks_lp_b'] - orca_x0(r, 'orca_tracks_b'))),
             paper=(2.0,), check='bound'),
        Spec('appA.orca.showers', 113, 'adding the showers lowers b (global minimum, x0)', '13',
             lambda r: r['orca']['orca_tracks_lp_b'] - orca_x0(r, 'orca_all_b')),
        Spec('appA.orca.s23', 113, 'adding the showers: change of sin^2 theta23_IO (global minimum, x0)', 'about 0.01',
             lambda r: orca_x0(r, 'orca_all_s23sq') - r['orca']['orca_tracks_lp_s23sq'], tol=0.005),
    ]


def app_b() -> list[Spec]:
    """App. B: the exchange of the orderings."""
    return [
        Spec('appB.exchange_numu', 160, '|x returned - x0| for a true IO, nu_mu', 'about 2',
             lambda r: abs(r['deepcore']['exchange_numu'])),
        Spec('appB.exchange_all', 160, '|x returned - x0| for a true IO, all channels', 'within 0.1',
             lambda r: abs(r['deepcore']['exchange_all']), check='bound'),
    ]


EEFF = {0.0: '3.5', 0.01: '3.6', 0.02: '3.9', juno.A_RES: '4.1', 0.05: '5.3', 0.08: '6.4'}    # a_res: E_eff [MeV]


def app_c() -> list[Spec]:
    """App. C (fig_juno_effective_energy): accelerators in matter, JUNO resolution and effective energy (Eq. 3.8)."""
    def t2k_like(r):
        v = r['vacuum']
        return max(np.max(np.abs(v[f't2k_like_{tag}'] - v[f'closed_{tag}'])) for tag in ('nufit', 'zero'))

    eeff_specs = [Spec(f'fig_juno_effective_energy.eeff.a{a * 100:.3g}', 237,
                       f'E_eff [MeV] at a_res = {a * 100:.3g}%', q,
                  lambda r, a=a: e_eff(r, a)) for a, q in EEFF.items()]
    return [
        Spec('appC.eeff_0', 231, 'E_eff [MeV] at a_res = 0', '3.5 MeV', lambda r: e_eff(r, 0.0)),
        Spec('appC.eeff_juno', 231, 'E_eff [MeV] at a_res = 2.61%', '4.1 MeV', lambda r: e_eff(r, juno.A_RES)),
        Spec('appC.ares_cross', 231, 'a_res [%] where JUNO b equals the 4 MeV closed form', '2.3%',
             lambda r: 100 * np.interp(r['vacuum']['juno_b_closed_4mev'], r['vacuum']['juno_b'],
                                       r['vacuum']['a_res'])),
        Spec('appC.offset', 220, 'DeepCore without matter - Eq. (3.4), theta23 fixed and fitted (max)', 'below 1',
             lambda r: (vacuum_limit(r, 'fixed'), vacuum_limit(r, 'fitted')), paper=(1.0,), check='bound'),
        Spec('appC.delta21_dyb', 220, 'Delta_21 at 1.65 km and 4 MeV', '0.04',
             lambda r: PHASE * NUFIT.dm21 * 1.65 / 4e-3),
        Spec('appC.delta21_810', 224, 'Delta_21 at 810 km and 2 GeV', '0.04', lambda r: PHASE * NUFIT.dm21 * 810 / 2),
        Spec('appC.nova_like', 227, 'Eq. (3.4) minus NOvA-like beam, neutrinos only, dCP 177, 0 deg', '4 to 7',
             lambda r: (-beam_shift(r, 'nova_like', 0, 'nufit'), -beam_shift(r, 'nova_like', 0, 'zero'))),
        Spec('appC.t2k_like', 227, 'T2K-like beam minus Eq. (3.4), all shares (max)', 'below 0.5', t2k_like,
             paper=(0.5,), check='bound'),
        Spec('appC.equal_shares', 227, 'b - Eq. (3.4), equal shares, dCP 177, 0: NOvA-like, T2K-like',
             'cancels for equal shares', lambda r: [beam_shift(r, beam, -1, tag) for beam in ('nova_like', 't2k_like')
                                                     for tag in ('nufit', 'zero')], check=''),
        Spec('appC.ahat_2gev', 227, 'A-hat at 2 GeV in the NOvA crust', '0.17',
             lambda r: A_PER_RHOYE_GEV * lbl_spectra.YE * lbl_spectra.inputs()['nova']['density'] * 2 / X0),
        Spec('appC.nova_narrower', 227, 'NOvA dis band narrower than the closed form [%] (min, max)',
             'about 11%', nova_narrower, paper=(11.0,), tol=0.5),
        Spec('appC.nova_center', 227, 'NOvA dis band center minus closed form center (max)',
             'close to that of the closed form', nova_center, check=''),
        Spec('appC.cf4_gap', 231, 'JUNO b at a_res = 2.61% minus the 4 MeV closed form', '2',
             lambda r: np.interp(juno.A_RES, r['vacuum']['a_res'], r['vacuum']['juno_b'])
             - float(r['vacuum']['juno_b_closed_4mev']), tol=0.5),
        Spec('appC.eq38', 231, '|Eq. (3.8) - weighted bin average| of JUNO at a_res = 2.61%', '1',
             lambda r: eq38_gap(r, juno.A_RES), check='bound'),
        *eeff_specs,
    ]


def app_d() -> list[Spec]:
    """App. D: parts and analytic forms of the atmospheric solution (Eqs. 4.1-4.2, D.4)."""
    def d(r):
        return r['deepcore']

    return [
        Spec('appD.y_20_50', 272, '|y_i| of the up-going 20-50 GeV bins, not gray (min, max)',
             '2.421 and 2.426', lambda r: preferred_range(r, 20.0, 50.0)),
        Spec('appD.solar_phase_mumu', 259, '|Dm2_phi,mumu| over the DeepCore bins, weighted mean',
             'negligible for DeepCore', lambda r: solar_phase_mumu(r)[1], check=''),
        Spec('appD.first_term', 270, '2 s13^2 cos 2theta12 Dm21 / c13^2', '1.3',
             lambda r: 2 * NUFIT.s13sq * (1 - 2 * NUFIT.s12sq) * NUFIT.dm21 / (1 - NUFIT.s13sq) * 1e6),
        Spec('appD.analytic_share', 275, 'frequency form: share [%] of delta_atm over x', '72--77%',
             lambda r: analytic_share(r, 'frequency_90'), tol=5.0),
        Spec('appD.omsd_share', 275, 'one mass scale: share [%] of delta_atm over x', '92--95%',
             lambda r: analytic_share(r, 'one_mass_scale')),
        Spec('appD.cp_amp_2', 275, 'exact b, NuFIT phases minus 90 deg, over x', '1.2--1.3', cp_term),
        Spec('appD.cp_mirror', 321, 'A-hat > 4 in the mirror octant (sin^2 theta23 = 0.439)', 'changes sign',
             lambda r: cp_cells(r, 2, 4.0), check=''),
    ]


def app_e() -> list[Spec]:
    """App. E and fig_deepcore_reproduction: the full fit of the DeepCore public release."""
    def ic(r):
        return r['icfit']

    return [
        Spec('appE.asimov_points', 346, 'Asimov points: number, first and last x', '25, 2.2--2.8',
             lambda r: (ic(r)['asimov_x'].size, ic(r)['asimov_x'][0] * 1e3, ic(r)['asimov_x'][-1] * 1e3)),
        Spec('appE.valid_tab', 384, '(i) tabulated response minus full fit (min, max over x)', '1.2--1.9',
             lambda r: above_full_fit(r, 'tab')),
        Spec('appE.valid_fixed', 384, '(ii) WBA on the fitter, splitting only, minus full fit (max)', '1.7',
             lambda r: above_full_fit(r, 'y')[1]),
        Spec('appE.valid_fixed_datm', 384, '(ii) over delta_atm of fig_deepcore_bin_weights', 'about a tenth',
             lambda r: above_full_fit(r, 'y')[1] / r['deepcore']['delta_atm'], paper=(0.1,), tol=0.03),
        Spec('appE.valid_all', 384, '(iii) WBA with all nuisances minus full fit (min, max)',
             'reaches the minimum of the full fit', lambda r: above_full_fit(r, 'all'), check=''),
        Spec('appE.nuisance', 398, 'nuisance parameters lower the WBA on the fitter (min, max over x)', '0.3--1.2',
             nuisance_effect),
        Spec('appE.repro_s23', 405, 'sin^2 theta23 of the NO data fit', '0.507 +- 0.058',
             lambda r: fitted(r, 'no', 's23sq')),
        Spec('appE.repro_dm32', 405, 'Dm32 of the NO data fit', '2.411 +- 0.078',
             lambda r: splitting(r, 'no', ic(r)['dm21_release'])),
        Spec('appE.hs_asimov_matched', 426, 'Asimov x0: b(true splitting) - b(matched reading)', '2',
             lambda r: ic(r)['asimov_x0_b_true'] - ic(r)['asimov_x0_b_matched']),
        Spec('appE.hs_asimov_naive', 426, 'Asimov x0: b(true splitting) - b(at |Dm31,IO|)', '14',
             lambda r: ic(r)['asimov_x0_b_true'] - ic(r)['asimov_x0_b_splitting']),
        Spec('appE.dm21_release', 428, 'Dm21 of the release fit [1e-5 eV^2]', '7.39',
             lambda r: ic(r)['dm21_release'] * 1e5),
        Spec('appE.data_no', 428, 'Dm31 of the NO data fit', '2.485 +- 0.078', lambda r: splitting(r, 'no')),
        Spec('appE.data_io', 429, 'Dm31 of the IO data fit', '-2.367 +- 0.076', lambda r: splitting(r, 'io')),
        Spec('appE.data_b', 429, 'b of the data fit', '117.8', lambda r: ic(r)['data_b']),
    ]


def app_f() -> list[Spec]:
    """App. F: the combination of two experiments."""
    return [
        Spec('appF.nova_dc_edge', 477, 'largest NOvA dis - DC nu_mu gap: true sin^2 theta23 below 0.53, 0.53 or more',
             '0.53 or more', nova_dc_edge, check=''),
        Spec('appF.joint_larger', 477, 'share of the true points where the joint NOvA-DC IO fit exceeds Eq. (1.1) '
             'at 1.5% [%]: dis, all', 'almost all', joint_larger, check=''),
        Spec('appF.joint_max', 477, 'largest joint NOvA-DC synergy at 1.5%: true sin^2 theta23, Eq. (1.1) there, '
             'joint value (dis, then all)', 'lower octant, nearly zero', joint_max, check=''),
    ]


SECTIONS = (    # title, label of the location, TeX file, numbers
    ('Abstract', 'Abstract', 'main', introduction),
    ('Sec. 2: the weighted bin average, fig_deepcore_bin_weights', 'Sec. 2', 'main', sec2),
    ('Sec. 3: reactors and accelerators, fig_accelerator_closed_form and fig_juno_intercept_vs_resolution',
     'Sec. 3', 'main', sec3),
    ('Sec. 4: atmospheric neutrinos, fig_deepcore_dcp, fig_deepcore_resolution, fig_deepcore_energy_window, and '
     'fig_deepcore_vs_orca6', 'Sec. 4', 'main', sec4),
    ('Sec. 5: unknown delta_CP and theta23, fig_solutions_plane, fig_intercept_ranges, fig_intercept_vs_truth, and '
     'fig_published_best_fits', 'Sec. 5', 'main', sec5),
    ('Sec. 6: synergies, fig_synergy_vs_precision and fig_synergy_rings', 'Sec. 6', 'main', sec6),
    ('Sec. 7: conclusions', 'Sec. 7', 'main', conclusions),
    ('App. A: inputs and detector models', 'App. A', 'app', app_a),
    ('App. B: the weighted bin average', 'App. B', 'app', app_b),
    ('App. C: reactors and accelerators, fig_juno_effective_energy', 'App. C', 'app', app_c),
    ('App. D: the atmospheric solution in matter', 'App. D', 'app', app_d),
    ('App. E: full fit of the DeepCore release, fig_full_fit_validation and fig_deepcore_reproduction',
     'App. E', 'app', app_e),
    ('App. F: combination of experiments', 'App. F', 'app', app_f),
)


def table(r: Results) -> tuple[list[tuple[str, list[Number]]], list[dict[str, str]]]:
    """The numbers of every section, and those that cannot be computed, with the reason."""
    groups, missing = [], []
    for title, label, tex, section in SECTIONS:
        rows = []
        for s in section():
            where = f'{label}, {tex}:{s.line}'
            try:
                rows.append(evaluate(s, r, where))
            except Missing as e:
                missing.append({'id': s.id, 'location': where, 'quantity': s.what, 'paper': s.quote,
                                'reason': str(e)})
        groups.append((title, rows))
    return groups, missing


def computed() -> tuple[dict[str, Number], dict[str, str]]:
    """Every computed number by id, and the reason for each number that cannot be computed."""
    groups, missing = table(Results())
    return {n.id: n for _, rows in groups for n in rows}, {m['id']: m['reason'] for m in missing}


def main() -> None:
    groups, missing = table(Results())
    for title, rows in groups:
        print(f'\n{title}')
        for n in rows:
            new = ', '.join(f'{v:.10g}' for v in n.new)
            mark = 'MOVED' if n.moved else ''
            print(f'  {n.id:<34} {n.where:<17} {n.what:<56} {n.quote:>22}  {new:<46} {mark}')
    print('\nNot computed')
    for m in missing:
        print(f'  {m["id"]:<34} {m["location"]:<17} {m["quantity"]:<56} {m["reason"]}')
    numbers = [{'id': n.id, 'section': title, 'location': n.where, 'quantity': n.what, 'paper': n.quote,
                'paper_values': list(n.paper), 'new': list(n.new), 'moved': n.moved}
               for title, rows in groups for n in rows]
    with open(RESULTS / 'numbers.json', 'w') as f:
        json.dump({'numbers': numbers, 'not_computed': missing}, f, indent=1)
    moved = sum(bool(n['moved']) for n in numbers)
    print(f'\n{len(numbers)} numbers, {moved} moved, {len(missing)} not computed: results/numbers.json')


if __name__ == '__main__':
    main()
