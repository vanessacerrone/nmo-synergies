"""Far-detector spectra of NOvA and T2K built from the public data releases (App. A.3).

A spectrum gives, per energy bin and beam mode (FHC: neutrino mode, RHC: antineutrino mode), the selected
events of each component if its oscillation probability were one:

    numu_mu, numubar_mu   surviving nu_mu and nubar_mu in the mu-like sample        x P_mumu, Pbar_mumu
    numu_e, numubar_e     appeared nu_e and nubar_e in the e-like sample             x P_mue, Pbar_mue
    nue_e, nuebar_e       beam nu_e and nubar_e in the e-like sample                 x P_ee, Pbar_ee
    nc_mu, nc_e           neutral currents, cosmic rays, other backgrounds           not oscillated

NOvA, in bins of reconstructed energy: the mu-like sample is the no-oscillation prediction of the 2024 data
release, split into nu_mu and nubar_mu by the ratio of their charged-current rates (near-detector flux of the
2017 flux release times the cross section); the appeared nu_e follow the same shapes, the beam nu_e the rates
of the beam nu_e, the e-like background the no-oscillation nu_mu shape. The mu-like background is the one of
the release; every other component is normalized to the predicted events of NOvA at its best fit
(arXiv:2509.04361, Table 2). The cross sections are those on CH2, the NOvA target, of GENIE 2.12.10
(data/genie_ch2_xsec, made by scripts/make_ch2_xsec.py).

T2K, in bins of true energy: flux at Super-Kamiokande of the 2020 flux release times the cross sections on
water of GENIE 2.12.10 (data/genie_water_xsec, made by scripts/make_water_xsec.py), with one efficiency per
component calibrated to the event numbers of the Hyper-Kamiokande Design Report (arXiv:1805.04163, Tables
XXXV-XXXVII), and one normalization per sample to the predicted events of T2K (arXiv:2101.03779, Table XI). The
e-like sample stops at 1.25 GeV.

Exposures, target masses, and the distance between the near and the far detector cancel in these
normalizations, so the rates below are in arbitrary units, common to all components of one beam mode. Numbers
typed from the publications are in data/lbl_inputs.toml, with their sources; the public files are in
data/external/.
"""
import tomllib
from dataclasses import dataclass

import numpy as np
import uproot

from .. import osc
from ..params import Params
from ..paths import DATA, EXTERNAL

CH2_XSEC = DATA / 'genie_ch2_xsec'             # GENIE 2.12.10, per nucleon of CH2 (NOvA)
WATER_XSEC = DATA / 'genie_water_xsec'         # GENIE 2.12.10, per nucleon of water (T2K)

MODES = ('FHC', 'RHC')
COMPONENTS = ('numu_mu', 'numubar_mu', 'nc_mu', 'numu_e', 'numubar_e', 'nue_e', 'nuebar_e', 'nc_e')
CHANNELS = {'mumu': (1, 1), 'mue': (0, 1), 'ee': (0, 0)}     # (beta, alpha) of P(alpha -> beta)
FLAVORS = ('numu', 'numubar', 'nue', 'nuebar')
YE = 0.5                            # electron fraction of the crust
N_SUB = 20                          # midpoints per bin in the integral of flux times cross section


@dataclass(frozen=True, eq=False)
class Spectra:
    """Selected events per bin if the oscillation probability were one (App. A.3).

    edges [GeV] are bins of true energy (true_energy=True) or of reconstructed energy;
    events[(mode, component)] has one entry per bin.
    """

    name: str
    l_km: float
    rhoye: float
    edges: np.ndarray
    true_energy: bool
    events: dict


def inputs() -> dict:
    """Numbers typed from the publications (data/lbl_inputs.toml)."""
    with open(DATA / 'lbl_inputs.toml', 'rb') as f:
        return tomllib.load(f)


def channels(prob: np.ndarray) -> dict:
    """P_mumu, P_mue, P_ee from the probabilities P[..., beta, alpha] of osc.probabilities."""
    return {k: prob[..., beta, alpha] for k, (beta, alpha) in CHANNELS.items()}


def samples(events: dict, mode: str, nu: dict, nubar: dict) -> tuple[np.ndarray, np.ndarray]:
    """mu-like and e-like events per bin of one beam mode (App. A.3).

    nu and nubar hold P_mumu, P_mue, P_ee of neutrinos and antineutrinos per bin; the backgrounds are not
    oscillated.
    """
    ev = {c: events[mode, c] for c in COMPONENTS}
    mu = nu['mumu'] * ev['numu_mu'] + nubar['mumu'] * ev['numubar_mu'] + ev['nc_mu']
    e = (nu['mue'] * ev['numu_e'] + nubar['mue'] * ev['numubar_e'] + nu['ee'] * ev['nue_e']
         + nubar['ee'] * ev['nuebar_e'] + ev['nc_e'])
    return mu, e


def _params(point: dict) -> Params:
    """Params of a published point given by sin^2 2theta12, sin^2 2theta13 (angles below pi/4), Dm21."""
    s12sq, s13sq = ((1 - np.sqrt(1 - point[k])) / 2 for k in ('sin2_2theta12', 'sin2_2theta13'))
    return Params(s12sq=s12sq, s13sq=s13sq, dm21=point['dm21'])


def _probabilities(e_gev, l_km, rhoye, point, p, dcp):
    """P_mumu, P_mue, P_ee at constant density of neutrinos and of antineutrinos, as (nu, nubar).

    point holds the published dm32 (normal ordering), dm21, and s23sq; p the other mixing parameters.
    """
    dm31 = point['dm32'] + point['dm21']
    return tuple(channels(osc.probabilities(osc.constant_amplitudes(e_gev, l_km, rhoye, dm31, p, anti),
                                            point['s23sq'], dcp, anti)) for anti in (False, True))


def _cross_sections(folder):
    """sigma/E per nucleon [1e-38 cm^2/GeV] of the GLoBES tables xsec_cc.dat and xsec_nc.dat in folder.

    CH2_XSEC (NOvA) and WATER_XSEC (T2K) hold sums of GENIE splines, smooth already, read as they are.
    Returns {'cc' or 'nc': (log10 E, {flavor: sigma/E})}.
    """
    columns = {'nue': 1, 'numu': 2, 'nuebar': 4, 'numubar': 5}
    out = {}
    for kind in ('cc', 'nc'):
        table = np.loadtxt(folder / f'xsec_{kind}.dat')
        out[kind] = (table[:, 0], {f: table[:, j] for f, j in columns.items()})
    return out


def _sigma(xsec, kind, flavor, e_gev):
    """Cross section per nucleon [1e-38 cm^2], zero below the table."""
    log_e, values = xsec[kind]
    v = values[flavor]
    return np.interp(np.log10(np.maximum(e_gev, 1e-3)), log_e, v, left=0.0, right=v[-1]) * e_gev


def _binned_rates(flux, xsec, edges):
    """Rates per bin, flux times cross section integrated over the bin.

    flux(flavor, e_gev) is the differential flux. The appeared nu_e and nubar_e use the nu_mu and nubar_mu
    flux with the nu_e and nubar_e cross sections; 'nc' sums the neutral currents of the four flavors.
    """
    t = (np.arange(N_SUB) + 0.5) / N_SUB
    e = edges[:-1, None] + np.diff(edges)[:, None] * t[None, :]
    width = np.diff(edges) / N_SUB

    def rate(flux_flavor, xsec_flavor, kind='cc'):
        return np.sum(flux(flux_flavor, e) * _sigma(xsec, kind, xsec_flavor, e), axis=1) * width

    return {'numu_mu': rate('numu', 'numu'), 'numubar_mu': rate('numubar', 'numubar'),
            'numu_e': rate('numu', 'nue'), 'numubar_e': rate('numubar', 'nuebar'),
            'nue_e': rate('nue', 'nue'), 'nuebar_e': rate('nuebar', 'nuebar'),
            'nc': sum(rate(f, f, 'nc') for f in FLAVORS)}


def _t2k_flux(mode):
    """T2K 2020 flux at Super-Kamiokande, histogram per flavor [per cm^2 per GeV per 1e21 POT], and edges."""
    horn = {'FHC': 'plus250kA', 'RHC': 'minus250kA'}[mode]
    table = np.genfromtxt(EXTERNAL / 't2k_flux_2020' / f't2kflux_2020_{horn}_nominal_sk.txt', skip_header=3,
                          usecols=(1, 3, 4, 5, 6, 7))
    edges = np.append(table[:, 0], table[-1, 1])
    values = {f: table[:, 2 + k] / 0.05 for k, f in enumerate(FLAVORS)}     # the file gives per 50 MeV

    def flux(flavor, e_gev):
        idx = np.searchsorted(edges, e_gev, side='right') - 1
        inside = (idx >= 0) & (idx < len(edges) - 1)
        return np.where(inside, values[flavor][np.clip(idx, 0, len(edges) - 2)], 0.0)

    return flux, edges


def _nova_flux(mode):
    """NOvA 2017 near-detector flux per flavor [per m^2 per GeV per 1e6 POT], linear between bin centers."""
    folder = EXTERNAL / 'nova_flux_2017'
    tables = {f: np.loadtxt(folder / f'{mode}_Flux_{f}_NOvA_ND_2017.txt') for f in FLAVORS}

    def flux(flavor, e_gev):
        lo, hi, n = tables[flavor][:, 0], tables[flavor][:, 1], tables[flavor][:, 2]
        out = np.interp(e_gev, (lo + hi) / 2, n / (hi - lo))
        return np.where((e_gev < lo[0]) | (e_gev > hi[-1]), 0.0, out)

    return flux


def _nova_release():
    """Edges and, per beam mode, the no-oscillation nu_mu CC prediction and mu-like background (NOvA 2024).

    The histograms sum the four quartiles of hadronic energy fraction; FHC and RHC have the same bins.
    """
    with uproot.open(EXTERNAL / 'nova_2024' / 'NOvA_2024_prediction_with_systs_histograms.root') as f:
        hists = {mode: f[f'prediction_components_numu_{mode.lower()}_all'] for mode in MODES}
        edges = hists['FHC']['NoOscillations_Signal'].axis().edges()
        release = {mode: (d['NoOscillations_Signal'].values(),
                          d['Oscillated_Beam_bkg'].values() + d['Cosmic_bkg'].values())
                   for mode, d in hists.items()}
    return edges, release


def nova() -> Spectra:
    """NOvA far-detector spectra in reconstructed energy (App. A.3)."""
    cfg = inputs()['nova']
    bf = cfg['best_fit']
    rhoye = YE * cfg['density']
    xsec = _cross_sections(CH2_XSEC)
    edges, release = _nova_release()
    e = (edges[:-1] + edges[1:]) / 2
    nu, nubar = _probabilities(e, cfg['baseline_km'], rhoye, bf, _params(bf), bf['dcp_over_pi'] * np.pi)
    events = {}
    for mode in MODES:
        base, background = release[mode]
        target = cfg['events'][mode]
        rates = _binned_rates(_nova_flux(mode), xsec, edges)
        share = rates['numu_mu'] / (rates['numu_mu'] + rates['numubar_mu'])
        shapes = {'numu_mu': (base * share, nu['mumu']), 'numubar_mu': (base * (1 - share), nubar['mumu']),
                  'numu_e': (base * share, nu['mue']), 'numubar_e': (base * (1 - share), nubar['mue'])}
        for c, (shape, prob_c) in shapes.items():
            events[mode, c] = target[c] * shape / np.sum(prob_c * shape)
        beam = np.sum(nu['ee'] * rates['nue_e'] + nubar['ee'] * rates['nuebar_e'])
        events[mode, 'nue_e'] = target['beam_e'] * rates['nue_e'] / beam
        events[mode, 'nuebar_e'] = target['beam_e'] * rates['nuebar_e'] / beam
        events[mode, 'nc_mu'] = background
        events[mode, 'nc_e'] = (target['nc_e'] + target['cosmic_e'] + target['other_e']) * base / base.sum()
    return Spectra('NOvA', cfg['baseline_km'], rhoye, edges, False, events)


def _calibrate(rates, nu, nubar, targets, e_like):
    """One constant efficiency per component: its events at the given probabilities equal the target."""
    weights = {'numu_mu': nu['mumu'], 'numubar_mu': nubar['mumu'], 'numu_e': nu['mue'],
               'numubar_e': nubar['mue'], 'nue_e': nu['ee'], 'nuebar_e': nubar['ee']}
    events = {}
    for c, w in weights.items():
        shape = rates[c] * e_like if c.endswith('_e') else rates[c]
        events[c] = targets[c] * shape / np.sum(w * shape)
    events['nc_mu'] = targets['nc_mu'] * rates['nc'] / rates['nc'].sum()
    events['nc_e'] = targets['nc_e'] * rates['nc'] * e_like / np.sum(rates['nc'] * e_like)
    return events


def t2k(e_max_gev: float = 10.0) -> Spectra:
    """T2K far-detector spectra in true energy (App. A.3), bins of the T2K flux release up to e_max_gev."""
    cfg = inputs()
    t2k_cfg, hk = cfg['t2k'], cfg['hyperk']
    l_km, rhoye = t2k_cfg['baseline_km'], YE * t2k_cfg['density']
    xsec = _cross_sections(WATER_XSEC)
    edges = _t2k_flux('FHC')[1]
    edges = edges[edges <= e_max_gev * (1 + 1e-9)]
    e = (edges[:-1] + edges[1:]) / 2
    e_like = (e < hk['e_like_max_gev']).astype(float)
    hk_ref, t2k_ref = hk['reference'], t2k_cfg['reference']
    prob_hk = _probabilities(e, l_km, rhoye, hk_ref, _params(hk_ref), hk_ref['dcp_over_pi'] * np.pi)
    p_t2k = Params(s12sq=t2k_ref['s12sq'], s13sq=t2k_ref['s13sq'], dm21=t2k_ref['dm21'])
    prob_t2k = _probabilities(e, l_km, rhoye, t2k_ref, p_t2k, t2k_cfg['events']['dcp_over_pi'] * np.pi)
    events = {}
    for mode in MODES:
        rates = _binned_rates(_t2k_flux(mode)[0], xsec, edges)
        h = hk['events'][mode]
        targets = {c: h[c] for c in ('numu_e', 'numubar_e', 'nue_e', 'nuebar_e', 'nc_e', 'nc_mu')}
        targets['numu_mu'] = h['numu_ccqe_mu'] + h['numu_ccnqe_mu']
        targets['numubar_mu'] = h['numubar_ccqe_mu'] + h['numubar_ccnqe_mu']
        events.update({(mode, c): v for c, v in _calibrate(rates, *prob_hk, targets, e_like).items()})
        mu, e_tot = (v.sum() for v in samples(events, mode, *prob_t2k))
        norm = {'mu': t2k_cfg['events'][f'{mode}_mu'] / mu, 'e': t2k_cfg['events'][f'{mode}_e'] / e_tot}
        for c in COMPONENTS:
            events[mode, c] = events[mode, c] * norm[c.rsplit('_', 1)[1]]
    first = np.argmax(np.any([v > 0 for v in events.values()], axis=0))     # empty below the cross sections
    events = {k: v[first:] for k, v in events.items()}
    return Spectra('T2K', l_km, rhoye, edges[first:], True, events)
