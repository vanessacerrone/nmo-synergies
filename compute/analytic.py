"""Analytic forms of the atmospheric solution (Sec. 4, App. D.1-D.2): fig_deepcore_analytic, its scalings with theta13
and density, and the CP residual of DeepCore split over the true cells.

The DeepCore intercept (nu_mu disappearance, tabulated response, theta23 and the phases fixed) is computed three
ways and compared:
  exact          the weighted bin average on the exact probabilities;
  frequency      every true cell prefers the intercept of Eq. (4.2) in its exact two-level form, and the cells are
                 averaged with the weights the detector gives them;
  one mass scale the weighted bin average on the probability with Dm21 = 0 (App. D.1), plus the decoupled value.
The two analytic forms use the length of each path from production and its mean density (earth.layers). Also the
matter shift of one mantle path in narrow energy bins, against the second-order formula of Eq. (4.2) (Sec. 4).
The CP residual (App. D.2): b(180, 180 deg) - b(0, 0) for nu_mu disappearance with theta23 fixed, to first order in
the expected counts a sum over the true cells, grouped by A-hat = A / Dm31 at the mean rho*Ye of each path, with
and without matter, at sin^2 theta23 = 0.561, at maximal mixing, and in the mirror octant; and without matter, b
minus Eq. (3.4) with the same delta_CP in both orderings (App. D.1). The amplitude term of Eq. (D.4) on the mantle
path (App. D.2): the cos(delta_CP) part of the exact P_mumu at constant density, per unit of its oscillation
sin^2, split into the part odd in cos 2theta23 (sin^2 theta23 = 0.561 against its mirror 0.439), which Eq. (D.4)
describes, and the even part, which it does not.
Output: results/analytic.npz. Run from the repository root: python -m compute.analytic
"""
from dataclasses import replace

import numpy as np

from nmosyn import osc
from nmosyn.average import gauss_newton, poisson_weights, sensitivity
from nmosyn.earth import mean_rhoye
from nmosyn.experiments.deepcore import DEPTH_KM, OVERBURDEN, DeepCore, _detector_flux
from nmosyn.params import DCP_IO, DCP_NO, NUFIT, S23SQ, X0, b_lbl, intercept
from nmosyn.paths import RESULTS

X_LINE = np.linspace(2.2e-3, 2.8e-3, 7)
X_SCALING = 2.5e-3
PHASES = {'90': (np.pi / 2, np.pi / 2), 'nufit': (DCP_NO, DCP_IO)}
PATH_KM, PATH_RHOYE = 9000.0, 2.29              # one mantle path (Sec. 4)
PATH_ENERGIES = np.geomspace(9.0, 250.0, 40)    # GeV; bins of +-3% around each, 120 points
CP_S23SQ = np.array([S23SQ, 0.5, 1 - S23SQ])     # App. D.2: NuFIT value, maximal mixing, mirror octant
AHAT_EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0, np.inf])
DCP_EQUAL = np.radians(np.arange(-180.0, 180.1, 30.0))
CP_AMP_AHAT = np.linspace(8.0, 18.0, 11)           # App. D.2: A-hat of the amplitude check on the mantle path


def decoupled(p):
    """2 (c12^2 - s13^2 s12^2) Dm21 / c13^2: what every bin prefers above the resonance without the matter shift."""
    return 2 * ((1 - p.s12sq) - p.s13sq * p.s12sq) * p.dm21 / (1 - p.s13sq)


def block(m, a, p):
    """The 2x2 block of nu_e and the third rotated state, R23-rotated Hamiltonian times 2E with Dm21 = 0 (App. D.1):
    diagonal (a + s13^2 m, c13^2 m), off-diagonal c13 s13 m, for the signed splitting m and potential a [eV^2]."""
    return a + p.s13sq * m, (1 - p.s13sq) * m, np.sqrt(p.s13sq * (1 - p.s13sq)) * m


def third_level(m, a, p):
    """Eigenvalue of the block continuously connected to c13^2 m: the level of the nu_mu - nu_tau system."""
    d_e, d_3, off = block(m, a, p)
    centre, half = 0.5 * (d_e + d_3), np.hypot(0.5 * (d_e - d_3), off)
    return np.where(d_3 >= d_e, centre + half, centre - half)


def exact(model, x, phases):
    """b of the weighted bin average on the exact probabilities, theta23 and the phases fixed (Eq. 2.3)."""
    dcp_no, dcp_io = phases
    y = gauss_newton(lambda v: model.counts(v, S23SQ, dcp_io), model.counts(x, S23SQ, dcp_no),
                     -x + b_lbl(dcp_no, dcp_io))
    return intercept(y, x)


def frequency(model, x, phases, density_scale=1.0):
    """b from the two-level frequency shift of every true cell, averaged with the detector weights (App. D.1).

    Linearizing the reconstructed bins i around the solution, cell c enters with the weight
    g_c sum_i K_ci W_i s_i: g_c the derivative of its count in Dm31, K the response, W_i = 1/N_i^NO, and
    s_i = sum_c K_ci g_c; it prefers b_c = [2 (c12^2 - s13^2 s12^2) Dm21 - lambda(x) - lambda(-x)] / c13^2.
    """
    p, (dcp_no, dcp_io) = model.p, phases
    y = exact(model, x, phases) * 1e-6 - x
    weight_bins = poisson_weights(model.counts(x, S23SQ, dcp_no))
    rhoye = density_scale * mean_rhoye(model.cosz, DEPTH_KM, OVERBURDEN)[1]
    slope, response = {}, {}
    for anti, kind in ((False, 'nu'), (True, 'nubar')):
        response[kind] = model.response(('cc', kind, 1))
        p_mumu = [osc.probabilities(model._amplitudes(v)[anti], S23SQ, dcp_io, anti)[:, 1, 1]
                  for v in (y + 1e-6, y - 1e-6)]
        slope[kind] = model.norm * model.flux['numu' + kind[2:]] * (p_mumu[0] - p_mumu[1]) / 2e-6
    s_bins = sum(slope[kind] @ response[kind] for kind in slope)
    total, norm = 0.0, 0.0
    for kind, sign in (('nu', 1.0), ('nubar', -1.0)):
        w = slope[kind] * (response[kind] @ (weight_bins * s_bins))
        a = sign * osc.A_PER_RHOYE_GEV * rhoye * model.e
        shift = -third_level(x, a, p) - third_level(-x, a, p)       # the c13^2 m terms of x and -x cancel
        total += np.sum(w * (decoupled(p) + shift / (1 - p.s13sq)))
        norm += np.sum(w)
    return total / norm * 1e6


def one_mass_scale(model, x, density_scale=1.0):
    """b of the weighted bin average on the one-mass-scale probability at the mean density of each path, plus the
    decoupled value (App. D.1): P_mumu = |c23^2 + s23^2 sum_k |v_3k|^2 exp(-i lambda_k L / 2E)|^2."""
    p = model.p
    length, rhoye = mean_rhoye(model.cosz, DEPTH_KM, OVERBURDEN)
    rhoye = density_scale * rhoye
    response = {kind: model.response(('cc', kind, 1)) for kind in ('nu', 'nubar')}

    def counts(m):
        out = 0.0
        for kind, sign in (('nu', 1.0), ('nubar', -1.0)):
            d_e, d_3, off = block(m, sign * osc.A_PER_RHOYE_GEV * rhoye * model.e, p)
            centre, half = 0.5 * (d_e + d_3), np.hypot(0.5 * (d_e - d_3), off)
            amplitude = (1 - S23SQ) + 0j
            for level in (centre + half, centre - half):
                share = (level - d_e) ** 2 / ((level - d_e) ** 2 + off ** 2)
                amplitude = amplitude + S23SQ * share * np.exp(-1j * osc.PHASE_2E * level * length / model.e)
            out = out + (model.flux['numu' + kind[2:]] * np.abs(amplitude) ** 2) @ response[kind]
        return out

    y = gauss_newton(counts, counts(x), -x + decoupled(p))
    return intercept(y, x) + decoupled(p) * 1e6


def cp_solution(model, dcp, s23sq):
    """Dm31_IO with delta_CP = dcp in both orderings and theta23 fixed at s23sq (Gauss-Newton, App. B.3)."""
    return gauss_newton(lambda v: model.counts(v, s23sq, dcp), model.counts(X0, s23sq, dcp),
                        -X0 + b_lbl(dcp, dcp, s23sq, s23sq))


def cp_cells(model, s23sq):
    """Contribution of each true fine cell to b(180, 180 deg) - b(0, 0), theta23 fixed (App. D.2) [1e-6 eV^2].

    Linearized around the solution y of one pair of phases, the Gauss-Newton step moves y by
    sum_i W_i s_i dr_i / sum_i W_i s_i^2 for the other pair, dr_i the change of the data minus that of the model
    at y. dr_i is the response applied to the changes of the flux at the detector in the true cells, so each cell
    enters with its own term. The linearizations around (0, 0) and around (180, 180 deg) are averaged.
    """
    def moved(ref, other):
        y = cp_solution(model, ref, s23sq)
        data = model.counts(X0, s23sq, ref)
        w = poisson_weights(data)
        s = sensitivity(lambda v: model.counts(v, s23sq, ref), y)
        rates = {(v, d): _detector_flux(model._amplitudes(v), s23sq, d, model.flux, model._sources)
                 for v in (X0, y) for d in (ref, other)}
        out = 0.0
        for key in model._kernels:
            change = (rates[X0, other][key[1:]] - rates[X0, ref][key[1:]]
                      - rates[y, other][key[1:]] + rates[y, ref][key[1:]])
            out = out + (model.response(key) @ (w * s)) * change
        return model.norm * out / np.sum(w * s * s)

    return 0.5 * (moved(0.0, np.pi) - moved(np.pi, 0.0)) * 1e6


def cp_split(numu):
    """CP residual of DeepCore nu_mu disappearance by A-hat, with and without matter (App. D.2)."""
    ahat = osc.A_PER_RHOYE_GEV * mean_rhoye(numu.cosz, DEPTH_KM, OVERBURDEN)[1] * numu.e / X0
    group = np.digitize(ahat, AHAT_EDGES) - 1
    out = {'cp_s23sq': CP_S23SQ, 'cp_ahat_edges': AHAT_EDGES}
    vacuum = DeepCore('numu', density_scale=0.0)
    for tag, model in (('matter', numu), ('vacuum', vacuum)):
        out[f'cp_cells_{tag}'] = np.array([np.bincount(group, cp_cells(model, s), AHAT_EDGES.size - 1)
                                           for s in CP_S23SQ])
    out['vacuum_equal_dcp'] = DCP_EQUAL
    out['vacuum_equal_residual'] = np.array([intercept(cp_solution(vacuum, d, S23SQ), X0) - b_lbl(d, d) * 1e6
                                             for d in DCP_EQUAL])
    out['cp_span'] = np.array([intercept(cp_solution(numu, np.pi, s), X0) - intercept(cp_solution(numu, 0.0, s), X0)
                               for s in CP_S23SQ])
    out['cp_span_vacuum'] = (b_lbl(np.pi, np.pi, CP_S23SQ, CP_S23SQ) - b_lbl(0.0, 0.0, CP_S23SQ, CP_S23SQ)) * 1e6
    return out


def cp_amplitude(p=NUFIT):
    """Amplitude term of Eq. (D.4) against the exact P_mumu on the mantle path, NO at x0, delta_CP^IO = delta_CP^NO.

    At each A-hat the cos(delta_CP) part of P_mumu, (P(0) - P(180 deg)) / 2, is divided by the oscillation
    sin^2(Omega L / 4E), taken from the even part as (1 - (P(0) + P(180 deg)) / 2) / (1 - cos^2 2theta23). Eq. (D.4)
    gives 2 cos 2theta23 epsilon, with epsilon at cos(delta_CP) = 1. Returns the parts odd and even in
    cos 2theta23, from sin^2 theta23 = S23SQ and 1 - S23SQ, each over Eq. (D.4) at S23SQ; (neutrinos,
    antineutrinos) x A-hat.
    """
    e = CP_AMP_AHAT * X0 / (osc.A_PER_RHOYE_GEV * PATH_RHOYE)
    c = 1 - 2 * S23SQ
    epsilon = np.sqrt(p.s13sq * 4 * p.s12sq * (1 - p.s12sq) * 4 * S23SQ * (1 - S23SQ)) * p.dm21 / X0
    odd, even = [], []
    for anti in (False, True):
        amp = osc.constant_amplitudes(e, PATH_KM, PATH_RHOYE, X0, p, anti)
        term = []
        for s23sq in (S23SQ, 1 - S23SQ):
            prob = [osc.probabilities(amp, s23sq, d, anti)[:, 1, 1] for d in (0.0, np.pi)]
            oscillation = (1 - 0.5 * (prob[0] + prob[1])) / (1 - (1 - 2 * s23sq) ** 2)
            term.append(0.5 * (prob[0] - prob[1]) / oscillation)
        odd.append(0.5 * (term[0] - term[1]) / (2 * c * epsilon))
        even.append(0.5 * (term[0] + term[1]) / (2 * c * epsilon))
    return np.array(odd), np.array(even)


def mantle_path(antineutrino, p=NUFIT):
    """b on the mantle path in narrow energy bins, delta_CP = 90 deg (Sec. 4, App. D.1): with matter and without,
    each (n_energies,)."""
    out = []
    for e in PATH_ENERGIES:
        grid = np.linspace(0.97 * e, 1.03 * e, 120)
        b = []
        for rhoye in (PATH_RHOYE, 0.0):
            def counts(y):
                amp = osc.constant_amplitudes(grid, PATH_KM, rhoye, y, p, antineutrino)
                return np.array([osc.probabilities(amp, S23SQ, np.pi / 2, antineutrino)[:, 1, 1].sum()])
            b.append(intercept(gauss_newton(counts, counts(X0), -X0 + decoupled(p)), X0))
        out.append(b)
    return np.array(out).T


def main():
    out = dict(x=X_LINE, x_scaling=X_SCALING, lead=2 * (1 - NUFIT.s12sq) * NUFIT.dm21 * 1e6)
    numu = DeepCore('numu')
    for tag, phases in PHASES.items():
        out[f'exact_{tag}'] = np.array([exact(numu, x, phases) for x in X_LINE])
        out[f'frequency_{tag}'] = np.array([frequency(numu, x, phases) for x in X_LINE])
    out['one_mass_scale'] = np.array([one_mass_scale(numu, x) for x in X_LINE])
    out['exact_nominal'] = exact(numu, X_SCALING, PHASES['90'])
    out['frequency_nominal'] = frequency(numu, X_SCALING, PHASES['90'])
    out['one_mass_scale_nominal'] = one_mass_scale(numu, X_SCALING)
    for factor in (0.8, 1.2):
        model = DeepCore('numu', p=replace(NUFIT, s13sq=NUFIT.s13sq * factor))
        out[f's13_{factor}'] = np.array([exact(model, X_SCALING, PHASES['90']),
                                         frequency(model, X_SCALING, PHASES['90']), one_mass_scale(model, X_SCALING)])
    for factor in (0.95, 1.05):
        model = DeepCore('numu', density_scale=factor)
        out[f'density_{factor}'] = np.array([exact(model, X_SCALING, PHASES['90']),
                                             frequency(model, X_SCALING, PHASES['90'], factor),
                                             one_mass_scale(numu, X_SCALING, factor)])
    out['path_energies'] = PATH_ENERGIES
    out['path_a_hat'] = osc.A_PER_RHOYE_GEV * PATH_RHOYE * PATH_ENERGIES / X0
    for kind, anti in (('nu', False), ('nubar', True)):
        matter, vacuum = mantle_path(anti)
        out[f'path_b_{kind}'], out[f'path_shift_{kind}'] = matter, matter - vacuum
    out.update(cp_split(numu))
    out['cp_amp_ahat'] = CP_AMP_AHAT
    out['cp_amp_odd'], out['cp_amp_even'] = cp_amplitude()
    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(RESULTS / 'analytic.npz', **out)


if __name__ == '__main__':
    main()
