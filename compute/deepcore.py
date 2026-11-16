"""IceCube DeepCore with the tabulated response of the public release (Sec. 4, App. B.2, App. D):
fig_deepcore_bin_weights, fig_deepcore_dcp, fig_deepcore_resolution, fig_deepcore_energy_window, and
fig_full_fit_validation.

fig_deepcore_bin_weights: weights and preferred values of the bins at the NuFIT point (nu_mu disappearance, theta23 and
the phases fixed). fig_deepcore_dcp: b against the true delta_CP, with and without matter, theta23 fixed in both
orderings (delta_CP^IO = 285 deg and the range over delta_CP^IO); also with theta23 at the global minimum with matter
and fitted at delta_CP^IO = 285 deg without (fig3_b, fig3_lo, fig3_hi, fig3_b_vacuum). fig_deepcore_resolution and
fig_deepcore_energy_window: Gaussian resolution and energy window. fig_full_fit_validation: all channels, delta_CP = 177
deg in both orderings, theta23 fitted. Fits start from the vacuum value (App. B.3). Also the numbers of the text: the
slope of the line and the exchange of the orderings. Output: results/deepcore.npz. Run from the repository root: python
-m compute.deepcore
"""
import numpy as np

from nmosyn import fit
from nmosyn.average import bin_solutions, gauss_newton
from nmosyn.experiments.deepcore import (DeepCore, E_MAX_SCAN, E_MIN_SCAN, SIGMA_SCAN, edges_one_sigma,
                                         window_edges)
from nmosyn.params import DCP_IO, DCP_NO, NUFIT, S23SQ, X0, X_GRID, b_lbl, intercept
from nmosyn.paths import RESULTS

DCP_SCAN_FIG3 = np.radians(np.arange(-180.0, 180.1, 15.0))
DCP_GRID = np.radians(np.arange(-180.0, 180.1, 30.0))
X_LINE = np.round(np.arange(2.2e-3, 2.8001e-3, 0.05e-3), 7)
LEAD = 2 * (1 - NUFIT.s12sq) * NUFIT.dm21           # 2 c12^2 Dm21, the common part of Eq. (4.1)


def fixed(model, x, dcp_no=DCP_NO, dcp_io=DCP_IO, s23sq=S23SQ):
    """Dm31_IO with theta23 and both phases fixed (Gauss-Newton, App. B.3)."""
    return gauss_newton(lambda v: model.counts(v, s23sq, dcp_io), model.counts(x, s23sq, dcp_no),
                        -x + b_lbl(dcp_no, dcp_io, s23sq, s23sq))


def theta_fitted(h, data, x, dcp_io, intervals=((0.0, 0.5), (0.5, 1.0)), start=None):
    """Minimum over (Dm31_IO, theta23) at fixed delta_CP^IO, one start per interval of sin^2 theta23."""
    y0 = -x + LEAD if start is None else start
    lo, hi = np.array(intervals).T
    c, y, s, _ = fit.minimize(h, data, np.full(len(lo), y0), np.clip(S23SQ, lo, hi), dcp_io,
                              free=(True, True, False), s23sq_bounds=(lo, hi))
    i = int(np.argmin(c))
    return c[i], y[i], s[i]


def main():
    out = dict(x=X_GRID, dcp_fig3=DCP_SCAN_FIG3, dcp_grid=DCP_GRID, x_line=X_LINE, sigma=SIGMA_SCAN,
               e_max=E_MAX_SCAN, e_min=E_MIN_SCAN)
    numu, allch = DeepCore('numu'), DeepCore('all')

    # fig_deepcore_bin_weights and the line with theta23 and the phases fixed
    y0 = fixed(numu, X0)
    data = numu.counts(X0, S23SQ, DCP_NO)
    y_i, omega = bin_solutions(lambda v: numu.counts(v, S23SQ, DCP_IO), data, y0)
    out.update(fig1_y=y0, fig1_y_i=y_i, fig1_omega=omega, e_edges=numu.e_edges, cosz_edges=numu.cosz_edges)
    out['numu_fixed_line'] = np.array([intercept(fixed(numu, x), x) for x in X_GRID])
    out['numu_fixed_grid'] = np.array([[intercept(fixed(numu, X0, dn, di), X0) for di in DCP_GRID]
                                       for dn in DCP_GRID])

    # fig_deepcore_dcp: theta23 at the global minimum with matter; theta23 fitted at delta_CP^IO = 285 deg without
    h = fit.Harmonics(numu)
    vacuum = DeepCore('numu', density_scale=0.0)
    hv = fit.Harmonics(vacuum)
    b285, lo, hi, s285, bvac, svac = [], [], [], [], [], []
    for d in DCP_SCAN_FIG3:
        data = numu.counts(X0, S23SQ, d)
        sol = fit.global_fit(h, X0, data, (70, 190))
        o = np.argmin(sol.profile['chi2'], axis=0)
        b_prof = intercept(sol.profile['y'][o, np.arange(o.size)], X0)
        lo.append(min(b_prof.min(), sol.b))
        hi.append(max(b_prof.max(), sol.b))
        _, y, s = theta_fitted(h, data, X0, DCP_IO, start=sol.y)
        b285.append(intercept(y, X0))
        s285.append(s)
        _, y, s = theta_fitted(hv, vacuum.counts(X0, S23SQ, d), X0, DCP_IO)
        bvac.append(intercept(y, X0))
        svac.append(s)
    out.update(fig3_b=np.array(b285), fig3_lo=np.array(lo), fig3_hi=np.array(hi), fig3_s23sq=np.array(s285),
               fig3_b_vacuum=np.array(bvac), fig3_s23sq_vacuum=np.array(svac))
    # the version of fig_deepcore_dcp in the paper: theta23 fixed at its true value in both orderings
    band = np.array([[intercept(fixed(numu, X0, d, di), X0) for di in DCP_SCAN_FIG3[:-1]] for d in DCP_SCAN_FIG3])
    out.update(fig3_fixed_b=np.array([intercept(fixed(numu, X0, d), X0) for d in DCP_SCAN_FIG3]),
               fig3_fixed_lo=band.min(axis=1), fig3_fixed_hi=band.max(axis=1),
               fig3_fixed_b_vacuum=np.array([intercept(fixed(vacuum, X0, d), X0) for d in DCP_SCAN_FIG3]))

    # fig_deepcore_resolution and fig_deepcore_energy_window: Gaussian resolution and window, theta23 and the phases
    # fixed
    out['fig4_b_sigma'] = np.array([intercept(fixed(numu.gaussian(edges_one_sigma(s), sigma_lne=s), X0), X0)
                                    for s in SIGMA_SCAN])
    out['fig4_b_release_resolution'] = intercept(fixed(numu.gaussian(), X0), X0)
    out['fig4_b_e_max'] = np.array([intercept(fixed(numu.gaussian(window_edges(e_max=e)), X0), X0) for e in E_MAX_SCAN])
    out['fig4_b_e_min'] = np.array([intercept(fixed(numu.gaussian(window_edges(e_min=e)), X0), X0) for e in E_MIN_SCAN])

    # fig_full_fit_validation: all channels, theta23 fitted, dCP = 177 deg in both orderings
    ha = fit.Harmonics(allch)
    line, s_line = [], []
    for x in X_LINE:
        _, y, s = theta_fitted(ha, allch.counts(x, S23SQ, DCP_NO), x, DCP_NO, intervals=((0.5, 0.7),))
        line.append(intercept(y, x))
        s_line.append(s)
    out['fig6_line'], out['fig6_s23sq'] = np.array(line), np.array(s_line)
    out['all_fixed_line'] = np.array([intercept(fixed(allch, x), x) for x in X_GRID])

    # numbers of App. B.2: the exchange of the orderings
    for name, model in (('numu', numu), ('all', allch)):
        y = fixed(model, X0)
        x_back = gauss_newton(lambda v: model.counts(v, S23SQ, DCP_NO), model.counts(y, S23SQ, DCP_IO), X0)
        out[f'exchange_{name}'] = (x_back - X0) * 1e6
    out['delta_atm'] = intercept(y0, X0) - LEAD * 1e6

    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(RESULTS / 'deepcore.npz', **out)


if __name__ == '__main__':
    main()
