"""IceCube DeepCore and KM3NeT/ORCA6 against the true splitting (Sec. 4, fig_deepcore_vs_orca6; App. A.5).

The inverted-ordering fit is at its global minimum over Dm31_IO, theta23 in both octants, and delta_CP^IO
(nmosyn.fit.global_fit) for DeepCore with all channels, ORCA6 with all classes, and the ORCA6 high-purity tracks.
Also the ORCA6 intercept with theta23 fixed and the NuFIT phases (App. A.5) for all classes, the high-purity
tracks, and the tracks of both purities, the latter also at the global minimum; the share of the weight carried by
the showers (Sec. 4), and the predicted against the observed total of the release.
Output: results/orca.npz. Run from the repository root: python -m compute.orca
"""
import numpy as np

from nmosyn import fit
from nmosyn.average import bin_solutions, gauss_newton
from nmosyn.experiments.deepcore import DeepCore
from nmosyn.experiments.orca import Orca6
from nmosyn.params import DCP_IO, DCP_NO, NUFIT, S23SQ, X0, intercept
from nmosyn.paths import RESULTS

X_ORCA = np.unique(np.append(np.linspace(2.2e-3, 2.8e-3, 7), X0))
LEAD = 2 * (1 - NUFIT.s12sq) * NUFIT.dm21


def main():
    out = dict(x=X_ORCA)
    models = {'dc_all': DeepCore('all'), 'orca_all': Orca6(), 'orca_tracks': Orca6(classes=(1,))}
    for name, model in models.items():
        h = fit.Harmonics(model)
        sols = [fit.global_fit(h, x, model.counts(x, S23SQ, DCP_NO), (70, 170)) for x in X_ORCA]
        out[f'{name}_b'] = np.array([s.b for s in sols])
        out[f'{name}_s23sq'] = np.array([s.s23sq for s in sols])
        out[f'{name}_dcp'] = np.array([s.dcp for s in sols])
        out[f'{name}_chi2'] = np.array([s.chi2 for s in sols])

    # App. A.5: the classes added one at a time at x0, with theta23 and the phases fixed and at the global minimum
    lp = Orca6(classes=(1, 3))
    sol = fit.global_fit(fit.Harmonics(lp), X0, lp.counts(X0, S23SQ, DCP_NO), (70, 170))
    out['orca_tracks_lp_b'], out['orca_tracks_lp_s23sq'] = sol.b, sol.s23sq
    for name, model in (('orca_tracks', models['orca_tracks']), ('orca_tracks_lp', lp)):
        data = model.counts(X0, S23SQ, DCP_NO)
        y = gauss_newton(lambda v: model.counts(v, S23SQ, DCP_IO), data, -X0 + LEAD)
        out[f'{name}_fixed_b'] = intercept(y, X0)

    orca = models['orca_all']
    data = orca.counts(X0, S23SQ, DCP_NO)
    io = lambda v: orca.counts(v, S23SQ, DCP_IO)
    y = gauss_newton(io, data, -X0 + LEAD)
    out['orca_fixed_b'] = intercept(y, X0)
    _, omega = bin_solutions(io, data, y)
    out['orca_fixed_class_weight'] = omega.reshape(3, -1).sum(axis=1) / omega.sum()
    i = int(np.argmin(np.abs(X_ORCA - X0)))
    _, omega = bin_solutions(lambda v: orca.counts(v, out['orca_all_s23sq'][i], out['orca_all_dcp'][i]), data,
                             out['orca_all_b'][i] * 1e-6 - X0)
    out['orca_global_class_weight'] = omega.reshape(3, -1).sum(axis=1) / omega.sum()
    out['orca_predicted_total'] = data.sum()
    out['orca_observed_total'] = orca.observed.sum()

    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(RESULTS / 'orca.npz', **out)


if __name__ == '__main__':
    main()
