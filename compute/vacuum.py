"""Reactors and accelerators near the vacuum limit (Sec. 3, App. C): fig_accelerator_closed_form,
fig_juno_intercept_vs_resolution, fig_accelerator_nubar_share, fig_juno_effective_energy, and the JUNO lines.

Generic beam (fig_accelerator_closed_form): b against the true delta_CP with delta_CP^IO = 285 deg, and on a 30-deg grid
of both phases for the band over delta_CP^IO. JUNO (fig_juno_intercept_vs_resolution): b against the resolution
sigma_E / E at 1 MeV, the three terms scaled together, in matter and in vacuum, and the line on X_GRID at the baseline,
2.94% at 1 MeV (a_res = 2.61%). fig_accelerator_nubar_share (not in the paper, values in App. C.2): the shift from the closed form against the
antineutrino share. fig_juno_effective_energy: the JUNO weights per bin and the effective energy. theta23 and the phases
are fixed in these fits, so the Gauss-Newton iteration of App. B.3 in Dm31_IO alone gives the solution. Sec. 5: the NOvA
disappearance band of fig_intercept_vs_truth (left) without matter, at x0 over the true delta_CP, as in compute.bands
(global minimum, valley rule of App. B.3, closed form over the valley); the band with matter is in results/bands.npz.
Output: results/vacuum.npz. Run from the repository root: python -m compute.vacuum
"""
from dataclasses import replace

import numpy as np

from compute.bands import DT, solve as band
from nmosyn import fit
from nmosyn.average import gauss_newton
from nmosyn.experiments import juno, lbl, lbl_spectra
from nmosyn.params import DCP_IO, DCP_NO, NUFIT, S23SQ, X0, X_GRID, b_lbl, dm2_phi_ee, intercept
from nmosyn.paths import RESULTS

DCP_LINE = np.radians(np.arange(-180.0, 180.1, 5.0))
DCP_GRID = np.radians(np.arange(-180.0, 180.1, 30.0))
R1_RES = np.sort(np.append(np.arange(0.0, 0.0901, 0.0025), juno.R1_RES))   # resolution at 1 MeV: scan and baseline
R1_EEFF = np.array([0.0, 0.01, 0.02, juno.R1_RES, 0.05, 0.08])
A_RES, A_EEFF = (r * juno.A_RES / juno.R1_RES for r in (R1_RES, R1_EEFF))     # the same, as stochastic terms
SHARES = np.linspace(0.0, 0.5, 11)


def solve(model, x, dcp_no, dcp_io, s23sq=S23SQ):
    """b with theta23 and both phases fixed."""
    y = gauss_newton(lambda v: model.counts(v, s23sq, dcp_io), model.counts(x, s23sq, dcp_no),
                     -x + b_lbl(dcp_no, dcp_io))
    return intercept(y, x)


def main():
    out = dict(dcp_line=DCP_LINE, dcp_grid=DCP_GRID, a_res=A_RES, a_eeff=A_EEFF, r1_res=R1_RES, r1_eeff=R1_EEFF,
               shares=SHARES, x=X_GRID)

    beam = lbl.generic_beam()
    out['beam_b_line'] = np.array([solve(beam, X0, d, DCP_IO) for d in DCP_LINE])
    out['beam_b_grid'] = np.array([[solve(beam, X0, dn, di) for di in DCP_GRID] for dn in DCP_GRID])
    out['beam_closed_line'] = b_lbl(DCP_LINE, DCP_IO) * 1e6

    out['juno_b'] = np.array([intercept(juno.Juno.scaled(a).solution(X0), X0) for a in A_RES])
    out['juno_b_vacuum'] = np.array([intercept(juno.Juno.scaled(a, rhoye=0.0).solution(X0), X0) for a in A_RES])
    design = juno.Juno()
    out['juno_y'] = np.array([design.solution(x) for x in X_GRID])
    out['juno_b_closed_4mev'] = juno.b_mbr(4.0) * 1e6

    for name, beam_shape in (('nova_like', lbl.NOVA_LIKE), ('t2k_like', lbl.T2K_LIKE)):
        for tag, dcp_no in (('nufit', DCP_NO), ('zero', 0.0)):
            out[f'{name}_{tag}'] = np.array([solve(lbl.peaked_beam(*beam_shape, f), X0, dcp_no, DCP_IO)
                                             for f in SHARES])
    out['closed_nufit'] = b_lbl(DCP_NO, DCP_IO) * 1e6
    out['closed_zero'] = b_lbl(0.0, DCP_IO) * 1e6

    weights, averages, b_eeff = [], [], []
    for a in A_EEFF:
        model = juno.Juno.scaled(a)
        y = model.solution(X0)
        weights.append(model.omega(X0, y))
        averages.append(model.solar_phase_average(X0, y))
        b_eeff.append(intercept(y, X0))
    out['eeff_bins'] = design.e_mev
    out['eeff_weights'] = np.array(weights)
    out['eeff_average'] = np.array(averages) * 1e6
    out['eeff_energy'] = np.array([juno.effective_energy(v) for v in averages])
    out['eeff_b'] = np.array(b_eeff)
    out['dm2_phi_bins'] = dm2_phi_ee(design.e_mev, juno.L_NOMINAL_KM, NUFIT) * 1e6
    out['dm2_phi_4mev'] = float(dm2_phi_ee(4.0, juno.L_NOMINAL_KM, NUFIT)) * 1e6

    nova = lbl.FarDetector(replace(lbl_spectra.nova(), rhoye=0.0), NUFIT)
    nova_band = band(fit.Harmonics(fit.Bins(nova, nova.disappearance)), X0, [(S23SQ, d) for d in DT], 'nova_dis')
    out['nova_dis_vacuum_dcp'] = DT
    out['nova_dis_vacuum_range'] = nova_band['range_paper']
    out['nova_dis_vacuum_cf'] = nova_band['cf']

    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(RESULTS / 'vacuum.npz', **out)


if __name__ == '__main__':
    main()
