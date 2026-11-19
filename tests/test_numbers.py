"""Regression of the paper numbers of paper_numbers.py, and invariants of the cached results.

REFERENCE holds the values of paper_numbers.py on the cached results of this version, in its units (b in
1e-6 eV^2, splittings in 1e-3 eV^2, shares in %). Each tolerance absorbs the rounding of the reference and the
convergence of the fits, and catches a real change: 0.01-0.05 on intercepts, 0.02 on synergy terms. A test
whose result file is missing is skipped with the reason.
"""
import numpy as np
import pytest

import paper_numbers
from nmosyn.params import DCP_IO, X0, b_lbl, intercept
from nmosyn.paths import RESULTS

REFERENCE = {    # id: (values, absolute tolerance)
    # Abstract
    'abs.dc_independent': ((88.748, 95.311), 0.05),
    # Sec. 2: the weighted bin average, fig_deepcore_bin_weights
    'sec2.slope_sbr_lbl': ((-1, -1), 1e-06),
    'sec2.weight_20_50': ((65.53,), 0.1),
    'sec2.weight_core': ((17.04,), 0.1),
    # Sec. 3: reactors and accelerators, fig_accelerator_closed_form and fig_juno_intercept_vs_resolution
    'sec3.lbl_agree': ((0.0918,), 0.005),
    'sec3.eq36_103p8': ((103.8114,), 0.0001),
    'sec3.eq36_11p6': ((11.57215,), 0.0001),
    'sec3.lbl_unequal': ((-4.185, -6.782), 0.01),
    'sec3.dphi_above_2p5': ((2.46,), 0.011),
    'sec3.eeff_juno': ((4.1490,), 0.005),
    'sec3.juno_resolution': ((2.94308,), 0.0001),
    'sec3.juno_b': ((25.662,), 0.01),
    'sec3.juno_2to4': ((9.8016,), 0.02),
    # Sec. 4: atmospheric neutrinos, fig_deepcore_dcp, fig_deepcore_resolution, fig_deepcore_energy_window, and
    # fig_deepcore_vs_orca6
    'sec4.msw_energy': ((5.7718, 9.5633), 0.001),
    'sec4.b_reference': ((119.747,), 0.01),
    'sec4.weight_above_10': ((97.74,), 0.1),
    'sec4.freq_share': ((69.13,), 0.2),
    'sec4.amp_rest': ((0.4978,), 0.005),
    'sec4.datm_x_dependence': ((1.088,), 0.02),
    'fig_deepcore_resolution.sigma_lnE': ((0.2909, 0.3388), 0.002),
    'sec4.res_scan': ((10.925,), 0.02),
    'sec4.emax_raise': ((21.722, 23.05, 40.026), 0.05),
    'sec4.direction': ((6.506, 4.975, 10.925), 0.02),
    'sec4.orca_trk_close': ((3.675,), 0.05),
    'sec4.orca_vs_dc': ((11.277,), 0.05),
    'sec4.orca_slope': ((-1.0227, -0.979), 0.002),
    'sec4.orca_shower_weight': ((9.858,), 0.05),
    'sec4.valid_tab': ((1.891,), 0.02),
    # Sec. 5: unknown delta_CP and theta23, fig_solutions_plane, fig_intercept_ranges, fig_intercept_vs_truth, and
    # fig_published_best_fits
    'fig_solutions_plane.s23_range': ((0.43, 0.596), 1e-06),
    'fig_intercept_ranges.width.nova_dis.dcp': ((40.482,), 0.05),
    'fig_intercept_ranges.width.nova_dis.t23': ((22.21,), 0.05),
    'fig_intercept_ranges.width.nova_dis.both': ((41.424,), 0.05),
    'fig_intercept_ranges.width.nova.dcp': ((25.364,), 0.05),
    'fig_intercept_ranges.width.nova.t23': ((5.645,), 0.05),
    'fig_intercept_ranges.width.nova.both': ((28.748,), 0.05),
    'fig_intercept_ranges.width.t2k_dis.dcp': ((41.016,), 0.05),
    'fig_intercept_ranges.width.t2k_dis.t23': ((23.534,), 0.05),
    'fig_intercept_ranges.width.t2k_dis.both': ((41.886,), 0.05),
    'fig_intercept_ranges.width.t2k.dcp': ((27.657,), 0.05),
    'fig_intercept_ranges.width.t2k.t23': ((2.791,), 0.05),
    'fig_intercept_ranges.width.t2k.both': ((31.023,), 0.05),
    'fig_intercept_ranges.width.dcn.dcp': ((1.992,), 0.05),
    'fig_intercept_ranges.width.dcn.t23': ((21.469,), 0.05),
    'fig_intercept_ranges.width.dcn.both': ((24.691,), 0.05),
    'fig_intercept_ranges.width.dca.dcp': ((2.596,), 0.05),
    'fig_intercept_ranges.width.dca.t23': ((4.617,), 0.05),
    'fig_intercept_ranges.width.dca.both': ((7.152,), 0.05),
    'sec5.valley_acc_dis': ((396, 396), 0),
    'sec5.valley_dc_app': ((339, 396, 0, 0), 5),
    'fig_intercept_vs_truth.valley_tol': ((0.01,), 0),
    'fig_intercept_vs_truth.dc_range': ((2.202,), 0.02),
    'fig_intercept_vs_truth.t2k_same_dis': ((-2.309, 0.351, -0.515, 3.05), 0.05),
    'sec5.nova_t2k_few': ((0.486, 0.024, -0.462, -2.737), 0.05),
    'sec5.app_nova_t2k': ((19.582,), 0.05),
    'sec5.app_t2k': ((221.74,), 0.5),
    'sec5.app_3pi2': ((257.52, 276.86), 0.5),
    'sec5.nova_vacuum': ((0.164, 0.175), 0.05),
    'sec5.dca_constant': ((1.946, 3.824), 0.03),
    'sec5.dyb_on_line': ((44.8, 45.989), 0.01),
    'sec5.juno_measured_b': ((27,), 0.01),
    'sec5.juno_on_line': ((25.496,), 0.01),
    'sec5.nova_in_app_band': ((109.8, 89.09, 118.1, 109.9, 90, 118.84, 99.8, 88.89, 118.07), 0.05),
    'sec5.nufit_in_app_band': ((98.9, 89.409, 118.157), 0.05),
    'sec5.dcdata_in_dca_band': ((117.787, 113.988, 121.074), 0.05),
    'sec5.dc93_above_dca': ((128.8, 113.905, 120.978), 0.05),
    # Sec. 6: synergies, fig_synergy_vs_precision and fig_synergy_rings
    'fig_synergy_vs_precision.gap.dis.JUNO_NOvA': ((56.324, 82.447), 0.05),
    'fig_synergy_vs_precision.gap.dis.JUNO_DeepCore': ((87.741, 112.347), 0.05),
    'fig_synergy_vs_precision.gap.dis.NOvA_DeepCore': ((0, 40.026), 0.05),
    'fig_synergy_vs_precision.gap.all.JUNO_NOvA': ((63.75, 92.496), 0.05),
    'fig_synergy_vs_precision.gap.all.JUNO_DeepCore': ((88.748, 95.311), 0.05),
    'fig_synergy_vs_precision.gap.all.NOvA_DeepCore': ((0.2, 30.305), 0.05),
    'sec6.precisions': ((1.0363, 1.3519, 1.8742), 0.005),
    'sec6.solar_sep': ((2.037,), 0.005),
    'sec6.joint_larger': ((100.0, 98.99), 0.5),
    'fig_synergy_rings.ring.dca.j0.5.p1.0': ((9.813, 11.318), 0.02),
    'fig_synergy_rings.ring.dca.j0.8.p1.5': ((4.244, 4.895), 0.02),
    'fig_synergy_rings.ring.dca.j1.0.p1.0': ((6.133, 7.074), 0.02),
    'fig_synergy_rings.ring.dca.j1.0.p2.0': ((2.453, 2.829), 0.02),
    'fig_synergy_rings.ring.nova_all.j0.5.p1.0': ((5.063, 10.659), 0.02),
    'fig_synergy_rings.ring.nova_all.j0.8.p1.5': ((2.190, 4.61), 0.02),
    'fig_synergy_rings.ring.nova_all.j1.0.p1.0': ((3.164, 6.662), 0.02),
    'fig_synergy_rings.ring.nova_all.j1.0.p2.0': ((1.266, 2.665), 0.02),
    'sec6.ring_ratio': ((1.153, 2.1053), 0.005),
    'sec6.equal_prec': ((6.133, 7.074, 3.164, 6.662), 0.02),
    # Sec. 7: conclusions
    'concl.slope_juno': ((0.772,), 0.02),
    'concl.slope_dc': ((1.088,), 0.02),
    'concl.orca_slope': ((-1.0227, -0.979), 0.002),
    'concl.dc_nearly_fixed': ((1.153, 2.1053), 0.005),
    # App. A: inputs and detector models
    'appA.b_sbr': ((45.9886,), 0.0001),
    'appA.b_lead': ((103.8114,), 0.0001),
    'appA.juno_abc': ((2.61, 0.64, 1.2), 1e-06),
    'appA.juno_rhoye': ((1.225,), 1e-06),
    'appA.lbl_density': ((2.84, 2.6), 1e-06),
    'appA.dc.events': ((21401.834,), 0.01),
    'appA.dc.bins': ((10, 6.31, 88.199, 158.49), 0.01),
    'appA.dc.cosz_bins': ((-1, 0.1), 1e-06),
    'appA.dc.grid_converged': ((0.0343,), 0.003),
    'appA.dc.direction': ((6.506, 4.975), 0.01),
    'appA.dc.gauss_vs_tab': ((3.293,), 0.02),
    'appA.orca.total': ((0.684,), 0.02),
    'appA.orca.weights': ((64.996, 25.146, 9.858), 0.05),
    'appA.orca.lp_add': ((0.756, 1.907), 0.02),
    'appA.orca.showers': ((13.045,), 0.02),
    'appA.orca.s23': ((0.01278,), 0.0003),
    # App. B: the weighted bin average
    'appB.exchange_numu': ((2.309,), 0.02),
    'appB.exchange_all': ((0.096,), 0.01),
    # App. C: reactors and accelerators, fig_juno_effective_energy
    'appC.offset': ((0.36, 0.918), 0.01),
    'appC.delta21_dyb': ((0.03914,), 0.0001),
    'appC.delta21_810': ((0.03843,), 0.0001),
    'appC.t2k_like': ((0.4575,), 0.005),
    'appC.ahat_2gev': ((0.1711,), 0.001),
    'appC.nova_narrower': ((10.783, 10.914), 0.05),
    'appC.nova_center': ((1.005,), 0.01),
    'appC.nova_like': ((4.185, 6.782), 0.01),
    'appC.equal_shares': ((-0.041, 0.092, -0.03, 0.046), 0.01),
    'appC.cf4_gap': ((2.0365,), 0.01),
    'appC.eq38': ((0.402,), 0.01),
    'fig_juno_effective_energy.eeff.r0': ((3.4539,), 0.005),
    'fig_juno_effective_energy.eeff.r1': ((3.5477,), 0.005),
    'fig_juno_effective_energy.eeff.r2': ((3.8060,), 0.005),
    'fig_juno_effective_energy.eeff.r2.94': ((4.1490,), 0.005),
    'fig_juno_effective_energy.eeff.r5': ((5.0176,), 0.005),
    'fig_juno_effective_energy.eeff.r8': ((6.1043,), 0.005),
    # App. D: the atmospheric solution in matter
    'appD.solar_phase_mumu': ((0.0143,), 0.002),
    'appD.first_term': ((1.29769,), 0.0001),
    'appD.analytic_share': ((72.31, 76.95), 0.2),
    'appD.omsd_share': ((91.89, 94.69), 0.2),
    'appD.cp_amp_2': ((1.195, 1.348), 0.01),
    'appD.cp_mirror': ((-2.92,), 0.01),
    # App. E: full fit of the DeepCore release, fig_full_fit_validation and fig_deepcore_reproduction
    'appE.asimov_points': ((25, 2.2, 2.8), 1e-06),
    'appE.valid_tab': ((1.19, 1.891), 0.02),
    'appE.valid_fixed': ((1.728,), 0.03),
    'appE.valid_fixed_datm': ((0.1084,), 0.002),
    'appE.valid_all': ((0.0014, 0.0166), 0.005),
    'appE.nuisance': ((0.334, 1.172), 0.02),
    'appC.eeff_0': ((3.4539,), 0.005),
    'appC.eeff_juno': ((4.1490,), 0.005),
    'appC.res_cross': ((2.5607,), 0.005),
    'appD.y_20_50': ((2.42052, 2.425704), 2e-05),
    'appE.repro_s23': ((0.5068, 0.0583), 0.001),
    'appE.repro_dm32': ((2.4107, 0.0779), 0.001),
    'appE.dm21_release': ((7.39,), 0.0001),
    'appE.data_no': ((2.4846, 0.0779), 0.001),
    'appE.data_io': ((-2.36685, 0.07647), 0.0001),
    'appE.data_b': ((117.787,), 0.05),
    'appE.hs_asimov_matched': ((1.925,), 0.05),
    'appE.hs_asimov_naive': ((13.553,), 0.05),
    # App. F: combination of experiments
    'appF.nova_dc_edge': ((18.995, 40.026), 0.05),
    'appF.joint_larger': ((100.0, 98.99), 0.5),
    'appF.joint_max': ((0.43, 0.0, 1.228, 0.43, 0.029, 0.524), 0.005),
}


@pytest.fixture(scope='module')
def numbers():
    return paper_numbers.computed()


@pytest.mark.parametrize('number_id', REFERENCE)
def test_paper_number(numbers, number_id):
    values, missing = numbers
    if number_id in missing:
        pytest.skip(missing[number_id])
    expected, tol = REFERENCE[number_id]
    assert np.allclose(values[number_id].new, expected, rtol=0, atol=tol), values[number_id].new


def test_every_number_is_computed_or_explained(numbers):
    values, missing = numbers
    assert not set(values) & set(missing)
    assert all(reason for reason in missing.values())
    assert set(REFERENCE) <= set(values) | set(missing)


def results(name):
    """A cached result file, or a skip when it is not there."""
    path = RESULTS / f'{name}.npz'
    if not path.exists():
        pytest.skip(f'results/{name}.npz not found')
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def test_weights_solution_is_the_weighted_mean_of_the_bins():
    """Eq. (2.3): the weighted mean of the values preferred by the bins is the solution of the fit."""
    d = results('deepcore')
    ok = np.isfinite(d['fig1_y_i'])
    mean = np.sum(d['fig1_omega'][ok] * d['fig1_y_i'][ok]) / np.sum(d['fig1_omega'][ok])
    assert abs(intercept(mean, X0) - intercept(d['fig1_y'], X0)) < 1e-6


def test_juno_resolution():
    """A worse resolution raises b toward 2 s12^2 Dm21 and raises the effective energy; matter lowers b
    (App. C.3)."""
    v = results('vacuum')
    assert np.all(np.diff(v['juno_b']) > 0) and np.all(np.diff(v['juno_b_vacuum']) > 0)
    assert np.all(v['juno_b_vacuum'] > v['juno_b'])
    assert np.all(np.diff(v['eeff_energy']) > 0)


def test_accelerator_closed_form():
    """The generic beam in matter follows Eq. (3.6) to 0.1 x 1e-6 eV^2 over the true delta_CP (Sec. 3)."""
    v = results('vacuum')
    assert np.allclose(v['beam_closed_line'], b_lbl(v['dcp_line'], DCP_IO) * 1e6)
    assert np.max(np.abs(v['beam_b_line'] - v['beam_closed_line'])) < 0.1


def test_disappearance_is_even_in_dcp():
    """nu_mu disappearance along a path whose density profile is symmetric depends on cos delta_CP only, so
    b(delta_CP) = b(-delta_CP) (Secs. 3-5): exactly in vacuum and at constant density. The atmospheric paths
    cross the atmosphere and the ice at one end only, which adds a small sin delta_CP term (below 0.03)."""
    v, d, b = results('vacuum'), results('deepcore'), results('bands')
    assert np.allclose(v['beam_b_line'], v['beam_b_line'][::-1], rtol=0, atol=1e-6)
    assert np.allclose(d['fig3_b_vacuum'], d['fig3_b_vacuum'][::-1], rtol=0, atol=1e-4)
    for key in ('fig3_b', 'fig3_lo', 'fig3_hi'):
        assert np.allclose(d[key], d[key][::-1], rtol=0, atol=0.03), key
    grid = d['numu_fixed_grid']
    assert np.allclose(grid, grid[::-1], rtol=0, atol=0.03)            # true delta_CP
    assert np.allclose(grid, grid[:, ::-1], rtol=0, atol=0.03)         # delta_CP^IO
    ix = int(np.argmin(np.abs(b['x'] - X0)))
    for sample, tol in (('nova_dis', 1e-4), ('dcn', 0.03)):
        scan = b[f'{sample}/scan_dcp/b/{ix}']      # true delta_CP from 0 to 2 pi
        assert np.allclose(scan, scan[::-1], rtol=0, atol=tol), sample


def test_bands_contain_the_global_minimum():
    """Under the rule of App. B.3 every range of b contains b at the global minimum of the fit."""
    b = results('bands')
    for key in [k for k in b if '/range_paper/' in k]:
        lo, hi = b[key].T
        best = b[key.replace('range_paper', 'b')]
        assert np.all(lo <= best) and np.all(best <= hi), key


def test_neutrino_events_of_the_release():
    """The normalization N_EVENTS of the tabulated response is the observed total minus the expected muons."""
    from nmosyn.experiments.deepcore import N_EVENTS
    r = results('release')
    assert round(float(r['observed'] - r['muons'])) == N_EVENTS

