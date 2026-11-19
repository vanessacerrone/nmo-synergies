"""Unit checks of the oscillation kernel, the weighted bin average, and the harmonic expansion of the fit."""
import numpy as np
import pytest

from nmosyn import average, earth, fit, osc
from nmosyn.params import NUFIT, PHASE, X0, b_lbl, kappa_mu

E = np.geomspace(0.5, 50.0, 400)


def vacuum_probability(e, l_km, dm31, s23sq, dcp, p=NUFIT, antineutrino=False):
    """P[..., beta, alpha] from the PMNS matrix, for comparison."""
    s12, c12 = np.sqrt(p.s12sq), np.sqrt(1 - p.s12sq)
    s13, c13 = np.sqrt(p.s13sq), np.sqrt(1 - p.s13sq)
    s23, c23 = np.sqrt(s23sq), np.sqrt(1 - s23sq)
    d = -dcp if antineutrino else dcp
    u = np.array([[1, 0, 0], [0, c23, s23], [0, -s23, c23]]) @ np.array(
        [[c13, 0, s13 * np.exp(-1j * d)], [0, 1, 0], [-s13 * np.exp(1j * d), 0, c13]]) @ np.array(
        [[c12, s12, 0], [-s12, c12, 0], [0, 0, 1]])
    phases = np.exp(-2j * PHASE * np.outer(l_km / e, [0.0, p.dm21, dm31]))
    amp = np.einsum('bi,ni,ai->nba', u, phases, u.conj())
    return np.abs(amp) ** 2


@pytest.mark.parametrize('dm31', [2.5e-3, -2.4e-3])
@pytest.mark.parametrize('antineutrino', [False, True])
def test_vacuum_limit(dm31, antineutrino):
    amp = osc.constant_amplitudes(E, 1300.0, 0.0, dm31, NUFIT, antineutrino)
    p = osc.probabilities(amp, 0.45, 1.2, antineutrino)
    assert np.allclose(p, vacuum_probability(E, 1300.0, dm31, 0.45, 1.2, antineutrino=antineutrino), atol=1e-12)


def test_unitarity_and_layers():
    for cosz in (-0.9, 0.5):
        amp = osc.earth_amplitudes(E, np.full(E.size, cosz), 2.5e-3, NUFIT, 2.0, 0.51)
        p = osc.probabilities(amp, 0.561, 3.0)
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-4) and np.allclose(p.sum(axis=2), 1.0, atol=1e-4)


@pytest.mark.parametrize('cosz', [-1.0, -0.6, -0.05, 0.0, 0.05, 0.6, 1.0])
def test_path_geometry(cosz):
    depth, overburden = 2.45, 0.57
    lengths, rhoye = earth.layers([cosz], depth, overburden)
    lengths, rhoye = lengths[0][lengths[0] > 0], rhoye[0][lengths[0] > 0]
    r_det, top = earth.R_EARTH - depth, earth.R_EARTH + earth.HEIGHT_KM
    q2, p = r_det ** 2 * (1 - cosz ** 2), r_det * cosz
    assert np.isclose(lengths.sum(), np.sqrt(top ** 2 - q2) - p, rtol=1e-12)
    assert np.isclose(lengths.sum(), earth.path_length(cosz, depth), rtol=1e-12)
    # production -> detector: the atmosphere first, then the matter up to the surface on the far side
    assert rhoye[0] == 0.0 and np.isclose(lengths[0], np.sqrt(top ** 2 - q2) - np.sqrt(earth.R_EARTH ** 2 - q2))
    assert np.isclose(lengths[1:].sum(), np.sqrt(earth.R_EARTH ** 2 - q2) - p, rtol=1e-12)
    assert np.all(rhoye[1:] > 0)
    if cosz >= 0:
        assert lengths.size == 2 and rhoye[1] == overburden
    else:
        prem = rhoye[1:] != overburden
        assert np.isclose(lengths[1:][prem].sum(), -2 * p, rtol=1e-12) and rhoye[1] == overburden


def test_layers_converged():
    """The intercepts are converged in the largest layer length (earth.py): DeepCore nu_mu, theta23 and the phases
    fixed, at x0, with layers of at most MAX_KM = 500 km and at most 100 km."""
    from nmosyn.experiments.deepcore import RELEASE, DeepCore
    from nmosyn.params import DCP_IO, DCP_NO, S23SQ, intercept
    if not RELEASE.exists():
        pytest.skip('the IceCube DeepCore release is not in data/external/ (scripts/download_data.py)')
    b = []
    for max_km in (earth.MAX_KM, 100.0):
        model = DeepCore('numu', max_km=max_km)
        y = average.gauss_newton(lambda v: model.counts(v, S23SQ, DCP_IO), model.counts(X0, S23SQ, DCP_NO),
                                 -X0 + b_lbl(DCP_NO, DCP_IO))
        b.append(intercept(y, X0))
    assert abs(b[0] - b[1]) < 0.003, b


def test_constant_density_is_one_layer_split_in_two():
    one = osc.constant_amplitudes(E, 800.0, 1.4, 2.5e-3, NUFIT)
    two = osc.path_amplitudes(E, np.array([[400.0, 400.0]]), np.array([[1.4, 1.4]]), 2.5e-3, NUFIT)
    assert np.allclose(one, two, atol=1e-6)


class Beam:
    """nu_mu disappearance at 810 km in vacuum, Gaussian spectrum, 10% resolution: the closed form of Eq. (3.6)."""

    def __init__(self):
        from scipy.special import erf
        self.e = np.arange(0.5, 5.0, 0.002)
        edges = np.arange(0.5, 5.0 + 1e-9, 0.1)
        r2 = np.sqrt(2) * 0.1 * self.e
        k = 0.5 * (erf((edges[1:, None] - self.e) / r2) - erf((edges[:-1, None] - self.e) / r2))
        self.r = 1e4 * np.exp(-0.5 * ((self.e - 2) / 0.3) ** 2) * self.e * k
        self.n_bins = len(self.r)

    def counts(self, y, s23sq, dcp):
        amp = osc.constant_amplitudes(self.e, 810.0, 0.0, y, NUFIT)
        return self.r @ osc.probabilities(amp, s23sq, dcp)[:, 1, 1]


@pytest.fixture(scope='module')
def beam():
    return Beam()


def test_gauss_newton_is_the_minimum_and_the_bin_average(beam):
    data = beam.counts(X0, 0.561, 3.0)
    model = lambda y: beam.counts(y, 0.561, 4.9)
    y = average.gauss_newton(model, data, -X0 + 100e-6)
    w = average.poisson_weights(data)
    chi2 = lambda v: np.sum(w * (model(v) - data) ** 2)
    assert chi2(y) <= min(chi2(y + 1e-8), chi2(y - 1e-8))
    y_i, omega = average.bin_solutions(model, data, y)
    ok = np.isfinite(y_i)
    assert abs(np.sum(omega[ok] * y_i[ok]) / np.sum(omega[ok]) - y) < 1e-12
    assert abs((y + X0) - b_lbl(3.0, 4.9, 0.561, 0.561)) < 0.05e-6     # vacuum closed form, Eq. (3.6)


def test_harmonics_are_exact(beam):
    h = fit.Harmonics(beam)
    for y, s23sq, dcp in ((-X0 + 101.37e-6, 0.47, 0.4), (X0 + 3.3e-7, 0.58, 5.0)):
        assert np.allclose(h.counts(y, s23sq, dcp), beam.counts(y, s23sq, dcp), rtol=1e-10)


def test_kappa_first_order():
    """kappa_mu of Eq. (3.6) is c12^2 - A_CP, up to terms of order s13^2."""
    d = np.linspace(0, 2 * np.pi, 13)
    a_cp = np.cos(d) * np.sqrt(NUFIT.s13sq) * 2 * np.sqrt(NUFIT.s12sq * (1 - NUFIT.s12sq)) * np.sqrt(0.561 / 0.439)
    assert np.max(np.abs(kappa_mu(d, 0.561) - (1 - NUFIT.s12sq - a_cp))) < 3 * NUFIT.s13sq
