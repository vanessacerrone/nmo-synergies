"""Three-flavor oscillation probabilities in vacuum, at constant density, and through the layered Earth.

A numba port of the algorithms of NuFast (P. B. Denton and S. J. Parke, arXiv:2405.02400,
https://github.com/PeterDenton/NuFast-LBL) for the eigenvalues and eigenvectors in matter, and of NuFast-Earth
(P. B. Denton and S. J. Parke, arXiv:2511.04735, https://github.com/PeterDenton/NuFast-Earth) for the amplitudes of
the layers and the rotation by theta23 and delta_CP. Both are MIT licensed; their copyright and permission notices
are in THIRD_PARTY_LICENSES. tests/test_nufast_earth.py checks this module against the original NuFast-Earth.

The Hamiltonian is H = R23 Gamma H0 Gamma^dagger R23^T (App. D.2), with H0 independent of theta23 and delta_CP.
In every layer of constant density the eigenvalues and eigenvectors of H0 are computed in closed form, and the
amplitudes of the layers are multiplied along the path. theta23 and delta_CP enter last, as a rotation of the
product. The amplitudes therefore depend on Dm31 (and the solar parameters and theta13) only, and every value of
theta23 and delta_CP costs one 3x3 rotation per path.

Conventions: amp[..., beta, alpha] is the amplitude of nu_alpha -> nu_beta, flavors ordered (e, mu, tau).
Antineutrinos have the opposite matter potential and the opposite CP phase.
"""
import numpy as np
from numba import njit, prange

from .earth import MAX_KM, layers

A_PER_RHOYE_GEV = 1.526493231029146e-4          # A [eV^2] = A_PER_RHOYE_GEV * rho*Ye [g/cm^3] * E [GeV]
PHASE_2E = 1e-9 / 1.973269804e-7 * 1e3 / 2.0
N_NEWTON = 1                             # Newton iterations on the largest eigenvalue, as in NuFast-Earth


@njit(cache=True)
def _layer_amplitude(l_over_e, a, s12sq, s13sq, dm21, dm31, out):
    """Amplitude of one layer of constant density in the basis without theta23 and delta_CP (NuFast-Earth)."""
    c12sq = 1.0 - s12sq
    c13sq = 1.0 - s13sq
    s13xc13 = np.sqrt(s13sq * c13sq)
    dmee = dm31 - s12sq * dm21
    see = dm21 + dmee * c13sq
    sem = -dm21 * np.sqrt(c13sq * s12sq * c12sq)
    set_ = -dmee * s13xc13
    tee = dm21 * dm31 * c13sq * c12sq
    tem = dm31 * sem
    tet = -dm21 * dm31 * s13xc13 * c12sq
    if a == 0.0:
        lam1, lam2, lam3 = 0.0, dm21, dm31
        s12, c12, s13, c13 = np.sqrt(s12sq), np.sqrt(c12sq), np.sqrt(s13sq), np.sqrt(c13sq)
        ve2sq, ve3sq = s12sq * c13sq, s13sq
        vm2sq, vm3sq = c12sq, 0.0
        vt2sq, vt3sq = s12sq * s13sq, c13sq
        ve2vm2, ve2vt2, vm2vt2 = s12 * c12 * c13, -s12sq * c13 * s13, -c12 * s12 * s13
        ve3vm3, ve3vt3, vm3vt3 = 0.0, s13 * c13, 0.0
    else:
        big_a = dm21 + dm31 + a
        big_b = dm21 * dm31 + a * see
        big_c = a * tee
        xmat = a / dmee
        lam3 = dm31 + 0.5 * dmee * (xmat - 1.0 + np.sqrt((1.0 - xmat) ** 2 + 4.0 * s13sq * xmat))
        for _ in range(N_NEWTON):
            lam3 = (lam3 * lam3 * (lam3 + lam3 - big_a) + big_c) / (lam3 * (2.0 * (lam3 - big_a) + lam3) + big_b)
        disc = (big_a - lam3) ** 2 - 4.0 * big_c / lam3
        dl21 = np.sqrt(max(disc, 0.0))
        lam2 = 0.5 * (big_a - lam3 + dl21)
        lam1 = lam2 - dl21
        d21 = lam2 - lam1
        d31 = lam3 - lam1
        inv_pid = 1.0 / (d31 * (d31 - d21) * d21)
        xp2 = -d31 * inv_pid
        xp3 = d21 * inv_pid
        ve3sq = (lam3 * (lam3 - see) + tee) * xp3
        ve2sq = (lam2 * (lam2 - see) + tee) * xp2
        ve3vm3 = (-lam3 * sem + tem) * xp3
        ve3vt3 = (-lam3 * set_ + tet) * xp3
        r3 = ve3vm3 / ve3sq
        vm3vt3 = ve3vt3 * r3
        vm3sq = ve3vm3 * r3
        vt3sq = 1.0 - ve3sq - vm3sq
        ve2vm2 = (-lam2 * sem + tem) * xp2
        ve2vt2 = (-lam2 * set_ + tet) * xp2
        r2 = ve2vm2 / ve2sq
        vm2vt2 = ve2vt2 * r2
        vm2sq = ve2vm2 * r2
        vt2sq = 1.0 - ve2sq - vm2sq
    ph21 = PHASE_2E * (lam2 - lam1) * l_over_e
    ph31 = PHASE_2E * (lam3 - lam1) * l_over_e
    e21 = complex(np.cos(ph21) - 1.0, -np.sin(ph21))
    e31 = complex(np.cos(ph31) - 1.0, -np.sin(ph31))
    out[0, 0] = 1.0 + ve2sq * e21 + ve3sq * e31
    out[1, 1] = 1.0 + vm2sq * e21 + vm3sq * e31
    out[2, 2] = 1.0 + vt2sq * e21 + vt3sq * e31
    out[0, 1] = out[1, 0] = ve2vm2 * e21 + ve3vm3 * e31
    out[0, 2] = out[2, 0] = ve2vt2 * e21 + ve3vt3 * e31
    out[1, 2] = out[2, 1] = vm2vt2 * e21 + vm3vt3 * e31


@njit(cache=True)
def _matmul3(a, b, out):
    for i in range(3):
        for j in range(3):
            out[i, j] = a[i, 0] * b[0, j] + a[i, 1] * b[1, j] + a[i, 2] * b[2, j]


@njit(parallel=True, cache=True)
def _path_amplitudes(e, lengths, rhoye, sign, s12sq, s13sq, dm21, dm31, out):
    n, n_layers = lengths.shape
    for i in prange(n):
        total = np.eye(3, dtype=np.complex128)
        layer = np.empty((3, 3), dtype=np.complex128)
        product = np.empty((3, 3), dtype=np.complex128)
        for k in range(n_layers):
            if lengths[i, k] == 0.0:
                continue
            _layer_amplitude(lengths[i, k] / e[i], sign * A_PER_RHOYE_GEV * rhoye[i, k] * e[i],
                             s12sq, s13sq, dm21, dm31, layer)
            _matmul3(layer, total, product)
            total[:, :] = product
        out[i] = total


def path_amplitudes(e_gev, lengths, rhoye, dm31, p, antineutrino=False):
    """Amplitudes (n, 3, 3) of paths made of layers of constant density, without theta23 and delta_CP.

    e_gev (n,); lengths [km] and rhoye [g/cm^3] (n, n_layers), traversed in order; a layer of zero length is skipped.
    """
    e = np.ascontiguousarray(e_gev, dtype=float).ravel()
    lengths = np.ascontiguousarray(np.broadcast_to(lengths, (e.size, np.shape(lengths)[-1])), dtype=float)
    rhoye = np.ascontiguousarray(np.broadcast_to(rhoye, lengths.shape), dtype=float)
    out = np.empty((e.size, 3, 3), dtype=np.complex128)
    _path_amplitudes(e, lengths, rhoye, -1.0 if antineutrino else 1.0, p.s12sq, p.s13sq, p.dm21, float(dm31), out)
    return out


def constant_amplitudes(e_gev, l_km, rhoye, dm31, p, antineutrino=False):
    """Amplitudes (n, 3, 3) at constant density rho*Ye [g/cm^3] over the baseline l_km (vacuum for rhoye = 0)."""
    return path_amplitudes(e_gev, np.array([[float(l_km)]]), np.array([[float(rhoye)]]), dm31, p, antineutrino)


def earth_amplitudes(e_gev, cosz, dm31, p, depth_km, overburden, antineutrino=False, max_km=MAX_KM,
                     density_scale=1.0):
    """Amplitudes (n, 3, 3) from production to a detector depth_km under the overburden rho*Ye [g/cm^3], for
    each pair (E, cos theta_z).

    The layers are those of earth.layers (atmosphere, overburden, and PREM cut at its discontinuities). A model that
    calls this for many values of Dm31 should compute earth.layers once and call path_amplitudes.
    """
    lengths, rhoye = layers(cosz, depth_km, overburden, max_km, density_scale)
    return path_amplitudes(e_gev, lengths, rhoye, dm31, p, antineutrino)


@njit(parallel=True, cache=True)
def _rotate(amp, c23, s23, cd, sd, out):
    ep = complex(cd, sd)
    em = complex(cd, -sd)
    for i in prange(amp.shape[0]):
        a = amp[i]
        ate = ep * a[2, 0]
        aet = em * a[0, 2]
        amt = em * a[1, 2] + ep * a[2, 1]
        out[i, 0, 0] = abs(a[0, 0]) ** 2
        out[i, 1, 0] = abs(c23 * a[1, 0] + s23 * ate) ** 2
        out[i, 2, 0] = 1.0 - out[i, 0, 0] - out[i, 1, 0]
        out[i, 0, 1] = abs(c23 * a[0, 1] + s23 * aet) ** 2
        out[i, 1, 1] = abs(c23 * c23 * a[1, 1] + s23 * s23 * a[2, 2] + s23 * c23 * amt) ** 2
        out[i, 2, 1] = 1.0 - out[i, 0, 1] - out[i, 1, 1]
        out[i, 0, 2] = 1.0 - out[i, 0, 0] - out[i, 0, 1]
        out[i, 1, 2] = 1.0 - out[i, 1, 0] - out[i, 1, 1]
        out[i, 2, 2] = 1.0 - out[i, 2, 0] - out[i, 2, 1]


def probabilities(amp, s23sq, dcp, antineutrino=False):
    """Probabilities P[..., beta, alpha] of nu_alpha -> nu_beta from amplitudes without theta23 and delta_CP."""
    d = -dcp if antineutrino else dcp
    out = np.empty(amp.shape, dtype=float)
    _rotate(amp, np.sqrt(1.0 - s23sq), np.sqrt(s23sq), np.cos(d), np.sin(d), out)
    return out
