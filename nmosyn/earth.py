"""The Earth: PREM density and the layers of constant density along a path to the detector (Sec. 4).

A neutrino is produced at the height HEIGHT_KM above the surface and reaches a detector depth_km below it. Along
its path the atmosphere is vacuum, the shell between the surface and the detector depth is the overburden of
the detector (sea water or ice, constant density), and the Earth below the detector depth is PREM. Down-going
paths cross the atmosphere and the overburden; up-going paths cross the atmosphere, the overburden at the
entry point, and PREM. PREM ends with its 3 km ocean at 6368-6371 km; above the detector depth the overburden
replaces it, so the water above KM3NeT/ORCA is counted once. As in NuFast-Earth, every path is cut where it
crosses a discontinuity of PREM; each PREM piece between two cuts is divided into layers of at most MAX_KM, and
every layer takes the mean density along it. With MAX_KM = 500 km the intercepts of the paper are converged to
0.003e-6 eV^2.

HEIGHT_KM is one fixed production height, 20 km, and the detector depth of DeepCore is 2 km, the values of IceCube's
oscillation framework PISA (pisa_examples/resources/settings/osc/earth.cfg: prop_height = 20 km, detector_depth
= 2 km). PISA can also average uniformly over a range of heights (prop_height_range of its nuSQuIDS stage), which is
not done here. The azimuth-averaged Honda flux tables used here give the flux only. Honda et al. publish the
distribution of production heights separately (Phys. Rev. D 92, 023004, Sec. VI). Between 15 and 25 km the DeepCore
intercept changes by 0.13e-6 eV^2 (nu_mu disappearance) and 0.007e-6 eV^2 (all channels).
"""
import numpy as np
from numba import njit

R_EARTH = 6371.0         # km
SHELLS = np.array([1221.5, 3480.0, 5701.0, 5771.0, 5971.0, 6151.0, 6346.6, 6356.0, 6368.0])   # PREM radii [km]
MAX_KM = 500.0
HEIGHT_KM = 20.0         # production height [km]
WATER_YE = 0.555         # electron fraction of water and ice, 10 electrons per 18.015 u
CUT_KM = 1e-6
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)


@njit(cache=True)
def prem_rhoye(r):
    """Density times electron fraction [g/cm^3] at radius r [km], PREM with Ye = 0.466 in the core and 0.494 above."""
    x = r / R_EARTH
    if r < 1221.5:
        rho = 13.0885 - 8.8381 * x ** 2
    elif r < 3480.0:
        rho = 12.5815 - 1.2638 * x - 3.6426 * x ** 2 - 5.5281 * x ** 3
    elif r < 5701.0:
        rho = 7.9565 - 6.4761 * x + 5.5283 * x ** 2 - 3.0807 * x ** 3
    elif r < 5771.0:
        rho = 5.3197 - 1.4836 * x
    elif r < 5971.0:
        rho = 11.2494 - 8.0298 * x
    elif r < 6151.0:
        rho = 7.1089 - 3.8045 * x
    elif r < 6346.6:
        rho = 2.6910 + 0.6924 * x
    elif r < 6356.0:
        rho = 2.9
    elif r < 6368.0:
        rho = 2.6
    elif r <= R_EARTH:
        rho = 1.02
    else:
        return 0.0
    return rho * (0.466 if r < 3480.0 else 0.494)


@njit(cache=True)
def _path(cosz, r_det, top, overburden, max_km, radii, gl_x, gl_w, lengths, rhoye):
    """Layers of one path from the production point, at radius top, to the detector at radius r_det, in the
    order the neutrino crosses them; returns their number.

    s is the distance from the detector toward the production point: r(s)^2 = (s + p)^2 + q2 with p = r_det cosz
    and q2 = r_det^2 (1 - cosz^2). The path is cut where it crosses the radii (the PREM discontinuities below
    the detector, the detector depth, and the surface); cuts closer than CUT_KM to an end are rounding.
    """
    p = r_det * cosz
    q2 = r_det * r_det - p * p
    total = np.sqrt(top * top - q2) - p
    cuts = np.empty(2 * radii.size + 2)
    cuts[0], cuts[1] = 0.0, total
    n_cuts = 2
    for radius in radii:
        if radius * radius > q2:
            half = np.sqrt(radius * radius - q2)
            for s in (-p - half, -p + half):
                if CUT_KM < s < total - CUT_KM:
                    cuts[n_cuts] = s
                    n_cuts += 1
    cuts = np.sort(cuts[:n_cuts])
    n = 0
    for j in range(n_cuts - 1, 0, -1):
        a, b = cuts[j - 1], cuts[j]
        if b - a <= 0.0:
            continue
        mid = 0.5 * (a + b)
        r_mid = np.sqrt((mid + p) ** 2 + q2)
        if r_mid >= r_det:
            lengths[n] = b - a
            rhoye[n] = overburden if r_mid <= R_EARTH else 0.0
            n += 1
            continue
        pieces = max(1, int(np.ceil((b - a) / max_km)))
        for k in range(pieces - 1, -1, -1):
            lo = a + (b - a) * k / pieces
            hi = a + (b - a) * (k + 1) / pieces
            mean = 0.0
            for g in range(gl_x.size):
                d = 0.5 * (hi - lo) * gl_x[g] + 0.5 * (hi + lo)
                mean += 0.5 * gl_w[g] * prem_rhoye(np.sqrt((d + p) ** 2 + q2))
            lengths[n] = hi - lo
            rhoye[n] = mean
            n += 1
    return n


@njit(cache=True)
def _layers(cosz, r_det, top, overburden, max_km, radii, gl_x, gl_w, lengths, rhoye):
    for i in range(cosz.size):
        _path(cosz[i], r_det, top, overburden, max_km, radii, gl_x, gl_w, lengths[i], rhoye[i])


def layers(cosz, depth_km, overburden, max_km=MAX_KM, density_scale=1.0, height_km=HEIGHT_KM):
    """Lengths [km] and rho*Ye [g/cm^3] of the layers along the path of each direction, shape (n, n_layers),
    from the production point to the detector.

    The detector sits depth_km below the surface, under an overburden of rho*Ye = overburden [g/cm^3]; the
    neutrinos are produced height_km above the surface. Unused entries have zero length. max_km is the largest
    length of a PREM layer; density_scale multiplies every density (0 gives vacuum).
    """
    if max_km < 50.0:
        raise ValueError(f'max_km = {max_km} is the largest layer length in km, not a number of layers')
    r_det = R_EARTH - depth_km
    radii = np.append(SHELLS[SHELLS < r_det], [r_det, R_EARTH])
    unique, index = np.unique(np.asarray(cosz, dtype=float).ravel(), return_inverse=True)
    capacity = 4 * radii.size + 4 + int(np.ceil(2 * R_EARTH / max_km))
    lengths = np.zeros((unique.size, capacity))
    rhoye = np.zeros((unique.size, capacity))
    _layers(unique, r_det, R_EARTH + height_km, float(overburden), float(max_km), radii, _GL_X, _GL_W, lengths,
            rhoye)
    used = max(1, int(np.max(np.count_nonzero(lengths, axis=1))))
    return lengths[index, :used], density_scale * rhoye[index, :used]


def path_length(cosz, depth_km, height_km=HEIGHT_KM):
    """Distance [km] from the production point to the detector, the vacuum chord of the shell of radius R + h."""
    c = np.asarray(cosz, dtype=float)
    r_det = R_EARTH - depth_km
    return np.sqrt((R_EARTH + height_km) ** 2 - r_det ** 2 * (1 - c ** 2)) - r_det * c


def mean_rhoye(cosz, depth_km, overburden):
    """Length [km] and mean rho*Ye [g/cm^3] of each path from production, used by the analytic forms of App. D.1."""
    lengths, rhoye = layers(cosz, depth_km, overburden)
    total = lengths.sum(axis=1)
    return total, (lengths * rhoye).sum(axis=1) / total
