"""Oscillation inputs: the NuFIT 6.0 normal-ordering best fit (Eq. A.1) and the ranges of the unknown parameters.

The solar parameters and theta13 are the same in both orderings and in every experiment. The splitting, theta23,
and the CP phase are passed separately, because the inverted-ordering fit adjusts them.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Params:
    """Parameters fixed in both orderings: sin^2 theta12, sin^2 theta13, Dm21 [eV^2]."""

    s12sq: float = 0.307
    s13sq: float = 0.02195
    dm21: float = 7.49e-5


NUFIT = Params()
X0 = 2.534e-3                       # Dm31_NO, the true splitting [eV^2]
S23SQ = 0.561                       # true sin^2 theta23
DCP_NO = np.radians(177.0)          # true delta_CP
DCP_IO = np.radians(285.0)          # delta_CP of the inverted ordering at the NuFIT 6.0 IO best fit
X_GRID = np.linspace(2.2e-3, 2.8e-3, 25)

# NuFIT 6.0 3 sigma ranges for normal ordering (Sec. 5)
S23SQ_RANGE = (0.430, 0.596)
DCP_RANGE = np.radians((96.0, 422.0))

PHASE = 1e-9 / 1.973269804e-7 * 1e3 / 4     # Delta_ij = PHASE * Dm2_ij [eV^2] L [km] / E [GeV], hbar c in eV m


def intercept(y, x):
    """Intercept b = y + x of Eq. (2.4), in 1e-6 eV^2."""
    return (np.asarray(y) + np.asarray(x)) * 1e6


def u_mu(dcp, s23sq=S23SQ, p=NUFIT):
    """|U_mu1|^2 and |U_mu2|^2 in the PDG parameterization."""
    s12, c12 = np.sqrt(p.s12sq), np.sqrt(1 - p.s12sq)
    s13 = np.sqrt(p.s13sq)
    s23, c23 = np.sqrt(s23sq), np.sqrt(1 - s23sq)
    phase = np.exp(1j * np.asarray(dcp))
    return np.abs(-s12 * c23 - c12 * s23 * s13 * phase) ** 2, np.abs(c12 * c23 - s12 * s23 * s13 * phase) ** 2


def kappa_mu(dcp, s23sq=S23SQ, p=NUFIT):
    """kappa_mu = |U_mu2|^2 / (|U_mu1|^2 + |U_mu2|^2) of Eq. (3.2), exact in the PDG parameterization (Eq. (3.5))."""
    u1, u2 = u_mu(dcp, s23sq, p)
    return u2 / (u1 + u2)


def b_sbr(p=NUFIT):
    """Short-baseline reactor intercept 2 s12^2 Dm21, Eq. (3.7) [eV^2]."""
    return 2 * p.s12sq * p.dm21


def b_lbl(dcp_no, dcp_io, s23sq_no=S23SQ, s23sq_io=S23SQ, p=NUFIT):
    """Accelerator closed form (kappa_mu^NO + kappa_mu^IO) Dm21, Eq. (3.4) [eV^2]."""
    return (kappa_mu(dcp_no, s23sq_no, p) + kappa_mu(dcp_io, s23sq_io, p)) * p.dm21


def dm2_phi(u1, u2, e_gev, l_km, p=NUFIT):
    """Solar-phase term Dm2_phi = 4 E phi / L of a survival probability with u_k = |U_alpha k|^2 (App. C.1)
    [eV^2]: phi from tan phi of the effective-frequency form, at energies e_gev [GeV] and baseline l_km."""
    k = PHASE * l_km / np.asarray(e_gev)
    d21 = k * p.dm21
    num = u1 * np.sin(2 * u2 * d21 / (u1 + u2)) - u2 * np.sin(2 * u1 * d21 / (u1 + u2))
    den = u1 * np.cos(2 * u2 * d21 / (u1 + u2)) + u2 * np.cos(2 * u1 * d21 / (u1 + u2))
    return np.arctan2(num, den) / k


def dm2_phi_ee(e_mev, l_km, p=NUFIT):
    """Solar-phase term Dm2_phi,ee of the reactor survival probability (App. C.1) [eV^2]."""
    return dm2_phi((1 - p.s12sq) * (1 - p.s13sq), p.s12sq * (1 - p.s13sq), np.asarray(e_mev) * 1e-3, l_km, p)


def dm2_phi_mumu(e_gev, l_km, dcp=DCP_NO, s23sq=S23SQ, p=NUFIT):
    """Solar-phase term Dm2_phi,mumu of the nu_mu survival probability in vacuum (App. C.1, App. D.1) [eV^2]."""
    return dm2_phi(*u_mu(dcp, s23sq, p), e_gev, l_km, p)
