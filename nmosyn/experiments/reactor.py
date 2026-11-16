"""Reactor antineutrino spectrum detected by inverse beta decay (App. A.2).

The emitted spectrum is the Huber-Mueller model, P. Huber, Phys. Rev. C 84 (2011) 024617 (235U, 239Pu,
241Pu) and Th. A. Mueller et al., Phys. Rev. C 83 (2011) 054615 (238U), summed with the fission fractions of a
commercial pressurized-water reactor. The cross section is the approximation of Eq. (25) of A. Strumia and
F. Vissani, Phys. Lett. B 564 (2003) 42. The short-baseline closed form (Eq. 3.7) and the solar phase
(App. C.1) are in `nmosyn.params` (`b_sbr`, `dm2_phi_ee`).
"""
import numpy as np
from numpy.typing import ArrayLike

# isotope, fission fraction in a pressurized-water reactor, coefficients a_0 ... a_5 of
# phi(E) = exp(sum_k a_k E^k) antineutrinos per fission per MeV, E in MeV
ISOTOPES = (
    ('U235', 0.58, (4.367, -4.577, 2.100, -5.294e-1, 6.186e-2, -2.777e-3)),
    ('Pu239', 0.30, (4.757, -5.392, 2.563, -6.596e-1, 7.820e-2, -3.536e-3)),
    ('U238', 0.07, (4.833e-1, 1.927e-1, -1.283e-1, -6.762e-3, 2.233e-3, -1.536e-4)),
    ('Pu241', 0.05, (2.990, -2.882, 1.278, -3.343e-1, 3.905e-2, -1.754e-3)),
)

DELTA_NP = 1.293            # neutron-proton mass difference [MeV]
M_E = 0.511                 # electron mass [MeV]
THRESHOLD_MEV = 1.806       # [MeV]


def huber_mueller_flux(e_mev: ArrayLike) -> np.ndarray:
    """Antineutrinos per fission per MeV of a pressurized-water reactor, Huber-Mueller model (App. A.2)."""
    e = np.asarray(e_mev, dtype=float)
    return sum(f * np.exp(np.polynomial.polynomial.polyval(e, a)) for _, f, a in ISOTOPES)


def ibd_cross_section(e_mev: ArrayLike) -> np.ndarray:
    """Inverse-beta-decay cross section [cm^2], Strumia-Vissani, zero below threshold (App. A.2)."""
    e = np.asarray(e_mev, dtype=float)
    e_e = e - DELTA_NP
    p_e = np.sqrt(np.maximum(e_e ** 2 - M_E ** 2, 0.0))
    e_above = np.maximum(e, THRESHOLD_MEV)
    log_e = np.log(e_above)
    sigma = 1e-43 * p_e * e_e * e_above ** (-0.07056 + 0.02018 * log_e - 0.001953 * log_e ** 3)
    return np.where(e > THRESHOLD_MEV, sigma, 0.0)


def ibd_spectrum(e_mev: ArrayLike) -> np.ndarray:
    """Spectrum before oscillation, flux times cross section [cm^2 per fission per MeV] (App. A.2)."""
    return huber_mueller_flux(e_mev) * ibd_cross_section(e_mev)


def prompt_energy(e_mev: ArrayLike) -> np.ndarray:
    """Prompt energy E_nu - (m_n - m_p) + m_e of the positron signal [MeV] (App. A.2)."""
    return np.asarray(e_mev, dtype=float) - (DELTA_NP - M_E)
