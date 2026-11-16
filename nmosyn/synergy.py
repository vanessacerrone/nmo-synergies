"""The synergy term of two experiments (Eq. 1.1) and its extensions to real data and shared parameters (App. F)."""
import numpy as np


def dchi2_syn(b_a, b_b, sigma_a, sigma_b):
    """(f_A - f_B)^2 / (sigma_A^2 + sigma_B^2), Eq. (1.1), with f_A - f_B = b_A - b_B (App. F.1)."""
    return (np.asarray(b_a) - np.asarray(b_b)) ** 2 / (np.asarray(sigma_a) ** 2 + np.asarray(sigma_b) ** 2)


def dchi2_syn_data(db, dx, sigma_a, sigma_b, sigma_a_no, sigma_b_no):
    """Synergy term when the two experiments measure different splittings x_A, x_B, Eq. (F.1)."""
    return (db - dx) ** 2 / (sigma_a ** 2 + sigma_b ** 2) - dx ** 2 / (sigma_a_no ** 2 + sigma_b_no ** 2)


def dchi2_syn_shared(f_a, f_b, cov_a, cov_b):
    """Synergy term with shared oscillation parameters, Eq. (F.2): a quadratic form in the difference of solutions."""
    d = np.asarray(f_a) - np.asarray(f_b)
    return float(d @ np.linalg.solve(np.asarray(cov_a) + np.asarray(cov_b), d))


def gap(range_a, range_b):
    """Smallest |b_A - b_B| between two ranges of equally good solutions, one row (b_lo, b_hi) per true point.

    Where an experiment has a degenerate valley (accelerator disappearance), the joint inverted-ordering fit takes
    the point of the valley closest to the partner (Sec. 6).
    """
    a, b = np.atleast_2d(range_a), np.atleast_2d(range_b)
    return np.maximum(0.0, np.maximum(b[:, 0] - a[:, 1], a[:, 0] - b[:, 1]))


def synergy_range(range_a, range_b, sigma_a, sigma_b):
    """Smallest and largest synergy term over the true points shared by the two experiments (fig_synergy_vs_precision
    and fig_synergy_rings)."""
    g = gap(range_a, range_b)
    return dchi2_syn(g.min(), 0.0, sigma_a, sigma_b), dchi2_syn(g.max(), 0.0, sigma_a, sigma_b)
