"""The wrong-ordering solution as a weighted bin average (Sec. 2 and App. B).

For NO Asimov data N^NO at the true splitting x, the IO fit minimizes chi^2(y) = sum_i W_i [N_i(y) - N_i^NO]^2,
Eq. (2.1) written in counts, with W_i = 1 / N_i^NO. Each bin prefers the value y_i that reproduces it alone, and the
solution is their average weighted by omega_i = W_i s_i^2, Eq. (2.3). The Gauss-Newton iteration of App. B.3 solves
it without forming the y_i; with nuisance parameters the sensitivity s_i is replaced by the projected one of
Eq. (B.5), Eq. (B.6).
"""
import numpy as np


def poisson_weights(data):
    """W_i = 1 / N_i^NO of Eq. (2.1); bins without expected events get no weight."""
    data = np.asarray(data, dtype=float)
    return np.where(data > 0, 1.0 / np.where(data > 0, data, 1.0), 0.0)


def sensitivity(counts, y, h=1e-7):
    """s_i = dN_i/dy at y, by central differences (App. B.3)."""
    return (counts(y + h) - counts(y - h)) / (2 * h)


def gauss_newton(counts, data, y0, weights=None, h=1e-7, tol=1e-13, max_iter=100):
    """Minimum of sum_i W_i [counts(y)_i - data_i]^2 by the Gauss-Newton iteration of App. B.3.

    counts(y) returns the expected counts of the inverted-ordering model; weights default to 1/data.
    """
    w = poisson_weights(data) if weights is None else weights
    y = float(y0)
    for _ in range(max_iter):
        s = sensitivity(counts, y, h)
        step = np.sum(w * s * (data - counts(y))) / np.sum(w * s * s)
        y += step
        if abs(step) < tol:
            break
    return y


def bin_solutions(counts, data, y, weights=None, h=1e-7):
    """Values y_i preferred by the single bins and their weights omega_i = W_i s_i^2 at the solution y (Eq. 2.3).

    Bins with s_i = 0 prefer no value (NaN) and carry no weight. The weighted mean of y_i is the solution y.
    """
    w = poisson_weights(data) if weights is None else weights
    s = sensitivity(counts, y, h)
    resid = data - counts(y)
    ok = s != 0
    y_i = np.where(ok, y + resid / np.where(ok, s, 1.0), np.nan)
    return y_i, w * s * s


def projected_sensitivity(s, jac, weights, prior_inv):
    """s~ = s - J (J^T W J + Sigma^-1)^-1 J^T W s, Eq. (B.5): the part of s no nuisance parameter can mimic.

    s (n_bins,), jac (n_bins, n_par) the Jacobian of the counts in the nuisance parameters, weights (n_bins,) the
    diagonal of W, prior_inv (n_par, n_par) the inverse prior covariance (zero rows for parameters without prior).
    """
    a = jac.T @ (weights[:, None] * jac) + prior_inv
    return s - jac @ np.linalg.solve(a, jac.T @ (weights * s))


def profiled_step(resid, s, jac, weights, prior_inv, prior_pull):
    """One Gauss-Newton step in (y, theta) with the nuisance parameters theta profiled (Eqs. B.5-B.6).

    resid = data - counts at the current point, prior_pull = Sigma^-1 (theta - theta_0) at the current point.
    The step in y is sum W s r~ / sum W s s~, with r~ the residual after the nuisance parameters have absorbed what
    they can (for theta at the prior centers sum W s r~ = sum W s~ r, and with r_i = s_i (y_i - y) the step is
    the average of Eq. B.6); the step in theta is the conditional minimum given that step. Returns (dy, dtheta).
    """
    a = jac.T @ (weights[:, None] * jac) + prior_inv
    s_tilde = projected_sensitivity(s, jac, weights, prior_inv)
    r_tilde = resid - jac @ np.linalg.solve(a, jac.T @ (weights * resid) - prior_pull)
    dy = np.sum(weights * s * r_tilde) / np.sum(weights * s * s_tilde)
    dtheta = np.linalg.solve(a, jac.T @ (weights * (resid - s * dy)) - prior_pull)
    return dy, dtheta
