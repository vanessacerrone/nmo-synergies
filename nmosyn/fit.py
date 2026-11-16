"""The inverted-ordering fit: the minimum of the chi^2 of Eq. (2.1) over Dm31_IO, theta23, and delta_CP^IO.

theta23 and delta_CP enter every expected count through the rotation of App. D.2, for any density profile. At fixed
Dm31 the counts are therefore trigonometric polynomials of degree 2 in delta_CP and in 2 theta23, and 5 x 5 samples
determine them exactly. `Harmonics` keeps these coefficients on a grid of Dm31 with step 1e-6 eV^2 and interpolates
them with four-point Lagrange polynomials, so that the chi^2 and its derivatives cost a few matrix products at any
point. `global_fit` takes the minimum over Dm31_IO, theta23 in both octants, and delta_CP^IO (Sec. 5) and keeps the
profile of the chi^2 in delta_CP^IO for each octant. Where the minimum is degenerate along delta_CP^IO, as in
accelerator disappearance, a `ValleyRule` turns the profile into a range of b (App. B.3).
"""
from dataclasses import dataclass

import numpy as np

from .average import poisson_weights

T_NODES = np.arange(5) * np.pi / 8              # theta23 in [0, pi/2]
D_NODES = np.arange(5) * 2 * np.pi / 5
T_CHECK, D_CHECK = 0.3, 1.1                      # extra point that checks every set of coefficients
S23SQ_SCAN = np.arange(0.01, 0.9901, 0.02)       # coarse scan for the starting points, over the whole range
DCP_COARSE = np.radians(np.arange(0.0, 360.0, 10.0))
DCP_SCAN = np.radians(np.arange(0.0, 360.0, 2.0))  # points of the profile in delta_CP^IO


def _t_basis(theta, order=0):
    """[1, cos 2t, sin 2t, cos 4t, sin 4t] and its derivatives in t."""
    t = np.asarray(theta, dtype=float)
    f = [np.ones_like(t), np.cos(2 * t), np.sin(2 * t), np.cos(4 * t), np.sin(4 * t)]
    if order == 1:
        f = [np.zeros_like(t), -2 * f[2], 2 * f[1], -4 * f[4], 4 * f[3]]
    return np.stack(f, axis=-1)


def _d_basis(dcp, order=0):
    """[1, cos d, sin d, cos 2d, sin 2d] and its derivatives in d."""
    d = np.asarray(dcp, dtype=float)
    f = [np.ones_like(d), np.cos(d), np.sin(d), np.cos(2 * d), np.sin(2 * d)]
    if order == 1:
        f = [np.zeros_like(d), -f[2], f[1], -2 * f[4], 2 * f[3]]
    return np.stack(f, axis=-1)


def _basis(theta, dcp, dt=0, dd=0):
    """Products of the theta23 and delta_CP harmonics, shape (..., 25)."""
    theta, dcp = np.broadcast_arrays(np.asarray(theta, dtype=float), np.asarray(dcp, dtype=float))
    t, d = _t_basis(theta, dt), _d_basis(dcp, dd)
    return (t[..., :, None] * d[..., None, :]).reshape(theta.shape + (25,))


def theta(s23sq):
    return np.arcsin(np.sqrt(s23sq))


class Bins:
    """A model restricted to some of its analysis bins (e.g. the disappearance samples of an accelerator)."""

    def __init__(self, model, mask):
        self.model, self.mask = model, np.asarray(mask, dtype=bool)
        self.n_bins = int(self.mask.sum())

    def counts(self, y, s23sq, dcp):
        return self.model.counts(y, s23sq, dcp)[self.mask]


class Harmonics:
    """Expected counts of a model at any (Dm31, theta23, delta_CP) from its exact samples (see the module docstring).

    model.counts(y, s23sq, dcp) is sampled at the 25 nodes for each Dm31 of the grid and checked at one more point;
    scale multiplies the counts (e.g. to the event numbers of a release).
    """

    def __init__(self, model, scale=1.0, step=1e-6):
        self.model, self.scale, self.step = model, scale, step
        self.n_bins = model.n_bins
        self._inverse = np.linalg.inv(_basis(np.repeat(T_NODES, 5), np.tile(D_NODES, 5)))
        self._nodes = {}

    def exact(self, y):
        """Coefficients (25, n_bins) at y, from the model."""
        samples = np.array([self.model.counts(y, np.sin(t) ** 2, d) for t in T_NODES for d in D_NODES])
        coef = self._inverse @ samples
        check = self.model.counts(y, np.sin(T_CHECK) ** 2, D_CHECK)
        if np.max(np.abs(_basis(T_CHECK, D_CHECK) @ coef - check)) > 1e-9 * np.max(np.abs(check)):
            raise ValueError(f'counts are not a trigonometric polynomial in theta23 and delta_CP at y = {y}')
        return self.scale * coef

    def _node(self, k):
        if k not in self._nodes:
            self._nodes[k] = self.exact(k * self.step)
        return self._nodes[k]

    def _lagrange(self, y):
        """First node index and four-point Lagrange weights (value and derivative in y) at the values y."""
        u = np.asarray(y, dtype=float) / self.step
        k0 = np.floor(u).astype(int) - 1
        t = u - k0
        w = np.stack([-(t - 1) * (t - 2) * (t - 3) / 6, t * (t - 2) * (t - 3) / 2,
                      -t * (t - 1) * (t - 3) / 2, t * (t - 1) * (t - 2) / 6], axis=-1)
        dw = np.stack([-(3 * t * t - 12 * t + 11) / 6, (3 * t * t - 10 * t + 6) / 2,
                       -(3 * t * t - 8 * t + 3) / 2, (3 * t * t - 6 * t + 2) / 6], axis=-1) / self.step
        return k0, w, dw

    def _stencils(self, k0):
        """Coefficients of the four nodes from each k0, shape (M, 4, 25, n_bins)."""
        ks = np.unique(k0[:, None] + np.arange(4))
        table = np.array([self._node(k) for k in ks])
        return table[np.searchsorted(ks, k0[:, None] + np.arange(4))]

    def coefficients(self, y):
        """Coefficients at the values y (M,) and their derivatives in y, each (M, 25, n_bins)."""
        k0, w, dw = self._lagrange(y)
        nodes = self._stencils(k0)
        return np.einsum('mj,mjcn->mcn', w, nodes), np.einsum('mj,mjcn->mcn', dw, nodes)

    def exact_counts(self, y, s23sq, dcp):
        """Expected counts at one point, from the model (e.g. the NO data)."""
        return self.scale * self.model.counts(y, s23sq, dcp)

    def counts(self, y, s23sq, dcp):
        """Expected counts at one point, interpolated in y."""
        c, _ = self.coefficients([y])
        return _basis(theta(s23sq), dcp) @ c[0]


def _chi2_and_jacobian(h, data, w, y, t, d):
    """chi^2 and its Gauss-Newton gradient and Hessian in (y, theta, delta) at M points."""
    k0, lw, dlw = h._lagrange(y)
    b = np.stack([_basis(t, d), _basis(t, d, dt=1), _basis(t, d, dd=1)], axis=1)        # (M, 3, 25)
    g = b[:, None] @ h._stencils(k0)                                                      # (M, 4, 3, n_bins)
    pred = np.einsum('mj,mjn->mn', lw, g[:, :, 0])
    jac = np.stack([np.einsum('mj,mjn->mn', dlw, g[:, :, 0]), np.einsum('mj,mjn->mn', lw, g[:, :, 1]),
                    np.einsum('mj,mjn->mn', lw, g[:, :, 2])], axis=1)                   # (M, 3, n_bins)
    r = pred - data
    chi2 = (r * r) @ w
    grad = jac @ (w * r)[..., None]
    hess = (jac * w) @ jac.transpose(0, 2, 1)
    return chi2, grad[..., 0], hess


def minimize(h, data, y, s23sq, dcp, free=(True, True, True), weights=None, s23sq_bounds=(0.0, 1.0), n_iter=50,
             chunk=64):
    """Damped Gauss-Newton minimization of the chi^2 from M starting points at once.

    free selects which of (y, theta23, delta_CP) are fitted; s23sq_bounds = (lower, upper), scalars or arrays of
    shape (M,), limit theta23. A point stops when its step falls below 1e-12 eV^2 in Dm31 and 1e-7 rad in the
    angles. Returns arrays (chi2, y, s23sq, dcp) of shape (M,).
    """
    w = poisson_weights(data) if weights is None else weights
    p = np.stack(np.broadcast_arrays(np.asarray(y, float), theta(np.asarray(s23sq, float)),
                                     np.asarray(dcp, float)), axis=-1).reshape(-1, 3).copy()
    lo, hi = (np.broadcast_to(theta(np.asarray(v, float)), len(p)).copy() for v in s23sq_bounds)
    p[:, 1] = np.clip(p[:, 1], lo, hi)
    mask = np.array(free, dtype=bool)
    scale = np.array([1e-6, 1.0, 1.0])            # parameters in comparable units for the damping
    lam = np.full(len(p), 1e-3)
    chi2, grad, hess = np.empty(len(p)), np.empty((len(p), 3)), np.empty((len(p), 3, 3))
    for part in np.array_split(np.arange(len(p)), max(1, int(np.ceil(len(p) / chunk)))):
        chi2[part], grad[part], hess[part] = _chi2_and_jacobian(h, data, w, *p[part].T)
    active = np.arange(len(p))
    for _ in range(n_iter):
        g = (grad[active] * scale)[:, mask]
        a = (hess[active] * scale[:, None] * scale[None, :])[:, mask][:, :, mask]
        a = a + lam[active, None, None] * np.einsum('mii->mi', a)[:, :, None] * np.eye(mask.sum())
        trial = p[active].copy()
        trial[:, mask] -= np.linalg.solve(a, g[..., None])[..., 0] * scale[mask]
        trial[:, 1] = np.clip(trial[:, 1], lo[active], hi[active])
        step = np.abs(trial - p[active])
        still = np.ones(len(active), dtype=bool)
        for part in np.array_split(np.arange(len(active)), max(1, int(np.ceil(len(active) / chunk)))):
            idx = active[part]
            c2, g2, h2 = _chi2_and_jacobian(h, data, w, *trial[part].T)
            better = c2 <= chi2[idx]
            take = idx[better]
            p[take], chi2[take], grad[take], hess[take] = trial[part][better], c2[better], g2[better], h2[better]
            lam[idx] = np.where(better, np.maximum(lam[idx] / 10, 1e-9), lam[idx] * 10)
            converged = (step[part, 0] < 1e-12) & np.all(step[part, 1:] < 1e-7, axis=1)
            still[part] = ~(converged | (lam[idx] > 1e8))
        active = active[still]
        if len(active) == 0:
            break
    return chi2, p[:, 0], np.sin(p[:, 1]) ** 2, np.mod(p[:, 2], 2 * np.pi)


@dataclass(frozen=True)
class ValleyRule:
    """How a minimum that is degenerate in delta_CP^IO is reported as a range of b (App. B.3).

    kind 'point': b at the global minimum only. 'tol': every delta_CP^IO whose profile chi^2 is within tol of the
    minimum. 'count': as 'tol' if more than `count` values of delta_CP^IO in steps of `step_deg` are within tol,
    else 'point' (App. B.3). 'extent': as 'tol' if the points within tol cover more than the share `extent` of the
    circle, else 'point'. 'circle': the profile minimum b(delta_CP^IO) over the whole circle, with no tolerance.
    both_octants adds the profile of the other octant under the same rule.
    """

    kind: str = 'count'
    tol: float = 0.01
    count: int = 3
    step_deg: float = 10.0
    extent: float = 0.5
    both_octants: bool = False


@dataclass
class Solution:
    """Inverted-ordering fit at its global minimum, with the profile in delta_CP^IO of each octant."""

    x: float
    y: float
    s23sq: float
    dcp: float
    chi2: float
    profile: dict            # dcp (n,), and chi2, y, s23sq of shape (2, n): lower and upper octant

    @property
    def b(self):
        return (self.y + self.x) * 1e6

    def best_octant(self):
        return int(self.s23sq > 0.5)

    def b_range(self, rule=ValleyRule()):
        """(b_lo, b_hi) in 1e-6 eV^2 under a ValleyRule."""
        if rule.kind == 'point':
            return self.b, self.b
        octants = [0, 1] if rule.both_octants else [self.best_octant()]
        c, y = self.profile['chi2'][octants], self.profile['y'][octants]
        if rule.kind == 'circle':
            keep = np.ones_like(c, dtype=bool)
        else:
            keep = c <= self.chi2 + rule.tol
            if rule.kind == 'extent' and keep[0].mean() <= rule.extent:
                return self.b, self.b
            if rule.kind == 'count':
                on_grid = np.isclose(np.mod(np.degrees(self.profile['dcp']) + 1e-9, rule.step_deg), 0, atol=1e-6)
                if np.sum(keep[0] & on_grid) <= rule.count:
                    return self.b, self.b
        bs = (np.append(y[keep], self.y) + self.x) * 1e6
        return bs.min(), bs.max()


def _scan(h, data, w, ks, t_scan):
    """chi^2 on the nodes ks x t_scan x DCP_COARSE, from the quadratic form phi^T Q phi - 2 phi^T r + sum w d^2."""
    basis = _basis(t_scan[:, None], DCP_COARSE[None, :]).reshape(-1, 25)
    c = np.array([h._node(k) for k in ks])                                  # (n_k, 25, n_bins)
    q = np.einsum('kcn,n,kdn->kcd', c, w, c)
    r = c @ (w * data)
    out = np.sum((basis[None] @ q) * basis[None], axis=2) - 2 * r @ basis.T
    return out.reshape(len(ks), len(t_scan), len(DCP_COARSE)) + np.sum(w * data * data)


def global_fit(h, x, data, b_range=(40.0, 200.0), weights=None):
    """Global minimum over Dm31_IO, theta23 in both octants, and delta_CP^IO (Sec. 5, App. B.3).

    data are the NO counts at the true splitting x (same scale as h). The chi^2 is scanned on every other Dm31 node
    in b_range [1e-6 eV^2], on S23SQ_SCAN, and on DCP_COARSE; for every delta_CP^IO of DCP_SCAN it is then minimized
    over (Dm31_IO, theta23) within each octant from the best start in that octant, which gives the two profiles, and the
    global minimum is refined in all three parameters. A minimum on the edge of b_range raises ValueError.
    """
    w = poisson_weights(data) if weights is None else weights
    ks = np.arange(int(np.floor((-x + b_range[0] * 1e-6) / h.step)),
                   int(np.ceil((-x + b_range[1] * 1e-6) / h.step)) + 1)
    t_scan = theta(S23SQ_SCAN)
    coarse = ks[::2]
    chi2 = _scan(h, data, w, coarse, t_scan)                                # (n_k, n_t, n_coarse)
    nd = len(DCP_SCAN)
    nearest = np.round(DCP_SCAN / (DCP_COARSE[1] - DCP_COARSE[0])).astype(int) % len(DCP_COARSE)
    y0, s0, lo, hi = [], [], [], []
    for octant, bounds in ((t_scan <= np.pi / 4, (0.0, 0.5)), (t_scan >= np.pi / 4, (0.5, 1.0))):
        sub = np.where(octant[None, :, None], chi2, np.inf)
        best = sub.transpose(2, 0, 1).reshape(len(DCP_COARSE), -1).argmin(axis=1)[nearest]
        ik, it = np.unravel_index(best, (len(coarse), len(t_scan)))
        y0.append(coarse[ik] * h.step)
        s0.append(S23SQ_SCAN[it])
        lo.append(np.full(nd, bounds[0]))
        hi.append(np.full(nd, bounds[1]))
    c, y, s, _ = minimize(h, data, np.concatenate(y0), np.concatenate(s0), np.tile(DCP_SCAN, 2),
                          free=(True, True, False), weights=w, s23sq_bounds=(np.concatenate(lo), np.concatenate(hi)))
    prof = dict(dcp=DCP_SCAN, chi2=c.reshape(2, nd), y=y.reshape(2, nd), s23sq=s.reshape(2, nd))
    o, j = np.unravel_index(np.argmin(prof['chi2']), prof['chi2'].shape)
    cg, yg, sg, dg = (v[0] for v in minimize(h, data, prof['y'][o, j], prof['s23sq'][o, j], DCP_SCAN[j], weights=w))
    if not ks[0] + 2 < yg / h.step < ks[-1] - 2:
        raise ValueError(f'minimum b = {(yg + x) * 1e6:.2f} on the edge of b_range {b_range}')
    return Solution(x=x, y=yg, s23sq=sg, dcp=dg, chi2=cg, profile=prof)
