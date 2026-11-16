"""Joint inverted-ordering fit of NOvA and DeepCore with shared Dm31_IO, theta23^IO, and delta_CP^IO (Sec. 6, App. F.2).

At every true point of the bands of Sec. 5 (sin^2 theta23 in S0, delta_CP in DN, true splitting X0) the synergy
term of Eq. (1.1), gap^2 / (sigma_A^2 + sigma_B^2) with the closest gap of the two ranges of b (rule of App. B.3), is
compared with the exact joint IO fit, the minimum of chi2_A + chi2_B over the shared (Dm31_IO, theta23^IO,
delta_CP^IO) minus the two separate IO minima, the quantity that Eq. (F.2) approximates. The joint minimum is taken on
a grid (every Dm31 node of b = 40-215, sin^2 theta23 0.30-0.70 in steps of 0.005, delta_CP^IO in steps of 5 deg) and
refined by L-BFGS-B from the best grid point of each octant. sigma of each experiment is the half width of its IO
profile in b at chi2_min + 1 on the same grid. Two panels as in fig_synergy_vs_precision: 'dis' (NOvA nu_mu
disappearance with DeepCore nu_mu) and 'app' (NOvA all channels with DeepCore all channels). Both quantities are given
at the real exposures (raw) and with each chi2 scaled so that its width in b is 1.5% of |Dm31| (the common precision of
fig_synergy_vs_precision). Output: results/joint.npz, keys '<panel>/<quantity>'. About fifteen minutes in two processes.
Run from the repository root: python -m compute.joint
"""
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

from nmosyn import fit, synergy
from nmosyn.average import poisson_weights
from nmosyn.params import X0
from nmosyn.paths import RESULTS

from compute.bands import B_RANGE, DN, S0, model

PANELS = {'dis': ('nova_dis', 'dcn'), 'app': ('nova_all', 'dca')}
SIG_COMMON = 0.015 * X0 * 1e6               # 1.5% of |Dm31| [1e-6 eV^2]
B_LO, B_HI = 40, 215
S_GRID = np.round(np.arange(0.30, 0.7001, 0.005), 3)
D_GRID = np.radians(np.arange(0, 360, 5.0))
T_GRID = fit.theta(S_GRID)
KS = np.arange(int(np.floor((-X0 + B_LO * 1e-6) / 1e-6)), int(np.ceil((-X0 + B_HI * 1e-6) / 1e-6)) + 1)
B_GRID = (KS * 1e-6 + X0) * 1e6
BASIS = fit._basis(T_GRID[:, None], D_GRID[None, :]).reshape(-1, 25)


def grid_chi2(h, data, w):
    """chi2 on (Dm31 node, sin^2 theta23, delta_CP^IO), from the harmonic coefficients of each node."""
    c = np.array([h._node(k) for k in KS])
    q = np.einsum('kcn,n,kdn->kcd', c, w, c)
    r = c @ (w * data)
    out = np.einsum('pc,kcd,pd->kp', BASIS, q, BASIS) - 2 * r @ BASIS.T + np.sum(w * data * data)
    return out.reshape(len(KS), len(S_GRID), len(D_GRID))


def width(g):
    """Half width in b of the profile of the grid chi2 at its minimum + 1 (nan on a side that does not reach it)."""
    prof = g.reshape(len(KS), -1).min(1)
    j = prof.argmin()
    m = prof[j]
    left = np.interp(m + 1, prof[j::-1], B_GRID[j::-1]) if prof[:j + 1].max() > m + 1 else np.nan
    right = np.interp(m + 1, prof[j:], B_GRID[j:]) if prof[j:].max() > m + 1 else np.nan
    return 0.5 * ((B_GRID[j] - left) + (right - B_GRID[j]))


def joint(hs, datas, ws, gs, k):
    """Minimum of k_A chi2_A + k_B chi2_B over the shared (Dm31_IO, theta23, delta_CP^IO): value, b, s23sq, dcp."""
    g = k[0] * gs[0] + k[1] * gs[1]

    def f(p):
        return sum(kk * float(np.sum(w * (h.counts(p[0] * 1e-6, np.sin(p[1]) ** 2, p[2]) - d) ** 2))
                   for kk, h, d, w in zip(k, hs, datas, ws))
    best = None
    for sel, lo, hi in ((S_GRID <= 0.5, 0.0, np.pi / 4), (S_GRID >= 0.5, np.pi / 4, np.pi / 2)):
        sub = np.where(sel[None, :, None], g, np.inf)
        ik, it, idd = np.unravel_index(sub.argmin(), g.shape)
        start = np.array([KS[ik], T_GRID[it], D_GRID[idd]], dtype=float)
        r = minimize(f, start, method='L-BFGS-B', bounds=[(KS[0], KS[-1]), (lo, hi), (None, None)],
                     options=dict(ftol=1e-13, gtol=1e-9))
        val, p = (r.fun, r.x) if r.fun <= sub.min() else (sub.min(), start)
        if best is None or val < best[0]:
            best = (val, p)
    val, p = best
    return val, p[0] + X0 * 1e6, np.sin(p[1]) ** 2, np.degrees(p[2]) % 360


def panel(name):
    hs = [fit.Harmonics(model(s)) for s in PANELS[name]]
    rows = []
    for s0 in S0:
        for d0 in DN:
            datas = [h.exact_counts(X0, s0, d0) for h in hs]
            ws = [poisson_weights(d) for d in datas]
            sols = [fit.global_fit(h, X0, d, B_RANGE[s], weights=w) for h, d, w, s in zip(hs, datas, ws, PANELS[name])]
            gs = [grid_chi2(h, d, w) for h, d, w in zip(hs, datas, ws)]
            sig = [width(g) for g in gs]
            gap = synergy.gap(*(s.b_range(fit.ValleyRule('count')) for s in sols))[0]
            cmin = [s.chi2 for s in sols]
            raw = joint(hs, datas, ws, gs, (1.0, 1.0))
            k = ((sig[0] / SIG_COMMON) ** 2, (sig[1] / SIG_COMMON) ** 2)
            com = joint(hs, datas, ws, gs, k)
            rows.append((s0, np.degrees(d0) % 360, gap, *sig,
                         synergy.dchi2_syn(gap, 0, *sig), raw[0] - sum(cmin),
                         synergy.dchi2_syn(gap, 0, SIG_COMMON, SIG_COMMON), com[0] - k[0] * cmin[0] - k[1] * cmin[1],
                         *com[1:]))
        print(f'{name}: sin^2 theta23 = {s0:.3f} done', flush=True)
    keys = ('s23sq', 'dcp', 'gap', 'sigma_nova', 'sigma_dc', 'eq11_raw', 'joint_raw', 'eq11', 'joint',
            'fit_b', 'fit_s23sq', 'fit_dcp')
    return {f'{name}/{k}': v for k, v in zip(keys, np.array(rows).T)}


def main():
    with Pool(len(PANELS)) as pool:
        parts = pool.map(panel, PANELS)
    out = {k: v for part in parts for k, v in part.items()}
    np.savez_compressed(RESULTS / 'joint.npz', sig_common=SIG_COMMON, **out)
    for name in PANELS:
        e, j = out[f'{name}/eq11'], out[f'{name}/joint']
        i = j.argmax()
        print(f'{name}: joint > Eq. (1.1) at {np.mean(j > e):.1%} of the true points; largest joint {j[i]:.3f} at '
              f'sin^2 theta23 = {out[f"{name}/s23sq"][i]:.3f}, delta_CP = {out[f"{name}/dcp"][i]:.0f} deg')


if __name__ == '__main__':
    main()
