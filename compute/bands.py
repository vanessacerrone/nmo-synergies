"""Wrong-ordering solutions of NOvA, T2K, and IceCube DeepCore over the unknown true delta_CP and theta23 (Sec. 5).

For every sample, true splitting, and true point (sin^2 theta23, delta_CP) the inverted-ordering fit is taken at its
global minimum over Dm31_IO, theta23 in both octants, and delta_CP^IO (nmosyn.fit.global_fit). The range of b at each
true point is stored under several valley rules (App. B.3), so that the figures can use any of them:
  paper   the rule of App. B.3: the range over delta_CP^IO within 0.01 of the minimum when more than three values
          of delta_CP^IO in steps of 10 deg are within it, else b at the minimum
  point   b at the global minimum
  tol     every delta_CP^IO within 0.01 of the minimum (the valley of accelerator disappearance)
  extent  as tol where those points cover more than half of the circle, else the point
  circle  the profile minimum over the whole circle of delta_CP^IO
  both    as tol, with the profiles of both octants
True points: sin^2 theta23 in S0 and delta_CP in DN (NuFIT 6.0 3 sigma ranges), plus the theta23 scan at the NuFIT
delta_CP; at the NuFIT splitting also the scans of fig_intercept_vs_truth. Output: results/bands.npz.
Run from the repository root: python -m compute.bands
"""
import argparse
from multiprocessing import Pool

import numba
import numpy as np

from nmosyn import fit
from nmosyn.experiments import deepcore, lbl
from nmosyn.params import DCP_NO, NUFIT, S23SQ, X0, kappa_mu
from nmosyn.paths import RESULTS

SAMPLES = ('nova_dis', 'nova_all', 't2k_dis', 't2k_all', 'dcn', 'dca')
X_BANDS = (2.2e-3, X0, 2.8e-3)
S0 = np.append(np.round(np.arange(0.43, 0.5901, 0.01), 3), 0.596)
DN = np.radians(np.arange(96.0, 422.1, 15.0))
DT = np.radians(np.arange(0.0, 360.01, 10.0))                      # fig_intercept_vs_truth, left: true delta_CP
ST = np.append(np.round(np.arange(0.43, 0.5951, 0.005), 3), 0.596) # fig_intercept_vs_truth, right: true sin^2 theta23
RULES = {'paper': fit.ValleyRule('count'), 'point': fit.ValleyRule('point'), 'tol': fit.ValleyRule('tol'),
         'extent': fit.ValleyRule('extent'), 'circle': fit.ValleyRule('circle'),
         'both': fit.ValleyRule('tol', both_octants=True)}
B_RANGE = {'nova_dis': (40, 200), 'nova_all': (40, 200), 't2k_dis': (40, 200), 't2k_all': (40, 200),
           'dcn': (70, 190), 'dca': (70, 190)}


def model(sample):
    if sample.startswith('dc'):
        return deepcore.DeepCore('numu' if sample == 'dcn' else 'all', NUFIT)
    m = lbl.nova(NUFIT) if sample.startswith('nova') else lbl.t2k(NUFIT)
    return fit.Bins(m, m.disappearance) if sample.endswith('dis') else m


def solve(h, x, truths, sample):
    """Global fits at the true points [(s23sq, dcp)]: b, s23sq, dcp, chi2, ranges per rule, closed form."""
    out = {k: [] for k in ('b', 's23sq', 'dcp', 'chi2', 'cf')}
    out.update({f'range_{r}': [] for r in RULES})
    for s0, d0 in truths:
        sol = fit.global_fit(h, x, h.exact_counts(x, s0, d0), B_RANGE[sample])
        for k in ('b', 's23sq', 'dcp', 'chi2'):
            out[k].append(getattr(sol, k))
        for r, rule in RULES.items():
            out[f'range_{r}'].append(sol.b_range(rule))
        o = sol.best_octant()                    # closed form of Eq. (5.1) over the valley of the tol rule
        keep = sol.profile['chi2'][o] <= sol.chi2 + RULES['tol'].tol
        dio, sio = np.append(sol.profile['dcp'][keep], sol.dcp), np.append(sol.profile['s23sq'][o, keep], sol.s23sq)
        cf = (kappa_mu(d0, s0) + kappa_mu(dio, sio)) * NUFIT.dm21 * 1e6
        out['cf'].append((cf.min(), cf.max()))
    return {k: np.array(v) for k, v in out.items()}


PARTS = RESULTS / 'bands_parts'


def job(args):
    """One sample at one true splitting; the result is saved on its own, so that an interrupted run resumes."""
    sample, ix = args
    path = PARTS / f'{sample}_{ix}.npz'
    if path.exists():
        return path
    numba.set_num_threads(1)
    x = X_BANDS[ix]
    h = fit.Harmonics(model(sample))
    grid = [(s0, d0) for s0 in S0 for d0 in DN]
    res = {'grid': solve(h, x, grid, sample), 't23': solve(h, x, [(s0, DCP_NO) for s0 in S0], sample)}
    if ix == 1 and sample in ('nova_dis', 'nova_all', 'dcn', 'dca'):
        res['scan_dcp'] = solve(h, x, [(S23SQ, d) for d in DT], sample)
        res['scan_s23'] = solve(h, x, [(s, DCP_NO) for s in ST], sample)
    np.savez_compressed(path, **{f'{part}/{k}': v for part, arrays in res.items() for k, v in arrays.items()})
    print(f'{sample} x = {x * 1e3:.3f}: done', flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('samples', nargs='*', default=SAMPLES, choices=SAMPLES, help='samples to compute')
    samples = parser.parse_args().samples
    PARTS.mkdir(parents=True, exist_ok=True)
    jobs = [(s, i) for s in samples for i in range(len(X_BANDS))]
    jobs.sort(key=lambda j: (j[0] not in ('t2k_all', 'dca', 'dcn'), j[1] != 1))     # longest first
    with Pool(6) as pool:
        for _ in pool.imap_unordered(job, jobs):
            pass
    if any(not (PARTS / f'{s}_{i}.npz').exists() for s in SAMPLES for i in range(len(X_BANDS))):
        return
    out = dict(x=np.array(X_BANDS), s0=S0, dn=DN, dt=DT, st=ST)
    for sample, ix in ((s, i) for s in SAMPLES for i in range(len(X_BANDS))):
        with np.load(PARTS / f'{sample}_{ix}.npz') as part:
            out.update({f'{sample}/{k}/{ix}': part[k] for k in part.files})
    np.savez_compressed(RESULTS / 'bands.npz', **out)


if __name__ == '__main__':
    main()
