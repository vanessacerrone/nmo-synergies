"""Full fit of the IceCube DeepCore public release (App. E): fig_full_fit_validation, fig_deepcore_reproduction, and the
numbers of App. E.

1. The observed data in both orderings, with the oscillation parameters of the release (App. E.3-E.4), with
   the NuFIT 6.0 Dm21, and with the inverted-ordering hypersurfaces read at |Dm31,IO| (App. E.1).
2. The profiled chi^2 of the normal ordering on the grid of fig_deepcore_reproduction.
3. NO Asimov data at X_GRID with NuFIT 6.0 and dCP = 177 deg in both orderings, the nuisance parameters at the best
   fit of the data: the inverted-ordering fit of all 13 parameters (fig_full_fit_validation), the weighted bin average
   on the fitter's expectation with only Dm31, with theta23, and with every parameter profiled (App. E.2), and at x0 the
   inverted-ordering fit with the other readings of the hypersurfaces (App. E.1).
This is the slow validation step (about half an hour in four processes). Output: results/icfit.npz.
Run from the repository root: python -m compute.icecube_fit
"""
from dataclasses import replace
from functools import cache
from multiprocessing import Pool

import numba
import numpy as np

from nmosyn.experiments import icfit
from nmosyn.params import DCP_NO, NUFIT, X0, X_GRID, intercept
from nmosyn.paths import RESULTS

WBA_POINTS = X_GRID


@cache
def fitter(kind):
    """The fitter of the data, of the data with the NuFIT Dm21, or of the Asimov checks; one per process."""
    if kind == 'data':
        return icfit.DeepCoreFit()
    if kind == 'data_dm21':
        return icfit.DeepCoreFit(p=replace(icfit.RELEASE_OSC, dm21=NUFIT.dm21))
    return icfit.DeepCoreFit(p=NUFIT, dcp=DCP_NO)


def task(args):
    """One unit of work, run in a worker process."""
    numba.set_num_threads(2)
    kind, arg, nuisances = args
    if kind == 'grid_row':
        f = fitter('data')
        return kind, arg, np.array([f.fit('NO', fixed={'s23sq': arg, 'dm31': m + f.p.dm21}).chi2
                                    for m in icfit.DM32_GRID])
    if kind == 'data_io_splitting':
        r = fitter('data').fit('IO', hs_reading='splitting')
        return kind, arg, (r.chi2, r.values['dm31'])
    if kind == 'data_dm21':
        f = fitter('data_dm21')
        return kind, arg, tuple(f.fit(o).values['dm31'] for o in ('NO', 'IO'))
    if kind == 'asimov':
        r = fitter('asimov').asimov_io_fit(arg, nuisances)
        return kind, arg, (r.values['dm31'], r.values['s23sq'], r.chi2)
    if kind == 'asimov_reading':
        x, reading = arg
        r = fitter('asimov').asimov_io_fit(x, nuisances, hs_reading=reading)
        return kind, arg, (r.values['dm31'], r.chi2)
    if kind == 'wba':
        x, case = arg
        point, _ = fitter('asimov').weighted_bin_average(x, nuisances, icfit.WBA_CASES[case])
        return kind, arg, point['dm31']
    raise ValueError(kind)


def main():
    data = fitter('data')
    no, io = data.fit('NO', hesse=True), data.fit('IO', hesse=True)
    nuisances = tuple(sorted((k, no.values[k]) for k in icfit.NOMINAL))
    out = {'names': np.array(icfit.NAMES)}
    for tag, r in (('no', no), ('io', io)):
        out[f'data_{tag}_values'] = np.array([r.values[k] for k in icfit.NAMES])
        out[f'data_{tag}_errors'] = np.array([r.errors[k] for k in icfit.NAMES])
        out[f'data_{tag}_chi2'] = r.chi2
        out[f'data_{tag}_dm31'] = r.values['dm31']
        out[f'data_{tag}_dm31_err'] = r.errors['dm31']
    out['data_b'] = intercept(io.values['dm31'], no.values['dm31'])
    out['dm21_release'] = data.p.dm21

    tasks = [('grid_row', s, None) for s in icfit.S23SQ_GRID]
    tasks += [('data_io_splitting', None, None), ('data_dm21', None, None)]
    tasks += [('asimov', x, dict(nuisances)) for x in X_GRID]
    tasks += [('wba', (x, case), dict(nuisances)) for x in WBA_POINTS for case in icfit.WBA_CASES]
    tasks += [('asimov_reading', (X0, reading), dict(nuisances)) for reading in ('matched', 'splitting', X0)]
    with Pool(4) as pool:
        done = pool.map(task, tasks, chunksize=1)

    grid = np.array([v for k, _, v in done if k == 'grid_row'])
    line = icfit.contour(icfit.S23SQ_GRID, icfit.DM32_GRID, grid, min(grid.min(), no.chi2))
    out.update(grid_s23sq=icfit.S23SQ_GRID, grid_dm32=icfit.DM32_GRID, grid_chi2=grid, contour=line,
               contour_extent=np.array([line[:, 0].min(), line[:, 0].max(), line[:, 1].min(), line[:, 1].max()]))
    for kind, arg, value in done:
        if kind == 'data_io_splitting':
            out['data_io_splitting_chi2'], out['data_io_splitting_dm31'] = value
            out['data_b_splitting'] = intercept(value[1], no.values['dm31'])
        elif kind == 'data_dm21':
            out['data_b_dm21_nufit'] = intercept(value[1], value[0])
    asimov = {arg: value for kind, arg, value in done if kind == 'asimov'}
    out['asimov_x'] = X_GRID
    out['asimov_b'] = np.array([intercept(asimov[x][0], x) for x in X_GRID])
    out['asimov_s23sq'] = np.array([asimov[x][1] for x in X_GRID])
    out['asimov_chi2'] = np.array([asimov[x][2] for x in X_GRID])
    wba = {arg: value for kind, arg, value in done if kind == 'wba'}
    out['wba_x'] = WBA_POINTS
    for case in icfit.WBA_CASES:
        out[f'wba_{case}'] = np.array([intercept(wba[(x, case)], x) for x in WBA_POINTS])
    for kind, arg, value in done:
        if kind == 'asimov_reading':
            name = arg[1] if isinstance(arg[1], str) else 'true'
            out[f'asimov_x0_b_{name}'] = intercept(value[0], X0)
            out[f'asimov_x0_chi2_{name}'] = value[1]
    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(RESULTS / 'icfit.npz', **out)


if __name__ == '__main__':
    main()
