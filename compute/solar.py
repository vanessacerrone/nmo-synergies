"""Dependence of the intercepts on the solar parameters (Sec. 6, footnote): JUNO, the NOvA closed form, DeepCore.

Dm21 and sin^2 theta12 are moved by +-REL, the relative precision of the JUNO measurement after 207 days
(JUNO, arXiv:2609.37895: 1.1% on Dm21, 2.1% on sin^2 theta12), and each intercept at x0 changes by half the
difference: JUNO (Eq. 2.3, design configuration), the closed form of Eq. (3.6) for NOvA on the 30-deg grid of both
phases, and DeepCore with all channels, theta23 and both phases fixed. Output: results/solar.npz.
Run from the repository root: python -m compute.solar
"""
from dataclasses import replace

import numpy as np

from nmosyn.average import gauss_newton
from nmosyn.experiments import juno
from nmosyn.experiments.deepcore import DeepCore
from nmosyn.params import DCP_IO, DCP_NO, NUFIT, S23SQ, X0, b_lbl, intercept
from nmosyn.paths import RESULTS

REL = {'dm21': 0.011, 's12sq': 0.021}
DCP_GRID = np.radians(np.arange(-180.0, 180.1, 30.0))


def intercepts(p):
    """b of JUNO, of the NOvA closed form on the grid (dCP_NO, dCP_IO), and of DeepCore all channels, at x0."""
    deepcore = DeepCore('all', p=p)
    y = gauss_newton(lambda v: deepcore.counts(v, S23SQ, DCP_IO), deepcore.counts(X0, S23SQ, DCP_NO),
                     -X0 + b_lbl(DCP_NO, DCP_IO, p=p))
    return (intercept(juno.Juno(p=p).solution(X0), X0),
            b_lbl(DCP_GRID[:, None], DCP_GRID[None, :], p=p) * 1e6,
            intercept(y, X0))


def main():
    out = dict(parameters=np.array(list(REL)), rel=np.array(list(REL.values())), dcp_grid=DCP_GRID)
    shifts = []
    for name, rel in REL.items():
        up, down = (intercepts(replace(NUFIT, **{name: getattr(NUFIT, name) * (1 + sign * rel)}))
                    for sign in (1, -1))
        shifts.append([(np.asarray(u) - np.asarray(d)) / 2 for u, d in zip(up, down)])
    out['juno'], out['nova'], out['deepcore'] = (np.array(s) for s in zip(*shifts))
    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(RESULTS / 'solar.npz', **out)


if __name__ == '__main__':
    main()
