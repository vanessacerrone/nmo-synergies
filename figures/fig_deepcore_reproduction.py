"""fig_deepcore_reproduction: the IceCube DeepCore measurement reproduced from the public release (App. E.3).

The 90% CL contour of the normal-ordering fit to the observed data (nmosyn.experiments.icfit.contour, from the
chi^2 profiled over the 11 nuisance parameters on a grid of sin^2 theta23 and Dm32), against the published contour
of PRD 108, 012014, Fig. 26, read from the vector paths of the figure by scripts/digitize_icecube_fig26.py. Reads
results/icfit.npz and data/icecube_prd108_fig26_contour90.json.
Run from the repository root: python -m figures.fig_deepcore_reproduction
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from nmosyn import style
from nmosyn.paths import DATA, RESULTS


def main() -> None:
    style.use()
    fit = np.load(RESULTS / 'icfit.npz')
    published = json.loads((DATA / 'icecube_prd108_fig26_contour90.json').read_text())
    contour = np.array(published['contour'])                        # (sin^2 theta23, Dm32 [1e-3 eV^2])
    line = fit['contour']
    best = dict(zip(fit['names'], fit['data_no_values']))
    color = style.COLORS['dc_all']

    fig, ax = plt.subplots(figsize=(7.38, 6.0))
    ax.plot(line[:, 0], line[:, 1] * 1e3, '-', color=color, lw=2.2, label=r'This work, public MC + data (90\% CL)')
    ax.plot(contour[:, 0], contour[:, 1], 'k--', lw=1.6,
            label=r'IceCube DeepCore, published (90\% CL)')
    ax.plot(best['s23sq'], (best['dm31'] - float(fit['dm21_release'])) * 1e3, 'o', color=color, mec='k', ms=8,
            label='This work, best fit')
    ax.plot(published['best_fit']['sin2_theta23'], published['best_fit']['dm32'], 'k*', ms=13,
            label='IceCube best fit')
    ax.set(xlabel=r'$\sin^2\theta_{23}$', ylabel=r'$\Delta m^2_{32}\ [10^{-3}\,\mathrm{eV}^2]$',
           xlim=(0.35, 0.68), ylim=(fit['grid_dm32'][0] * 1e3, 2.75))
    ax.legend(loc='upper left')
    fig.tight_layout()
    print(style.save(fig, 'fig_deepcore_reproduction', style.W_NARROW))


if __name__ == '__main__':
    main()
