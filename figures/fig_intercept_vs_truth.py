"""fig_intercept_vs_truth: the intercept b at the NuFIT 6.0 splitting against the true delta_CP and the true theta23
(Sec. 5).

NOvA disappearance: the range of b of the valley rule of App. B.3, and the closed form of Eq. (5.1) over the
same valley. NOvA with appearance and IceCube DeepCore: b at the global minimum of the inverted-ordering fit.
Left: true sin^2 theta23 = 0.561; right: true delta_CP = 177 deg. Reads results/bands.npz.
Run from the repository root: python -m figures.fig_intercept_vs_truth
"""
from collections.abc import Callable, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from nmosyn import style
from nmosyn.paths import RESULTS

BL = r'$b=\Delta m^2_{31,\mathrm{IO}}+\Delta m^2_{31,\mathrm{NO}}\ [10^{-6}\,\mathrm{eV}^2]$'
IX = 1                      # index of the NuFIT splitting in results/bands.npz
B_LIM = (79.3, 137.0)       # y range [1e-6 eV^2] of the paper figure; every curve lies inside
LINES = (('nova_all', style.COLORS['nova_app']), ('dcn', style.COLORS['dc_numu']),
         ('dca', style.COLORS['dc_all']))


def wrap(dcp: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """True delta_CP on [0, 2 pi] (last point = first) to delta_CP / pi on [-1, 1], periodic."""
    x = (dcp[:-1] / np.pi + 1) % 2 - 1
    order = np.argsort(x)
    v = values[:-1][order]
    return np.append(x[order], 1.0), np.append(v, v[:1], axis=0)


def panel(ax: Axes, bands: Mapping[str, np.ndarray], scan: str, x: np.ndarray,
          to_axis: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]) -> None:
    """Every curve of one panel; to_axis maps (x, values) to the plotted (x, values)."""
    xs, rng = to_axis(x, bands[f'nova_dis/{scan}/range_paper/{IX}'])
    ax.fill_between(xs, rng[:, 0], rng[:, 1], facecolor=style.opaque(style.COLORS['nova'], 0.20), lw=0)
    for edge in rng.T:
        ax.plot(xs, edge, '-', color=style.COLORS['nova'], lw=2.0)
    for sample, color in LINES:
        ax.plot(*to_axis(x, bands[f'{sample}/{scan}/b/{IX}']), '-', color=color, lw=2.4)
    xs, cf = to_axis(x, bands[f'nova_dis/{scan}/cf/{IX}'])
    for edge in cf.T:
        ax.plot(xs, edge, ':', color='k', lw=1.6)


def main() -> None:
    style.use()
    bands = np.load(RESULTS / 'bands.npz')
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.85, 6.6), sharey=True)
    fig.subplots_adjust(wspace=0.05, bottom=0.25)

    panel(left, bands, 'scan_dcp', bands['dt'], wrap)
    left.set_title(r'True $\sin^2\theta_{23}=0.561$')
    left.set_xlim(-1, 1)
    left.set_xticks([-1, -0.5, 0, 0.5, 1], [r'$-\pi$', r'$-\pi/2$', '$0$', r'$\pi/2$', r'$\pi$'])
    left.set_xlabel(r'True $\delta_{\rm CP}$')
    left.set_ylabel(BL)
    left.set_ylim(*B_LIM)

    panel(right, bands, 'scan_s23', bands['st'], lambda x, v: (x, v))
    right.set_title(r'True $\delta_{\rm CP}=177^\circ$')
    right.set_xlim(0.43, 0.596)
    right.set_xticks([0.44, 0.48, 0.52, 0.56], ['0.44', '0.48', '0.52', '0.56'])
    right.set_xlabel(r'True $\sin^2\theta_{23}$')

    shade = style.opaque(style.COLORS['nova'], 0.20)
    nova_band = (Patch(facecolor=shade, edgecolor=shade, lw=0),
                 Line2D([], [], color=style.COLORS['nova'], lw=2.0))
    handles = [nova_band, Line2D([], [], color=style.COLORS['nova_app'], lw=2.4),
               Line2D([], [], color='k', ls=':', lw=1.6),
               Line2D([], [], color=style.COLORS['dc_numu'], lw=2.4),
               Line2D([], [], color=style.COLORS['dc_all'], lw=2.4)]
    labels = [r'NOvA, $\nu_\mu$ disappearance', r'NOvA, $\nu_\mu$ disappearance + $\nu_e$ appearance',
              r'Closed form, accelerator disappearance (vacuum)',
              r'IceCube DeepCore, $\nu_\mu$ disappearance', 'IceCube DeepCore, all channels']
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    print(style.save(fig, 'fig_intercept_vs_truth', style.W_WIDE))


if __name__ == '__main__':
    main()
