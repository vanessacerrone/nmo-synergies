"""fig_deepcore_resolution and fig_deepcore_energy_window: the intercept b of IceCube DeepCore against the energy
resolution and the energy window (Sec. 4).

Acceptance of the public release (App. A.4), nu_mu CC, theta23 and the phases fixed at the NuFIT values, true Dm31_NO =
2.534e-3 eV^2, the tabulated response replaced by Gaussian smearing in ln E (widths of the release with the variance of
the analysis bins removed, App. A.4). Left, fig_deepcore_resolution: width sigma_lnE of the resolution, with
reconstructed bins one sigma wide; the tabulated response for reference. Right, fig_deepcore_energy_window: upper or
lower edge of the energy window varied, with the resolution of the release. Reads results/deepcore.npz. Run from the
repository root: python -m figures.fig_deepcore_resolution
"""
from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from nmosyn import style
from nmosyn.params import X0, intercept
from nmosyn.paths import RESULTS

BL = r'$b=\Delta m^2_{31,\mathrm{IO}}+\Delta m^2_{31,\mathrm{NO}}\ [10^{-6}\,\mathrm{eV}^2]$'
POINT = dict(mec='k', mew=0.8, ms=6.5)
CANVAS = (5.5, 4.2)
PAIR_HEIGHT = 5.05      # common page height of the two panels [in], so that they align at the top


def window(d: Mapping[str, np.ndarray]) -> tuple[Figure, Axes]:
    """Panel fig_deepcore_energy_window: b against the varied edge of the energy window."""
    fig, ax = plt.subplots(figsize=CANVAS)
    color = style.COLORS['dc_numu']
    e_max, b_max = d['e_max'], d['fig4_b_e_max']
    e_min, b_min = d['e_min'], d['fig4_b_e_min']
    ax.plot(e_max, b_max, '-', color=color, lw=2.2)
    ax.plot(e_max[::2], b_max[::2], 'o', color=color, **POINT)
    ax.plot(e_min, b_min, '--', color=color, lw=2.0)
    ax.plot(e_min[::2], b_min[::2], 's', color=color, **POINT)
    ax.axhline(float(d['fig4_b_release_resolution']), color='k', ls=':', lw=1.6)
    ax.set_xscale('log')
    ax.set_xticks([7, 10, 20, 50, 100, 158], ['7', '10', '20', '50', '100', '158'])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(6.0, e_max[-1])         # room for the first point, at 6.31 GeV, inside the frame
    ax.set_ylim(top=150)
    ax.set_xlabel(r'Varied edge of the energy window [GeV]')
    ax.set_ylabel(BL)
    ax.text(30, 136, 'upper edge', color=color, fontsize=13)
    ax.text(28, 112.6, 'lower edge', color=color, fontsize=13)
    # legend handles show line and marker together, as drawn
    handles = [Line2D([], [], ls='-', lw=2.2, color=color, marker='o', **POINT),
               Line2D([], [], ls='--', lw=2.0, color=color, marker='s', **POINT),
               Line2D([], [], ls=':', lw=1.6, color='k')]
    labels = ['Upper edge varied, lower edge at 6.31 GeV', 'Lower edge varied, upper edge at 158.5 GeV',
              'Full window 6.31--158.5 GeV']
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.01), handlelength=2.6,
              title='DeepCore resolution (public simulation)')
    return fig, ax


def resolution(d: Mapping[str, np.ndarray], b_tabulated: float, top: float) -> Figure:
    """Panel fig_deepcore_resolution: b against the Gaussian resolution; `top` [in] is the content above the axes of
    fig_deepcore_energy_window, reserved here too so that the two axes print at the same height."""
    fig, ax = plt.subplots(figsize=CANVAS)
    color = style.COLORS['dc_numu']
    sigma, b = d['sigma'], d['fig4_b_sigma']
    ax.plot(sigma, b, '-', color=color, lw=2.2)
    ax.plot(sigma[::2], b[::2], 'o', color=color, **POINT,
            label=r'Gaussian resolution, bins one $\sigma$ wide')
    ax.axhline(b_tabulated, color='k', ls='--', lw=1.6, label='DeepCore, tabulated response')
    ax.set_xlim(sigma[0], sigma[-1])
    ax.set_xlabel(r'Energy resolution $\sigma_{\ln E}$')
    ax.set_ylabel(BL)
    ax.legend(loc='lower center', title=r'DeepCore acceptance, window 6.31--158.5 GeV')
    y0 = ax.get_position().y1
    fig.add_artist(Line2D([0.5, 0.5], [y0, y0 + top / fig.get_figheight()], transform=fig.transFigure,
                          ls='none'))
    return fig


def main() -> None:
    style.use()
    d = np.load(RESULTS / 'deepcore.npz')
    b_tabulated = float(intercept(d['fig1_y'], X0))

    fig_win, ax_win = window(d)
    fig_win.draw_without_rendering()
    inches = fig_win.dpi_scale_trans.inverted()
    top = fig_win.get_tightbbox().y1 - ax_win.get_window_extent().transformed(inches).y1
    fig_res = resolution(d, b_tabulated, top)

    for fig, name in ((fig_res, 'fig_deepcore_resolution'), (fig_win, 'fig_deepcore_energy_window')):
        out = style.save(fig, name, style.W_PAIR, PAIR_HEIGHT)
        print(out)
    i = int(np.argmax(d['fig4_b_sigma']))
    b_release = float(d['fig4_b_release_resolution'])
    print(f'tabulated response {b_tabulated:.3f}, resolution of the release {b_release:.3f}, '
          f'Gaussian peak {d["fig4_b_sigma"][i]:.3f} at sigma = {d["sigma"][i]:.3f}; '
          f'content above the axes {top:.4f} in')


if __name__ == '__main__':
    main()
