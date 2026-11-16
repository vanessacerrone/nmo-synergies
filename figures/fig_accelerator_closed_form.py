"""fig_accelerator_closed_form and fig_juno_intercept_vs_resolution: the intercept b of an accelerator (left) and of
JUNO (right) near the vacuum limit (Sec. 3).

Left, fig_accelerator_closed_form: generic 810 km beam, b against the true delta_CP at delta_CP^IO = 285 deg, the closed
form of Eq. (3.4), and the range of b over delta_CP^IO. Right, fig_juno_intercept_vs_resolution: JUNO, b against the
stochastic term of the energy resolution, the short-baseline value of Eq. (3.7), and Eq. (3.8) with the solar phase at 4
MeV. Reads results/vacuum.npz. Run from the repository root: python -m figures.fig_accelerator_closed_form
"""
from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np

from nmosyn import style
from nmosyn.params import DCP_IO, b_lbl, b_sbr
from nmosyn.paths import RESULTS

BL = r'$b=\Delta m^2_{31,\mathrm{IO}}+\Delta m^2_{31,\mathrm{NO}}\ [10^{-6}\,\mathrm{eV}^2]$'
POINT = dict(ls='none', marker='o', ms=6.5, mec='k', mew=0.8)
CANVAS = (5.5, 5.0)
PAIR_HEIGHT = 4.55      # common page height of the two panels [in], so that they align at the top


def accelerator(v: Mapping[str, np.ndarray]) -> None:
    """Left panel, fig_accelerator_closed_form: points every 15 deg, band over delta_CP^IO between the 30-deg nodes."""
    fig, ax = plt.subplots(figsize=CANVAS)
    color = style.COLORS['nova']
    grid_deg = np.degrees(v['dcp_grid'])
    ax.fill_between(grid_deg, v['beam_b_grid'].min(axis=1), v['beam_b_grid'].max(axis=1),
                    color=style.opaque(color, 0.25), lw=0, label=r'all values of $\delta^{\rm IO}_{\rm CP}$')
    closed_deg = np.arange(-180.0, 180.1, 1.0)
    ax.plot(closed_deg, b_lbl(np.radians(closed_deg), DCP_IO) * 1e6, '-', color='k', lw=1.6,
            label=r'Closed form, $(\kappa_\mu^{\rm NO}+\kappa_\mu^{\rm IO})\Delta m^2_{21}$')
    ax.plot(np.degrees(v['dcp_line'])[::3], v['beam_b_line'][::3], color=color, **POINT,
            label='Weighted bin average, matter,' + '\n' + r'equal $\nu$ and $\bar\nu$ event samples')
    ax.set_xticks([-180, -90, 0, 90, 180], [r'$-\pi$', r'$-\pi/2$', '$0$', r'$\pi/2$', r'$\pi$'])
    ax.set_xlim(-180, 180)
    ax.set_ylim(70, 175)
    ax.set_xlabel(r'True $\delta_{\rm CP}$')
    ax.set_ylabel(BL)
    ax.legend(loc='upper center',
              title=r'Accelerator, 810 km, $\sigma_E/E=10\%$; $\delta^{\rm IO}_{\rm CP}=285^\circ$')
    style.save(fig, 'fig_accelerator_closed_form', style.W_PAIR, PAIR_HEIGHT)


def juno(v: Mapping[str, np.ndarray]) -> None:
    """Right panel, fig_juno_intercept_vs_resolution: line on every a_res, points every 1%."""
    fig, ax = plt.subplots(figsize=CANVAS)
    color = style.COLORS['juno']
    sbr = b_sbr() * 1e6
    ax.axhline(sbr, color=style.COLORS['sbr'], lw=2.2,
               label='Short baseline, $2s^2_{12}\\Delta m^2_{21}$\n(any resolution)')
    ax.axhline(float(v['juno_b_closed_4mev']), color='k', ls='--', lw=1.6,
               label='Closed form, solar phase at 4 MeV')
    a_pct = v['a_res'] * 100
    ax.plot(a_pct, v['juno_b'], '-', color=color, lw=2.2)
    every_pct = np.isclose(a_pct, np.round(a_pct))    # whole percents; the scan also holds the baseline 2.61%
    ax.plot(a_pct[every_pct], v['juno_b'][every_pct], color=color, **POINT,
            label='Weighted bin average,\nJUNO (eight cores, matter)')
    baseline = np.isclose(a_pct, 2.61)
    ax.plot(a_pct[baseline], v['juno_b'][baseline], color='w', **{**POINT, 'marker': 'D', 'ms': 7.5},
            label=r'JUNO resolution, $a_{\rm res}=2.61\%$')
    ax.set_xlim(a_pct[0], a_pct[-1])
    ax.set_ylim(0, sbr + 5)
    ax.set_xlabel(r'Stochastic term $a_{\rm res}$ of the energy resolution [\%]')
    ax.set_ylabel(BL)
    ax.legend(loc='lower right')
    style.save(fig, 'fig_juno_intercept_vs_resolution', style.W_PAIR, PAIR_HEIGHT)


def main() -> None:
    plt.switch_backend('agg')     # Agg measures the text that sets the page box, the same on every platform
    style.use()
    v = np.load(RESULTS / 'vacuum.npz')
    accelerator(v)
    juno(v)


if __name__ == '__main__':
    main()
