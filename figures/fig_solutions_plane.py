"""fig_solutions_plane: wrong-ordering solutions in the (Dm31_NO, Dm31_IO) plane, delta_CP and theta23 unknown (Sec. 5).

Lines: the short-baseline reactor (Eq. 3.7) and JUNO at a_res = 2.61%. Bands: NOvA, T2K, and IceCube DeepCore, the range
of b over the true points of the NuFIT 6.0 3 sigma ranges and over the valley rule of App. B.3, at three true
splittings; the band edges are linear in Dm31_NO between them. Reads results/vacuum.npz and results/bands.npz. Run from
the repository root: python -m figures.fig_solutions_plane
"""
from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np

from nmosyn import style
from nmosyn.params import b_sbr
from nmosyn.paths import RESULTS

XL = r'$\Delta m^2_{31,\mathrm{NO}}\ [10^{-3}\,\mathrm{eV}^2]$'
YL = r'$\Delta m^2_{31,\mathrm{IO}}\ [10^{-3}\,\mathrm{eV}^2]$'
XZ = np.linspace(2.425e-3, 2.625e-3, 41)
LEGEND_ROOM = 0.10      # the y axis extends this much lower [1e-3 eV^2], to make room for the legends

# title, accelerator channel and its label, DeepCore sample, color, and label
PANELS = (
    ('Disappearance only', 'dis', r'$\nu_\mu$ disappearance',
     'dcn', style.COLORS['dc_numu'], r'IceCube DeepCore, $\nu_\mu$ disappearance'),
    ('Disappearance + appearance', 'all', r'$\nu_\mu$ disappearance + $\nu_e$ appearance',
     'dca', style.COLORS['dc_all'], 'IceCube DeepCore, all channels'),
)


def band(bands: Mapping[str, np.ndarray], sample: str) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper edge in Dm31_IO [1e-3 eV^2] on XZ of the range of b over all true points."""
    ranges = np.array([bands[f'{sample}/grid/range_paper/{ix}'] for ix in range(len(bands['x']))])
    edges = ranges[..., 0].min(axis=1), ranges[..., 1].max(axis=1)
    return tuple((np.interp(XZ, bands['x'], b) * 1e-6 - XZ) * 1e3 for b in edges)


def main() -> None:
    style.use()
    bands = np.load(RESULTS / 'bands.npz')
    vac = np.load(RESULTS / 'vacuum.npz')
    x = XZ * 1e3
    fig, axs = plt.subplots(1, 2, figsize=(12.55, 5.9), sharey=True)
    fig.subplots_adjust(wspace=0.05)
    for ax, (title, channel, channel_label, dc, dc_color, dc_label) in zip(axs, PANELS):
        ax.plot(x, (b_sbr() - XZ) * 1e3, '-', color=style.COLORS['sbr'], lw=2.2,
                label='Short-baseline reactor (Daya Bay)')
        ax.plot(x, np.interp(XZ, vac['x'], vac['juno_y']) * 1e3, '-', color=style.COLORS['juno'], lw=2.2,
                label=r'JUNO ($a_{\rm res}=2.61\%$)')
        nova_lo, nova_hi = band(bands, f'nova_{channel}')
        dc_lo, dc_hi = band(bands, dc)
        style.bands(ax, x, [dict(lo=nova_lo, hi=nova_hi, color=style.COLORS['nova'], shade=0.35,
                                 label=f'NOvA, {channel_label}'),
                            dict(lo=dc_lo, hi=dc_hi, color=dc_color, shade=0.15)])
        lo, hi = band(bands, f't2k_{channel}')
        ax.plot(x, lo, '--', color=style.COLORS['t2k'], lw=1.8, zorder=4, label=f'T2K, {channel_label}')
        ax.plot(x, hi, '--', color=style.COLORS['t2k'], lw=1.8, zorder=4)
        ax.plot(x, dc_lo, '-', color=dc_color, lw=2.0, zorder=5, label=dc_label)
        ax.plot(x, dc_hi, '-', color=dc_color, lw=2.0, zorder=5)
        ax.set_title(title)
        ax.set_xlim(2.425, 2.625)
        ax.set_xticks([2.45, 2.50, 2.55, 2.60], ['2.45', '2.50', '2.55', '2.60'])
        ax.set_xlabel(XL)
    axs[0].set_ylabel(YL)
    lo, hi = axs[0].get_ylim()
    axs[0].set_ylim(lo - LEGEND_ROOM, hi)
    for ax in axs:
        ax.legend(loc='lower left', handlelength=1.8,
                  title=r'Bands: $\delta_{\rm CP}$ and $\theta_{23}$ unknown')
    print(style.save(fig, 'fig_solutions_plane', style.W_WIDE))


if __name__ == '__main__':
    main()
