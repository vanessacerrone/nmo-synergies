"""fig_deepcore_vs_orca6: the intercept b of IceCube DeepCore and KM3NeT/ORCA6 against the true splitting (Sec. 4, App.
A.5).

Weighted bin average with the NuFIT 6.0 truth; the inverted-ordering fit at its global minimum over Dm31_IO, theta23,
and delta_CP^IO. Reads results/orca.npz. Run from the repository root: python -m figures.fig_deepcore_vs_orca6
"""
import matplotlib.pyplot as plt
import numpy as np

from nmosyn import style
from nmosyn.paths import RESULTS

XL = r'$\Delta m^2_{31,\mathrm{NO}}\ [10^{-3}\,\mathrm{eV}^2]$'
BL = r'$b=\Delta m^2_{31,\mathrm{IO}}+\Delta m^2_{31,\mathrm{NO}}\ [10^{-6}\,\mathrm{eV}^2]$'
POINT = dict(ls='none', ms=6.5, mec='k', mew=0.8)

# key, label, color, marker, line style, marker face
SERIES = (
    ('dc_all_b', 'IceCube DeepCore, all channels', style.COLORS['dc_all'], 's', '-', None),
    ('orca_all_b', 'KM3NeT/ORCA6, all classes', style.COLORS['orca'], 'o', '-', None),
    ('orca_tracks_b', 'KM3NeT/ORCA6, high-purity tracks', style.COLORS['orca'], '^', '--', 'white'),
)


def main() -> None:
    style.use()
    r = np.load(RESULTS / 'orca.npz')
    x = r['x'] * 1e3
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for key, label, color, marker, ls, face in SERIES:
        ax.plot(x, r[key], ls=ls, color=color, lw=2.2)
        ax.plot(x, r[key], marker=marker, color=color, mfc=face or color, label=label, **POINT)
    ax.set_ylim(100, 135)
    ax.set_xlabel(XL)
    ax.set_ylabel(BL)
    ax.legend(loc='upper left', title=r'Weighted bin average, NuFIT 6.0; IO fit at its global minimum')
    print(style.save(fig, 'fig_deepcore_vs_orca6', style.W_NARROW))


if __name__ == '__main__':
    main()
