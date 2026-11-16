"""fig_accelerator_nubar_share: shift of the accelerator intercept from the closed form of Eq. (3.4), against the nubar
share.

NOvA-like (810 km, 2 GeV) and T2K-like (295 km, 0.6 GeV) beams in matter, sigma_E/E = 10%,
delta_CP^IO = 285 deg, at the true delta_CP of NuFIT 6.0 and at zero. Not in the paper: App. C.2 quotes its values
for neutrinos alone and for equal shares. Reads results/vacuum.npz.
Run from the repository root: python -m figures.fig_accelerator_nubar_share
"""
import matplotlib.pyplot as plt
import numpy as np

from nmosyn import style
from nmosyn.paths import RESULTS

BEAMS = (('nova_like', 'NOvA-like, 810 km, 2 GeV', style.COLORS['nova']),
         ('t2k_like', 'T2K-like, 295 km, 0.6 GeV', style.COLORS['t2k']))
PHASES = (('nufit', '-', 'o', r', $\delta^{\rm NO}_{\rm CP}=177^\circ$'),
          ('zero', '--', 's', r', $\delta^{\rm NO}_{\rm CP}=0$'))


def main() -> None:
    plt.switch_backend('agg')     # Agg measures the text that sets the page box, the same on every platform
    style.use()
    v = np.load(RESULTS / 'vacuum.npz')
    share_pct = v['shares'] * 100
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for name, label, color in BEAMS:
        for tag, ls, marker, phase_label in PHASES:
            shift = v[f'{name}_{tag}'] - v[f'closed_{tag}']
            ax.plot(share_pct, shift, ls, color=color, lw=2.2, marker=marker, mec='k', mew=0.8, ms=6.5,
                    label=label + phase_label)
    ax.axhline(0.0, color='k', lw=1.0)
    ax.set(xlim=(share_pct[0], share_pct[-1]), xlabel=r'Antineutrino share of the events [\%]',
           ylabel=r'$b-b_{\rm closed}$ $[10^{-6}\,\mathrm{eV}^2]$')
    ax.legend(loc='lower right', title=r'Matter, $\sigma_E/E=10\%$, $\delta^{\rm IO}_{\rm CP}=285^\circ$')
    style.save(fig, 'fig_accelerator_nubar_share', style.W_NARROW)


if __name__ == '__main__':
    main()
