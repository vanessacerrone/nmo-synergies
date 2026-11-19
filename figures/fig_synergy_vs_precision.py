"""fig_synergy_vs_precision: the synergy term of two experiments against their common precision (Eq. 1.1, Sec. 6).

At the NuFIT splitting, for every true point (sin^2 theta23, delta_CP) of the bands of Sec. 5, the joint
inverted-ordering fit of two experiments takes the closest points of their ranges of b (nmosyn.synergy.gap);
each band spans the smallest and the largest synergy term over the true points. JUNO has one value of b, from
its line at 2.94% at 1 MeV (a_res = 2.61%). Reads results/bands.npz and results/vacuum.npz.
Run from the repository root: python -m figures.fig_synergy_vs_precision
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from nmosyn import style, synergy
from nmosyn.params import X0, intercept
from nmosyn.paths import RESULTS

SIGMA_PCT = np.linspace(0.5, 2.5, 300)     # common precision sigma / |Dm31| of each experiment [%]
BANDS_TITLE = r'Bands: $\delta_{\rm CP}$ and $\theta_{23}$ unknown'
HATCH = '////'

# per panel: (title, pairs); pair: (sample A, sample B, label, color, alpha, hatched), drawn in this order
PANELS = (
    ('Disappearance only', (
        ('juno', 'nova_dis', r'JUNO + NOvA ($\nu_\mu$ dis.)', style.COLORS['nova'], 0.25, False),
        ('juno', 'dcn', r'JUNO + DeepCore ($\nu_\mu$ dis.)', style.COLORS['dc_numu'], 0.25, False),
        ('nova_dis', 'dcn', r'NOvA (dis.) + DeepCore ($\nu_\mu$ dis.)', 'k', 0.15, False),
    )),
    ('Disappearance + appearance', (
        ('juno', 'nova_all', r'JUNO + NOvA (dis. + $\nu_e$ app.)', style.COLORS['nova'], 0.25, True),
        ('juno', 'dca', 'JUNO + DeepCore (all channels)', style.COLORS['dc_all'], 0.45, False),
        ('nova_all', 'dca', 'NOvA (dis. + app.) + DeepCore (all channels)', 'k', 0.15, False),
    )),
)


def ranges_at_x0() -> dict[str, np.ndarray]:
    """(b_lo, b_hi) per true point at X0 under the rule of App. B.3; JUNO: its single b, from its line."""
    bands = np.load(RESULTS / 'bands.npz')
    ix = int(np.argmin(np.abs(bands['x'] - X0)))
    out = {s: bands[f'{s}/grid/range_paper/{ix}'] for s in ('nova_dis', 'nova_all', 'dcn', 'dca')}
    v = np.load(RESULTS / 'vacuum.npz')
    b_juno = np.interp(X0, v['x'], intercept(v['juno_y'], v['x']))
    out['juno'] = np.array([b_juno, b_juno])
    return out


def main() -> None:
    style.use()
    ranges = ranges_at_x0()
    sigma = SIGMA_PCT / 100 * X0 * 1e6
    # the hatch line width is read when the hatched band and the legend are built
    with plt.rc_context({'hatch.linewidth': 0.5}):
        fig, axs = plt.subplots(1, 2, figsize=(12.7, 4.9), sharey=True)
        fig.subplots_adjust(wspace=0.15)
        for ax, (title, pairs) in zip(axs, PANELS):
            specs = []
            for a, b, label, color, alpha, hatched in pairs:
                lo, hi = synergy.synergy_range(ranges[a], ranges[b], sigma, sigma)
                specs.append(dict(lo=lo, hi=hi, color=color, shade=alpha, lw=1.6,
                                  hatch=HATCH if hatched else None))
            handles = style.bands(ax, SIGMA_PCT, specs)
            ax.legend(handles, [pair[2] for pair in pairs], loc='upper center', bbox_to_anchor=(0.5, -0.15),
                      title=BANDS_TITLE)
            ax.set_title(title)
            ax.set_xlim(0.5, 2.5)
            ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
            ax.set_xlabel(r'Common precision $\sigma/|\Delta m^2_{31}|$ of each experiment [\%]')
        axs[0].set_ylim(0, 20)
        axs[0].set_yticks([0, 5, 10, 15, 20])
        axs[0].get_yticklabels()[0].set_verticalalignment('bottom')     # the 0 clear of the 0.5 below it
        axs[0].set_ylabel(r'Synergy term $\Delta\chi^2_{\rm syn}$')
        print(style.save(fig, 'fig_synergy_vs_precision', style.W_WIDE))


if __name__ == '__main__':
    main()
