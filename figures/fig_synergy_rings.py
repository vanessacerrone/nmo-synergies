"""fig_synergy_rings: the synergy term of JUNO with IceCube DeepCore and with NOvA, both with appearance, for three
precisions of each experiment (Eq. 1.1, Sec. 6).

Each ring spans the smallest (inner boundary) and the largest (outer boundary) synergy term over the true points (sin^2
theta23, delta_CP) of the bands of Sec. 5, at the NuFIT splitting. As in fig_synergy_vs_precision, the joint fit takes
the point of the partner's range of b closest to JUNO (nmosyn.synergy.gap). The area of a ring scales with the synergy
term. Reads results/bands.npz and results/vacuum.npz. Run from the repository root: python -m figures.fig_synergy_rings
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from nmosyn import style, synergy
from nmosyn.params import X0, intercept
from nmosyn.paths import RESULTS

JUNO_PCT = (0.5, 0.8, 1.0)          # sigma_JUNO / |Dm31| [%], rows
PARTNER_PCT = (1.0, 1.5, 2.0)       # sigma / |Dm31| of DeepCore or NOvA [%], columns
RING_PT = 8.47                      # radius of a ring [pt] per unit of sqrt(Delta chi2_syn), as in the paper figure
HATCH = '////'
RING_LABEL = r'Ring: range of $\Delta\chi^2_{\rm syn}$ over unknown $\delta_{\rm CP}$ and $\theta_{23}$'

# per panel: (sample, color, hatched, title, xlabel)
PANELS = (
    ('dca', style.COLORS['dc_all'], False, r'JUNO + IceCube DeepCore (all channels)',
     r'DeepCore precision $\sigma_{\rm ATM} / |\Delta m^2_{31}|$ [\%]'),
    ('nova_all', style.COLORS['nova'], True, r'JUNO + NOvA ($\nu_\mu$ disappearance + $\nu_e$ appearance)',
     r'NOvA precision $\sigma_{\rm LBL} / |\Delta m^2_{31}|$ [\%]'),
)


def ranges_at_x0() -> dict[str, np.ndarray]:
    """(b_lo, b_hi) per true point at X0 under the rule of App. B.3, as in fig_synergy_vs_precision; JUNO: its single
    b."""
    bands = np.load(RESULTS / 'bands.npz')
    ix = int(np.argmin(np.abs(bands['x'] - X0)))
    out = {s: bands[f'{s}/grid/range_paper/{ix}'] for s in ('nova_all', 'dca')}
    v = np.load(RESULTS / 'vacuum.npz')
    b_juno = np.interp(X0, v['x'], intercept(v['juno_y'], v['x']))
    out['juno'] = np.array([b_juno, b_juno])
    return out


def rings(ax: Axes, partner: np.ndarray, juno: np.ndarray, color: str, hatched: bool) -> None:
    """One ring per pair of precisions: smallest and largest synergy term of the partner with JUNO."""
    for juno_pct in JUNO_PCT:
        for i, partner_pct in enumerate(PARTNER_PCT):
            c_min, c_max = synergy.synergy_range(partner, juno, partner_pct / 100 * X0 * 1e6,
                                                 juno_pct / 100 * X0 * 1e6)
            r_out, r_in = np.sqrt(c_max) * RING_PT, np.sqrt(c_min) * RING_PT
            ax.scatter(i, juno_pct, s=(2 * r_out) ** 2, facecolors=[style.opaque(color, 0.25)],
                       edgecolors=[style.opaque(color, 0.4375)], linewidths=2.0, zorder=3)
            if hatched:
                ax.scatter(i, juno_pct, s=(2 * r_out) ** 2, facecolors='none', edgecolors=color, hatch=HATCH,
                           linewidths=0, zorder=3)
            ax.scatter(i, juno_pct, s=(2 * r_in) ** 2, facecolors='white', edgecolors=color,
                       linewidths=2.0, ls='--', zorder=4)
            ax.annotate(f'$[{c_min:.1f},\\,{c_max:.1f}]$', xy=(i, juno_pct), xytext=(0, -r_out - 6),
                        textcoords='offset points', ha='center', va='top', fontsize=12, color='0.15',
                        zorder=7)


def main() -> None:
    style.use()
    ranges = ranges_at_x0()
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 5.6), sharey=True)
    fig.subplots_adjust(left=0.075345, right=0.982273, bottom=0.135659, top=0.90, wspace=0.08)
    for ax, (sample, color, hatched, title, xlabel) in zip(axs, PANELS):
        rings(ax, ranges[sample], ranges['juno'], color, hatched)
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(0.355, 1.1375)       # rows at their values, the margins of the equally spaced grid
        ax.set_xticks([0, 1, 2], [f'{p:g}' for p in PARTNER_PCT])
        ax.set_yticks(JUNO_PCT, [f'{p:g}' for p in JUNO_PCT])
        ax.set_xlabel(xlabel, labelpad=7)
        ax.set_title(title, pad=9)
    axs[0].set_ylabel(r'JUNO precision $\sigma_{\rm JUNO} / |\Delta m^2_{31}|$ [\%]', labelpad=7)
    handles = [
        Patch(facecolor=style.opaque('0.6', 0.3), edgecolor=style.opaque('0.4', 0.51), label=RING_LABEL),
        Line2D([0], [0], ls='--', color='0.3', lw=1.5, label=r'Inner: smallest $\Delta\chi^2_{\rm syn}$'),
        Line2D([0], [0], ls='-', color='0.3', lw=1.5, label=r'Outer: largest $\Delta\chi^2_{\rm syn}$'),
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.99), ncol=3, columnspacing=1.5)
    print(style.save(fig, 'fig_synergy_rings', style.W_WIDE))


if __name__ == '__main__':
    main()
