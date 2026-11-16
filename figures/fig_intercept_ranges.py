"""fig_intercept_ranges: range of the intercept b at the NuFIT 6.0 splitting, delta_CP, theta23, or both unknown (Sec.
5).

For every sample three bars: the true delta_CP over its 3 sigma range at true sin^2 theta23 = 0.56, the true theta23
over its 3 sigma range at true delta_CP = 177 deg, and both over their ranges (as fig_solutions_plane). Each true point
contributes the range of b of the valley rule of App. B.3. The printed number is the width of the bar. Reads
results/bands.npz. Run from the repository root: python -m figures.fig_intercept_ranges
"""
from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from nmosyn import style
from nmosyn.paths import RESULTS

S23SQ_DCP_BAR = 0.56        # true sin^2 theta23 of the delta_CP bar: the true grid value closest to NuFIT
IX = 1                      # index of the NuFIT splitting in results/bands.npz
ROW_STEP = 1.3
BAR_STEP = 0.32
BAR_HEIGHT = 0.26
XLABEL = (r'$b=\Delta m^2_{31,\mathrm{IO}}+\Delta m^2_{31,\mathrm{NO}}\ [10^{-6}\,\mathrm{eV}^2]$ at '
          r'$\Delta m^2_{31,\mathrm{NO}}=2.534\times10^{-3}\,\mathrm{eV}^2$')

# rows from the bottom: sample, label, color
ROWS = (
    ('dca', 'IceCube DeepCore, all channels', style.COLORS['dc_all']),
    ('dcn', r'IceCube DeepCore, $\nu_\mu$ disappearance', style.COLORS['dc_numu']),
    ('t2k_all', r'T2K, $\nu_\mu$ disappearance + $\nu_e$ appearance', style.COLORS['t2k']),
    ('t2k_dis', r'T2K, $\nu_\mu$ disappearance', style.COLORS['t2k']),
    ('nova_all', r'NOvA, $\nu_\mu$ disappearance + $\nu_e$ appearance', style.COLORS['nova']),
    ('nova_dis', r'NOvA, $\nu_\mu$ disappearance', style.COLORS['nova']),
)


def bars(bands: Mapping[str, np.ndarray], sample: str) -> list[tuple[float, float]]:
    """(b_lo, b_hi) with delta_CP unknown, theta23 unknown, and both unknown."""
    grid = bands[f'{sample}/grid/range_paper/{IX}']
    t23 = bands[f'{sample}/t23/range_paper/{IX}']
    s0 = bands['s0']
    dcp = grid.reshape(s0.size, -1, 2)[np.argmin(np.abs(s0 - S23SQ_DCP_BAR))]
    return [(r[:, 0].min(), r[:, 1].max()) for r in (dcp, t23, grid)]


def main() -> None:
    style.use()
    bands = np.load(RESULTS / 'bands.npz')
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    for row, (sample, _, color) in enumerate(ROWS):
        kinds = (dict(facecolor=style.opaque(color, 0.30), edgecolor=style.opaque(color, 0.51)),
                 dict(facecolor='none', edgecolor=color, hatch='xx'),
                 dict(facecolor=style.opaque(color, 0.70), edgecolor=style.opaque(color, 0.91)))
        for k, ((lo, hi), kind) in enumerate(zip(bars(bands, sample), kinds)):
            y = ROW_STEP * row + (1 - k) * BAR_STEP
            ax.barh(y, hi - lo, left=lo, height=BAR_HEIGHT, lw=1.2, **kind)
            ax.text(hi + 0.8, y, f'{hi - lo:.0f}', va='center', fontsize=12, color='0.25')
    ax.set_yticks(ROW_STEP * np.arange(len(ROWS)), [label for _, label, _ in ROWS])
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(74, 150)
    ax.set_xlabel(XLABEL)
    light, dark = style.opaque('0.5', 0.3), style.opaque('0.5', 0.7)
    proxies = [Patch(facecolor=light, edgecolor=light), Patch(facecolor='none', edgecolor='0.4', hatch='xx'),
               Patch(facecolor=dark, edgecolor=dark)]
    labels = [r'$\delta_{\rm CP}$ unknown', r'$\theta_{23}$ unknown', 'Both unknown']
    ax.legend(proxies, labels, loc='lower right', bbox_to_anchor=(1.0, 1.01), ncol=3, frameon=False,
              columnspacing=1.2)
    print(style.save(fig, 'fig_intercept_ranges', style.W_WIDE))


if __name__ == '__main__':
    main()
