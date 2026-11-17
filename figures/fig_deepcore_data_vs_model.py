"""fig_deepcore_data_vs_model: the observed counts of the IceCube DeepCore release against the expected counts at
the best fits of the data (not in the paper; the fits are those of App. E.1 and E.4).

Three hypotheses: the normal ordering, the inverted ordering with the hypersurfaces read at |Dm31,IO| + 2 c12^2 Dm21
(the reading of the paper), and the inverted ordering with the hypersurfaces read at |Dm31,IO|. The pull of a bin is
(n - N) / sqrt(N + sigma^2_MC), the term of the modified Pearson chi^2; chi^2_bins is its sum over the 200 bins,
without the prior terms. The panels show the counts of each event class summed over cos theta_z and over the
energy, with the pulls of these sums. Reads results/icfit.npz.
Run from the repository root: python -m figures.fig_deepcore_data_vs_model
"""
import matplotlib.pyplot as plt
import numpy as np

from nmosyn import style
from nmosyn.experiments.icfit import CLASS_EDGES, COSZ_EDGES, E_EDGES
from nmosyn.paths import RESULTS

SHAPE = (E_EDGES.size - 1, COSZ_EDGES.size - 1, CLASS_EDGES.size - 1)     # the flattening order of icfit
HYPOTHESES = (('io', r'IO, hypersurfaces at $|\Delta m^2_{31,\mathrm{IO}}|+2c_{12}^2\Delta m^2_{21}$',
               style.COLORS['magenta'], '-', 2.6),
              ('io_splitting', r'IO, hypersurfaces at $|\Delta m^2_{31,\mathrm{IO}}|$', style.COLORS['orange'], '--',
               1.8),
              ('no', 'NO', style.COLORS['blue'], '-', 1.2))      # drawn in this order, NO on top
CLASSES = (r'$0.55\le\mathrm{PID}<0.75$', r'$\mathrm{PID}\ge0.75$')
E_LABEL = r'$E_\mathrm{reco}$ [GeV]'
COSZ_LABEL = r'$\cos\theta_{z,\mathrm{reco}}$'
PULL_LABEL = 'Pull'


def main() -> None:
    style.use()
    fit = np.load(RESULTS / 'icfit.npz')
    observed = fit['observed'].reshape(SHAPE)
    expected = {tag: fit[f'expected_{tag}'].reshape(SHAPE) for tag, *_ in HYPOTHESES}
    variance = {tag: fit[f'variance_{tag}'].reshape(SHAPE) for tag, *_ in HYPOTHESES}
    chi2 = {tag: float(np.sum((observed - expected[tag]) ** 2 / (expected[tag] + variance[tag])))
            for tag, *_ in HYPOTHESES}

    fig, axes = plt.subplots(5, 2, figsize=(11.0, 11.5), sharex='col',
                             gridspec_kw={'height_ratios': [3, 1.3, 0.5, 3, 1.3], 'hspace': 0.0, 'wspace': 0.22})
    for ax in axes[2]:
        ax.set_visible(False)                                   # space between the two classes
    for c, name in enumerate(CLASSES):
        for col, (edges, axis) in enumerate(((E_EDGES, 1), (COSZ_EDGES, 0))):
            ax, pull_ax = axes[3 * c, col], axes[3 * c + 1, col]
            n = observed[..., c].sum(axis=axis)
            centers = np.sqrt(edges[1:] * edges[:-1]) if col == 0 else 0.5 * (edges[1:] + edges[:-1])
            for tag, label, color, ls, lw in HYPOTHESES:
                big_n = expected[tag][..., c].sum(axis=axis)
                ax.stairs(big_n, edges, baseline=None, color=color, ls=ls, lw=lw,
                          label=f'{label}, $\\chi^2_\\mathrm{{bins}}={chi2[tag]:.1f}$')
                pull = (n - big_n) / np.sqrt(big_n + variance[tag][..., c].sum(axis=axis))
                pull_ax.stairs(pull, edges, baseline=None, color=color, ls=ls, lw=lw)
            ax.errorbar(centers, n, np.sqrt(n), fmt='o', color='k', ms=4.5, lw=1.2, label='Observed')
            ax.set_ylim(0, 1.3 * n.max())
            ax.text(0.04, 0.92, name, transform=ax.transAxes, va='top')
            pull_ax.axhline(0, color='0.5', lw=0.8)
            pull_ax.set_ylim(-3.5, 3.5)
            if col == 0:
                ax.set_xscale('log')
                ax.set_ylabel('Events')
                pull_ax.set_ylabel(PULL_LABEL)
            pull_ax.set_xlim(edges[0], edges[-1])
    axes[-1, 0].set_xlabel(E_LABEL)
    axes[-1, 1].set_xlabel(COSZ_LABEL)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.88)
    fig.legend([handles[i] for i in (3, 2, 0, 1)], [labels[i] for i in (3, 2, 0, 1)], loc='lower center',
               bbox_to_anchor=(0.5, 0.885), ncol=2)
    print(style.save(fig, 'fig_deepcore_data_vs_model', style.W_WIDE))


if __name__ == '__main__':
    main()
