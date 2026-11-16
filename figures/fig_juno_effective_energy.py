"""fig_juno_effective_energy: the JUNO weights per bin and the effective energy of the solar-phase term (Eq. (3.8), App.
C.3).

The curve is Dm2_phi,ee(E) at 52.5 km. Each point sits at the energy where the curve equals its average
weighted by omega_i = W_i s_i^2, for six values of the stochastic term a_res. The shaded areas are the weights
omega_i per MeV, drawn as a 15-bin running mean so that the shapes can be told apart. They overlap, so they are
one opaque image of the colors that transparent fills would show over white: the figure has no transparency.
Reads results/vacuum.npz.
Run from the repository root: python -m figures.fig_juno_effective_energy
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

from nmosyn import style
from nmosyn.paths import RESULTS

RUNNING_MEAN_BINS = 15
X_RANGE = (1.85, 9.0)            # reconstructed energy [MeV]
Y2_RANGE = (0.0, 0.75)           # weight per MeV
SHADE_ALPHA = 0.18               # transparency the shaded weights stand for
IMAGE = (2000, 800)              # pixels of the shaded image in energy and in weight


def shaded(e_mev: np.ndarray, weights: list[np.ndarray], colors) -> np.ndarray:
    """Image (weight, energy, RGB) of the weights filled from zero in turn with transparency SHADE_ALPHA over
    white: in each pixel, the opaque color of the fills that cover it."""
    e = np.linspace(*X_RANGE, IMAGE[0] + 1)
    w = np.linspace(*Y2_RANGE, IMAGE[1] + 1)
    e, w = 0.5 * (e[1:] + e[:-1]), 0.5 * (w[1:] + w[:-1])
    image = np.ones((IMAGE[1], IMAGE[0], 3))
    for per_mev, color in zip(weights, colors):
        covered = w[:, None] < np.interp(e, e_mev, per_mev, left=0.0, right=0.0)[None, :]
        image[covered] = SHADE_ALPHA * np.array(color[:3]) + (1 - SHADE_ALPHA) * image[covered]
    return image


def main() -> None:
    plt.switch_backend('agg')     # Agg measures the text that sets the page box, the same on every platform
    style.use()
    v = np.load(RESULTS / 'vacuum.npz')
    e_mev = v['eeff_bins']
    bin_mev = e_mev[1] - e_mev[0]
    colors = colormaps['Blues'](np.linspace(0.35, 1.0, len(v['a_eeff'])))

    fig = plt.figure(figsize=(style.W_WIDE, 4.4))
    ax = fig.add_axes((0.08, 0.15, 0.66, 0.80))
    ax2 = ax.twinx()
    ax.set_zorder(ax2.get_zorder() + 1)     # curve and points over the shaded weights
    ax.patch.set_visible(False)

    kernel = np.ones(RUNNING_MEAN_BINS) / RUNNING_MEAN_BINS
    weights = [np.convolve(w, kernel, mode='same') / bin_mev for w in v['eeff_weights']]
    ax2.imshow(shaded(e_mev, weights, colors), extent=(*X_RANGE, *Y2_RANGE), origin='lower', aspect='auto',
               interpolation='none', zorder=0)
    for per_mev, color in zip(weights, colors):
        ax2.plot(e_mev, per_mev, '-', color=style.opaque(color, 0.8), lw=1.0)

    ax.plot(e_mev, v['dm2_phi_bins'], '-', color='k', lw=2.2, label=r'$\Delta m^2_{\varphi,ee}(E)$', zorder=5)
    for a, e_eff, avg, color in zip(v['a_eeff'], v['eeff_energy'], v['eeff_average'], colors):
        ax.plot(e_eff, avg, 'o', color=color, mec='k', mew=0.8, ms=8, zorder=6,
                label=fr'$a_{{\rm res}}={a * 100:.3g}\%$: {e_eff:.1f} MeV')
    ax.plot([4.0], [float(v['dm2_phi_4mev'])], '*', color='k', ms=13, zorder=7, label='Closed form, 4 MeV')

    ax.set(xlim=X_RANGE, ylim=(0, 70), xlabel=r'Reconstructed energy [MeV]',
           ylabel=r'$\Delta m^2_{\varphi,ee}$ $[10^{-6}\,\mathrm{eV}^2]$')
    weight_gray = '0.4'                     # the right axis, of the shaded weights, in gray
    ax2.set(ylim=Y2_RANGE)
    ax2.set_ylabel(r'Weight $\omega$ per MeV (shaded)', color=weight_gray)
    ax2.tick_params(axis='y', colors=weight_gray)
    ax.legend(loc='center left', bbox_to_anchor=(1.12, 0.5), ncol=1, handletextpad=0.3, borderaxespad=0.0,
              fontsize=11, title_fontsize=11, title=r'Points at $E_{\rm eff}$')
    style.save(fig, 'fig_juno_effective_energy', style.W_WIDE)


if __name__ == '__main__':
    main()
