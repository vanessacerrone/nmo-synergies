"""fig_deepcore_bin_weights: the weighted bin average of IceCube DeepCore, bin by bin (Sec. 2, Eq. 2.3).

Tabulated response of the public release (App. A.4), nu_mu CC, theta23 and the phases fixed at the NuFIT values, true
Dm31_NO = 2.534e-3 eV^2. Left: share of the statistical weight omega_i of each reconstructed bin. Right: the value
|Dm31_IO|_i preferred by each bin alone, in the bins with more than 0.05% of the weight. Reads results/deepcore.npz. Run
from the repository root: python -m figures.fig_deepcore_bin_weights
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from nmosyn import style
from nmosyn.params import X0, intercept
from nmosyn.paths import RESULTS

CZ_CORE = -0.837        # cos theta_z below which the path crosses the Earth's core
SHARE_MIN = 5e-4        # bins with a smaller share of the weight are left gray in panel (b)
HALF_RANGE = 0.040      # color range of panel (b) around the average [1e-3 eV^2]
LABEL = dict(color='0.15', fontsize=13, ha='left', va='bottom')
BOX = dict(boxstyle='square,pad=0.15', facecolor='white', edgecolor='white', linewidth=0)    # labels of (a)
TITLES = (r'\textbf{(a) Statistical weight distribution}',
          r'\textbf{(b) Value $|y_i|$ preferred by each bin}')


def main() -> None:
    style.use()
    d = np.load(RESULTS / 'deepcore.npz')
    e, cz = d['e_edges'], d['cosz_edges']
    shape = (len(e) - 1, len(cz) - 1)       # bins are ordered with cos theta_z fastest
    share = (d['fig1_omega'] / d['fig1_omega'].sum()).reshape(shape).T
    y_abs = -float(d['fig1_y']) * 1e3
    y_i_abs = (-d['fig1_y_i'] * 1e3).reshape(shape).T
    shown = np.where(share > SHARE_MIN, np.clip(y_i_abs, y_abs - HALF_RANGE, y_abs + HALF_RANGE), np.nan)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.14, 5.0), gridspec_kw=dict(wspace=0.30))
    mesh = a1.pcolormesh(e, cz, share * 100, cmap='RdPu', shading='flat')
    fig.colorbar(mesh, ax=a1, pad=0.03).set_label(r'Share of statistical weight $\omega_i/\sum\omega$ [\%]',
                                                   fontsize=15)
    a2.set_facecolor('0.78')
    mesh = a2.pcolormesh(e, cz, shown, cmap='RdBu', shading='flat', vmin=y_abs - HALF_RANGE,
                         vmax=y_abs + HALF_RANGE)
    fig.colorbar(mesh, ax=a2, pad=0.03).set_label(
        'Value preferred by the bin\n' + r'$|y_i|\ [10^{-3}\,\mathrm{eV}^2]$',
        fontsize=15)

    for ax, title in zip((a1, a2), TITLES):
        box = BOX if ax is a1 else None
        ax.set_xscale('log')
        ax.set_xlim(e[0], e[-1])
        ax.set_ylim(cz[0], cz[-1])
        ax.set_xticks([10, 20, 50, 100], ['10', '20', '50', '100'])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('Reconstructed energy [GeV]')
        ax.set_ylabel(r'Reconstructed $\cos\theta_z$')
        ax.axhline(CZ_CORE, color='0.3', ls='--', lw=1.1)
        ax.text(24.0, CZ_CORE + 0.02, 'Core-mantle boundary', bbox=box, **LABEL)
        ax.axhline(0.0, color='0.3', ls=':', lw=1.1)
        ax.text(24.0, 0.02, r'Horizon ($\cos\theta_z=0$)', bbox=box, **LABEL)
        # titles as annotations, so that the one-line and the two-line title start at the same height
        ax.annotate(title, xy=(0.5, 1), xycoords='axes fraction', xytext=(0, 46), textcoords='offset points',
                    ha='center', va='top', multialignment='center', fontsize=15)

    out = style.save(fig, 'fig_deepcore_bin_weights', style.W_WIDE)
    print(f'{out}: b = {intercept(d["fig1_y"], X0):.3f} 1e-6 eV^2 (caption), '
          f'max share {share.max() * 100:.3f}%, {np.isnan(shown).sum()} gray bins')


if __name__ == '__main__':
    main()
