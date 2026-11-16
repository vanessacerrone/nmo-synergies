"""fig_full_fit_validation: validation of the weighted bin average against the full fit of the IceCube DeepCore release
(Sec. 4, App. E.2).

NuFIT 6.0 with delta_CP = 177 deg in both orderings. Upper panel: b from the full fit, from the weighted bin
average on the tabulated response with all channels (theta23 fitted), and on the full fitter's expectation
with every nuisance parameter in the weights (Eq. B.6). Lower panel: the difference of each from the full fit
[1e-6 eV^2]. Reads results/deepcore.npz and results/icfit.npz.
Run from the repository root: python -m figures.fig_full_fit_validation
"""
import matplotlib.pyplot as plt
import numpy as np

from nmosyn import style
from nmosyn.paths import RESULTS

WBA = 'Weighted bin average'
XL = r'$\Delta m^2_{31,\mathrm{NO}}\ [10^{-3}\,\mathrm{eV}^2]$'
BL = r'$b=\Delta m^2_{31,\mathrm{IO}}+\Delta m^2_{31,\mathrm{NO}}\ [10^{-6}\,\mathrm{eV}^2]$'
POINT = dict(ms=6.5, mec='k', mew=0.8)


def main() -> None:
    style.use()
    dc, fit = np.load(RESULTS / 'deepcore.npz'), np.load(RESULTS / 'icfit.npz')
    x_fit, b_fit = fit['asimov_x'], fit['asimov_b']
    tabulated = np.interp(x_fit, dc['x_line'], dc['fig6_line'])

    def difference(x, b):
        """Difference from the full fit [1e-6 eV^2]."""
        return b - np.interp(x, x_fit, b_fit)

    color = style.COLORS['dc_all']
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.7, 7.0), sharex=True,
                                      gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.06))
    top.plot(dc['x_line'] * 1e3, dc['fig6_line'], '-', color=color, lw=2.2)
    top.plot(x_fit * 1e3, tabulated, 'o', color=color, ls='-', lw=2.2, label=f'{WBA}, tabulated response, all channels',
             **POINT)
    top.plot(fit['wba_x'] * 1e3, fit['wba_all'], 'D', color='0.25', ms=6.0, mec='k', mew=0.8, zorder=4)
    top.scatter(x_fit * 1e3, b_fit, s=11 ** 2, marker='*', facecolors='none', edgecolors='k', linewidths=1.1,
                zorder=5, label='Full fit (all channels, nuisance parameters)')     # open marker without transparency
    top.set_ylabel(BL)
    top.set_ylim(110.5, max(120.5, np.ceil(dc['fig6_line'].max() + 0.3)))

    bottom.plot(x_fit * 1e3, difference(x_fit, tabulated), 'o', color=color, ls='none', **POINT)
    bottom.plot(fit['wba_x'] * 1e3, difference(fit['wba_x'], fit['wba_y']), 'D', color='w', ls='none',
                label=f'{WBA}, full-fit model, only ' + r'$\Delta m^2_{31,\rm IO}$' + ' fitted (lower panel)', **POINT)
    bottom.plot(fit['wba_x'] * 1e3, difference(fit['wba_x'], fit['wba_all']), 'D', color='0.25', ls='none',
                label=f'{WBA}, full-fit model, ' + r'$\theta_{23}$' + ' and nuisance parameters profiled',
                **POINT)
    bottom.axhline(0, color='0.6', lw=0.8)
    bottom.set_ylim(-0.35, 2.45)
    bottom.set_xlim(2.2, 2.8)
    bottom.set_xlabel(XL)
    bottom.set_ylabel('Difference from\n' + r'full fit [$10^{-6}\,\mathrm{eV}^2$]', labelpad=2)

    handles, labels = top.get_legend_handles_labels()
    lower_h, lower_l = bottom.get_legend_handles_labels()
    top.legend(handles + lower_h[::-1], labels + lower_l[::-1], loc='lower left', bbox_to_anchor=(-0.14, 1.01),
               ncol=1, frameon=False,
               title=r'IceCube DeepCore, NuFIT 6.0, $\delta_{\rm CP}=177^\circ$ in both orderings')
    fig.align_ylabels((top, bottom))
    print(style.save(fig, 'fig_full_fit_validation', style.W_NARROW))


if __name__ == '__main__':
    main()
