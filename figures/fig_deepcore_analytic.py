"""fig_deepcore_analytic: the intercept b of IceCube DeepCore nu_mu disappearance against the true splitting, exact and
in the analytic forms of App. D.1.

Exact weighted bin average (theta23 fixed) at the NuFIT phases and at delta_CP = 90 deg; the weighted bin
average on the one-mass-scale probability in matter; the frequency shift of the two-level form averaged with
the detector weights; the value without matter, 2 c12^2 Dm21. Reads results/analytic.npz.
Run from the repository root: python -m figures.fig_deepcore_analytic
"""
import matplotlib.pyplot as plt
import numpy as np

from nmosyn import style
from nmosyn.paths import RESULTS

XL = r'$\Delta m^2_{31,\mathrm{NO}}\ [10^{-3}\,\mathrm{eV}^2]$'
BL = r'$b=\Delta m^2_{31,\mathrm{IO}}+\Delta m^2_{31,\mathrm{NO}}\ [10^{-6}\,\mathrm{eV}^2]$'
POINT = dict(ms=6.5, mec='k', mew=0.8)


def main() -> None:
    style.use()
    r = np.load(RESULTS / 'analytic.npz')
    x = r['x'] * 1e3
    color = style.COLORS['dc_numu']
    fig, ax = plt.subplots(figsize=(7.8, 6.9))
    ax.plot(x, r['exact_nufit'], '--', color=color, lw=2.2,
            label=r'Exact, NuFIT $\delta^{\rm NO,IO}_{\rm CP}$')
    ax.plot(x, r['exact_90'], '-', color=color, lw=2.2, label=r'Exact, $\delta_{\rm CP}=90^\circ$')
    ax.plot(x, r['one_mass_scale'], '-', marker='D', color='0.45', lw=1.2, **POINT,
            label=r'One-mass-scale probability in matter, $\delta_{\rm CP}=90^\circ$')
    ax.plot(x, r['frequency_90'], ls='none', marker='o', color='white', mfc='white', **POINT,
            label=r'Frequency shift, two-level form (no $\delta_{\rm CP}$ dependence)')
    lead = float(r['lead'])
    ax.axhline(lead, color='0.6', ls=':', lw=1.2)
    ax.text(2.21, lead + 0.6, r'$2c^2_{12}\Delta m^2_{21}$ (no matter)', fontsize=13, color='0.4', ha='left',
            va='bottom')
    ax.set_xlim(2.2, 2.8)
    ax.set_ylim(101, 123)
    ax.set_xlabel(XL)
    ax.set_ylabel(BL, labelpad=2.4)     # the y label and the last x tick label then fit the page width
    ax.legend(loc='lower left', bbox_to_anchor=(-0.14, 1.01),
              title=r'IceCube DeepCore, $\nu_\mu$ disappearance, detector weights')
    print(style.save(fig, 'fig_deepcore_analytic', style.W_NARROW))


if __name__ == '__main__':
    main()
