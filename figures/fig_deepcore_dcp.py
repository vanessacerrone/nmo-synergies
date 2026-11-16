"""fig_deepcore_dcp: the intercept b of IceCube DeepCore against the true delta_CP, with and without matter (Sec. 4).

Tabulated response of the public release (App. A.4), nu_mu disappearance, true Dm31_NO = 2.534e-3 eV^2 and sin^2 theta23
= 0.561, fixed in both orderings. With matter: b at delta_CP^IO = 285 deg (line) and its range over delta_CP^IO (band).
Without matter: b at delta_CP^IO = 285 deg. The accelerator closed form of Eq. (3.4) and the common part 2 c12^2 Dm21 of
Eq. (4.1) are evaluated here. Reads results/deepcore.npz. Run from the repository root: python -m
figures.fig_deepcore_dcp
"""
import matplotlib.pyplot as plt
import numpy as np

from nmosyn import style
from nmosyn.params import DCP_IO, NUFIT, b_lbl
from nmosyn.paths import RESULTS

BL = r'$b=\Delta m^2_{31,\mathrm{IO}}+\Delta m^2_{31,\mathrm{NO}}\ [10^{-6}\,\mathrm{eV}^2]$'
POINT = dict(ls='none', marker='o', ms=6.5, mec='k', mew=0.8)


def main() -> None:
    style.use()
    d = np.load(RESULTS / 'deepcore.npz')
    color = style.COLORS['dc_numu']
    deg = np.degrees(d['dcp_fig3'])
    closed_deg = np.arange(-180.0, 180.1, 1.0)
    lead = 2 * (1 - NUFIT.s12sq) * NUFIT.dm21 * 1e6

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.fill_between(deg, d['fig3_fixed_lo'], d['fig3_fixed_hi'], color=style.opaque(color, 0.22), lw=0,
                    label=r'DeepCore with matter, all values of $\delta^{\rm IO}_{\rm CP}$')
    ax.plot(closed_deg, b_lbl(np.radians(closed_deg), DCP_IO) * 1e6, '--', color='k', lw=1.6,
            label=r'Accelerator closed form, $(\kappa_\mu^{\rm NO}+\kappa_\mu^{\rm IO})\Delta m^2_{21}$')
    ax.plot(deg, d['fig3_fixed_b_vacuum'], 'o', color='k', mfc='w', ms=6.5, label='DeepCore without matter')
    ax.plot(deg, d['fig3_fixed_b'], '-', color=color, lw=2.2)
    ax.plot(deg, d['fig3_fixed_b'], color=color, **POINT, label='DeepCore with Earth matter')
    ax.axhline(lead, color='0.6', ls=':', lw=1.2)
    ax.text(0, lead + 1.5, r'$2c^2_{12}\Delta m^2_{21}$', fontsize=13, color='0.4', ha='center')
    ax.set_xticks([-180, -90, 0, 90, 180], [r'$-\pi$', r'$-\pi/2$', '$0$', r'$\pi/2$', r'$\pi$'])
    ax.set_xlim(-180, 180)
    ax.set_ylim(80, 172)
    ax.set_xlabel(r'True $\delta_{\rm CP}$')
    ax.set_ylabel(BL)
    ax.legend(loc='upper center', ncol=1,
              title=r'IceCube DeepCore, $\nu_\mu$ disappearance; $\delta^{\rm IO}_{\rm CP}=285^\circ$, '
                    r'$\theta_{23}$ fixed')
    out = style.save(fig, 'fig_deepcore_dcp', style.W_NARROW)
    print(f'{out}: with matter {d["fig3_fixed_b"].min():.2f}-{d["fig3_fixed_b"].max():.2f}, band '
          f'{d["fig3_fixed_lo"].min():.2f}-{d["fig3_fixed_hi"].max():.2f}, without matter '
          f'{d["fig3_fixed_b_vacuum"].min():.2f}-{d["fig3_fixed_b_vacuum"].max():.2f}; 2 c12^2 Dm21 = {lead:.4f}')


if __name__ == '__main__':
    main()
