"""fig_published_best_fits: published best fits in the (Dm31_NO, Dm31_IO) plane with the computed lines and bands (Sec.
5).

Lines: the short-baseline reactor (Eq. 3.9) and JUNO at 2.94% at 1 MeV (a_res = 2.61%). Bands: NOvA with appearance and IceCube DeepCore
with all channels, delta_CP and theta23 unknown (as fig_solutions_plane). Points: our fit of the DeepCore public data
(App. E) and the published best fits of data/published_best_fits.json, Dm32 + Dm21 for the experiments that quote Dm32.
Reads results/vacuum.npz, results/bands.npz, results/measured.npz, and results/icfit.npz. Run from the repository root:
python -m figures.fig_published_best_fits
"""
import json
from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np

from nmosyn import style
from nmosyn.params import NUFIT, b_sbr
from nmosyn.paths import DATA, RESULTS

XL = r'$\Delta m^2_{31,\mathrm{NO}}\ [10^{-3}\,\mathrm{eV}^2]$'
YL = r'$\Delta m^2_{31,\mathrm{IO}}\ [10^{-3}\,\mathrm{eV}^2]$'
XZ = np.linspace(2.40e-3, 2.66e-3, 80)
POINTS = {      # label of data/published_best_fits.json: color, marker
    'Daya Bay 2022': (style.COLORS['sbr'], 's'),
    'JUNO 2026': (style.COLORS['juno'], 'o'),
    'NOvA 2025': (style.COLORS['nova'], 'D'),
    'T2K 2024': ('#D62728', 'v'),
    'T2K + NOvA 2025': (style.COLORS['t2k'], '^'),
    'NuFIT 6.0 (global)': ('k', '*'),
    'IceCube DeepCore 9.3 yr': (style.COLORS['dc_all'], 'p'),
}


def band(bands: Mapping[str, np.ndarray], sample: str) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper edge in Dm31_IO [1e-3 eV^2] on XZ of the range of b over all true points."""
    ranges = np.array([bands[f'{sample}/grid/range_paper/{ix}'] for ix in range(len(bands['x']))])
    edges = ranges[..., 0].min(axis=1), ranges[..., 1].max(axis=1)
    return tuple((np.interp(XZ, bands['x'], b) * 1e-6 - XZ) * 1e3 for b in edges)


def pair(err: float | np.ndarray) -> np.ndarray:
    """Error bar (lower, upper) of one point as matplotlib takes it, from one symmetric error or a pair."""
    return np.broadcast_to(np.atleast_1d(err), (2,)).reshape(2, 1)


def data_fit() -> Mapping[str, float | np.ndarray]:
    """Our fit of the DeepCore public data (App. E) [eV^2]."""
    with np.load(RESULTS / 'icfit.npz') as r:
        return {k: r[k] for k in ('data_no_dm31', 'data_no_dm31_err', 'data_io_dm31', 'data_io_dm31_err')}


def published(measured: Mapping[str, np.ndarray]) -> list[tuple[str, float, np.ndarray, float, np.ndarray]]:
    """(label, Dm31_NO, its error, Dm31_IO, its error) [1e-3 eV^2] of every published best fit."""
    with open(DATA / 'published_best_fits.json') as f:
        entries = json.load(f)['measurements']
    points = []
    for e in entries:
        dm21 = e.get('dm21', NUFIT.dm21 * 1e3)
        quantity = [q.strip() for q in e['quantity'].split(',')]
        shift_no = dm21 if quantity[0].startswith('dm32') else 0.0
        shift_io = dm21 if quantity[-1].startswith('dm32') else 0.0
        no = e['no']
        if isinstance(e['io'], str):        # the IceCube 9.3-year inverted ordering, from its released map
            io = [float(measured['ic93_io_dm32']) * 1e3, *(measured['ic93_io_dm32_err'] * 1e3)]
        else:
            io = e['io']
        points.append((e['label'], no[0] + shift_no, pair(no[1:]), io[0] + shift_io, pair(io[1:])))
    return points


def main() -> None:
    style.use()
    bands = np.load(RESULTS / 'bands.npz')
    vac = np.load(RESULTS / 'vacuum.npz')
    x = XZ * 1e3
    fig, ax = plt.subplots(figsize=(10.1, 5.6))
    fig.subplots_adjust(left=0.10, right=0.68, top=0.96, bottom=0.13)
    ax.plot(x, (b_sbr() - XZ) * 1e3, '-', color=style.COLORS['sbr'], lw=2.2, label='Short-baseline reactor (Daya Bay)')
    ax.plot(x, np.interp(XZ, vac['x'], vac['juno_y']) * 1e3, '-', color=style.COLORS['juno'], lw=2.2,
            label=r'JUNO (2.94\% at 1 MeV)')
    (nova_lo, nova_hi), (dc_lo, dc_hi) = band(bands, 'nova_all'), band(bands, 'dca')
    style.bands(ax, x, [dict(lo=nova_lo, hi=nova_hi, color=style.COLORS['nova'], shade=0.25, lw=1.2,
                             label=r'NOvA, $\nu_\mu$ dis. + $\nu_e$ app. (band)'),
                        dict(lo=dc_lo, hi=dc_hi, color=style.COLORS['dc_all'], shade=0.40, lw=1.6,
                             label='IceCube DeepCore, all channels (band)')])
    fit = data_fit()
    ax.errorbar(fit['data_no_dm31'] * 1e3, fit['data_io_dm31'] * 1e3,
                xerr=pair(fit['data_no_dm31_err'] * 1e3), yerr=pair(fit['data_io_dm31_err'] * 1e3),
                fmt='o', color='w', mec=style.COLORS['dc_all'], mew=2.0, ms=9, capsize=3,
                ecolor=style.COLORS['dc_all'], lw=1.4, label='IceCube DeepCore data (this work)', zorder=6)
    for label, x_no, err_no, y_io, err_io in published(np.load(RESULTS / 'measured.npz')):
        color, marker = POINTS[label]
        ax.errorbar(x_no, y_io, xerr=err_no, yerr=err_io, fmt=marker, color=color, mec='k', mew=0.9,
                    ms=11 if marker == '*' else 8, capsize=3, lw=1.4, label=label, zorder=5)
    ax.set_xlim(2.40, 2.66)
    ax.set_ylim(-2.58, -2.28)
    ax.set_xticks([2.40, 2.45, 2.50, 2.55, 2.60, 2.65], ['2.40', '2.45', '2.50', '2.55', '2.60', '2.65'])
    ax.set_xlabel(XL)
    ax.set_ylabel(YL)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), ncol=1, frameon=False)
    print(style.save(fig, 'fig_published_best_fits', style.W_WIDE))


if __name__ == '__main__':
    main()
