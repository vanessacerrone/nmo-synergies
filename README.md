<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo_dark.svg">
    <img src="docs/logo.svg" alt="nmo-synergies" width="460">
  </picture>
</h1>

<p align="center"><a href="https://doi.org/10.5281/zenodo.24304049"><img src="https://zenodo.org/badge/1460194275.svg" alt="DOI"></a></p>

Code and data that reproduce every figure and number of

> V. Cerrone and M. Grassi, *Phenomenology of neutrino mass ordering synergies in reactor, accelerator, and
> atmospheric neutrino experiments, and the role of the detector response*.

## What the code computes

When the data of an experiment, generated in the normal mass ordering (NO), are fitted under the inverted
ordering (IO), the fit returns a wrong-ordering solution of the splitting, Dm31_IO, and each experiment returns
its own. The paper writes this solution as a weighted average of the values preferred by the single bins of the
experiment (Eq. 2.3) and describes each experiment by one number, the intercept b = Dm31_IO + Dm31_NO. Two
experiments with different intercepts disagree in the inverted ordering, so together they disfavor it more than
the sum of their separate results: this is their synergy (Eq. 1.1).

From public configurations and data releases, the package `nmosyn` computes

- the intercepts of Daya Bay, JUNO, NOvA, T2K, IceCube DeepCore, and KM3NeT/ORCA6,
- their dependence on the unknown delta_CP and theta23, on matter along the path, and on the detector response
  (energy resolution, energy window, direction resolution),
- the synergy of pairs of experiments,
- the full fit of the IceCube DeepCore public release, which reproduces the published measurement and validates
  the weighted bin average (App. E).

A first look, in a few seconds:

```python
from nmosyn.experiments.juno import Juno
from nmosyn.params import DCP_IO, DCP_NO, X0, b_lbl, b_sbr, intercept

y = Juno().solution(X0)              # JUNO at the true splitting X0: the wrong-ordering solution, Eq. (2.3)
print(intercept(y, X0))              # its intercept b: 25.66 (in 1e-6 eV^2)
print(b_sbr() * 1e6)                 # short-baseline reactor, Eq. (3.9): 45.99
print(b_lbl(DCP_NO, DCP_IO) * 1e6)   # accelerator closed form, Eq. (3.6): 110.56
```

## Quick start

Python 3.11 or later with numpy (2.0 or later), scipy, matplotlib, numba, pandas, uproot, and iminuit, and pytest
for the tests. The figure labels use LaTeX when it is installed and matplotlib's mathtext otherwise. Everything
runs from the repository root.

```
git clone https://github.com/vanessacerrone/nmo-synergies
cd nmo-synergies
pip install -e .
make figures      # the 16 PDFs of the paper and two more, from the cached results, in seconds
make numbers      # every number of the paper, against the value quoted in it
```

The cached results `results/*.npz` are committed, and the figure scripts and `paper_numbers.py` only read them, so
these two steps need no download and none of the long computations of `compute/`. To recompute everything:

```
make data         # the public releases, about 84 MB, checked against SHA-256 sums
make results      # compute/*.py -> results/*.npz (the grid check and the joint fit take about ten and fifteen minutes)
make test         # pytest
make validate     # the full fit of the IceCube DeepCore release (App. E; about half an hour in four processes)
```

## How the pieces fit together

```
nmosyn/  ->  compute/*.py  ->  results/*.npz  ->  figures/*.py      ->  output/*.pdf
                                              ->  paper_numbers.py  ->  every number of the paper
```

`nmosyn/` is the library: the physics and the models of the experiments, with no script logic. Each script of
`compute/` uses it for one part of the paper and saves the result to `results/<name>.npz`. The figure scripts and
`paper_numbers.py` only read these files. To change the physics, edit `nmosyn/`. To change what a figure or a
section uses, edit its script in `compute/`.

| path | content |
|---|---|
| `nmosyn/params.py` | NuFIT 6.0 inputs (App. A.1), closed forms of Eqs. (3.6) and (3.9), solar phase (App. C.1) |
| `nmosyn/earth.py` | PREM and the layers of constant density along a path, from production in the atmosphere to the detector |
| `nmosyn/osc.py` | three-flavor probabilities in vacuum, at constant density, and through the Earth |
| `nmosyn/flux.py` | Honda atmospheric neutrino flux tables |
| `nmosyn/average.py` | weighted bin average and its Gauss-Newton iteration (Eq. 2.3, App. B.3), nuisance parameters (Eqs. B.5-B.6) |
| `nmosyn/fit.py` | inverted-ordering fit at its global minimum over Dm31_IO, theta23, and delta_CP^IO |
| `nmosyn/synergy.py` | synergy term (Eq. 1.1) and its extensions (Eqs. F.1-F.2) |
| `nmosyn/style.py` | the figure style and the colors of the experiments |
| `nmosyn/experiments/` | the reactor spectrum and JUNO, the accelerators (generic beam, NOvA, T2K), IceCube DeepCore (tabulated response and full fit), KM3NeT/ORCA6 |
| `compute/` | one script per cached result in `results/` |
| `figures/` | one script per figure of the paper, named after the PDF it writes |
| `paper_numbers.py` | every number of the paper, against the value quoted in it |
| `data/` | small inputs and `PROVENANCE.md`, which documents every input (the public releases go to `data/external/`) |
| `scripts/` | the download of the public releases, the CH2 and water cross sections of NOvA and T2K from the GENIE splines, the digitization of one published contour |
| `tests/` | the numbers of the paper and unit checks of the physics |
| `examples/examples.ipynb` | seven short cases: JUNO against its energy resolution, the accelerator band over delta_CP, your own experiment, DeepCore with and without matter and its response, DeepCore and ORCA6, the global minimum with free theta23 and delta_CP, the synergy of two experiments |
| `docs/` | the conventions of the code (`CODE_STYLE.md`) and the logo |

## Figures

Figures are named after the PDF file that the paper includes, so the names stay the same when the paper
renumbers its figures. Each script `figures/<name>.py` writes `output/<name>.pdf`. Two scripts write two panels
each: `fig_accelerator_closed_form.py` also writes `fig_juno_intercept_vs_resolution.pdf`, and `fig_deepcore_resolution.py` also writes `fig_deepcore_energy_window.pdf`.

| PDF | where | content | results |
|---|---|---|---|
| `fig_deepcore_bin_weights` | Sec. 2 | weighted bin average of IceCube DeepCore, bin by bin | `deepcore` |
| `fig_accelerator_closed_form`, `fig_juno_intercept_vs_resolution` | Sec. 3 | intercept of an accelerator and of JUNO near the vacuum limit | `vacuum` |
| `fig_deepcore_dcp` | Sec. 4 | DeepCore intercept against the true delta_CP, with and without matter | `deepcore` |
| `fig_deepcore_resolution`, `fig_deepcore_energy_window` | Sec. 4 | DeepCore intercept against the energy resolution and the energy window | `deepcore` |
| `fig_deepcore_vs_orca6` | Sec. 4 | DeepCore and ORCA6 intercepts against the true splitting | `orca` |
| `fig_solutions_plane` | Sec. 5 | wrong-ordering solutions in the (Dm31_NO, Dm31_IO) plane | `bands`, `vacuum` |
| `fig_intercept_ranges` | Sec. 5 | range of the intercepts, delta_CP and theta23 unknown | `bands` |
| `fig_intercept_vs_truth` | Sec. 5 | intercepts against the true delta_CP and the true theta23 | `bands` |
| `fig_published_best_fits` | Sec. 5 | published best fits with the computed lines and bands | `bands`, `icfit`, `measured`, `vacuum` |
| `fig_synergy_vs_precision` | Sec. 6 | synergy of two experiments against their common precision | `bands`, `vacuum` |
| `fig_synergy_rings` | Sec. 6 | synergy of JUNO with DeepCore and with NOvA | `bands`, `vacuum` |
| `fig_accelerator_nubar_share` | not in the paper | shift of the accelerator intercept from Eq. (3.6), against the antineutrino share; its values are quoted in App. C.2 | `vacuum` |
| `fig_juno_effective_energy` | App. C | JUNO weights and the effective energy of the solar phase | `vacuum` |
| `fig_deepcore_analytic` | not in the paper | DeepCore nu_mu disappearance, exact and analytic; its shares are quoted in App. D.1 | `analytic` |
| `fig_full_fit_validation` | App. E | weighted bin average against the full fit of the DeepCore release | `deepcore`, `icfit` |
| `fig_deepcore_reproduction` | App. E | the IceCube DeepCore measurement reproduced from its release | `icfit` |
| `fig_deepcore_data_vs_model` | not in the paper | observed DeepCore counts against the expected counts at the best fits of App. E: NO, and IO with the hypersurfaces read at both splittings of App. E.1, with the pulls | `icfit` |

The other results hold numbers quoted in the text: `direction` (direction resolution, Sec. 4), `solar` (solar
parameters, Sec. 6), `grid` (convergence of the DeepCore response, App. A.4), and `release` (event totals and
resolution of the DeepCore release, App. A.4 and App. E).

## Notation

Names follow the paper.

| code | paper | unit |
|---|---|---|
| `x` | true splitting $\Delta m^2_{31,\rm NO}$ | eV² |
| `y` | inverted-ordering splitting $\Delta m^2_{31,\rm IO}$ (negative) | eV² |
| `b` | intercept $b=y+x$ (Eq. 2.4) | 10⁻⁶ eV² |
| `w`, `omega` | weights $W_i$, $\omega_i=W_i s_i^2$ (Eqs. 2.1, 2.3) | |
| `s`, `s_tilde` | sensitivity $s_i=\partial P_i^{\rm IO}/\partial y$, projected $\tilde s_i$ (Eq. B.5) | |
| `y_i`, `b_i` | value preferred by bin $i$, and its intercept | eV², 10⁻⁶ eV² |
| `s23sq`, `dcp_no`, `dcp_io` | $\sin^2\theta_{23}$, $\delta_{\rm CP}^{\rm NO}$, $\delta_{\rm CP}^{\rm IO}$ | –, rad |

Inside the code the units are eV², GeV (MeV for reactors, in names ending in `_mev`), km, g/cm³, and radians.
Intercepts are stored, printed, and plotted in 10⁻⁶ eV², and degrees appear only in figures and printed text.

## Methods in brief

- **Probabilities.** `nmosyn/osc.py` is a numba port of NuFast (eigenvalues and eigenvectors in matter) and
  NuFast-Earth (amplitudes of layers of constant density, theta23 and delta_CP applied as a rotation), by
  P. B. Denton and S. J. Parke. Atmospheric neutrinos start 20 km above the surface and cross the atmosphere
  (vacuum), the water or ice above the detector, and PREM, cut at its discontinuities into layers of at most
  500 km with their mean density.
- **Inverted-ordering fit.** theta23 and delta_CP enter every expected count through a rotation of the flavor
  basis (App. D.2), so at fixed Dm31 the counts are trigonometric polynomials in theta23 and delta_CP, fixed
  exactly by 25 evaluations. The fit computes them on a grid of Dm31 in steps of 1e-6 eV², interpolates between
  the grid points, and minimizes the chi² of Eq. (2.1) directly, with no grid in the angles. When more than three
  values of delta_CP^IO in steps of 10 degrees lie within 0.01 of the minimum chi², the solution is a band that
  contains every 2-degree point within 0.01. Otherwise the solution is the global minimum (App. B.3).
- **Speed.** The amplitude of a path through any sequence of layers is computed once per value of Dm31 and
  cached, so every theta23 and delta_CP costs one 3×3 rotation per cell.
- **Experiments.** All inputs are public: the JUNO configuration of its physics papers, with the energy
  resolution of its full detector simulation (a = 2.61%, b = 0.64%, c = 1.20%, arXiv:2405.17860) and
  rho Y_e = 0.5 x 2.45 g/cm^3 along the baselines (arXiv:2204.13249, Sec. 2.2); the NOvA and
  T2K far-detector spectra built from their data and flux releases, with the GENIE 2.12.10 cross sections on CH2
  for NOvA and on water for T2K; the IceCube
  DeepCore and KM3NeT/ORCA6 open data; and the Honda flux tables.

## Checks

- `make numbers` prints every number of the paper that the code computes, with its place in the TeX source, the
  quoted value, and the computed one, and marks with MOVED any number that no longer agrees.
- `make test` checks the same numbers within their tolerances, and the physics: the vacuum limit and unitarity of
  the probabilities, the path geometry and the convergence of the layers, the closed form of Eq. (3.6) recovered
  by the weighted bin average, the Gauss-Newton solution equal to the minimum of the chi², and the exact
  harmonic expansion of the fit. With a C++ compiler, it also compares `nmosyn/osc.py` with the original
  NuFast-Earth.
- `compute/grid.py` repeats the DeepCore intercepts on a grid of true cells five times finer (App. A.4).
- `compute/joint.py` compares Eq. (1.1) for NOvA and DeepCore with their exact joint IO fit, which shares
  Dm31_IO, theta23^IO, and delta_CP^IO (Eq. F.2), at every true point of the bands (Sec. 6, App. F.2).
- `make validate` runs the full fit of the DeepCore release, which reproduces the published measurement (`fig_deepcore_reproduction`).

## Data

Every input is public. `data/PROVENANCE.md` lists the source, DOI or URL, license, and checksum of each one, and
what was taken from it. `scripts/download_data.py` fetches the releases into `data/external/` and checks their
SHA-256 sums: the IceCube DeepCore releases (doi:10.7910/DVN/B4RITM, doi:10.7910/DVN/U20MMB), the KM3NeT/ORCA6
open data, the NOvA data and flux releases, the T2K flux release, the GENIE cross sections of the DUNE simulation
configuration (whose energy grid the derived cross-section tables share), the Honda flux tables, the IceCube DeepCore article (arXiv:2304.12236) whose published contour
`fig_deepcore_reproduction` compares with, and the NuFast-Earth sources for the comparison test. The cross
sections of the NOvA and T2K spectra, `data/genie_ch2_xsec/` and `data/genie_water_xsec/`, are in the repository:
`scripts/make_ch2_xsec.py` and `scripts/make_water_xsec.py` make them from the GENIE 2.12.10 splines, which `python scripts/download_data.py genie_xsec_2_12_10` fetches on request
(498 MiB). The published best fits of `fig_published_best_fits` are in `data/published_best_fits.json`, with their sources.

## How to cite

If you use this code, please cite the paper and the software:

> V. Cerrone and M. Grassi, *Phenomenology of neutrino mass ordering synergies in reactor, accelerator, and
> atmospheric neutrino experiments, and the role of the detector response*, JHEP (2026), arXiv:2611.21252.

> V. Cerrone and M. Grassi, *nmo-synergies: wrong-ordering solutions of reactor, accelerator, and atmospheric
> neutrino experiments as weighted bin averages*, version 1.3.0, Zenodo (2026), doi:10.5281/zenodo.XXXXXXX.

```bibtex
@article{Cerrone:2026nmo,
    author  = "Cerrone, Vanessa and Grassi, Marco",
    title   = "{Phenomenology of neutrino mass ordering synergies in reactor, accelerator, and atmospheric
               neutrino experiments, and the role of the detector response}",
    journal = "JHEP",
    year    = "2026",
    eprint  = "2611.21252",
    archivePrefix = "arXiv"
}
@software{nmosynergies:2026,
    author  = "Cerrone, Vanessa and Grassi, Marco",
    title   = "{nmo-synergies: wrong-ordering solutions of reactor, accelerator, and atmospheric neutrino
               experiments as weighted bin averages}",
    version = "1.3.0",
    year    = "2026",
    doi     = "10.5281/zenodo.XXXXXXX",
    url     = "https://github.com/vanessacerrone/nmo-synergies"
}
```

`CITATION.cff` holds the same information for the "Cite this repository" button of GitHub.

## Credits and license

MIT license (`LICENSE`). `nmosyn/osc.py` translates code of NuFast and NuFast-Earth, which are MIT licensed, and
their notices are in `THIRD_PARTY_LICENSES`. If you use the probabilities, please also cite P. B. Denton and S. J. Parke,
arXiv:2405.02400 and arXiv:2511.04735. The data releases are cited in `data/PROVENANCE.md`. The conventions of
the code are in `docs/CODE_STYLE.md`.
