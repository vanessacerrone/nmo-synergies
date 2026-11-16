# Code conventions

The rules that keep the repository small, readable, and fast. The README describes what the code computes, its
layout, the notation, and the methods. This file lists only how the code is written.

## Code

- Python 3.11 or later, PEP 8, lines of at most 120 characters, 4-space indentation, one statement per line.
- Names follow the paper (the notation table of the README). Type hints on public functions. A docstring has one
  summary line that cites the equation or appendix it implements, e.g. `"""Weighted bin average, Eq. (2.3)."""`,
  and then arguments and units only when the notation table does not already give them.
- Functions are pure: inputs in, results out. Parameters travel in the frozen dataclass `params.Params`, never as
  module globals, and no module changes the attributes of another.
- The package returns and the scripts print: no `print` inside `nmosyn/`. No environment-variable switches, no
  commented-out code, no dead branches, no comments about the history of the code.
- numba compiles only the inner kernels: the probabilities and rotations of `osc.py` and the path layers of
  `earth.py`. Everything else is vectorized numpy.
- Comments explain why, not what. American English, with the words of the paper: wrong-ordering solution,
  intercept, weighted bin average, nuisance parameters constrained by pull terms.

## Units

- eV², GeV (MeV for reactors, in names ending in `_mev`), km, g/cm³, and radians. Degrees appear only in figure
  axes and printed text.
- Intercepts are stored, printed, and plotted in 10⁻⁶ eV². The conversion `b = (y + x) * 1e6` happens in one
  function, `params.intercept`.

## Results and figures

- A script of `compute/` writes one file, `results/<name>.npz`, and states in its docstring which figures and
  sections use it. A figure script only reads `results/` and plots: it computes no physics.
- One style, `nmosyn/style.py`: the rcParams of the paper (serif, LaTeX text, inward ticks on all sides, no grid,
  no legend frame) and one color per experiment, from the IBM color-blind-safe set.
- Every figure is saved at the width at which the paper prints it, so that all text prints at the same size, and
  with opaque colors only (no transparency).

## Data

- Every input is published. `data/PROVENANCE.md` lists for each file its source, DOI or URL, license, what was
  taken (table, page, figure), and whether it was digitized.
- Large public releases are not committed: `scripts/download_data.py` fetches them into `data/external/` and
  checks their SHA-256 sums, which `tests/test_provenance.py` compares with `PROVENANCE.md`.

## Tests

- Every number of `paper_numbers.py` has a test with its tolerance, and the tests run in under a minute on the
  cached results.
- A change that moves a number of the paper is reported with the number, its old and new value, and the reason.
