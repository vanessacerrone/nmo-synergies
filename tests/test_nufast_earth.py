"""nmosyn.osc against the original NuFast-Earth engine (arXiv:2511.04735), compiled from the pinned release.

The sources are fetched by scripts/download_data.py (entry nufast_earth); the test is skipped without them or
without a C++ compiler. Both evaluate the same layers of constant density, so they agree to rounding (1e-10).
"""
import shutil
import subprocess

import numpy as np
import pytest

from nmosyn import earth, osc
from nmosyn.params import NUFIT
from nmosyn.paths import EXTERNAL, ROOT

SOURCES = EXTERNAL / 'nufast_earth'
COMPILER = shutil.which('c++')


@pytest.fixture(scope='module')
def engine(tmp_path_factory):
    if COMPILER is None or not (SOURCES / 'NuFastEarth.cpp').exists():
        pytest.skip('NuFast-Earth sources (scripts/download_data.py nufast_earth) or a C++ compiler missing')
    binary = tmp_path_factory.mktemp('nufast_earth') / 'driver'
    units = [SOURCES / f'{name}.cpp' for name in ('NuFastEarth', 'Oscillation', 'Earth', 'Geometry', 'Solar')]
    subprocess.run([COMPILER, '-O2', '-std=c++17', f'-I{SOURCES}', *map(str, units),
                    str(ROOT / 'tests' / 'nufast_earth_driver.cpp'), '-o', str(binary)], check=True)
    return binary


def original(binary, lengths, rhoye, energies, s23sq, dcp, dm31, neutrino):
    """P[..., beta, alpha] from the compiled NuFast-Earth along one path."""
    p = NUFIT
    text = (f'{p.s12sq} {p.s13sq} {s23sq} {dcp} {p.dm21} {dm31} {int(neutrino)}\n{len(lengths)} '
            + ' '.join(f'{float(l)!r} {float(r)!r}' for l, r in zip(lengths, rhoye))
            + f'\n{len(energies)} ' + ' '.join(repr(float(e)) for e in energies) + '\n')
    out = subprocess.run([str(binary)], input=text, capture_output=True, text=True, check=True).stdout
    return np.array(out.split(), dtype=float).reshape(len(energies), 3, 3).transpose(0, 2, 1)


@pytest.mark.parametrize('cosz', [-0.95, -0.6, -0.2, 0.3])
@pytest.mark.parametrize('dm31', [2.5e-3, -2.4e-3])
@pytest.mark.parametrize('neutrino', [True, False])
def test_layered_earth(engine, cosz, dm31, neutrino):
    energies = np.geomspace(2.0, 80.0, 25)
    lengths, rhoye = earth.layers([cosz], 2.0, 0.51)
    used = lengths[0] > 0
    amp = osc.path_amplitudes(energies, lengths, rhoye, dm31, NUFIT, antineutrino=not neutrino)
    ours = osc.probabilities(amp, 0.561, 3.09, antineutrino=not neutrino)
    theirs = original(engine, lengths[0, used], rhoye[0, used], energies, 0.561, 3.09, dm31, neutrino)
    assert np.max(np.abs(ours - theirs)) < 1e-9
