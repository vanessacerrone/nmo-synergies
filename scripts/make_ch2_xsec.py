"""Neutrino cross sections on CH2 (the NOvA target) per nucleon, from GENIE, as scripts/make_water_xsec.py does for
water.

Same splines, processes, grid, and layout; CH2 per nucleon is (sigma_C12 + 2 sigma_H1) / 14.
Run from the repository root:
    python -I scripts/make_ch2_xsec.py <gxspl-FNALsmall.xml or .xml.gz> data/genie_ch2_xsec
"""
import importlib.util
import sys
from pathlib import Path

TARGETS = {'1000060120': 1, '1000010010': 2}                 # C12 and two H1 per molecule
NUCLEONS = 14

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    # loaded by path: python -I leaves the folder of the script out of sys.path
    spec = importlib.util.spec_from_file_location('make_water_xsec', Path(__file__).with_name('make_water_xsec.py'))
    water = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(water)
    water.build(Path(sys.argv[1]), Path(sys.argv[2]), TARGETS, NUCLEONS, 'CH2', 'scripts/make_ch2_xsec.py')
