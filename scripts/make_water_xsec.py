"""Neutrino cross sections on water per nucleon, in the GLoBES layout of data/external/dune_globes_xsec, from GENIE.

Reads the pre-computed splines of GENIE 2.12.10 (DefaultPlusValenciaMEC, file gxspl-FNALsmall.xml, see
data/PROVENANCE.md), sums for nu_e, nu_mu, nubar_e, nubar_mu on O16 and H1 every spline of the charged current
(Weak[CC]) and of the neutral current (Weak[NC]), without neutrino-electron scattering (NuEEL) and inverse muon
decay (IMD), and writes the water cross section per nucleon, (sigma_O16 + 2 sigma_H1) / 18, divided by E, on the
75-point grid of log10(E/GeV) of the argon tables (their only use; scripts/make_ch2_xsec.py calls build() for the
CH2 tables of NOvA):

    xsec_cc.dat, xsec_nc.dat   log10(E/GeV), sigma/E of nu_e, nu_mu, nu_tau, nubar_e, nubar_mu, nubar_tau
                               [1e-38 cm^2/GeV per nucleon]; the nu_tau columns are zero

GENIE splines are cross sections per nucleus in natural units, GeV^-2. The file is parsed as a stream.
Run from the repository root:
    python -I scripts/make_water_xsec.py <gxspl-FNALsmall.xml or .xml.gz> data/genie_water_xsec
"""
import gzip
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

GEV2_TO_1E38CM2 = 0.389379e-27 / 1e-38    # 1 GeV^-2 = (hbar c)^2 = 0.389379 mb
TARGETS = {'1000080160': 1, '1000010010': 2}                 # O16 and two H1 per molecule
NUCLEONS = 18
FLAVORS = {12: 1, 14: 2, -12: 4, -14: 5}                      # PDG code: column of the GLoBES table
SKIP = ('NuEEL', 'IMD')                                       # scattering on electrons
ARGON = Path(__file__).resolve().parent.parent / 'data' / 'external' / 'dune_globes_xsec'


def read_splines(path: Path, log_e: np.ndarray, targets: dict = TARGETS) -> tuple[dict, set]:
    """{(kind, nu, tgt): summed sigma [GeV^-2] at log10(E/GeV) = log_e} for the flavors and targets above, and the
    processes. Each spline is interpolated linearly in log10 E (its knots start at its threshold, zero below)."""
    opener = gzip.open if str(path).endswith('.gz') else open
    sums, procs = {}, set()
    with opener(path, 'rb') as f:
        for _, elem in ET.iterparse(f, events=('end',)):
            if elem.tag != 'spline':
                continue
            fields = dict(item.split(':', 1) for item in elem.get('name').split('/')[-1].split(';') if ':' in item)
            nu, tgt, proc = int(fields.get('nu', 0)), fields.get('tgt'), fields.get('proc', '')
            if nu in FLAVORS and tgt in targets and not any(s in proc for s in SKIP):
                kind = {'Weak[CC]': 'cc', 'Weak[NC]': 'nc'}.get(proc.split(',')[0])
                if kind is None:
                    raise SystemExit(f'unknown process {proc!r} in {elem.get("name")}')
                procs.add((kind, proc))
                knots = np.array([(float(k.findtext('E')), float(k.findtext('xsec'))) for k in elem.iter('knot')])
                if knots[-1, 0] < 10 ** log_e[-1]:
                    raise SystemExit(f'the grid of log10 E goes beyond the knots of {elem.get("name")}')
                key = (kind, nu, tgt)
                sums[key] = sums.get(key, 0.0) + np.interp(log_e, np.log10(knots[:, 0]), knots[:, 1], left=0.0)
            elem.clear()
    return sums, procs


def build(spline_file: Path, out: Path, targets: dict, nucleons: int, material: str, script: str) -> None:
    """Write xsec_cc.dat and xsec_nc.dat of material per nucleon, sum_t n_t sigma_t / nucleons, into out.
    targets: {GENIE target code: nuclei per molecule}; script: the name written in the header."""
    log_e = np.loadtxt(ARGON / 'xsec_cc.dat')[:, 0]
    sums, procs = read_splines(spline_file, log_e, targets)
    missing = [(k, nu, t) for k in ('cc', 'nc') for nu in FLAVORS for t in targets if (k, nu, t) not in sums]
    if missing:
        raise SystemExit(f'splines missing for (kind, nu, target): {missing}')
    print('processes summed:', ', '.join(sorted(f'{k}:{p}' for k, p in procs)))
    out.mkdir(parents=True, exist_ok=True)
    tables = {}
    for kind in ('cc', 'nc'):
        table = np.zeros((log_e.size, 7))
        table[:, 0] = log_e
        for nu, col in FLAVORS.items():
            sigma = sum(n * sums[kind, nu, t] for t, n in targets.items()) / nucleons * GEV2_TO_1E38CM2
            table[:, col] = sigma / 10 ** log_e
        tables[kind] = table
        np.savetxt(out / f'xsec_{kind}.dat', table, fmt='%.6g',
                   header=f'GENIE 2.12.10 DefaultPlusValenciaMEC, {material} per nucleon, {kind.upper()}: '
                          'log10(E/GeV), sigma/E [1e-38 cm^2/GeV] of nu_e nu_mu nu_tau nubar_e nubar_mu nubar_tau '
                          f'({script})')

    cc = tables['cc']

    def at(table, e, col):
        return np.interp(np.log10(e), table[:, 0], table[:, col])

    print(f'CC sigma/E at 50 GeV [1e-38 cm^2/GeV per nucleon]: nu_mu {at(cc, 50, 2):.3f}, '
          f'nubar_mu {at(cc, 50, 5):.3f} (isoscalar ~0.68, ~0.33)')
    argon = np.loadtxt(ARGON / 'xsec_cc.dat')
    for e in (0.4, 0.6, 1.0, 2.0):
        print(f'E = {e} GeV  {material}/argon sigma/E: nu_mu CC {at(cc, e, 2) / at(argon, e, 2):.3f}, '
              f'nubar_mu CC {at(cc, e, 5) / at(argon, e, 5):.3f}')
    if not (0.6 < at(cc, 50, 2) < 0.75 and 0.28 < at(cc, 50, 5) < 0.38):
        raise SystemExit('sanity check failed: CC sigma/E at 50 GeV is not isoscalar-like')


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    build(Path(sys.argv[1]), Path(sys.argv[2]), TARGETS, NUCLEONS, 'water', 'scripts/make_water_xsec.py')

if __name__ == '__main__':
    main()
