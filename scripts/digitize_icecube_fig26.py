"""Read the published 90% CL contour of the IceCube DeepCore golden event sample from the vector paths of Fig. 26
of PRD 108, 012014 (arXiv:2304.12236v2, page 27) and write data/icecube_prd108_fig26_contour90.json.

The figure is vector graphics, so the contour is read exactly rather than traced by eye. The page is converted to
SVG and its text boxes are listed with poppler (pdftocairo, pdftotext). Both axes are calibrated on their tick
marks: each tick is paired with the number printed next to it and a straight line is fitted through the pairs.
The contour of the golden sample is the closed blue path of the figure, and its best fit the center of the blue
cross. Every point is converted with the calibration.

    python scripts/download_data.py icecube_prd108     # the PDF, checked against its SHA-256
    python scripts/digitize_icecube_fig26.py

Requires numpy and the poppler command-line tools. Run from the repository root.
"""
import hashlib
import json
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from download_data import EXTERNAL, SOURCES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'icecube_prd108_fig26_contour90.json'
NAME = '2304.12236v2.pdf'
PDF = EXTERNAL / 'icecube_prd108' / NAME
PAGE = 27
BOX = (330.0, 490.0, 550.0, 650.0)        # Fig. 26 on the page [pt]: x_min, y_min, x_max, y_max (y downward)
BLUE = 'rgb(11.799622%, 56.498718%, 100%)'   # 'DeepCore 8 years - golden event sample'
BLACK = 'rgb(0%, 0%, 0%)'
SVG = '{http://www.w3.org/2000/svg}'


def matrix(node):
    m = re.match(r'matrix\(([^)]*)\)', node.get('transform', ''))
    return [float(v) for v in re.split(r'[ ,]+', m.group(1).strip())] if m else [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def compose(p, q):
    """The affine map p applied after q, as SVG matrices (a, b, c, d, e, f)."""
    a, b, c, d, e, f = p
    A, B, C, D, E, F = q
    return [a * A + c * B, b * A + d * B, a * C + c * D, b * C + d * D, a * E + c * F + e, b * E + d * F + f]


def vertices(d, m):
    """End points of the segments of an SVG path (M, L, C, Z) in page coordinates."""
    tokens = re.findall(r'[MLCZ]|-?\d*\.?\d+(?:e-?\d+)?', d)
    out, i, command = [], 0, None
    while i < len(tokens):
        if tokens[i] in 'MLCZ':
            command = tokens[i]
            i += 1
            continue
        n = 6 if command == 'C' else 2
        x, y = (float(v) for v in tokens[i + n - 2:i + n])
        out.append((m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]))
        i += n
    return np.array(out)


def paths(svg, color):
    """Vertices of every path stroked with color that lies inside BOX."""
    found = []

    def walk(node, m):
        m = compose(m, matrix(node))
        if node.tag == SVG + 'path' and node.get('stroke') == color:
            v = vertices(node.get('d'), m)
            if len(v) and np.all((v[:, 0] > BOX[0]) & (v[:, 0] < BOX[2]) & (v[:, 1] > BOX[1]) & (v[:, 1] < BOX[3])):
                found.append(v)
        for child in node:
            if child.tag != SVG + 'defs':
                walk(child, m)

    walk(ET.parse(svg).getroot(), [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    return found


def numbers(html):
    """Printed numbers inside BOX: (value, x center, y center) [pt]."""
    out = []
    for w in re.finditer(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>',
                         html):
        x0, y0, x1, y1 = (float(w.group(k)) for k in range(1, 5))
        if re.fullmatch(r'\d+\.\d+', w.group(5)) and BOX[0] < x0 and x1 < BOX[2] and BOX[1] < y0 and y1 < BOX[3]:
            out.append((float(w.group(5)), 0.5 * (x0 + x1), 0.5 * (y0 + y1)))
    return out


def calibration(ticks, labels, along):
    """Linear map from page coordinate to value: each tick paired with the printed number nearest to it along the
    axis (along = 0 for x, 1 for y); returns the fit and the largest residual over the ticks."""
    pairs = []
    for t in ticks:
        value, *center = min(labels, key=lambda lab: abs(lab[1 + along] - t))
        assert abs(center[along] - t) < 2.0, (t, value, center)
        pairs.append((t, value))
    t, v = np.array(pairs).T
    assert len(set(v)) == len(v) >= 5, pairs
    fit = np.polyfit(t, v, 1)
    return fit, float(np.max(np.abs(np.polyval(fit, t) - v)))


def main() -> None:
    digest = hashlib.sha256(PDF.read_bytes()).hexdigest()
    assert digest == SOURCES['icecube_prd108'].sha256[NAME], 'run scripts/download_data.py icecube_prd108 first'
    with tempfile.TemporaryDirectory() as tmp:
        svg, html = Path(tmp) / 'page.svg', Path(tmp) / 'page.html'
        subprocess.run(['pdftocairo', '-svg', '-f', str(PAGE), '-l', str(PAGE), str(PDF), str(svg)], check=True)
        subprocess.run(['pdftotext', '-bbox', '-f', str(PAGE), '-l', str(PAGE), str(PDF), str(html)], check=True)
        blue, black, printed = paths(svg, BLUE), paths(svg, BLACK), numbers(html.read_text())

    # tick marks: short straight black segments, vertical on the lower axis and horizontal on the left one
    marks = [v for v in black if len(v) == 2 and np.hypot(*(v[1] - v[0])) < 3.0]
    x_ticks = sorted({round(v[0, 0], 3) for v in marks if abs(v[0, 0] - v[1, 0]) < 1e-6})
    y_ticks = sorted({round(v[0, 1], 3) for v in marks if abs(v[0, 1] - v[1, 1]) < 1e-6})
    x_axis = max(v[:, 1].max() for v in marks if abs(v[0, 0] - v[1, 0]) < 1e-6)
    y_axis = min(v[:, 0].min() for v in marks if abs(v[0, 1] - v[1, 1]) < 1e-6)
    fx, res_x = calibration(x_ticks, [n for n in printed if n[2] > x_axis], along=0)
    fy, res_y = calibration(y_ticks, [n for n in printed if n[1] < y_axis], along=1)

    contour = max(blue, key=len)
    assert np.allclose(contour[0], contour[-1]), 'the contour is not closed'
    cross = next(v for v in blue if len(v) == 4 and np.ptp(v, axis=0).max() < 10.0)
    s23, dm32 = np.polyval(fx, contour[:, 0]), np.polyval(fy, contour[:, 1])
    best = float(np.polyval(fx, cross[:, 0].mean())), float(np.polyval(fy, cross[:, 1].mean()))

    out = {
        'what': '90% C.L. allowed region (Wilks, two parameters) of sin^2 theta23 and Dm32 in the normal ordering, '
                'and its best fit, from the IceCube DeepCore 8-year golden event sample',
        'source': {
            'reference': 'IceCube Collaboration, R. Abbasi et al., Measurement of atmospheric neutrino mixing with '
                         'improved IceCube DeepCore calibration and data processing, Phys. Rev. D 108, 012014 '
                         '(2023), doi:10.1103/PhysRevD.108.012014',
            'arxiv': '2304.12236v2', 'license': 'CC BY 4.0', 'pdf_sha256': digest,
            'figure': f'Fig. 26, page {PAGE}: blue curve "DeepCore 8 years - golden event sample" and its blue cross',
        },
        'method': 'vector paths of the figure read by scripts/digitize_icecube_fig26.py; both axes calibrated on '
                  'their tick marks and printed tick values',
        'calibration': {'ticks_sin2_theta23': len(x_ticks), 'ticks_dm32': len(y_ticks),
                        'largest_tick_residual': {'sin2_theta23': round(res_x, 7), 'dm32': round(res_y, 7)}},
        'units': {'sin2_theta23': '1', 'dm32': '1e-3 eV^2'},
        'best_fit': {'sin2_theta23': round(best[0], 5), 'dm32': round(best[1], 5)},
        'contour': [[round(float(a), 5), round(float(b), 5)] for a, b in zip(s23, dm32)],
    }
    text = json.dumps(out, indent=1)
    text = re.sub(r'\[\n\s+(-?[\d.]+),\n\s+(-?[\d.]+)\n\s+\]', r'[\1, \2]', text)   # one point per line
    OUT.write_text(text + '\n')
    print(f'{OUT.relative_to(ROOT)}: {len(s23)} points, best fit {best[0]:.5f}, {best[1]:.5f}; '
          f'tick residuals {res_x:.1e}, {res_y:.1e}')


if __name__ == '__main__':
    main()
