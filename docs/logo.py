"""The logo of the repository: docs/logo.svg (dark text, for light themes), docs/logo_dark.svg (light text, for
dark themes), and docs/logo_symbol.svg (the symbol alone, for icons).

The symbol is the picture of the paper drawn in cells: the wrong-ordering solutions of a reactor, an accelerator,
and an atmospheric experiment are parallel lines of slope -1 in the plane of the two splittings, the reactor line
apart from the other two. The reactor gives one line; the accelerator and the atmospheric lines spread over their
neighboring cells, as their solutions are bands over the unknown delta_CP and theta23. The name is set in DejaVu
Sans (shipped with matplotlib and outlined here), nmo in bold, the hyphen in the colors of the three lines.
Run from the repository root: python -m docs.logo
"""
from pathlib import Path

from matplotlib import get_data_path
from matplotlib.font_manager import FontProperties
from matplotlib.ft2font import FT2Font
from matplotlib.path import Path as MPath
from matplotlib.textpath import TextPath

from nmosyn.paths import ROOT
from nmosyn.style import COLORS

FONTS = Path(get_data_path()) / 'fonts' / 'ttf'
REACTOR, BEAM, ATM = COLORS['juno'], COLORS['nova'], COLORS['dc_all']
LINES = ((3, REACTOR), (-1, BEAM), (-2, ATM))                 # the cells with row - column = d
SPREAD = ((0, BEAM), (-3, ATM))   # lighter neighbors of the accelerator and atmospheric lines, outside the pair
THEMES = {'light': {'tile': '#F3F5F8', 'axis': '#57606A', 'ink': '#1F2328', 'blend': '#FFFFFF', 'spread': 0.72},
          'dark': {'tile': '#1B2230', 'axis': '#8A96AB', 'ink': '#E6EDF3', 'blend': '#1B2230', 'spread': 0.55}}
N, PITCH, X0, Y0 = 9, 14, 79, 57                              # cell grid inside the axis corner, in px
SIZE, X_TEXT = 104, 296                                       # type size and start of the name, in px


def blend(color: str, toward: str, a: float) -> str:
    c, t = ([int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in (color, toward))
    return '#%02X%02X%02X' % tuple(round(x + (y - x) * a) for x, y in zip(c, t))


def rounded(x: float, y: float, w: float, h: float, r: float) -> str:
    return (f'M{x + r:.2f},{y:.2f} H{x + w - r:.2f} A{r},{r} 0 0 1 {x + w:.2f},{y + r:.2f} V{y + h - r:.2f} '
            f'A{r},{r} 0 0 1 {x + w - r:.2f},{y + h:.2f} H{x + r:.2f} A{r},{r} 0 0 1 {x:.2f},{y + h - r:.2f} '
            f'V{y + r:.2f} A{r},{r} 0 0 1 {x + r:.2f},{y:.2f} Z')


def symbol(theme: dict) -> list[str]:
    """Tile, axis corner, and the three lines in cells, on a 256 x 256 canvas."""
    out = [f'<path fill="{theme["tile"]}" d="{rounded(8, 8, 240, 240, 52)}"/>',
           f'<path fill="{theme["axis"]}" d="{rounded(52, 48, 16, 160, 8)} {rounded(52, 192, 156, 16, 8)}"/>']
    cells = {}
    for d, color in LINES:
        cells.update({(i, i + d): color for i in range(N) if 0 <= i + d < N})
    for d, color in SPREAD:
        for i in range(N):
            if 0 <= i + d < N:
                cells.setdefault((i, i + d), blend(color, theme['blend'], theme['spread']))
    by_color = {}
    for (i, j), color in cells.items():
        by_color.setdefault(color, []).append(rounded(X0 + i * PITCH + 1.3, Y0 + j * PITCH + 1.3, PITCH - 2.6,
                                                      PITCH - 2.6, 3))
    return out + [f'<path fill="{c}" d="{" ".join(d)}"/>' for c, d in by_color.items()]


def outline(text: str, font: str, x: float, baseline: float, tracking: float = 0.0) -> tuple[str, float]:
    """Path data of text set in a font of FONTS, with its advances plus tracking (in em), and its width."""
    ft = FT2Font(str(FONTS / font))
    ft.set_size(SIZE, 72)
    prop, parts, pen = FontProperties(fname=str(FONTS / font)), [], 0.0
    for ch in text:
        for verts, code in TextPath((0, 0), ch, size=SIZE, prop=prop).iter_segments():
            pts = ' '.join(f'{x + pen + verts[k]:.2f},{baseline - verts[k + 1]:.2f}' for k in range(0, len(verts), 2))
            parts.append('Z' if code == MPath.CLOSEPOLY else
                         {MPath.MOVETO: 'M', MPath.LINETO: 'L', MPath.CURVE3: 'Q', MPath.CURVE4: 'C'}[code] + pts)
        pen += ft.load_char(ord(ch)).linearHoriAdvance / 65536 + tracking * SIZE
    return ' '.join(parts), pen - tracking * SIZE


def name(theme: dict) -> tuple[list[str], float]:
    """nmo (bold), the hyphen in three colors, synergies (book), centered on the symbol; and the width."""
    baseline = 162
    nmo, w_nmo = outline('nmo', 'DejaVuSans-Bold.ttf', X_TEXT, baseline, -0.02)
    x = X_TEXT + w_nmo + 0.05 * SIZE
    third = 0.40 * SIZE / 3
    bar = [f'<rect x="{x + k * third:.2f}" y="{baseline - 0.37 * SIZE:.2f}" width="{third + 0.4:.2f}" '
           f'height="{0.12 * SIZE:.2f}" fill="{c}"/>' for k, c in enumerate((REACTOR, BEAM, ATM))]
    x += 3 * third + 0.05 * SIZE
    rest, w_rest = outline('synergies', 'DejaVuSans.ttf', x, baseline, -0.01)
    return [f'<path fill="{theme["ink"]}" d="{nmo} {rest}"/>'] + bar, x + w_rest - X_TEXT


def write(out: Path, elements: list[str], width: float) -> None:
    out.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} 256" width="{width:.0f}" '
                   f'height="256">\n<title>nmo-synergies</title>\n' + '\n'.join(elements) + '\n</svg>\n')


if __name__ == '__main__':
    for key, file in (('light', 'logo.svg'), ('dark', 'logo_dark.svg')):
        words, width = name(THEMES[key])
        write(ROOT / 'docs' / file, symbol(THEMES[key]) + words, X_TEXT + width + 24)
    write(ROOT / 'docs' / 'logo_symbol.svg', symbol(THEMES['light']), 256)
