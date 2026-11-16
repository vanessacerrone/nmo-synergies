"""The matplotlib style of the paper, the colors of the experiments, and the printed widths of the figures.

Every figure is saved at a width that the paper scales down by the same factor SCALE = 0.553, so that the
label sizes of STYLE (16, 15, 13 pt) print as 8.9, 8.3, 7.2 pt in all of them. The figures use no transparency
(JHEP author manual): a light shade is the opaque color that the transparent one shows over white (`opaque`),
and overlapping bands are drawn with the colors their transparent layers would give (`bands`).
"""
import shutil
import warnings
from pathlib import Path

import matplotlib as mpl
import numpy as np
from cycler import cycler
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

from nmosyn import paths

STYLE = {
    'axes.prop_cycle': cycler('color', ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000', '#40B0A6',
                                        '#006CD1', '#B00068', '#994F00']),
    'figure.figsize': (8, 6),
    'xtick.direction': 'in',
    'xtick.major.size': 8,
    'xtick.major.width': 1.0,
    'xtick.minor.size': 3,
    'xtick.minor.width': 0.8,
    'xtick.minor.visible': True,
    'xtick.top': True,
    'xtick.bottom': True,
    'ytick.direction': 'in',
    'ytick.major.size': 8,
    'ytick.major.width': 1.0,
    'ytick.minor.size': 3,
    'ytick.minor.width': 0.8,
    'ytick.minor.visible': True,
    'ytick.right': True,
    'ytick.left': True,
    'axes.labelsize': 16,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'axes.titlesize': 15,
    'legend.fontsize': 13,
    'legend.title_fontsize': 13,
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.2,
    'axes.facecolor': 'white',
    'axes.edgecolor': '.15',
    'legend.frameon': False,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'savefig.dpi': 400,
    'font.family': 'serif',
    'font.size': 13.0,
    'mathtext.fontset': 'dejavuserif',
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage[T1]{fontenc} \usepackage{amsmath} \usepackage{amssymb}',
}

# IBM color-blind-safe palette, plus forestgreen for the short-baseline reactor. A channel keeps its color in
# every figure; a variant of one experiment (resolution, energy window, octant) changes the line style or the
# marker, not the color.
COLORS = {
    'sbr': 'forestgreen',           # short-baseline reactor, Daya Bay
    'juno': '#006CD1',
    'nova': '#FFB000',              # NOvA and the generic 810 km beam
    'nova_app': '#B07800',          # NOvA disappearance + appearance, dark gold line over the gold bands
    't2k': '#FE6100',
    'dc_numu': '#DC267F',           # DeepCore numu disappearance
    'dc_all': '#785EF0',            # DeepCore, all channels
    'orca': '#40B0A6',
    'blue': '#648FFF',
    'purple': '#785EF0',
    'magenta': '#DC267F',
    'orange': '#FE6100',
    'gold': '#FFB000',
    'teal': '#40B0A6',
    'darkblue': '#006CD1',
    'darkmagenta': '#B00068',
    'pink': '#D35FB7',
    'brown': '#994F00',
    'green': 'forestgreen',
}

# Saved widths [in]: the paper includes the figures at \textwidth, at 0.66\textwidth (one panel), at
# 0.49\textwidth (two panels next to each other, each saved alone), and at 0.8\textwidth.
TEXTWIDTH = 430.20639 / 72.27  # JHEP \textwidth [in], 0.72 of the A4 width
W_WIDE = 777.70838 / 72.27
W_NARROW = 0.66 * W_WIDE
W_PAIR = 0.49 * W_WIDE
W_BARS = 0.8 * W_WIDE
SCALE = TEXTWIDTH / W_WIDE  # printed size / saved size, the same for every figure


def opaque(color, alpha: float) -> tuple[float, float, float]:
    """The opaque color that color with transparency alpha shows over white."""
    return tuple(float(alpha * c + 1 - alpha) for c in mpl.colors.to_rgb(color))


def over(top, alpha_top: float, bottom, alpha_bottom: float = 1.0) -> tuple[float, float, float]:
    """The opaque color of top with transparency alpha_top over bottom with alpha_bottom, over white."""
    below = mpl.colors.to_rgb(opaque(bottom, alpha_bottom))
    return tuple(float(alpha_top * t + (1 - alpha_top) * b) for t, b in zip(mpl.colors.to_rgb(top), below))


def bands(ax, x, specs) -> list:
    """Bands over the same x, drawn in order without transparency, as transparent ones would look over white.

    specs: dicts with lo, hi, color, shade (the transparency the band stands for), and optionally lw (outline
    width), hatch, zorder (default 1, below the ticks of the axes), and label. Each band is filled with
    opaque(color, shade), outlined with the color of a transparent outline over its own face, and hatched in its
    full color. Where two bands overlap, the overlap is
    filled with the color of the later one over the earlier one, and the outline and hatch of the earlier band are
    drawn again inside the later one with the color they show through it. Returns the legend handles, one per
    band.
    """
    handles, drawn = [], []
    for k, b in enumerate(specs):
        z = b.get('zorder', 1) + 0.01 * k
        alpha, color, lw, hatch = b['shade'], b['color'], b.get('lw', 0.0), b.get('hatch')
        edge = opaque(color, 1 - (1 - alpha) ** 2)
        fill = opaque(color, alpha)
        face = ax.fill_between(x, b['lo'], b['hi'], facecolor=fill, edgecolor=edge if lw else fill, lw=lw, zorder=z,
                               label=b.get('label'))
        path = face.get_paths()[0]
        for prev in drawn:
            lo, hi = np.maximum(prev['lo'], b['lo']), np.minimum(prev['hi'], b['hi'])
            if np.any(hi > lo):
                ax.fill_between(x, lo, hi, where=hi > lo, interpolate=True, lw=0, zorder=z + 0.001,
                                facecolor=over(color, alpha, prev['color'], prev['shade']))
            if prev.get('hatch'):
                inner = ax.fill_between(x, prev['lo'], prev['hi'], facecolor='none', lw=0, hatch=prev['hatch'],
                                        edgecolor=over(color, alpha, prev['color']), zorder=z + 0.002)
                inner.set_clip_path(path, ax.transData)
            if prev.get('lw'):
                inner = ax.add_patch(mpl.patches.PathPatch(prev['path'], facecolor='none', lw=prev['lw'],
                                                           edgecolor=over(color, alpha, prev['edge']),
                                                           zorder=z + 0.003))
                inner.set_clip_path(path, ax.transData)
        if hatch:
            ax.fill_between(x, b['lo'], b['hi'], facecolor='none', edgecolor=color, hatch=hatch, lw=0,
                            zorder=z + 0.004)
        if lw:
            ax.add_patch(mpl.patches.PathPatch(path, facecolor='none', edgecolor=edge, lw=lw, zorder=z + 0.005))
        drawn.append(dict(b, edge=edge, path=path))
        patch = mpl.patches.Patch(facecolor=fill, edgecolor=edge if lw else fill, lw=lw)
        handles.append((patch, mpl.patches.Patch(facecolor='none', edgecolor=color, hatch=hatch, lw=0))
                       if hatch else patch)
    return handles


def use() -> None:
    """Apply STYLE. Text is set by LaTeX when latex and dvipng are installed, by mathtext otherwise."""
    mpl.rcParams.update(STYLE)
    if not (shutil.which('latex') and shutil.which('dvipng')):
        mpl.rcParams['text.usetex'] = False
        warnings.warn('latex or dvipng not found: text drawn with mathtext, unlike the paper', stacklevel=2)


def _stretch(lo: float, hi: float, size: float | None, anchor: str, axis: str) -> tuple[float, float]:
    """Interval [lo, hi] grown to size: about its middle ('center') or downward from hi ('top')."""
    if size is None:
        return lo, hi
    if hi - lo > size + 1e-3:
        warnings.warn(f'figure content is {hi - lo:.3f} in along {axis}, more than the {size:.3f} in of the page',
                      stacklevel=4)
        return lo, hi
    if anchor == 'center':
        extra = 0.5 * (size - (hi - lo))
        return lo - extra, hi + extra
    return hi - size, hi


def _page(fig: Figure, width: float | None, height: float | None) -> Bbox:
    """Page [in] around the content: its tight box padded by savefig.pad_inches, centered horizontally in
    `width` and aligned at the top of `height`, so that two panels printed next to each other share their axes
    heights."""
    fig.draw_without_rendering()
    inches = fig.dpi_scale_trans.inverted()
    content = [fig.get_tightbbox()]
    for ax in fig.get_axes():           # get_tightbbox leaves out an axis label that overhangs its axis
        for axis in (ax.xaxis, ax.yaxis):
            if ax.get_visible() and axis.label.get_visible() and axis.label.get_text():
                content.append(axis.label.get_window_extent().transformed(inches))
    box = Bbox.union(content).padded(mpl.rcParams['savefig.pad_inches'])
    x0, x1 = _stretch(box.x0, box.x1, width, 'center', 'x')
    y0, y1 = _stretch(box.y0, box.y1, height, 'top', 'y')
    return Bbox([[x0, y0], [x1, y1]])


def save(fig: Figure, name: str, width: float | None = None, height: float | None = None) -> Path:
    """Write FIGURES/<name>.pdf without axes grids, on a page `width` x `height` [in], tight where None.

    The background of each axes gets an edge of its own color (not drawn, width 0): matplotlib writes an edge of
    color 'none' as a transparent stroke, and the figures use no transparency.
    """
    for ax in fig.get_axes():
        ax.grid(False, which='both')
        ax.patch.set_edgecolor(ax.patch.get_facecolor())
    out = paths.FIGURES / f'{name}.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches=_page(fig, width, height), pad_inches=0, metadata={'CreationDate': None})
    return out
