"""Shared style for the report figures: sized for a ~6 in text width so that fonts print at ~9-10 pt,
and clean outline overlays of 3-class masks (0 other, 1 clad, 2 remelt) on micrographs."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FULL = 7.0  # width (in) of a full-text-width figure
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 11.5, 'axes.labelsize': 11, 'legend.fontsize': 10,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'figure.titlesize': 12,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 220,
                     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04})
CLAD, REMELT, SURF = '#39e639', '#ff3030', '#00e5ff'


def surface_line(m):
    """Straight line through the clad/remelt interface (as in unet_seg.measure); None if no clad."""
    xs, ys = [], []
    for x in np.flatnonzero(((m == 1) | (m == 2)).any(0)):
        a, b = np.flatnonzero(m[:, x] == 1), np.flatnonzero(m[:, x] == 2)
        if len(a):
            xs.append(x); ys.append((a.max() + b.min()) / 2 if len(b) else a.max() + .5)
    if len(xs) < 5:
        return None
    k, c = np.polyfit(xs, ys, 1)
    return lambda x: k * np.asarray(x) + c


def crop_box(masks, shape, pad=0.35, aspect=4 / 3):
    """Window around the union of the masks' foreground, padded and widened to a fixed aspect ratio."""
    h, w = shape
    fg = np.zeros(shape, bool)
    for m in masks:
        fg |= m > 0
    if not fg.any():
        return 0, h, 0, w
    ys, xs = np.nonzero(fg)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    bw, bh = (np.ptp(xs) + 1) * (1 + 2 * pad), (np.ptp(ys) + 1) * (1 + 2 * pad)
    bw = min(max(bw, bh * aspect), w); bh = min(bw / aspect, h)
    x0 = int(np.clip(cx - bw / 2, 0, w - bw)); y0 = int(np.clip(cy - bh / 2, 0, h - bh))
    return y0, y0 + int(bh), x0, x0 + int(bw)


def outline(ax, g, m, box=None, surf=True, lw=1.6):
    """Grey image with the clad (green) and remelt (red) outlines and the substrate surface (cyan)."""
    y0, y1, x0, x1 = box or (0, g.shape[0], 0, g.shape[1])
    ax.imshow(g[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=255 if g.max() > 1 else 1)
    mm = m[y0:y1, x0:x1]
    for c, col in ((1, CLAD), (2, REMELT)):
        if (mm == c).any():
            ax.contour((mm == c).astype(float), [0.5], colors=col, linewidths=lw)
    line = surface_line(m) if surf else None
    if line is not None:
        ax.plot([0, x1 - x0 - 1], [line(x0) - y0, line(x1 - 1) - y0], color=SURF, lw=lw * .8, ls='--')
    ax.set_xlim(-.5, x1 - x0 - .5); ax.set_ylim(y1 - y0 - .5, -.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fill(ax, g, m, box=None, alpha=.65):
    """Grey image with clad and remelt filled semi-transparently (for network predictions)."""
    y0, y1, x0, x1 = box or (0, g.shape[0], 0, g.shape[1])
    g = g.astype(float) / (255 if g.max() > 1 else 1)
    rgb = np.stack([g] * 3, -1) * .85
    for c, col in ((1, (0.2, 0.85, 0.2)), (2, (0.95, 0.2, 0.2))):
        rgb[m == c] = rgb[m == c] * (1 - alpha) + np.array(col) * alpha
    ax.imshow(np.clip(rgb[y0:y1, x0:x1], 0, 1)); ax.axis('off')


def mask_legend(fig, surf=True, filled=False, **kw):
    """Shared colour key for mask figures, placed by the caller (loc / bbox_to_anchor in kw)."""
    if filled:
        from matplotlib.patches import Patch
        h = [Patch(fc=(0.2, 0.85, 0.2), label='clad (deposit above the surface)'),
             Patch(fc=(0.95, 0.2, 0.2), label='remelt (melted substrate)')]
    else:
        h = [Line2D([], [], color=CLAD, lw=2.5, label='clad outline'),
             Line2D([], [], color=REMELT, lw=2.5, label='fusion line (remelt zone)')]
        if surf:
            h.append(Line2D([], [], color=SURF, lw=2.5, ls='--', label='substrate surface'))
    return fig.legend(handles=h, ncol=len(h), frameon=False, **kw)
