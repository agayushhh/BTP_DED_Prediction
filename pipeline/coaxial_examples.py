"""Coaxial-camera example frames for the report: (top) sections at low/medium/high energy per unit length,
(bottom) two repeats of one setting whose penetration differs, with the fixed-level isotherm contours used as
features. Frames are the middle frame of each section's registered window (same logic as coaxial_features).
Usage: python -I pipeline/coaxial_examples.py <BTP root> <out.png>"""
import sys, os
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coaxial_features import track_frames, frame_feats, snap_edges, LEVELS, PX_PER_MM
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

root, out = sys.argv[1], sys.argv[2]
coax = os.path.join(root, 'data/optomec/coaxial')
sec = pd.read_csv(os.path.join(root, 'results/optomec/optomec_sections_mm.csv'))
cx = pd.read_csv(os.path.join(root, 'results/optomec/coaxial_section_features.csv'))
m = sec.merge(cx, on=['id', 'sec'])


def section_frame(tid, s):
    """Middle frame of section s of track tid after 1/V-prior + snap registration."""
    t = sec[sec.id == tid].sort_values('sec')
    imgs = [img for _, img in track_frames(os.path.join(coax, f'{tid}_denoise.zip'))]
    F = [frame_feats(i) for i in imgs]
    on = [i for i, f in enumerate(F) if f is not None]
    imgs, F = imgs[on[0]:on[-1] + 1], F[on[0]:on[-1] + 1]
    edges = snap_edges(F, t.V.values)
    k = list(t.sec).index(s)
    j = int((edges[k] + edges[k + 1]) / 2)
    while F[j] is None:
        j += 1
    return imgs[j]


def show(ax, img, r, title):
    g = cv2.GaussianBlur(img, (5, 5), 0)
    ys, xs = np.nonzero(g > 30)
    cy, cx_ = int(np.median(ys)), int(np.median(xs))
    half = int(0.8 * PX_PER_MM)
    y0, x0 = max(cy - half, 0), max(cx_ - half, 0)
    crop = g[y0:y0 + 2 * half, x0:x0 + 2 * half]
    ax.imshow(crop, cmap='gray', vmin=0, vmax=255)
    cs = ax.contour(crop, levels=list(LEVELS), colors=['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c'], linewidths=1)
    ax.plot([8, 8 + PX_PER_MM / 2], [crop.shape[0] - 8] * 2, 'w-', lw=3)
    ax.text(8, crop.shape[0] - 14, '0.5 mm', color='w', fontsize=8)
    ax.axis('off')
    ax.set_title(title, fontsize=9.5)
    ax.text(0.5, -0.03, f'H {r.H:.2f}  W {r.W:.2f}  D {r.D:.2f} mm\nglow area $A_{{30}}$ = {r.cx_A30:.2f} mm$^2$',
            transform=ax.transAxes, ha='center', va='top', fontsize=9)
    return cs


m = m[m.H >= 0.1]  # flush tracks are outside the scope of this work
# top row: narrow / medium / wide beads: the camera sees the visible bead
q = m.sort_values('W')
top = [q.iloc[int(f * (len(q) - 1))] for f in (0.1, 0.5, 0.9)]
# bottom row: same setting, different tracks, nearly equal glow area (< 3 %) but the largest difference in D
best, gap = None, 0
for c, g in m.groupby('cond'):
    rows = list(g.itertuples())
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if a.id != b.id and abs(a.cx_A30 - b.cx_A30) < 0.03 * (a.cx_A30 + b.cx_A30) / 2 and abs(a.D - b.D) > gap:
                best, gap = sorted([a, b], key=lambda r: r.D), abs(a.D - b.D)
fig, axs = plt.subplots(2, 3, figsize=(12, 9.6))
for ax, r in zip(axs[0], top):
    show(ax, section_frame(r.id, r.sec), r, f'{r.id[-4:]}{r.sec}: P {r.P:.0f} W, V {r.V:.0f} mm/min,\nF {r.F:.1f} g/min')
for ax, r, lab in zip(axs[1], best, ('repeat 1', 'repeat 2')):
    cs = show(ax, section_frame(r.id, r.sec), r, f'{lab}: {r.id[-4:]}{r.sec}, P {r.P:.0f} W, V {r.V:.0f} mm/min,\nF {r.F:.1f} g/min')
ax = axs[1, 2]; ax.axis('off')
for lev, col in zip(LEVELS, ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c']):
    ax.plot([], [], color=col if col != '#ffffbf' else '#e6d600', lw=2, label=f'grey level {lev}')
ax.legend(loc='center', title='isotherm-like contours', fontsize=10)
fig.tight_layout(h_pad=2.5, rect=(0, 0, 1, 0.97))
fig.savefig(out, dpi=125)
print('top', [f'{r.id}{r.sec}' for r in top], 'repeats', [f'{r.id}{r.sec} D={r.D:.3f} A30={r.cx_A30:.3f}' for r in best])
