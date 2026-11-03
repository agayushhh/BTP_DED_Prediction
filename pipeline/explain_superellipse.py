"""Visual explainer: clad outline, bead centre and superellipse fit, step by step (mirrors run_one in
segment_optomec.py; handles both mount types).
Usage: python -I pipeline/explain_superellipse.py <BTP root> <out.png> [image id, default DED_SS316L_0098D] [--en]"""
import sys, os
import numpy as np, pandas as pd, cv2
from scipy.ndimage import binary_fill_holes
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

EN = '--en' in sys.argv  # English labels for the report
sys.argv = [a for a in sys.argv if a != '--en']
tr = lambda en, hi: en if EN else hi
root, out = sys.argv[1], sys.argv[2]
key = sys.argv[3] if len(sys.argv) > 3 else 'DED_SS316L_0098D'
sys.path.insert(0, os.path.join(root, 'pipeline'))
import segment_optomec as so
PX = so.PX

g0 = np.array(Image.open(os.path.join(root, 'data/optomec/micro_tif', key + '.tif')).convert('L').resize((640, 480), Image.BILINEAR), np.uint8)
h, w = g0.shape
g = cv2.GaussianBlur(g0, (0, 0), 4)
yy = np.arange(h)[:, None]
bright_mount = g0[:15].mean() > 100
if bright_mount:  # smooth mount vs scratched metal: local standard deviation (texture), as in run_one
    f = g0.astype(float)
    mu = cv2.GaussianBlur(f, (0, 0), 2)
    sd = cv2.GaussianBlur(np.sqrt(np.maximum(cv2.GaussianBlur(f * f, (0, 0), 2) - mu * mu, 0)), (0, 0), 2)
    cue = np.clip(sd * 255 / max(sd.max(), 1e-6), 0, 255).astype(np.uint8)
    t1 = cv2.threshold(cue, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
else:
    cue, t1 = g, so.threshold_multiotsu(g)[0]
metal = cv2.morphologyEx((cue > t1).astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(metal)
metal = binary_fill_holes((lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])) | (yy > h - 3))
top = np.array([np.argmax(metal[:, x]) for x in range(w)], float)
line = so.fit_line(top)
above = yy < line[None, :] - 2
if bright_mount:
    body = metal
else:
    t_up = cv2.threshold(g[above].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    up = cv2.morphologyEx(((g > min(t_up, t1)) & above).astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)).astype(bool)
    body = binary_fill_holes(up | ~above)
    n, lab, st, _ = cv2.connectedComponentsWithStats(body.astype(np.uint8))
    body = lab == lab[h - 1, w // 2]
top = np.array([np.argmax(body[:, x]) for x in range(w)], float)
rise = line - top
on = rise > 0.03 * PX
peak = int(np.argmax(rise)); l = r = peak
while l > 0 and on[l - 1]: l -= 1
while r < w - 1 and on[r + 1]: r += 1
H, W = rise[peak], r - l + 1
xc0 = (l + r) / 2

gs_ = cv2.GaussianBlur(g0, (0, 0), 1.5)
bh = cv2.morphologyEx(gs_, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))).astype(float)
gm = cv2.GaussianBlur(g0, (0, 0), 4).astype(float)
t = np.linspace(0, np.pi, 120)

def curve(D, a, nexp, xc):
    cx, sy = np.cos(t), np.sin(t)
    px = xc + a * np.sign(cx) * np.abs(cx) ** (2 / nexp)
    py = D * np.abs(sy) ** (2 / nexp)
    return px, py

def score(D, a, nexp, xc):
    """Identical to the grid-search score in run_one."""
    px, py = curve(D, a, nexp, xc)
    xi = np.clip(px.astype(int), 0, w - 1)
    yi = np.clip((line[xi] + py).astype(int), 0, h - 1)
    keep = py > max(0.15 * D, 0.03 * PX)
    if keep.sum() < 10:
        return None
    def at(f, off):
        x = np.clip((xc + f * (px - xc)).astype(int), 0, w - 1)
        return gm[np.clip((line[x] + f * py + off).astype(int), 0, h - 1), x][keep]
    return bh[yi[keep], xi[keep]].mean() + 0.5 * (at(1.1, 4) - at(0.9, -4)).mean()

Ds = np.arange(15, 0.95 * (h - line.max()), 2)
best, perD = (-1e9, None), []
for D in Ds:
    bD = -1e9
    for af in (0.8, 0.9, 1.0, 1.1, 1.2):
        for nexp in (1.6, 2.0, 2.6, 3.4):
            for dx in (-0.1 * W, 0, 0.1 * W):
                s = score(D, af * W / 2, nexp, xc0 + dx)
                if s is None: continue
                bD = max(bD, s)
                if s > best[0]: best = (s, (D, af * W / 2, nexp, xc0 + dx))
    perD.append(bD)
perD = np.array(perD)
Db, ab, nb, xcb = best[1]
ref = pd.read_csv(os.path.join(root, 'results/optomec_seg/optomec_segmented.csv'))
ref = ref[(ref.id + ref.sec) == key].iloc[0]
print('grid size', len(Ds) * 5 * 4 * 3, 'best D mm', Db / PX, 'n', nb, 'a/(W/2)', ab / (W / 2), 'expert D', ref.D_ref)

from figstyle import plt, FULL  # noqa: E402  (report style; replaces the plain pyplot settings)

yc = line[int(xc0)]
X0, X1, Y0, Y1 = xc0 - 1.5 * W / 2 - 40, xc0 + 1.5 * W / 2 + 40, yc + 1.25 * Db + 20, yc - 1.3 * H - 20
crop = lambda ax: (ax.set_xlim(X0, X1), ax.set_ylim(Y0, Y1), ax.axis('off'))
fig, axs = plt.subplots(3, 2, figsize=(FULL, 8.6))
axs = axs.ravel()

ax = axs[0]; ax.imshow(g0, cmap='gray')
m = np.zeros((h, w, 4)); m[metal] = (1, 0.6, 0, 0.35); ax.imshow(m)
edge = int(w * 0.12)
for xa, xb in ((0, edge), (w - edge, w)): ax.axvspan(xa, xb, color='lime', alpha=0.15)
ax.plot(np.arange(w), line, 'c', lw=2)
ax.axis('off')
ax.set_title(tr('(a) Metal (orange) and surface line (cyan),\nfitted in the shaded outer columns',
                '1) Metal (orange), surface line (cyan)\nsirf green patti wale columns se'), fontsize=10.5)

ax = axs[1]; ax.imshow(g0, cmap='gray')
ax.plot(np.arange(l, r + 1), top[l:r + 1], 'lime', lw=2.2)
ax.plot(np.arange(w), line, 'c', lw=1.5)
for x in range(l, r + 1, 12): ax.plot([x, x], [line[x], top[x]], color='lime', lw=0.6, alpha=0.6)
ax.plot([l, r], [yc + 8, yc + 8], 'y-', lw=2.2); ax.text(xc0, yc + 30, f'W = {W / PX:.2f} mm', color='y', ha='center', fontsize=11,
                                                         bbox=dict(fc='k', alpha=0.6, lw=0))
ax.plot([peak, peak], [line[peak], top[peak]], 'm-', lw=2.2)
ax.text(X1 - 10, top[peak], f'H = {H / PX:.2f} mm', color='m', ha='right', fontsize=11, bbox=dict(fc='w', alpha=0.8, lw=0))
ax.plot(xc0, yc, 'o', color='yellow', ms=9, mec='k')
crop(ax); ax.set_title(tr('(b) Clad outline (green), bead feet,\ncentre (dot), $H$ and $W$',
                          '2) Clad outline, bead ke pair,\ncentre (dot), H aur W'), fontsize=10.5)

ax = axs[2]
for nexp, col in ((1.6, 'tab:red'), (2.0, 'tab:orange'), (2.6, 'tab:green'), (3.4, 'tab:blue')):
    px, py = curve(1, 1, nexp, 0); ax.plot(px, py, col, lw=2.2, label=f'$n$ = {nexp}')
ax.set_ylim(1.08, -0.05); ax.set_aspect('equal'); ax.axhline(0, color='k', lw=0.8)
ax.set_yticks([0, 0.5, 1]); ax.set_xticks([-1, 0, 1])
ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(.5, .9), ncol=2, frameon=False, handlelength=1.2,
          columnspacing=.8)  # inside the U, above the curves
ax.set_xlabel('$x / a$'); ax.set_ylabel(tr('$y / D$ (downwards)', 'y / D (neeche)'))
ax.set_title(tr('(c) Superellipse family:\nsmall $n$ pointed, large $n$ flat-bottomed',
                '3) Superellipse family: n chhota = nukeela,\nn bada = flat bottom'), fontsize=10.5)

ax = axs[3]; ax.imshow(bh, cmap='magma', vmax=np.percentile(bh, 99.5))
for D, col in ((0.55 * Db, 'w'), (Db, 'lime'), (1.5 * Db, 'c')):
    px, py = curve(D, ab, nb, xcb); ax.plot(px, line[np.clip(px.astype(int), 0, w - 1)] + py, color=col, lw=2,
                                            label=f'$D$ = {D / PX:.2f} mm, score {score(D, ab, nb, xcb):.0f}')
crop(ax); ax.legend(loc='lower center', fontsize=8.5, framealpha=0.9)
ax.set_title(tr('(d) Candidate curves on the\nblack-hat image', '4) Black-hat image par\ncandidate curves'), fontsize=10.5)

ax = axs[4]
ax.plot(Ds / PX, perD, 'k-', lw=1.5)
ax.axvline(Db / PX, color='limegreen', lw=2.2, label=f'best fit {Db / PX:.2f} mm')
ax.axvline(ref.D_ref, color='r', ls='--', lw=1.8, label=f'expert {ref.D_ref:.2f} mm')
ax.set_xlabel(tr('candidate depth $D$ (mm)', 'try kiya gaya depth D (mm)')); ax.set_ylabel('best score')
ax.legend(fontsize=9.5, loc='upper left'); ax.grid(alpha=0.3)
ax.set_title(tr(f'(e) Grid search over {len(Ds) * 60} curves', f'5) Grid search: {len(Ds) * 60} curves'), fontsize=10.5)

ax = axs[5]; ax.imshow(g0, cmap='gray')
px, py = curve(Db, ab, nb, xcb); ax.plot(px, line[np.clip(px.astype(int), 0, w - 1)] + py, 'w--', lw=2, label='best superellipse')
cont = so.refine(bh, gm, line, Db, ab, nb, xcb, foot=(l - 0.05 * PX, r + 0.05 * PX))
ax.plot(cont[:, 0], cont[:, 1], 'c', lw=2, label=tr('after dynamic programming', 'DP ke baad'))
ax.plot(np.arange(l, r + 1), top[l:r + 1], 'lime', lw=2, label='clad outline')
crop(ax); ax.legend(loc='lower center', fontsize=8.5, framealpha=0.9)
ax.set_title(tr('(f) Best superellipse and its\nrefinement by dynamic programming', '6) Superellipse + DP refinement'), fontsize=10.5)
fig.tight_layout(h_pad=1.0); fig.savefig(out)
