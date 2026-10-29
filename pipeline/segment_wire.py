"""Segment laser-wire DED micrographs (IN718 bead on SS304) -> H, W, D, A_d, A_s, dilution.

Input: one image holding a 3 x 3 grid of cross-section panels (rows P = 700/900/1100 W, columns
V = 450/600/750 mm/min), each with a 0.5 mm scale bar and possibly yellow annotation marks.
Etched IN718 bead is mid-grey, SS304 substrate is bright, mount is near-black, so the fusion line is a
plain intensity edge. Substrate surface = straight line fitted to the substrate top outside the bead.
Usage: python -I segment_wire.py <panel_grid.png> <out_dir>
"""
import sys, os, json
import cv2, numpy as np
from scipy.ndimage import binary_fill_holes as ndfill

COLS = [(106, 347), (362, 603), (619, 860)]
ROWS = [(9, 165), (179, 336), (350, 507)]
P = [700, 900, 1100]
V = [450, 600, 750]
SCALE_MM = 0.5


def scale_bar_px(rgb):
    """Length of the black bar inside the white scale box (bottom-right)."""
    g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    box = g[int(h * .75):, int(w * .6):]
    best = 0
    for row in (box < 90):  # dark pixels; the bar is the longest dark run inside a bright box
        r = np.flatnonzero(np.diff(np.r_[0, row.astype(int), 0]))
        for a, b in zip(r[::2], r[1::2]):
            if b - a > best and b - a < box.shape[1] * .9 and row.mean() < .6:
                best = b - a
    return best


def segment(rgb):
    """One panel: inpaint yellow annotation marks, mask the scale box, then measure()."""
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    ann = ((hsv[..., 0] > 15) & (hsv[..., 0] < 40) & (hsv[..., 1] > 90) & (hsv[..., 2] > 120)).astype(np.uint8)
    g = cv2.inpaint(g, cv2.dilate(ann, np.ones((5, 5), np.uint8)), 5, cv2.INPAINT_TELEA)
    g = cv2.GaussianBlur(g, (5, 5), 0).astype(float)
    ignore = np.zeros((h, w), bool)
    ignore[int(h * .78):, int(w * .62):] = True  # scale box
    return measure(g, ignore, bg_t=35, px_per_mm=95)  # mount <20 in every panel; etched bead top can be ~50


def measure(g, ignore, bg_t, px_per_mm):
    """Three-phase cross-section (dark mount / mid-grey bead / bright substrate) -> geometry in px.
    Substrate surface = straight line fitted to the substrate top in the outer 12% of columns."""
    h, w = g.shape
    k = max(3, int(round(0.05 * px_per_mm)) | 1)  # ~0.05 mm morphology kernel
    ker = np.ones((k, k), np.uint8)
    m = g[(g > bg_t) & ~ignore].astype(np.uint8).reshape(-1, 1)
    sub_t = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]  # bead vs substrate
    sub = (g > sub_t) & ~ignore
    n, lab = cv2.connectedComponents(sub.astype(np.uint8))
    keep = np.unique(lab[-3:][sub[-3:]])  # substrate touches the bottom edge
    sub = np.isin(lab, keep[keep > 0])
    sub = ndfill(cv2.morphologyEx(sub.astype(np.uint8), cv2.MORPH_CLOSE, ker).astype(bool))

    top = np.array([np.argmax(sub[:, x]) if sub[:, x].any() else -1 for x in range(w)], float)
    edge = int(w * .12)
    xs = np.r_[np.arange(3, edge), np.arange(w - edge, w - 3)]
    xs = xs[top[xs] > 0]
    kk, c = np.polyfit(xs, top[xs], 1)
    line = kk * np.arange(w) + c

    bead = (g > bg_t) & ~ignore & ~sub
    bead = cv2.morphologyEx(bead.astype(np.uint8), cv2.MORPH_OPEN, ker).astype(bool)  # drop thin edge strips along the surface
    n, lab, st, _ = cv2.connectedComponentsWithStats(bead.astype(np.uint8))
    i = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    bead = ndfill(lab == i)  # pores count as bead area

    yy, xx = np.mgrid[:h, :w]
    above = bead & (yy <= line[xx])
    below = bead & (yy > line[xx])
    cols = np.flatnonzero(bead.any(0))
    H = max(line[x] - np.argmax(bead[:, x]) for x in cols)
    D = max([(np.flatnonzero(bead[:, x])[-1] - line[x]) for x in cols] + [0])
    band = bead & (yy - line[xx] <= 2) & (line[xx] - yy <= 0.065 * px_per_mm)  # contact width just above the surface
    wcols = np.flatnonzero(band.any(0))
    W = wcols[-1] - wcols[0] + 1 if len(wcols) else 0
    Wmax = max(np.ptp(np.flatnonzero(r)) + 1 for r in bead if r.any())
    return dict(H=H, W=W, Wmax=Wmax, D=max(D, 0), Ad=above.sum(), As=below.sum()), bead, line, above, below


def main(src, out):
    os.makedirs(out, exist_ok=True)
    im = cv2.imread(src)
    rows, tiles = [], []
    for i, (r0, r1) in enumerate(ROWS):
        trow = []
        for j, (c0, c1) in enumerate(COLS):
            p = im[r0:r1, c0:c1].copy()
            px = scale_bar_px(p) / SCALE_MM  # px per mm
            m, bead, line, above, below = segment(p)
            os.makedirs(os.path.join(out, 'panels'), exist_ok=True)  # panel + 3-class mask (0 other, 1 clad, 2 remelt)
            cv2.imwrite(os.path.join(out, 'panels', f'{P[i]}_{V[j]}.png'), p)
            cv2.imwrite(os.path.join(out, 'panels', f'{P[i]}_{V[j]}_mask.png'), (above * 1 + below * 2).astype(np.uint8))
            mm = {k: m[k] / px for k in ('H', 'W', 'Wmax', 'D')}
            mm.update(Ad=m['Ad'] / px**2, As=m['As'] / px**2)
            mm['dilution'] = mm['As'] / (mm['Ad'] + mm['As'])
            mm['dil_depth'] = mm['D'] / (mm['H'] + mm['D'])
            rows.append(dict(P=P[i], V=V[j], px_per_mm=px, **{k: round(float(v), 4) for k, v in mm.items()}))
            ov = p.copy()
            ov[above] = (0.5 * ov[above] + [0, 90, 0]).astype(np.uint8)
            ov[below] = (0.5 * ov[below] + [0, 0, 120]).astype(np.uint8)
            cv2.line(ov, (0, int(line[0])), (p.shape[1] - 1, int(line[-1])), (255, 255, 0), 1)
            cv2.putText(ov, f"{P[i]}W {V[j]} dil={mm['dilution']*100:.0f}%", (3, 12), 0, .38, (255, 255, 255), 1)
            trow.append(ov)
        hmin = min(t.shape[0] for t in trow)
        tiles.append(np.hstack([t[:hmin] for t in trow]))
    wmin = min(t.shape[1] for t in tiles)
    cv2.imwrite(os.path.join(out, 'wire_seg_overlay.png'), np.vstack([t[:, :wmin] for t in tiles]))
    import csv
    with open(os.path.join(out, 'wire_geometry.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
    for r in rows:
        print(r)


if __name__ == '__main__':
    main(*sys.argv[1:3])
