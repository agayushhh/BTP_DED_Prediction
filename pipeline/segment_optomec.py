"""Segment Optomec 316L cross-section TIFFs -> H, W, D, A_d, A_s, dilution; validate vs expert H, W, D.

316L bead on 316L plate: bead and plate have the same brightness, so intensity thresholds cannot
separate them (tried; R2 < 0). Two robust cues instead:
  1. mount (dark, speckled) vs metal -> metal top profile -> surface line, H, contact W, A_d
  2. fusion line = thin dark etched curve below the surface -> grid-search the superellipse
     |x-xc|^n/a^n + (y/D)^n = 1 (below the line) that sits on the strongest dark-line response.
Images are 2560x1922 at 1202 px/mm; we work at 1/4 scale (300.5 px/mm).
Usage: python segment_optomec.py <tif_dir> <Final_total_measurement.xlsx> <out_dir>
"""
import sys, os, glob, re, math
import numpy as np, pandas as pd, cv2
from PIL import Image
from process_model import load

PX = 1202.0 / 4


def threshold_multiotsu(g):
    """3-class Otsu: thresholds (t1, t2) maximising between-class variance of the 8-bit histogram."""
    p = np.bincount(g.ravel(), minlength=256) / g.size
    i = np.arange(256)
    w, mu = np.cumsum(p), np.cumsum(p * i)
    best, bt = -1, (0, 0)
    for t1 in range(1, 254):
        for t2 in range(t1 + 1, 255):
            w0, w1, w2 = w[t1], w[t2] - w[t1], 1 - w[t2]
            if min(w0, w1, w2) <= 0:
                continue
            m0, m1, m2 = mu[t1] / w0, (mu[t2] - mu[t1]) / w1, (mu[-1] - mu[t2]) / w2
            v = w0 * m0**2 + w1 * m1**2 + w2 * m2**2
            if v > best:
                best, bt = v, (t1, t2)
    return bt


def fit_line(top, frac=.12):
    """Robust plate-surface line from the per-column top in the outer `frac` of columns: try left/right column
    pairs, keep the line most margin columns agree with (a stain or debris on one side otherwise tilts a
    least-squares fit); polished plates are near-flat, so |slope| <= 0.08."""
    w = len(top)
    edge = int(w * frac)
    xs = np.r_[np.arange(3, edge), np.arange(w - edge, w - 3)]
    L, Rt = xs[xs < w // 2], xs[xs >= w // 2]
    best_in, kk, c = -1, 0.0, np.median(top[xs])
    for xa in L[::4]:
        for xb in Rt[::4]:
            k1 = (top[xb] - top[xa]) / (xb - xa)
            if abs(k1) > 0.08:
                continue
            inl = np.abs(top[xs] - (top[xa] + k1 * (xs - xa))) < 6
            if inl.sum() > best_in:
                best_in, sel = inl.sum(), inl
    if best_in > 0:
        kk, c = np.polyfit(xs[sel], top[xs][sel], 1)
    return kk * np.arange(w) + c


def run_one(path):
    g0 = np.array(Image.open(path).convert('L').resize((640, 480), Image.BILINEAR), np.uint8)
    h, w = g0.shape
    g = cv2.GaussianBlur(g0, (0, 0), 4)
    from scipy.ndimage import binary_fill_holes
    yy = np.arange(h)[:, None]
    bright_mount = g0[:15].mean() > 100  # two imaging sets: dark speckled mount, or bright smooth mount
    if bright_mount:
        # bright smooth mount vs scratched (textured) metal: separate by local standard deviation, not brightness
        f = g0.astype(float)
        mu = cv2.GaussianBlur(f, (0, 0), 2)
        sd = cv2.GaussianBlur(np.sqrt(np.maximum(cv2.GaussianBlur(f * f, (0, 0), 2) - mu * mu, 0)), (0, 0), 2)  # small windows: less halo at the edge
        sd8 = np.clip(sd * 255 / max(sd.max(), 1e-6), 0, 255).astype(np.uint8)
        bg_t = cv2.threshold(sd8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        cue = sd8
    else:
        bg_t = threshold_multiotsu(g)[0]  # mount | bead | plate; bead can be much darker than plate
        cue = g
    metal = cv2.morphologyEx((cue > bg_t).astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(metal)
    metal = lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])  # plate + bead are one connected body
    metal = binary_fill_holes(metal | (yy > h - 3))

    top = np.array([np.argmax(metal[:, x]) if metal[:, x].any() else h for x in range(w)], float)
    line = fit_line(top)

    above = yy < line[None, :] - 2
    if bright_mount:
        body = metal
    else:
        # above the surface only mount and bead exist: a 2-class Otsu there separates them far better than the
        # whole-image cut (pulled up by the bright plate); the upper bead can be nearly as dark as the mount
        t_up = cv2.threshold(g[above].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        up = cv2.morphologyEx(((g > min(t_up, bg_t)) & above).astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)).astype(bool)
        body = binary_fill_holes(up | ~above)
        n, lab, st, _ = cv2.connectedComponentsWithStats(body.astype(np.uint8))
        body = lab == lab[h - 1, w // 2]  # the component holding the plate
    top = np.array([np.argmax(body[:, x]) for x in range(w)], float)
    rise = line - top  # bead height above the surface per column
    on = rise > 0.03 * PX
    if rise.max() < 0.1 * PX:
        # flush track (slow, low feed: H ~ 0, near-total remelt, diffuse low-contrast melt zone). A dedicated
        # fallback reached D within 0.1 mm in only 54 %, so scoped out: experts' H/W/D only.
        raise ValueError('flush regime: out of scope')
    # contact width: the contiguous run of raised columns containing the tallest point
    peak = int(np.argmax(rise))
    l, r = peak, peak
    while l > 0 and on[l - 1]:
        l -= 1
    while r < w - 1 and on[r + 1]:
        r += 1
    H = rise[peak]
    W = r - l + 1
    Ad = rise[l:r + 1].clip(min=0).sum()

    # dark-line response: black-hat highlights thin dark curves (the etched fusion boundary)
    gs = cv2.GaussianBlur(g0, (0, 0), 1.5)
    bh = cv2.morphologyEx(gs, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))).astype(float)
    gm = cv2.GaussianBlur(g0, (0, 0), 4).astype(float)  # smooth brightness for the inside/outside step
    xc0 = (l + r) / 2
    best = (-1e9, None)
    t = np.linspace(0, np.pi, 120)
    for D in np.arange(15, 0.95 * (h - line.max()), 2):
        for af in (0.8, 0.9, 1.0, 1.1, 1.2):
            a = af * W / 2
            for nexp in (1.6, 2.0, 2.6, 3.4):
                for dx in (-0.1 * W, 0, 0.1 * W):
                    xc = xc0 + dx
                    # superellipse lower half
                    cx = np.cos(t); sy = np.sin(t)
                    px = xc + a * np.sign(cx) * np.abs(cx) ** (2 / nexp)
                    py = D * np.abs(sy) ** (2 / nexp)
                    xi = np.clip(px.astype(int), 0, w - 1)
                    yi = np.clip((line[xi] + py).astype(int), 0, h - 1)
                    keep = py > max(0.15 * D, 0.03 * PX)  # skip the surface band: the mount/plate edge there is also a dark line
                    if keep.sum() < 10:
                        continue
                    # step across the curve: plate just outside should be brighter than bead just inside
                    def at(f, off):
                        x = np.clip((xc + f * (px - xc)).astype(int), 0, w - 1)
                        return gm[np.clip((line[x] + f * py + off).astype(int), 0, h - 1), x][keep]
                    step = (at(1.1, 4) - at(0.9, -4)).mean()
                    s = bh[yi[keep], xi[keep]].mean() + 0.5 * step
                    if s > best[0]:
                        best = (s, (D, a, nexp, xc))
    D, a, nexp, xc = best[1]
    cont = refine(bh, gm, line, D, a, nexp, xc, foot=(l - 0.05 * PX, r + 0.05 * PX))
    # fusion-line contour -> D, W at the surface, remelt area A_s (polygon closed along the surface line)
    depth = cont[:, 1] - line[np.clip(cont[:, 0].astype(int), 0, w - 1)]
    xl, xr = max(cont[:, 0].min(), 0), min(cont[:, 0].max(), w - 1)
    poly = np.vstack([cont, [[xl, line[int(xl)]], [xr, line[min(int(xr), w - 1)]]][::-1]]).astype(np.float32)
    As = cv2.contourArea(poly)
    # clad only between the fusion line's surface ends (drops debris/spatter lying on the plate beside the bead)
    span = slice(int(xl), int(xr) + 1)
    Ad = rise[span].clip(min=0).sum()
    H = rise[span].max()
    return (dict(H=H, W=W, W_fus=xr - xl, D=depth.max(), Ad=Ad, As=As, fit_score=best[0], n=nexp),
            g0, line, (int(np.clip(xl, 0, w - 1)), int(np.clip(xr, 0, w - 1)), top), cont)


def refine(bh, gm, line, D, a, nexp, xc, foot, n_ang=91, lo=0.7, hi=1.3, n_rad=61, jump=2):
    """Ray-by-ray boundary from the bead centre on the surface; dynamic programming picks one radius
    per angle maximising (dark line + bright-outside step) with at most `jump` radius steps between
    neighbouring rays. Starts from the superellipse, so it can only bend it, not wander off."""
    h, w = bh.shape
    yc = line[int(np.clip(xc, 0, w - 1))]
    th = np.linspace(np.radians(4), np.radians(176), n_ang)
    c, s = np.abs(np.cos(th)), np.sin(th)
    r0 = (c**nexp / a**nexp + s**nexp / D**nexp) ** (-1 / nexp)
    R = r0[:, None] * np.linspace(lo, hi, n_rad)[None, :]  # (angle, radius)

    def sample(img, rr):
        x = np.clip((xc + rr * np.cos(th)[:, None]).astype(int), 0, w - 1)
        y = np.clip((yc + rr * s[:, None]).astype(int), 0, h - 1)
        return img[y, x], x, y

    v, x, y = sample(bh, R)
    score = v + 0.5 * (sample(gm, R + 4)[0] - sample(gm, R - 4)[0])
    score[(y - line[x]) < 0.03 * PX] = 0  # surface band: no evidence either way
    score -= 60 * (R / r0[:, None] - 1) ** 2  # mild pull towards the global fit: stops runs along scratches
    score[(x < foot[0]) | (x > foot[1])] = -1e6  # fusion line must meet the plate inside the bead footprint
    acc = score[0].copy()
    back = np.zeros(score.shape, int)
    for i in range(1, n_ang):  # Viterbi over angles
        cand = np.stack([np.roll(acc, k) if k >= 0 else np.roll(acc, k) for k in range(-jump, jump + 1)])
        for row, k in enumerate(range(-jump, jump + 1)):  # invalidate wrapped-around entries
            if k > 0:
                cand[row, :k] = -np.inf
            elif k < 0:
                cand[row, k:] = -np.inf
        j = cand.argmax(0)
        back[i] = np.arange(n_rad) - (j - jump)
        acc = cand.max(0) + score[i]
    path = [int(acc.argmax())]
    for i in range(n_ang - 1, 0, -1):
        path.append(back[i][path[-1]])
    path = path[::-1]
    rr = R[np.arange(n_ang), path]
    return np.stack([xc + rr * np.cos(th), yc + rr * s], 1)


def overlay(g0, line, prof, cont, path, label=''):
    """Outlines only (not filled) so the real etched boundary stays visible underneath."""
    l, r, top = prof
    ov = cv2.cvtColor(g0, cv2.COLOR_GRAY2BGR)
    w = g0.shape[1]
    cv2.line(ov, (0, int(line[0])), (w - 1, int(line[-1])), (255, 255, 0), 1)  # cyan: plate surface
    clad = np.array([[x, top[x]] for x in range(l, r + 1)], np.int32)
    cv2.polylines(ov, [clad], False, (0, 220, 0), 1)  # green: bead outline above the surface
    cv2.polylines(ov, [cont.astype(np.int32)], False, (0, 0, 255), 1)  # red: fitted fusion line
    cv2.putText(ov, label, (5, 18), 0, .5, (255, 255, 255), 1)
    cv2.imwrite(path, ov)


def label_mask(shape, line, prof, cont):
    """0 = mount/plate, 1 = clad (bead above the surface), 2 = remelt (inside fusion line, below surface).
    Saved for every section; QC-passing ones become U-Net training labels."""
    h, w = shape
    l, r, top = prof
    m = np.zeros(shape, np.uint8)
    for x in range(l, r + 1):
        y0, y1 = int(top[x]), int(round(line[x]))
        if y1 > y0:
            m[y0:y1, x] = 1
    xl, xr = cont[:, 0].min(), cont[:, 0].max()
    poly = np.vstack([cont, [[xr, line[min(int(xr), w - 1)]], [xl, line[max(int(xl), 0)]]]]).astype(np.int32)
    rem = np.zeros(shape, np.uint8)
    cv2.fillPoly(rem, [poly], 1)
    m[(rem == 1) & (np.arange(h)[:, None] > line[None, :])] = 2
    return m


def process(args):
    """One image -> result row; writes its overlay and label mask. Top-level so worker processes can run it."""
    p, out, refv = args
    tid, sec = re.match(r'(DED_SS316L_\d+)([A-Z])', os.path.basename(p)).groups()
    try:
        m, g0, line, prof, fus = run_one(p)
    except Exception as e:  # record failure, keep going
        return f'fail {tid} {sec} {e}'
    r = dict(id=tid, sec=sec, H=m['H'] / PX, W=m['W'] / PX, W_fus=m['W_fus'] / PX, D=m['D'] / PX, Ad=m['Ad'] / PX**2, As=m['As'] / PX**2,
             fit_score=m['fit_score'], n=m['n'])
    r['dilution'] = r['As'] / (r['Ad'] + r['As'])
    if refv is not None:
        r.update(H_ref=refv[0], W_ref=refv[1], D_ref=refv[2])
    lab = f"{tid[-4:]}{sec}  D {r['D']:.2f} (ref {r.get('D_ref', np.nan):.2f})  W {r['W_fus']:.2f} (ref {r.get('W_ref', np.nan):.2f})"
    overlay(g0, line, prof, fus, os.path.join(out, 'overlays', f'{tid}{sec}.jpg'), lab)
    cv2.imwrite(os.path.join(out, 'masks', f'{tid}{sec}.png'), label_mask(g0.shape, line, prof, fus))
    return r


def main(tif_dir, xlsx, out, workers=6):
    from multiprocessing import Pool
    os.makedirs(os.path.join(out, 'overlays'), exist_ok=True)
    os.makedirs(os.path.join(out, 'masks'), exist_ok=True)
    ref = load(xlsx).set_index(['id', 'sec'])
    refd = {k: (v.H, v.W, v.D) for k, v in ref[['H', 'W', 'D']].iterrows()}
    jobs = []
    for p in sorted(glob.glob(os.path.join(tif_dir, '*.tif'))):
        tid, sec = re.match(r'(DED_SS316L_\d+)([A-Z])', os.path.basename(p)).groups()
        jobs.append((p, out, refd.get((tid, sec))))
    rows = []
    with Pool(int(workers)) as pool:
        for r in pool.imap(process, jobs, chunksize=4):
            if isinstance(r, str):
                print(r, flush=True)
            else:
                rows.append(r)
    # sections with two photos (..C-1/..C-2): keep the last, as the masks on disk do
    rows = list({(r['id'], r['sec']): r for r in rows}.values())
    d = pd.DataFrame(rows)
    # QC: our areas/dilution are trusted only where our H, W, D agree with the experts' manual values
    # (every section has them), i.e. new labels are human-validated, not blind
    d['qc_pass'] = ((d.H - d.H_ref).abs() < .1) & ((d.W_fus - d.W_ref).abs() < .1) & ((d.D - d.D_ref).abs() < .1)
    print(f"QC pass: {d.qc_pass.sum()} / {len(d)} segmented ({d.qc_pass.mean():.0%})")
    d.to_csv(os.path.join(out, 'optomec_segmented.csv'), index=False)
    v = d.dropna(subset=['H_ref'])
    for k, kr in [('H', 'H'), ('W', 'W'), ('W_fus', 'W'), ('D', 'D')]:
        err = v[k] - v[f'{kr}_ref']
        r2 = 1 - (err**2).sum() / ((v[f'{kr}_ref'] - v[f'{kr}_ref'].mean())**2).sum()
        print(f'{k}: n={len(v)} MAE={err.abs().mean():.3f} mm  median abs={err.abs().median():.3f}  R2={r2:.3f}  bias={err.mean():+.3f}')


if __name__ == '__main__':
    main(*sys.argv[1:5])
