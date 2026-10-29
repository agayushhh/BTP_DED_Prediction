"""Per-section melt-pool features from the coaxial denoise frames (Optomec 316L).

Registration: a track's sections A, B, ... are printed back-to-back at speeds V_i over (assumed) equal
lengths, so section i occupies a share (1/V_i) / sum(1/V_j) of the laser-on time. The original toolbox
splits frames into equal bins, which misassigns frames whenever speed changes along the track.
We keep the middle 40% of each section's window (cut is mid-section; transitions excluded).
Features per frame: areas above fixed grey levels (isotherm-like), total brightness, trail size (220 px = 1 mm).
Usage: python -I coaxial_features.py <coaxial_dir> <sections_mm.csv> <out.csv>
"""
import sys, os, re, zipfile, io
import numpy as np, pandas as pd, cv2

PX_PER_MM = 220.0


LEVELS = (30, 60, 100, 150, 250)  # fixed grey levels ~ isotherms (camera gain is fixed across the dataset)


def frame_feats(img):
    """Saturated core = laser spot (near-constant); the dim trail behind it carries the thermal information.
    Features: area above each fixed level (mm2), total brightness, trail length/width of the >30 region."""
    g = cv2.GaussianBlur(img, (5, 5), 0)
    if g.max() < 150:  # laser off / transition
        return None
    f = {f'A{t}': (g > t).sum() / PX_PER_MM**2 for t in LEVELS}
    f['Isum'] = g[g > 30].sum() / 255 / PX_PER_MM**2
    ys, xs = np.nonzero(g > 30)
    f['len30'] = np.ptp(xs) / PX_PER_MM
    f['wid30'] = np.ptp(ys) / PX_PER_MM
    return f


def track_frames(zpath):
    z = zipfile.ZipFile(zpath)
    names = [n for n in z.namelist() if n.lower().endswith('.jpg')]
    key = lambda n: int(re.search(r'prim[^\d]*(\d+)', os.path.basename(n)).group(1))
    names.sort(key=key)
    for n in names:
        img = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_GRAYSCALE)
        yield n, img


def snap_edges(F, V):
    """Section boundaries in frame index. Prior: equal-length sections -> time share ~ 1/V_i.
    Then each interior boundary moves to the strongest step of the glow area A30 within +-25% of the
    neighbouring section length (acceleration, unequal lengths). Steps exist because P, V or F change there."""
    w = (1 / V) / (1 / V).sum()
    edges = np.r_[0, np.cumsum(w)] * len(F)
    a = pd.Series([f['A30'] if f else np.nan for f in F]).interpolate().bfill().ffill().values
    a = pd.Series(a).rolling(9, center=True, min_periods=1).median().values
    for i in range(1, len(edges) - 1):
        L = min(edges[i] - edges[i - 1], edges[i + 1] - edges[i])
        m = max(5, int(0.15 * L))
        lo = int(max(edges[i - 1] + m, edges[i] - 0.25 * L))
        hi = int(min(edges[i + 1] - m, edges[i] + 0.25 * L))
        if hi <= lo:
            continue
        js = np.arange(lo, hi)
        step = [abs(a[j:j + m].mean() - a[j - m:j].mean()) for j in js]
        edges[i] = js[int(np.argmax(step))]
    return edges


def main(coax_dir, sections_csv, out):
    sec = pd.read_csv(sections_csv)
    rows = []
    for tid, s in sec.groupby('id', sort=True):
        zp = os.path.join(coax_dir, f'{tid}_denoise.zip')
        if not os.path.exists(zp) or os.path.exists(zp + '.done'):
            continue
        try:
            F = [frame_feats(img) for _, img in track_frames(zp)]
        except (zipfile.BadZipFile, AttributeError) as e:
            print('skip', tid, e); continue
        on = [i for i, f in enumerate(F) if f is not None]
        if len(on) < 20:
            continue
        F = F[on[0]:on[-1] + 1]  # laser-on span
        s = s.sort_values('sec')
        edges = snap_edges(F, s.V.values)
        for (_, r), a, b in zip(s.iterrows(), edges[:-1], edges[1:]):
            lo, hi = int(a + 0.3 * (b - a)), int(a + 0.7 * (b - a))
            fs = [f for f in F[lo:hi] if f is not None]
            if len(fs) < 3:
                continue
            df = pd.DataFrame(fs)
            feat = {f'cx_{k}': v for k, v in df.median().items()}
            feat.update({f'cx_{k}_sd': v for k, v in df[['A30', 'A100', 'Isum']].std().items()})
            rows.append(dict(id=tid, sec=r.sec, n_frames=len(fs), **feat))
        print(tid, len(F), 'frames', flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == '__main__':
    main(*sys.argv[1:4])
