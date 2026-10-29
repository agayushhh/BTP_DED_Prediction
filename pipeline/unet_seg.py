"""U-Net cross-section segmentation trained on our own expert-validated (QC-pass) classical masks.

Why: the classical pipeline is a chain of hand rules; it fails on unusual images (stains, dark beads, odd
framing). A U-Net learns appearance from the 3-class masks (0 other, 1 clad, 2 remelt) instead.
Labels = classical masks that passed QC (H, W, D within 0.1 mm of the experts), so training data is
human-checked. Split by TRACK (no sister sections across train/test).
Tests: (a) held-out QC-pass tracks, (b) the QC-FAIL sections, where the classical rules broke: does the
U-Net get closer to the experts' H, W, D there?
Usage: python unet_seg.py <tif_dir> <seg_dir (csv + masks/)> <out_dir> [epochs]
"""
import sys, os, time, glob
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

SZ = (320, 240)  # w, h; half of the classical 640x480 working size
PX = 1202.0 / 4 / 2  # px per mm at 320x240
torch.manual_seed(0); np.random.seed(0)


def load_img(tif_dir, key):
    # a few sections have two photos (..C-1.tif, ..C-2.tif); the classical run kept the last one, so do we
    path = sorted(glob.glob(os.path.join(tif_dir, key + '*.tif')))[-1]
    g = np.array(Image.open(path).convert('L').resize(SZ, Image.BILINEAR), np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, c=16, k=3):
        super().__init__()
        self.e = nn.ModuleList([block(1, c), block(c, 2 * c), block(2 * c, 4 * c), block(4 * c, 8 * c)])
        self.b = block(8 * c, 16 * c)
        self.u = nn.ModuleList([nn.ConvTranspose2d(16 * c // 2**i, 8 * c // 2**i, 2, 2) for i in range(4)])
        self.d = nn.ModuleList([block(16 * c // 2**i, 8 * c // 2**i) for i in range(4)])
        self.o = nn.Conv2d(c, k, 1)

    def forward(self, x):
        skips = []
        for e in self.e:
            x = e(x); skips.append(x); x = F.max_pool2d(x, 2)
        x = self.b(x)
        for u, d, s in zip(self.u, self.d, skips[::-1]):
            x = u(x)
            x = d(torch.cat([x, s], 1))
        return self.o(x)


def dice_loss(logits, y, k=3):
    p = logits.softmax(1)
    oh = F.one_hot(y, k).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3)); den = (p + oh).sum((0, 2, 3))
    return 1 - ((2 * inter + 1) / (den + 1))[1:].mean()  # foreground classes only


def augment(x, y):
    if np.random.rand() < .5:
        x, y = x[:, ::-1], y[:, ::-1]
    x = x * np.random.uniform(.8, 1.25) + np.random.uniform(-.3, .3)  # contrast / brightness jitter
    dy = np.random.randint(-20, 21)  # vertical shift: plate surface sits at different heights
    x, y = np.roll(x, dy, 0), np.roll(y, dy, 0)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def clean(m):
    """Shape prior a per-pixel classifier lacks: per class keep the largest connected region, close small
    gaps, fill holes (bead and remelt zone are each one solid blob)."""
    from scipy.ndimage import binary_fill_holes
    out = np.zeros_like(m)
    for c in (1, 2):
        b = cv2.morphologyEx((m == c).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(b)
        if n > 1:
            out[binary_fill_holes(lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])) & (out == 0)] = c
    return out


def measure(m):
    """Mask -> H, W, D (mm), A_d, A_s (mm2). Surface line = robust line through the clad/remelt
    interface (rows where class 1 sits directly above class 2), extended across the bead."""
    h, w = m.shape
    cols = np.flatnonzero(((m == 1) | (m == 2)).any(0))
    if len(cols) < 5 or not (m == 1).any():
        return None
    xs, ys = [], []
    for x in cols:
        c = m[:, x]
        a, b = np.flatnonzero(c == 1), np.flatnonzero(c == 2)
        if len(a) and len(b):
            xs.append(x); ys.append((a.max() + b.min()) / 2)
        elif len(a):
            xs.append(x); ys.append(a.max() + .5)  # clad sitting on unmelted plate
    if len(xs) < 5:
        return None
    k, c0 = np.polyfit(xs, ys, 1)
    line = k * np.arange(w) + c0
    H = max(line[x] - np.flatnonzero(m[:, x] == 1).min() for x in cols if (m[:, x] == 1).any())
    rem = [x for x in cols if (m[:, x] == 2).any()]
    D = max([np.flatnonzero(m[:, x] == 2).max() - line[x] for x in rem] + [0])
    W = (max(rem) - min(rem) + 1) if rem else (max(xs) - min(xs) + 1)
    Ad, As = (m == 1).sum(), (m == 2).sum()
    return dict(H=H / PX, W=W / PX, D=D / PX, Ad=Ad / PX**2, As=As / PX**2, dilution=As / (Ad + As))


def train(net, X, Y, keys, epochs, lr=1e-3, log=True):
    """Adam + cosine decay, batch 8, weighted CE + Dice, augmented. Used for training and fine-tuning."""
    if not epochs:
        return net
    opt = torch.optim.Adam(net.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    w = torch.tensor([1.0, 2.0, 2.0])  # foreground is a small part of the frame
    keys = list(keys)
    for ep in range(epochs):
        net.train(); np.random.shuffle(keys); tot = 0; t0 = time.time()
        for i in range(0, len(keys), 8):
            xb, yb = zip(*[augment(X[k], Y[k]) for k in keys[i:i + 8]])
            xb = torch.from_numpy(np.stack(xb))[:, None]; yb = torch.from_numpy(np.stack(yb))
            lo = net(xb)
            loss = F.cross_entropy(lo, yb, weight=w) + dice_loss(lo, yb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(xb)
        sched.step()
        if log:
            print(f'epoch {ep + 1}/{epochs} loss {tot / len(keys):.3f} ({time.time() - t0:.0f}s)', flush=True)
    return net


def predict(net, x):
    net.eval()
    with torch.no_grad():
        return clean(net(torch.from_numpy(x)[None, None]).argmax(1)[0].numpy().astype(np.uint8))


def main(tif_dir, seg_dir, out, epochs=40):
    epochs = int(epochs)
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)
    d = pd.read_csv(os.path.join(seg_dir, 'optomec_segmented.csv'))
    d['key'] = d.id + d.sec
    good = d[d.qc_pass].copy()
    tracks = good.id.unique(); np.random.shuffle(tracks)
    test_tracks = set(tracks[:max(1, len(tracks) // 5)])
    tr, te = good[~good.id.isin(test_tracks)], good[good.id.isin(test_tracks)]
    print(f'train {len(tr)} sections ({tr.id.nunique()} tracks) | test {len(te)} ({te.id.nunique()} tracks) | QC-fail {(~d.qc_pass).sum()}')

    def lab(key):
        return cv2.resize(cv2.imread(os.path.join(seg_dir, 'masks', key + '.png'), 0), SZ, interpolation=cv2.INTER_NEAREST).astype(np.int64)
    X = {k: load_img(tif_dir, k) for k in d.key}
    Y = {k: lab(k) for k in tr.key} if epochs else {}

    net = UNet()
    if epochs == 0:  # evaluate saved weights only
        net.load_state_dict(torch.load(os.path.join(out, 'unet.pt')))
    train(net, X, Y, list(tr.key), epochs)
    if epochs:
        torch.save(net.state_dict(), os.path.join(out, 'unet.pt'))

    net.eval(); rows = []
    os.makedirs(os.path.join(out, 'overlays'), exist_ok=True)
    with torch.no_grad():
        for _, r in d[d.key.isin(set(te.key) | set(d[~d.qc_pass].key))].iterrows():
            m = clean(net(torch.from_numpy(X[r.key])[None, None]).argmax(1)[0].numpy().astype(np.uint8))
            mm = measure(m)
            row = dict(key=r.key, set='heldout_qc_pass' if r.qc_pass else 'qc_fail', H_ref=r.H_ref, W_ref=r.W_ref, D_ref=r.D_ref,
                       H_cls=r.H, W_cls=r.W_fus, D_cls=r.D)
            if mm:
                row.update({f'{k}_unet': v for k, v in mm.items()})
            rows.append(row)
            g = cv2.normalize(X[r.key], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            ov = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            for c, col in ((1, (0, 220, 0)), (2, (0, 0, 255))):
                cs, _ = cv2.findContours((m == c).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(ov, cs, -1, col, 1)
            cv2.imwrite(os.path.join(out, 'overlays', r.key + '.jpg'), ov)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, 'unet_eval.csv'), index=False)
    for s_, g in res.groupby('set'):
        print(f'\n{s_}: n={len(g)}, U-Net produced a measurement for {g.H_unet.notna().sum()}')
        for k in 'HWD':
            ok = g[f'{k}_unet'].notna()
            eu = (g.loc[ok, f'{k}_unet'] - g.loc[ok, f'{k}_ref']).abs()
            ec = (g.loc[ok, f'{k}_cls'] - g.loc[ok, f'{k}_ref']).abs()
            print(f'  {k}: MAE U-Net {eu.mean():.3f} (median {eu.median():.3f}, <0.1 mm {100*(eu<.1).mean():.0f}%)'
                  f' | classical {ec.mean():.3f} (median {ec.median():.3f}, <0.1 mm {100*(ec<.1).mean():.0f}%)')


if __name__ == '__main__':
    main(*sys.argv[1:5])
