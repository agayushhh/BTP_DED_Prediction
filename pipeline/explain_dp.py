"""Visual explainer for the Viterbi fusion-line refinement on real 316L sections.
Usage: python -I pipeline/explain_dp.py <BTP root> <out.png> [pick] [--en]"""
import sys, os
import numpy as np, pandas as pd, cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

EN = '--en' in sys.argv  # English labels for the report
sys.argv = [a for a in sys.argv if a != '--en']
root, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(root, 'pipeline'))
import segment_optomec as so

cap = {}
_refine = so.refine
def spy(bh, gm, line, D, a, nexp, xc, foot, **k):
    cap.update(bh=bh, gm=gm, line=line, D=D, a=a, nexp=nexp, xc=xc, foot=foot)
    return _refine(bh, gm, line, D, a, nexp, xc, foot, **k)
so.refine = spy

def scores(c, n_ang=91, lo=0.7, hi=1.3, n_rad=61):
    """Same score as so.refine, kept separately so greedy and DP see identical numbers."""
    bh, gm, line, D, a, nexp, xc, foot = (c[k] for k in ('bh', 'gm', 'line', 'D', 'a', 'nexp', 'xc', 'foot'))
    h, w = bh.shape
    yc = line[int(np.clip(xc, 0, w - 1))]
    th = np.linspace(np.radians(4), np.radians(176), n_ang)
    cc, s = np.abs(np.cos(th)), np.sin(th)
    r0 = (cc**nexp / a**nexp + s**nexp / D**nexp) ** (-1 / nexp)
    R = r0[:, None] * np.linspace(lo, hi, n_rad)[None, :]
    def sample(img, rr):
        x = np.clip((xc + rr * np.cos(th)[:, None]).astype(int), 0, w - 1)
        y = np.clip((yc + rr * s[:, None]).astype(int), 0, h - 1)
        return img[y, x], x, y
    v, x, y = sample(bh, R)
    raw = v + 0.5 * (sample(gm, R + 4)[0] - sample(gm, R - 4)[0])
    raw[(y - line[x]) < 0.03 * so.PX] = 0
    pen = raw - 60 * (R / r0[:, None] - 1) ** 2
    pen[(x < foot[0]) | (x > foot[1])] = -1e6
    return raw, pen, R, th, xc, yc

def xy(R, th, xc, yc, path):
    rr = R[np.arange(len(th)), path]
    return xc + rr * np.cos(th), yc + rr * np.sin(th)

def depth_mm(px, py, line, w):
    return (py - line[np.clip(px.astype(int), 0, w - 1)]).max() / so.PX

d = pd.read_csv(os.path.join(root, 'results/optomec_seg/optomec_segmented.csv'))
d = d[(d.qc_pass == True) & (d.id.str[-4:].astype(int) <= 99)]
cands = []
for _, r in d.sample(60, random_state=1).iterrows():
    p = os.path.join(root, 'data/optomec/micro_tif', f'{r.id}{r.sec}.tif')
    if not os.path.exists(p):
        continue
    try:
        res, g0, line, prof, cont = so.run_one(p)
    except Exception:
        continue
    raw, pen, R, th, xc, yc = scores(cap)
    greedy = pen.argmax(1)  # same score DP uses; only the smoothness rule differs
    jumps = np.abs(np.diff(greedy)).mean()
    cands.append((jumps, r, g0, line, cont, raw, pen, R, th, xc, yc, greedy, dict(cap)))
cands.sort(key=lambda t: -t[0])
pick = int(sys.argv[3]) if len(sys.argv) > 3 else 0
jumps, r, g0, line, cont, raw, pen, R, th, xc, yc, greedy, c = cands[pick]
h, w = g0.shape
print('picked', r.id, r.sec, 'mean jump', round(jumps, 1), 'D_ref', round(r.D_ref, 3))

# DP path in index space (re-derive from the returned contour: nearest radius index per ray)
rr = np.hypot(cont[:, 0] - xc, cont[:, 1] - yc)
dp = np.abs(R - rr[:, None]).argmin(1)
mid = np.full(len(th), R.shape[1] // 2)

T = dict(  # (English, Hinglish)
    t1=('(a) Rays from the bead centre (orange); dashed: superellipse r0.\n61 candidate radii per ray, 0.7 r0 to 1.3 r0',
        '1) Centre se rays (orange). White dashed = superellipse\nandaza r0. Har ray par 0.7r0 se 1.3r0 tak 61 candidates'),
    greedy=('Greedy: best candidate on each ray', 'Greedy: har column ka sabse bright point'),
    dp=('Dynamic programming (Viterbi): max total score, smooth', 'DP (Viterbi): smooth path, max total score'),
    r0=('superellipse (r = r0)', 'superellipse andaza (r = r0)'),
    xl=('ray angle (degree): left edge of bead ... right edge', 'ray angle (degree)  ->  left side of bead ... right side'),
    yl=('radius / r0', 'radius / r0  (upar = andar, neeche = bahar)'),
    t2=('(b) Unrolled score map: one column per ray, one row per candidate radius.\n'
        'Bright = thin dark line with brighter substrate outside (fusion-line evidence)',
        '2) Rays ko "khol" ke ek image bana di: har column = ek ray, har row = ek candidate radius.\n'
        'Bright = dark-line jaisa (fusion line ka saboot). Ab problem = left se right ek path dhoondhna.'),
    t3=('({}) {} on the micrograph\nD = {:.3f} mm   (expert D = {:.3f} mm)',
        '{}) {} wapas photo par\nD = {:.3f} mm   (expert D = {:.3f} mm)'),
    box=('Constraint: the radius index changes\nby at most 2 between neighbouring rays\n\n'
         'best[i][j] = score[i][j]\n     + max(best[i-1][j-2 .. j+2])\n\n'
         'Take the largest best[] on the last\nray and follow the back-pointers.\n\n'
         'Both methods use the same scores;\nonly the smoothness constraint differs.',
         'Rule (jump <= 2 steps):\n\nray i par radius j choose kiya,\ntoh ray i+1 par sirf j-2 ... j+2 allowed\n\n'
         'best[i][j] = score[i][j]\n      + max(best[i-1][j-2 .. j+2])\n\n'
         'Last ray par sabse bada best,\nphir back-pointers se ulta chalo\n-> poora path.\n\n'
         'Greedy ek-ek ray dekhta hai -> stain/scratch\npar kood jaata hai.\n'
         'DP poora raasta ek saath dekhta hai ->\nek akela nakli bright point jeet nahi paata.'),
    sup=('Fusion-line refinement by dynamic programming: section {} (316L)', 'Fusion line kaise dhoondhi: {} (316L)'),
)
L = {k: v[0] if EN else v[1] for k, v in T.items()}

fig = plt.figure(figsize=(15, 9.5))
gsp = fig.add_gridspec(2, 3, height_ratios=[1, 1])
crop = lambda ax: (ax.set_xlim(xc - 1.6 * c['a'], xc + 1.6 * c['a']), ax.set_ylim(yc + 1.6 * c['D'], yc - 1.3 * c['D']))

ax = fig.add_subplot(gsp[0, 0]); ax.imshow(g0, cmap='gray')
ax.axhline(yc, color='c', lw=1)
x0, y0 = xy(R[:, :1], th, xc, yc, np.zeros(len(th), int)); x1, y1 = xy(R[:, -1:], th, xc, yc, np.zeros(len(th), int))
for i in range(0, len(th), 6):
    ax.plot([x0[i], x1[i]], [y0[i], y1[i]], color='orange', lw=0.8)
mx, my = xy(R, th, xc, yc, mid); ax.plot(mx, my, 'w--', lw=1.2)
ax.plot(xc, yc, 'yo')
crop(ax); ax.axis('off')
ax.set_title(L['t1'], fontsize=10)

ax = fig.add_subplot(gsp[0, 1:])
pm = np.where(pen < -1e5, np.nan, pen)
im = ax.imshow(np.clip(pm.T, np.nanpercentile(pm, 2), np.nanpercentile(pm, 99)), aspect='auto', cmap='magma', origin='upper',
               extent=[np.degrees(th[0]), np.degrees(th[-1]), 1.3, 0.7])
ax.plot(np.degrees(th), 0.7 + 0.6 * greedy / (R.shape[1] - 1), 'y.-', lw=1, ms=4, label=L['greedy'])
ax.plot(np.degrees(th), 0.7 + 0.6 * dp / (R.shape[1] - 1), 'c-', lw=2.5, label=L['dp'])
ax.axhline(1.0, color='w', ls='--', lw=1, label=L['r0'])
ax.set_xlabel(L['xl']); ax.set_ylabel(L['yl'])
ax.legend(loc='lower right', fontsize=9, framealpha=0.8)
ax.set_title(L['t2'], fontsize=10)
fig.colorbar(im, ax=ax, fraction=0.02, label='score')

for k, (path, col, name) in enumerate([(greedy, 'y', 'Greedy'), (dp, 'c', 'Dynamic programming' if EN else 'DP / Viterbi')]):
    ax = fig.add_subplot(gsp[1, k]); ax.imshow(g0, cmap='gray'); ax.axhline(yc, color='c', lw=0.6, alpha=0.5)
    px, py = xy(R, th, xc, yc, path); ax.plot(px, py, color=col, lw=1.6)
    crop(ax); ax.axis('off')
    ax.set_title(L['t3'].format('cd'[k] if EN else 3 + k, name, depth_mm(px, py, line, w), r.D_ref), fontsize=10)

ax = fig.add_subplot(gsp[1, 2]); ax.axis('off')
ax.text(0, 0.95, L['box'], va='top', family='monospace', fontsize=10)
fig.suptitle(L['sup'].format(r.id + r.sec), fontsize=13)
fig.tight_layout(); fig.savefig(out, dpi=110)
