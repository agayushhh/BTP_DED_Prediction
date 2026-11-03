"""Extra report figures: QC accept/reject examples, validation-split diagram, U-Net architecture + label example.
Usage: python -I pipeline/report_extra_figs.py <BTP root> <figs_dir> [qc splits unet mount]"""
import sys, os
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import plt, FULL, outline, crop_box, mask_legend
from matplotlib.patches import FancyBboxPatch
from sklearn.model_selection import KFold, GroupKFold

root, out = sys.argv[1], sys.argv[2]
seg = os.path.join(root, 'results/optomec_seg')


def section(key):
    """Grey 640x480 micrograph and its rule-based mask."""
    tif = os.path.join(root, 'data/optomec/micro_tif')
    p = sorted(f for f in os.listdir(tif) if f.startswith(key))[-1]
    g = np.array(Image.open(os.path.join(tif, p)).convert('L').resize((640, 480)))
    return g, np.array(Image.open(os.path.join(seg, 'masks', key + '.png')))


def errors(r):
    return (f'$|\\Delta H|$ {abs(r.H - r.H_ref) * 1000:.0f}, $|\\Delta W|$ {abs(r.W_fus - r.W_ref) * 1000:.0f}, '
            f'$|\\Delta D|$ {abs(r.D - r.D_ref) * 1000:.0f} µm')


def qc_examples():
    d = pd.read_csv(os.path.join(seg, 'optomec_segmented.csv'))
    d['n'] = d.id.str[-4:].astype(int)
    for k, km in (('H', 'H'), ('W', 'W_fus'), ('D', 'D')):  # same quantities as the QC in segment_optomec
        d['e' + k] = (d[km] - d[k + '_ref']).abs()
    acc = d[d.qc_pass == True]
    rej = d[(d.qc_pass == False) & d.eH.notna()]
    picks = [('(a) Accepted, dark mount', acc[acc.n <= 99].sort_values('eD').iloc[len(acc[acc.n <= 99]) // 2]),
             ('(b) Accepted, bright mount', acc[acc.n >= 200].sort_values('eD').iloc[len(acc[acc.n >= 200]) // 2]),
             ('(c) Rejected: faint fusion line missed', rej[(rej.eH < 0.05) & (rej.eW < 0.05) & (rej.eD > 0.15)].iloc[0]),
             ('(d) Rejected: boundary follows a scratch', rej[(rej.eH < 0.05) & (rej.eW > 0.15) & (rej.eD < 0.06)].iloc[0])]
    fig, axs = plt.subplots(2, 2, figsize=(FULL, 5.6))
    for ax, (title, r) in zip(axs.ravel(), picks):
        g, m = section(f'{r.id}{r.sec}')
        outline(ax, g, m, crop_box([m], m.shape, pad=.3))
        ax.set_title(f'{title}\nsection {r.id[-4:]}{r.sec}', fontsize=10.5, color='green' if r.qc_pass else '#c0392b')
        ax.set_xlabel(errors(r), fontsize=10)
    fig.tight_layout(h_pad=1.0)
    mask_legend(fig, loc='lower center', bbox_to_anchor=(.5, -.05))
    fig.savefig(os.path.join(out, 'fig_qc_examples.png'))
    print('qc picks', [f'{r.id}{r.sec}' for _, r in picks])


def validation_splits():
    d = pd.read_csv(os.path.join(root, 'results/optomec/optomec_sections_mm.csv'))
    c = d.groupby('cond').agg(P=('P', 'first'), V=('V', 'first'), F=('F', 'first'), n=('P', 'size')).reset_index()
    rng = np.random.default_rng(0)
    c['jx'] = c.V * np.exp(rng.uniform(-0.08, 0.08, len(c)))
    c['jy'] = c.P + rng.uniform(-12, 12, len(c))
    tr_i, te_i = next(KFold(5, shuffle=True, random_state=0).split(d))
    te_rand = d.iloc[te_i].groupby('cond').size().reindex(c.cond).fillna(0).values
    tr_rand = d.iloc[tr_i].groupby('cond').size().reindex(c.cond).fillna(0).values
    tr_g, te_g = next(GroupKFold(5).split(d, groups=d.cond))
    test_g = set(d.iloc[te_g].cond)
    fig, axs = plt.subplots(3, 1, figsize=(FULL, 8.4), sharex=True)
    s = 8 + 2.5 * c.n
    both = (te_rand > 0) & (tr_rand > 0)
    ax = axs[0]
    ax.scatter(c.jx[both], c.jy[both], s=s[both], c='tab:purple', alpha=0.7, label='sections in train and test')
    ax.scatter(c.jx[~both], c.jy[~both], s=s[~both], c='tab:blue', alpha=0.7, label='train only')
    ax.set_title(f'(a) Random split: {both.mean() * 100:.0f} % of the conditions have sections on both sides')
    ax = axs[1]
    m = c.cond.isin(test_g).values
    ax.scatter(c.jx[~m], c.jy[~m], s=s[~m], c='tab:blue', alpha=0.7, label='train')
    ax.scatter(c.jx[m], c.jy[m], s=s[m], c='tab:orange', alpha=0.8, label='test')
    ax.set_title('(b) Grouped by condition: test settings are never seen in training')
    ax = axs[2]
    m = (c.P >= 450).values
    ax.scatter(c.jx[~m], c.jy[~m], s=s[~m], c='tab:blue', alpha=0.7, label=f'train, $P \\leq$ 400 W ({(~m).sum()} conditions)')
    ax.scatter(c.jx[m], c.jy[m], s=s[m], c='tab:orange', alpha=0.8, label=f'test, $P \\geq$ 450 W ({m.sum()} conditions)')
    ax.axhline(425, color='k', ls='--', lw=1)
    ax.set_title('(c) Extrapolation in power: test power lies outside the training range')
    for ax in axs:
        ax.set_xscale('log'); ax.grid(alpha=0.3); ax.set_ylabel('laser power $P$ (W)'); ax.set_ylim(170, 620)
        ax.legend(loc='upper left', ncol=2, framealpha=.95, fontsize=9.5, markerscale=.6)
    axs[2].set_xlabel('travel speed $V$ (mm/min), log scale')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig_validation_splits.png'))


def unet_figure():
    fig = plt.figure(figsize=(FULL, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1])
    ax = fig.add_subplot(gs[0, :]); ax.axis('off'); ax.set_xlim(0, 12); ax.set_ylim(-0.9, 6.1)
    chans = [16, 32, 64, 128]

    def box(x, y, wd, ht, txt, col):
        ax.add_patch(FancyBboxPatch((x, y), wd, ht, boxstyle='round,pad=0.02', fc=col, ec='k', lw=0.8))
        ax.text(x + wd / 2, y + ht / 2, txt, ha='center', va='center', fontsize=9.5)
    ys = [4.6, 3.5, 2.4, 1.3]
    for i, (cc, y) in enumerate(zip(chans, ys)):
        box(0.2 + 0.75 * i, y, 1.9, 0.75, f'2 × conv 3×3\n{cc} channels', '#cfe2f3')
        box(9.9 - 0.75 * i, y, 1.9, 0.75, f'2 × conv 3×3\n{cc} channels', '#d9ead3')
        ax.annotate('', xy=(9.9 - 0.75 * i, y + 0.37), xytext=(2.1 + 0.75 * i, y + 0.37),
                    arrowprops=dict(arrowstyle='->', color='grey', ls='--'))
        if i < 3:
            ax.annotate('', xy=(0.2 + 0.75 * (i + 1) + 0.5, ys[i + 1] + 0.75), xytext=(0.2 + 0.75 * i + 0.5, y),
                        arrowprops=dict(arrowstyle='->', color='tab:red', lw=1.3))
            ax.annotate('', xy=(9.9 - 0.75 * i + 1.4, y), xytext=(9.9 - 0.75 * (i + 1) + 1.4, ys[i + 1] + 0.75),
                        arrowprops=dict(arrowstyle='->', color='tab:green', lw=1.3))
    box(4.8, 0.1, 2.4, 0.75, 'bottleneck\n256 channels', '#fff2cc')
    ax.annotate('', xy=(4.8, 0.47), xytext=(0.2 + 2.25 + 0.5, ys[3]), arrowprops=dict(arrowstyle='->', color='tab:red', lw=1.3))
    ax.annotate('', xy=(9.9 - 2.25 + 1.4, ys[3]), xytext=(7.2, 0.47), arrowprops=dict(arrowstyle='->', color='tab:green', lw=1.3))
    ax.text(0.2, 5.55, 'input: grey micrograph, 320 × 240', fontsize=10)
    ax.text(11.8, 5.55, 'output: other / clad / remelt', fontsize=10, ha='right')
    ax.text(6, -0.75, 'red: 2×2 max-pool    green: 2×2 transposed conv    grey dashed: skip connection\n'
            'each convolution followed by batch normalisation and ReLU; 1.9 M parameters', fontsize=9.5, ha='center')
    ax.set_title('(a) U-Net architecture', fontsize=11.5)
    d = pd.read_csv(os.path.join(seg, 'optomec_segmented.csv'))
    r = d[d.qc_pass == True].iloc[40]
    key = f'{r.id}{r.sec}'
    g, m = section(key)
    box_ = crop_box([m], m.shape, pad=.45)
    ax = fig.add_subplot(gs[1, 0]); outline(ax, g, np.zeros_like(m), box_, surf=False)
    ax.set_title(f'(b) Input: section {key[-5:]}')
    rgb = np.stack([g] * 3, -1).astype(float) / 255 * 0.6
    rgb[m == 1] = (0.2, 0.8, 0.2); rgb[m == 2] = (0.9, 0.2, 0.2)
    y0, y1, x0, x1 = box_
    ax = fig.add_subplot(gs[1, 1]); ax.imshow(rgb[y0:y1, x0:x1]); ax.axis('off')
    ax.set_title('(c) Label: green clad, red remelt')
    fig.tight_layout(h_pad=.5)
    fig.savefig(os.path.join(out, 'fig_unet.png'))


def mount_examples():
    """Random segmented sections from each imaging set, marked accepted/rejected (no cherry-picking)."""
    d = pd.read_csv(os.path.join(seg, 'optomec_segmented.csv'))
    d['path'] = [os.path.join(seg, 'masks', f'{a}{b}.png') for a, b in zip(d.id, d.sec)]
    d = d[d.path.map(os.path.exists) & d.D_ref.notna()].copy()
    n = d.id.str[-4:].astype(int)  # imaging sets as in the acceptance table: tracks 1-99 vs 200-284
    fig, axs = plt.subplots(4, 2, figsize=(FULL, 8.9))
    for col, (name, sub) in enumerate([('(a) Tracks 1–99 (mostly dark mount)', d[n <= 99]),
                                       ('(b) Tracks 200–284 (bright mount)', d[n >= 200])]):
        for ax, (_, r) in zip(axs[:, col], sub.sample(4, random_state=3).iterrows()):
            g, m = section(f'{r.id}{r.sec}')
            outline(ax, g, m, crop_box([m], m.shape, pad=.3, aspect=1.55), lw=1.4)
            ok = bool(r.qc_pass)
            ax.text(.02, .97, f'{r.id[-4:]}{r.sec}: ' + ('accepted' if ok else 'rejected'), transform=ax.transAxes,
                    va='top', fontsize=10, color='w', weight='bold',
                    bbox=dict(fc='green' if ok else '#c0392b', alpha=.85, lw=0, pad=2))
            ax.set_xlabel(errors(r), fontsize=9.5)
        axs[0, col].set_title(name, fontsize=11.5)
        print(name, 'acceptance in this set', round(sub.qc_pass.mean(), 2))
    fig.tight_layout(h_pad=.4)
    mask_legend(fig, loc='lower center', bbox_to_anchor=(.5, -.035))
    fig.savefig(os.path.join(out, 'fig_mount_examples.png'))


if __name__ == '__main__':
    which = sys.argv[3:] or ['qc', 'splits', 'unet', 'mount']
    for k, fn in [('qc', qc_examples), ('splits', validation_splits), ('unet', unet_figure), ('mount', mount_examples)]:
        if k in which:
            fn()
