"""Extra report figures: QC accept/reject examples, validation-split diagram, U-Net architecture + label example.
Usage: python -I pipeline/report_extra_figs.py <BTP root> <figs_dir> [qc splits unet mount]"""
import sys, os
import numpy as np, pandas as pd
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.model_selection import KFold, GroupKFold

root, out = sys.argv[1], sys.argv[2]
seg = os.path.join(root, 'results/optomec_seg')


def qc_examples():
    d = pd.read_csv(os.path.join(seg, 'optomec_segmented.csv'))
    d['n'] = d.id.str[-4:].astype(int)
    for k, km in (('H', 'H'), ('W', 'W_fus'), ('D', 'D')):  # same quantities as the QC in segment_optomec
        d['e' + k] = (d[km] - d[k + '_ref']).abs()
    ok = lambda r: os.path.exists(os.path.join(seg, 'overlays', f'{r.id}{r.sec}.jpg'))
    d = d[d.apply(ok, axis=1)]
    acc = d[d.qc_pass == True]
    rej = d[(d.qc_pass == False) & d.eH.notna()]
    picks = [('Accepted (dark mount)', acc[acc.n <= 99].sort_values('eD').iloc[len(acc[acc.n <= 99]) // 2]),
             ('Accepted (bright mount)', acc[acc.n >= 200].sort_values('eD').iloc[len(acc[acc.n >= 200]) // 2]),
             ('Rejected: faint fusion line missed ($D$)', rej[(rej.eH < 0.05) & (rej.eW < 0.05) & (rej.eD > 0.15)].iloc[0]),
             ('Rejected: boundary follows a scratch ($W$)', rej[(rej.eH < 0.05) & (rej.eW > 0.15) & (rej.eD < 0.06)].iloc[0])]
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.6))
    for ax, (title, r) in zip(axs.ravel(), picks):
        ax.imshow(Image.open(os.path.join(seg, 'overlays', f'{r.id}{r.sec}.jpg')))
        ax.axis('off')
        col = 'green' if r.qc_pass else 'red'
        ax.set_title(f'{title}\nsection {r.id[-4:]}{r.sec}', fontsize=11, color=col)
        ax.text(0.5, -0.03, f'|$\\Delta H$| = {r.eH * 1000:.0f} $\\mu$m   |$\\Delta W$| = {r.eW * 1000:.0f} $\\mu$m   '
                f'|$\\Delta D$| = {r.eD * 1000:.0f} $\\mu$m', transform=ax.transAxes, ha='center', va='top', fontsize=10)
    fig.tight_layout(h_pad=2.5)
    fig.savefig(os.path.join(out, 'fig_qc_examples.png'), dpi=130)
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
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    s = 12 + 4 * c.n
    both = (te_rand > 0) & (tr_rand > 0)
    ax = axs[0]
    ax.scatter(c.jx[both], c.jy[both], s=s[both], c='tab:purple', alpha=0.7, label='condition in train and test')
    ax.scatter(c.jx[~both], c.jy[~both], s=s[~both], c='tab:blue', alpha=0.7, label='train only')
    ax.set_title(f'(a) Random split: {both.mean() * 100:.0f} % of conditions\nhave sections in both train and test', fontsize=11)
    ax = axs[1]
    m = c.cond.isin(test_g).values
    ax.scatter(c.jx[~m], c.jy[~m], s=s[~m], c='tab:blue', alpha=0.7, label='train')
    ax.scatter(c.jx[m], c.jy[m], s=s[m], c='tab:orange', alpha=0.8, label='test')
    ax.set_title('(b) Grouped by condition: test settings\nare never seen in training', fontsize=11)
    ax = axs[2]
    m = (c.P >= 450).values
    ax.scatter(c.jx[~m], c.jy[~m], s=s[~m], c='tab:blue', alpha=0.7, label=f'train, P $\\leq$ 400 W ({(~m).sum()} conditions)')
    ax.scatter(c.jx[m], c.jy[m], s=s[m], c='tab:orange', alpha=0.8, label=f'test, P $\\geq$ 450 W ({m.sum()} conditions)')
    ax.axhline(425, color='k', ls='--', lw=1)
    ax.set_title('(c) Extrapolation in power: test power\nlies outside the training range', fontsize=11)
    for ax in axs:
        ax.set_xscale('log'); ax.set_xlabel('travel speed V (mm/min)'); ax.grid(alpha=0.3)
        ax.legend(loc='lower left', fontsize=8.5, framealpha=0.9)
    axs[0].set_ylabel('laser power P (W)')
    fig.suptitle('Process conditions of the public dataset (marker size: number of sections; small jitter added)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig_validation_splits.png'), dpi=130)


def unet_figure():
    fig = plt.figure(figsize=(15, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.3, 1, 1])
    ax = fig.add_subplot(gs[0]); ax.axis('off'); ax.set_xlim(0, 12); ax.set_ylim(-0.6, 6)
    chans = [16, 32, 64, 128]
    def box(x, y, wd, ht, txt, col):
        ax.add_patch(FancyBboxPatch((x, y), wd, ht, boxstyle='round,pad=0.02', fc=col, ec='k', lw=0.8))
        ax.text(x + wd / 2, y + ht / 2, txt, ha='center', va='center', fontsize=8.5)
    ys = [4.6, 3.5, 2.4, 1.3]
    for i, (cc, y) in enumerate(zip(chans, ys)):
        box(0.4 + 0.7 * i, y, 1.6, 0.7, f'2 x conv3x3\n{cc} ch', '#cfe2f3')
        box(9.6 - 0.7 * i, y, 1.6, 0.7, f'2 x conv3x3\n{cc} ch', '#d9ead3')
        ax.annotate('', xy=(9.6 - 0.7 * i, y + 0.35), xytext=(2.0 + 0.7 * i, y + 0.35),
                    arrowprops=dict(arrowstyle='->', color='grey', ls='--'))
        if i < 3:
            ax.annotate('', xy=(0.4 + 0.7 * (i + 1) + 0.5, ys[i + 1] + 0.7), xytext=(0.4 + 0.7 * i + 0.5, y),
                        arrowprops=dict(arrowstyle='->', color='tab:red'))
            ax.annotate('', xy=(9.6 - 0.7 * i + 1.1, y), xytext=(9.6 - 0.7 * (i + 1) + 1.1, ys[i + 1] + 0.7),
                        arrowprops=dict(arrowstyle='->', color='tab:green'))
    box(4.9, 0.2, 2.2, 0.7, 'bottleneck\n256 ch', '#fff2cc')
    ax.annotate('', xy=(4.9, 0.55), xytext=(0.4 + 2.1 + 0.5, ys[3]), arrowprops=dict(arrowstyle='->', color='tab:red'))
    ax.annotate('', xy=(9.6 - 2.1 + 1.1, ys[3]), xytext=(7.1, 0.55), arrowprops=dict(arrowstyle='->', color='tab:green'))
    ax.text(0.4, 5.55, 'input: grey micrograph\n320 x 240, standardised', fontsize=9)
    ax.text(9.0, 5.55, 'output: 3 classes per pixel\n(other / clad / remelt)', fontsize=9)
    ax.text(0.2, -0.45, 'red: 2x2 max-pool   green: 2x2 transposed conv   dashed: skip connection (concatenate)\n'
            'each conv: batch norm + ReLU;  1.9 M parameters', fontsize=8.5)
    ax.set_title('(a) U-Net used in this work', fontsize=11)
    d = pd.read_csv(os.path.join(seg, 'optomec_segmented.csv'))
    r = d[d.qc_pass == True].iloc[40]
    key = f'{r.id}{r.sec}'
    g = np.array(Image.open(os.path.join(root, 'data/optomec/micro_tif', key + '.tif')).convert('L').resize((640, 480)))
    m = np.array(Image.open(os.path.join(seg, 'masks', key + '.png')))
    ax = fig.add_subplot(gs[1]); ax.imshow(g, cmap='gray'); ax.axis('off'); ax.set_title(f'(b) Input: section {key[-5:]}', fontsize=11)
    rgb = np.stack([g] * 3, -1).astype(float) / 255 * 0.55
    rgb[m == 1] = (0.2, 0.8, 0.2); rgb[m == 2] = (0.9, 0.2, 0.2)
    ax = fig.add_subplot(gs[2]); ax.imshow(rgb); ax.axis('off')
    ax.set_title('(c) Training label (accepted rule-based\nmask): green clad, red remelt', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig_unet.png'), dpi=130)


def mount_examples():
    """Random segmented sections from each imaging set, marked accepted/rejected (no cherry-picking)."""
    d = pd.read_csv(os.path.join(seg, 'optomec_segmented.csv'))
    d['path'] = [os.path.join(seg, 'overlays', f'{a}{b}.jpg') for a, b in zip(d.id, d.sec)]
    d = d[d.path.map(os.path.exists) & d.D_ref.notna()].copy()
    n = d.id.str[-4:].astype(int)  # imaging sets as in the acceptance table: tracks 1-99 vs 200-284
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.4))
    for row, (name, sub) in enumerate([('tracks 1-99', d[n <= 99]), ('tracks 200-284', d[n >= 200])]):
        for ax, (_, r) in zip(axs[row], sub.sample(4, random_state=3).iterrows()):
            ax.imshow(Image.open(r.path)); ax.axis('off')
            ok = bool(r.qc_pass)
            ax.set_title(f'{name}, {r.id[-4:]}{r.sec}: ' + ('accepted' if ok else 'rejected'),
                         fontsize=10, color='green' if ok else 'red')
            ax.text(0.5, -0.03, f'|$\\Delta H$| {abs(r.H - r.H_ref) * 1000:.0f}, |$\\Delta W$| {abs(r.W_fus - r.W_ref) * 1000:.0f}, '
                    f'|$\\Delta D$| {abs(r.D - r.D_ref) * 1000:.0f} $\\mu$m', transform=ax.transAxes, ha='center', va='top', fontsize=9)
        print(name, 'acceptance in this set', round(sub.qc_pass.mean(), 2))
    fig.tight_layout(h_pad=2)
    fig.savefig(os.path.join(out, 'fig_mount_examples.png'), dpi=130)


if __name__ == '__main__':
    which = sys.argv[3:] or ['qc', 'splits', 'unet', 'mount']
    for k, fn in [('qc', qc_examples), ('splits', validation_splits), ('unet', unet_figure), ('mount', mount_examples)]:
        if k in which:
            fn()
