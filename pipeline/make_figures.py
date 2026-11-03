"""Report figures from the result files. Run from the project root (expects data/optomec/...).
Usage: python pipeline/make_figures.py <results_dir> <out_dir>"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import plt, FULL, outline, fill, crop_box, mask_legend
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

TIF = 'data/optomec/micro_tif'


def gray(key, size=(640, 480)):
    from PIL import Image
    p = sorted(f for f in os.listdir(TIF) if f.startswith(key))[-1]
    return np.array(Image.open(os.path.join(TIF, p)).convert('L').resize(size))


def seg_mask(res, key):
    import cv2
    return cv2.imread(os.path.join(res, 'optomec_seg', 'masks', key + '.png'), 0)


def pipeline_diagram(out):
    fig, ax = plt.subplots(figsize=(FULL, 6.4)); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    rows = [.86, .71, .56, .41, .26, .11]
    left = ['Public powder-DED dataset\n(316L): 1947 sections, 1519\nmicrographs, 243 coaxial videos',
            'Rule-based segmentation\n+ expert-agreement check',
            'Self-trained U-Net\n(checked masks as labels)',
            'Geometry labels\n$H$, $W$, $D$, $A_d$, $A_s$, dilution',
            'Process models\n(GPR, random forest, log-linear)\nthree validation protocols',
            'Coaxial camera: does it add\ninformation on $H$, $W$, $D$?']
    right = ['Laser-wire DED data\n(IN718 wire on SS304,\nMeltio M450)',
             'Three-phase segmentation\n(dissimilar metals)',
             'U-Net fine-tuned\non seven images',
             'Geometry and dilution\nof the IN718 beads',
             'Process prediction for\nlaser-wire DED\n(future work)']
    bw, bh = .38, .11

    def box(xc, y, t, dashed=False, fc='#eef2f7'):
        ax.add_patch(FancyBboxPatch((xc - bw / 2, y - bh / 2), bw, bh, boxstyle='round,pad=0.008', fc=fc,
                                    ec='#34495e', lw=1.2, ls='--' if dashed else '-'))
        ax.text(xc, y, t, ha='center', va='center', fontsize=9.5, linespacing=1.15)

    def arrow(a, b, txt=None, col='#34495e'):
        ax.annotate('', b, a, arrowprops=dict(arrowstyle='-|>', color=col, lw=1.4, mutation_scale=14))
        if txt:
            ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + .012, txt, ha='center', va='bottom', fontsize=9, color=col)

    xl, xr = .21, .79
    ax.text(xl, .97, 'Powder DED (source domain)', ha='center', fontsize=11.5, weight='bold')
    ax.text(xr, .97, 'Laser-wire DED (target domain)', ha='center', fontsize=11.5, weight='bold')
    for i, t in enumerate(left):
        box(xl, rows[i], t)
        if i:
            arrow((xl, rows[i - 1] - bh / 2), (xl, rows[i] + bh / 2))
    for i, t in enumerate(right):
        box(xr, rows[i], t, dashed=i == 4, fc='#fdf2e9')
        if i:
            arrow((xr, rows[i - 1] - bh / 2), (xr, rows[i] + bh / 2))
    arrow((xl + bw / 2 + .01, rows[2]), (xr - bw / 2 - .01, rows[2]), 'few-shot\ntransfer', '#c0392b')
    arrow((xl + bw / 2 + .01, rows[4]), (xr - bw / 2 - .01, rows[4]), 'residual\nGPR', '#c0392b')
    fig.savefig(os.path.join(out, 'fig1_pipeline.png')); plt.close(fig)


def parity(res, out):
    d = pd.read_csv(os.path.join(res, 'optomec_seg', 'optomec_segmented.csv'))
    u = pd.read_csv(os.path.join(res, 'unet', 'unet_eval.csv'))
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.9))
    for ax, (k, kr, lab) in zip(axs, [('H', 'H', 'Height $H$'), ('W_fus', 'W', 'Width $W$'), ('D', 'D', 'Depth $D$')]):
        q = d[d.qc_pass]
        ax.scatter(d[f'{kr}_ref'], d[k], s=3, c='#c8c8c8', label='rule-based, all segmented')
        ax.scatter(q[f'{kr}_ref'], q[k], s=4, c='C0', label='rule-based, accepted')
        uu = u[u.set == 'qc_fail']
        ok = ((uu.H_unet - uu.H_ref).abs() < .1) & ((uu.W_unet - uu.W_ref).abs() < .1) & ((uu.D_unet - uu.D_ref).abs() < .1)
        ax.scatter(uu[ok][f'{kr}_ref'], uu[ok][f'{kr}_unet'], s=4, c='C2', label='U-Net, recovered')
        lim = [0, 1.6]
        ax.plot(lim, lim, 'k-', lw=.8); ax.plot(lim, [l + .1 for l in lim], 'k:', lw=.8); ax.plot(lim, [l - .1 for l in lim], 'k:', lw=.8)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal'); ax.set_xticks([0, .5, 1, 1.5]); ax.set_yticks([0, .5, 1, 1.5])
        ax.set_xlabel('expert (mm)')
        e = (q[k] - q[f'{kr}_ref']).abs()
        ax.set_title(f'{lab}\nMAE {e.mean()*1000:.0f} µm (accepted)')
    axs[0].set_ylabel('automatic (mm)')
    fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=3, markerscale=3, frameon=False,
               bbox_to_anchor=(.5, -.1))
    fig.tight_layout(); fig.savefig(os.path.join(out, 'fig3_parity.png')); plt.close(fig)


def leakage(res, out):
    r = pd.read_csv(os.path.join(res, 'optomec', 'optomec_cv_results.csv'))
    pick = [('D', 'physics', 'GPR'), ('D', 'raw', 'RF'), ('D', 'physics', 'PowerLaw'), ('H', 'physics', 'RF'), ('W', 'physics', 'GPR'), ('dil', 'physics', 'PowerLaw')]
    fig, ax = plt.subplots(figsize=(FULL, 3.6))
    x = np.arange(len(pick)); w = .27
    for i, (sp, c, lab) in enumerate([('random', '#bbb', 'random split'), ('condition', 'C0', 'unseen settings'), ('extrap_P', 'C3', 'higher power (extrapolation)')]):
        v = [r[(r.target == t) & (r.features == f) & (r.model == m) & (r.split == sp)].R2.iloc[0] for t, f, m in pick]
        b = ax.bar(x + (i - 1) * w, v, w, color=c, label=lab)
        ax.bar_label(b, fmt='%.2f', fontsize=7, padding=1)
    names = {'PowerLaw': 'Log-linear', 'RF': 'Random forest'}
    ax.set_xticks(x, [f'{t if t != "dil" else "$D/(H+D)$"}\n{names.get(m, m)}\n({ {"physics": "grouped"}.get(f, f)})' for t, f, m in pick], fontsize=9.5)
    ax.set_ylabel('$R^2$'); ax.set_ylim(0, 1.08); ax.set_yticks([0, .2, .4, .6, .8, 1])
    ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(.5, 1.0), frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'fig4_leakage.png')); plt.close(fig)


def camera(res, out):
    p = os.path.join(res, 'optomec', 'coaxial_value.csv')
    if not os.path.exists(p):
        return
    r = pd.read_csv(p).set_index('target')
    fig, ax = plt.subplots(figsize=(FULL * .85, 3.3))
    x = np.arange(len(r)); w = .27
    for i, (c, col, lab) in enumerate([('R2_process', '#bbb', 'process parameters only'), ('R2_process_plus_camera', 'C0', 'process + camera'), ('R2_camera_only', 'C1', 'camera only')]):
        b = ax.bar(x + (i - 1) * w, r[c].clip(lower=0), w, color=col, label=lab)
        ax.bar_label(b, fmt='%.2f', fontsize=8.5, padding=1)
    ax.set_xticks(x, [{'dil': '$D/(H+D)$'}.get(t, f'${t}$') for t in r.index]); ax.set_ylabel('$R^2$ on unseen settings'); ax.set_ylim(0, 1.08)
    ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(.5, 1.0), frameon=False, fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'fig8_camera_value.png')); plt.close(fig)


def shape(res, out):
    d = pd.read_csv(os.path.join(res, 'optomec_seg', 'optomec_segmented.csv'))
    q = d[d.qc_pass]
    fig, axs = plt.subplots(1, 2, figsize=(FULL, 3.2))
    fd, fs = q.Ad / (q.W_fus * q.H), q.As / (q.W_fus * q.D)
    axs[0].hist(fd, bins=30, alpha=.7, label=f'clad, $A_d/(WH)$\nmedian {np.median(fd):.2f}')
    axs[0].hist(fs, bins=30, alpha=.7, label=f'remelt, $A_s/(WD)$\nmedian {np.median(fs):.2f}')
    axs[0].set_xlabel('fill fraction of bounding rectangle'); axs[0].set_ylabel('number of sections')
    axs[0].legend(loc='upper left', fontsize=9); axs[0].set_title('(a) Bead shape')
    prox = q.D_ref / (q.H_ref + q.D_ref)
    axs[1].scatter(q.dilution * 100, prox * 100, s=5)
    axs[1].plot([0, 100], [0, 100], 'k-', lw=.8)
    axs[1].set_xlabel('area dilution (%, segmented)'); axs[1].set_ylabel('$D/(H+D)$ (%, expert)')
    axs[1].set_title(f'(b) $r$ = {np.corrcoef(q.dilution, prox)[0,1]:.2f}, proxy {(prox - q.dilution).mean()*100:+.1f} points')
    axs[1].set_aspect('equal'); axs[1].set_xlim(0, 100); axs[1].set_ylim(0, 100)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'fig9_shape_factors.png')); plt.close(fig)


def schematic(out):
    """Cross-section definitions: H, W, D, clad A_d, remelt A_s."""
    fig, ax = plt.subplots(figsize=(FULL * .8, 3.4)); ax.axis('off')
    x = np.linspace(-1, 1, 200)
    ax.fill_between([-1.6, 1.6], -1.3, 0, color='#d9dde3')
    ax.fill_between(x, 0, .8 * np.sqrt(1 - x**2), color='#7f93a8')
    ax.fill_between(x, -.6 * np.sqrt(1 - x**2), 0, color='#a9b8c7')
    ax.plot([-1.6, 1.6], [0, 0], 'k--', lw=1)
    ax.annotate('', (0, .8), (0, 0), arrowprops=dict(arrowstyle='<->', lw=1.3)); ax.text(.06, .38, '$H$', fontsize=15)
    ax.annotate('', (0, -.6), (0, 0), arrowprops=dict(arrowstyle='<->', lw=1.3)); ax.text(.06, -.36, '$D$', fontsize=15)
    ax.annotate('', (1, .05), (-1, .05), arrowprops=dict(arrowstyle='<->', lw=1.3)); ax.text(-.92, .12, '$W$', fontsize=15)
    ax.text(-.62, .45, r'clad $A_d$', fontsize=13, color='w'); ax.text(-.88, -.3, r'remelt $A_s$', fontsize=13)
    ax.text(1.05, .07, 'original substrate\nsurface', fontsize=11, va='bottom'); ax.text(.7, -1.1, 'substrate', fontsize=12)
    ax.text(-1.55, -1.1, r'dilution $\delta = A_s/(A_d+A_s)$', fontsize=13)
    ax.set_xlim(-1.6, 2.2); ax.set_ylim(-1.3, 1.0)
    fig.savefig(os.path.join(out, 'fig0_cross_section.png')); plt.close(fig)


def examples(res, out):
    """Raw micrograph and rule-based segmentation for one accepted section of each imaging set."""
    d = pd.read_csv(os.path.join(res, 'optomec_seg', 'optomec_segmented.csv'))
    q = d[d.qc_pass].copy(); q['n'] = q.id.str[-4:].astype(int)
    fig, axs = plt.subplots(2, 2, figsize=(FULL, 5.6))
    for row, (name, sub) in enumerate([('Dark mount', q[q.n < 100]), ('Bright mount', q[q.n >= 200])]):
        r = sub.sample(1, random_state=3).iloc[0]; key = r.id + r.sec
        g, m = gray(key), seg_mask(res, key)
        box = crop_box([m], m.shape, pad=.3)
        outline(axs[row, 0], g, np.zeros_like(m), box, surf=False)
        outline(axs[row, 1], g, m, box)
        axs[row, 0].set_ylabel(f'{name}\nsection {key[-5:]}', fontsize=11)
        axs[row, 1].set_xlabel(f'$D$ = {r.D:.2f} mm (expert {r.D_ref:.2f})\n$W$ = {r.W_fus:.2f} mm (expert {r.W_ref:.2f})', fontsize=10)
    axs[0, 0].set_title('(a) Micrograph'); axs[0, 1].set_title('(b) Rule-based segmentation')
    fig.tight_layout(h_pad=1.2)
    mask_legend(fig, loc='lower center', bbox_to_anchor=(.5, -.04))
    fig.savefig(os.path.join(out, 'fig2a_examples.png')); plt.close(fig)


def registration_check(res, out, coax_dir='data/optomec/coaxial', tid='DED_SS316L_0017'):
    """Glow area A30 along one multi-speed track: equal bins vs 1/V prior vs snapped boundaries."""
    import re, zipfile, cv2
    from coaxial_features import frame_feats, snap_edges
    zp = os.path.join(coax_dir, f'{tid}_denoise.zip')
    if not os.path.exists(zp):
        return
    sec = pd.read_csv(os.path.join(res, 'optomec', 'optomec_sections_mm.csv'))
    z = zipfile.ZipFile(zp)
    names = sorted([n for n in z.namelist() if n.endswith('.jpg')], key=lambda n: int(re.search(r'prim(\d+)', n).group(1)))
    F = [frame_feats(cv2.imdecode(np.frombuffer(z.read(n), np.uint8), 0)) for n in names]
    on = [i for i, f in enumerate(F) if f]; F = F[on[0]:on[-1] + 1]
    s = sec[sec.id == tid].sort_values('sec')
    e = snap_edges(F, s.V.values); w = (1 / s.V.values) / (1 / s.V.values).sum(); e0 = np.r_[0, np.cumsum(w)] * len(F)
    fig, ax = plt.subplots(figsize=(FULL, 3.4)); ax.plot([f['A30'] if f else np.nan for f in F], '.', ms=2.5, c='k')
    for x in e0: ax.axvline(x, c='C3', alpha=.45, lw=1.5)
    for x in e: ax.axvline(x, c='C2', lw=2.2)
    for x in np.linspace(0, len(F), len(s) + 1): ax.axvline(x, c='C0', ls='--', lw=1.5)
    for (a, b), v in zip(zip(e[:-1], e[1:]), s.V):
        ax.text((a + b) / 2, 1.27, f'{v:.0f} mm/min', ha='center', fontsize=9.5)
    ax.set_xlabel('frame number'); ax.set_ylabel('glow area $A_{30}$ (mm$^2$)'); ax.set_ylim(.38, 1.32)
    ax.legend(handles=[Line2D([], [], c='C0', ls='--', lw=1.5, label='equal bins (dataset toolbox)'),
                       Line2D([], [], c='C3', alpha=.45, lw=1.5, label='$1/V$ prior'),
                       Line2D([], [], c='C2', lw=2.2, label='snapped to steps (this work)')],
              ncol=3, loc='lower center', bbox_to_anchor=(.5, 1.0), frameon=False, fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'fig7_camera_registration.png')); plt.close(fig)


def unet_compare(res, out):
    """Bright-mount rule-based rejects: rule-based vs U-Net (3 recovered, 3 still failing)."""
    import cv2, torch
    from unet_seg import UNet, load_img, predict
    p = os.path.join(res, 'unet', 'unet_eval.csv')
    if not os.path.exists(p):
        return
    r = pd.read_csv(p); r['n'] = r.key.str[11:15].astype(int)
    f = r[r.set == 'qc_fail'].copy()
    ok = ((f.H_unet - f.H_ref).abs() < .1) & ((f.W_unet - f.W_ref).abs() < .1) & ((f.D_unet - f.D_ref).abs() < .1)
    groups = [('Recovered by the U-Net', f[ok & (f.n >= 200)].sample(3, random_state=2)),
              ('Still difficult (faint fusion line)', f[~ok & (f.n >= 200)].sample(3, random_state=2))]
    net = UNet(); net.load_state_dict(torch.load(os.path.join(res, 'unet', 'unet.pt'))); net.eval()
    fig, axs = plt.subplots(3, 4, figsize=(FULL, 4.9))
    for gi, (gname, sub) in enumerate(groups):
        for i, (_, row) in enumerate(sub.iterrows()):
            g = gray(row.key)
            mr = seg_mask(res, row.key)
            mu = cv2.resize(predict(net, load_img(TIF, row.key)), (640, 480), interpolation=cv2.INTER_NEAREST)
            box = crop_box([mr, mu], mr.shape, pad=.2)
            for j, (m, Dv) in enumerate([(mr, row.D_cls), (mu, row.D_unet)]):
                ax = axs[i, 2 * gi + j]; outline(ax, g, m, box, lw=1.3)
                ax.set_xlabel(f'$D$ {Dv:.2f} (expert {row.D_ref:.2f})', fontsize=9)
            axs[i, 2 * gi].set_ylabel(row.key[-5:], fontsize=10)
        axs[0, 2 * gi].set_title('rule-based', fontsize=10.5); axs[0, 2 * gi + 1].set_title('U-Net', fontsize=10.5)
    fig.tight_layout(h_pad=.6, w_pad=.4, rect=(0, 0, 1, .94))
    for gi, (gname, _) in enumerate(groups):
        x0, x1 = axs[0, 2 * gi].get_position().x0, axs[0, 2 * gi + 1].get_position().x1
        fig.text((x0 + x1) / 2, .975, f'({"ab"[gi]}) {gname}', ha='center', va='top', fontsize=11.5, weight='bold')
        fig.add_artist(Line2D([x0, x1], [.94, .94], color='k', lw=.8))
    mask_legend(fig, loc='lower center', bbox_to_anchor=(.5, -.05))
    fig.savefig(os.path.join(out, 'fig2b_unet_vs_classical.png')); plt.close(fig)


def transfer_process_plot(res, out):
    r = pd.read_csv(os.path.join(res, 'transfer_learning', 'transfer_process_runs.csv'))
    meth = [('zero-shot', 'k', 'source model, no target data'), ('scratch', 'C3', 'target data only (from scratch)'),
            ('recal', 'C0', 'recalibration $a f_s + b$'), ('delta', 'C2', '$\\Delta$-learning (residual GPR)')]
    fig, axs = plt.subplots(2, 2, figsize=(FULL, 5.6))
    for ax, (t, lab, sc) in zip(axs.ravel(), [('H', '(a) Height $H$ (mm)', 1), ('W', '(b) Width $W$ (mm)', 1),
                                              ('D', '(c) Depth $D$ (mm)', 1), ('dil_area', '(d) Area dilution (points)', 100)]):
        for m, c, name in meth:
            g = r[(r.target == t) & (r.method == m)]
            if g.empty:
                continue
            s = g.groupby('k').MAE.quantile([.25, .5, .75]).unstack() * sc
            if m == 'zero-shot':
                ax.axhline(s[.5].iloc[0], color='k', ls='--', lw=1.2, label=name)
                continue
            s = s[s.index > 0]
            ax.plot(s.index, s[.5], 'o-', color=c, ms=4, lw=1.6, label=name)
            ax.fill_between(s.index, s[.25], s[.75], color=c, alpha=.15)
        ax.set_title(lab); ax.set_xticks([1, 2, 3, 5, 8, 12])
    for ax in axs[1]:
        ax.set_xlabel('target conditions used, $k$')
    for ax in axs[:, 0]:
        ax.set_ylabel('MAE, unseen targets')
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1.0))
    fig.savefig(os.path.join(out, 'fig6_transfer_process.png')); plt.close(fig)


def transfer_unet_plot(res, out):
    r = pd.read_csv(os.path.join(res, 'transfer_learning', 'unet_transfer_runs.csv'))
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 3.0))
    for ax, (c, lab) in zip(axs, [('MAE_D', '(a) Depth MAE (mm)'), ('all3_within_0p1', '(b) Share with $H$, $W$, $D$\nwithin 0.1 mm'),
                                  ('IoU_remelt', '(c) Remelt IoU')]):
        ax.axhline(r[r.method == 'source-only'][c].mean(), color='k', ls='--', lw=1.2, label='source model, no target labels')
        for m, col, name in [('fine-tune', 'C2', 'fine-tuned source model'), ('scratch', 'C3', 'trained from scratch')]:
            g = r[r.method == m].groupby('k')[c].agg(['mean', 'min', 'max'])
            ax.errorbar(g.index, g['mean'], yerr=[g['mean'] - g['min'], g['max'] - g['mean']], fmt='o-', color=col,
                        ms=4.5, lw=1.6, capsize=3, label=name)
        ax.set_title(lab, fontsize=10.5); ax.set_xticks([5, 10, 20, 40]); ax.set_xlabel('labelled images, $k$')
    fig.tight_layout(rect=(0, 0, 1, .9))
    fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(.5, 1.0))
    fig.savefig(os.path.join(out, 'fig5_transfer_unet.png')); plt.close(fig)


def wire_seg_plot(res, out):
    """3 x 3 grid of the IN718 panels (rows P, columns V) with the three-phase segmentation."""
    import cv2
    pd_ = os.path.join(res, 'wire', 'panels')
    geo = pd.read_csv(os.path.join(res, 'wire', 'wire_geometry.csv'))
    fig, axs = plt.subplots(3, 3, figsize=(FULL, 4.75))
    for i, P in enumerate([700, 900, 1100]):
        for j, V in enumerate([450, 600, 750]):
            k = f'{P}_{V}'
            g = cv2.cvtColor(cv2.imread(os.path.join(pd_, k + '.png')), cv2.COLOR_BGR2RGB)
            m = cv2.imread(os.path.join(pd_, k + '_mask.png'), 0)
            ax = axs[i, j]; fill(ax, cv2.cvtColor(g, cv2.COLOR_RGB2GRAY), m, alpha=.5)
            dl = geo[(geo.P == P) & (geo.V == V)].dilution.iloc[0] * 100
            ax.text(.02, .97, f'{dl:.0f} %', transform=ax.transAxes, va='top', fontsize=10.5, color='w', weight='bold',
                    bbox=dict(fc='k', alpha=.6, lw=0, pad=1.5))
            if i == 0:
                ax.set_title(f'$V$ = {V} mm/min')
            if j == 0:
                ax.axis('on'); ax.set_xticks([]); ax.set_yticks([]); ax.set_ylabel(f'$P$ = {P} W')
                for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(h_pad=.3, w_pad=.3)
    mask_legend(fig, filled=True, loc='lower center', bbox_to_anchor=(.5, -.06))
    fig.savefig(os.path.join(out, 'fig10a_target_lab_segmentation.png')); plt.close(fig)


def wire_unet_plot(res, out):
    """Label vs zero-shot / fine-tuned / scratch U-Net on three IN718 panels (masks saved by wire_unet_transfer)."""
    import cv2
    p = os.path.join(res, 'transfer_learning', 'wire_unet_preds.npz')
    if not os.path.exists(p):
        return
    z = np.load(p)
    t = pd.read_csv(os.path.join(res, 'transfer_learning', 'wire_unet_transfer.csv'))
    cols = [('label', 'label\n(three-phase)'), ('zero-shot', 'zero-shot\n(316L U-Net)'),
            ('fine-tuned (7 panels)', 'fine-tuned\n(7 images)'), ('scratch (7 panels)', 'from scratch\n(7 images)')]
    show = ['900_750', '1100_600', '1100_750']
    fig, axs = plt.subplots(len(show), 4, figsize=(FULL, 4.3))
    for i, k in enumerate(show):
        g = z[f'{k}|img']; y = z[f'{k}|label']
        for j, (name, title) in enumerate(cols):
            m = y if name == 'label' else z[f'{k}|{name}']
            ax = axs[i, j]; fill(ax, g, m)
            if name != 'label':
                rr = t[(t.panel == k) & (t.method == name)].iloc[0]
                ax.text(.5, -.04, f'IoU {rr.IoU_clad:.2f} / {rr.IoU_remelt:.2f}', transform=ax.transAxes,
                        ha='center', va='top', fontsize=9)
            if i == 0:
                ax.set_title(title, fontsize=10)
        P, V = k.split('_')
        axs[i, 0].axis('on'); axs[i, 0].set_xticks([]); axs[i, 0].set_yticks([])
        axs[i, 0].set_ylabel(f'{P} W\n{V} mm/min', fontsize=10)
        for s in axs[i, 0].spines.values(): s.set_visible(False)
    fig.tight_layout(h_pad=1.2, w_pad=.3)
    mask_legend(fig, filled=True, loc='lower center', bbox_to_anchor=(.5, -.07))
    fig.savefig(os.path.join(out, 'fig10c_target_lab_unet_transfer.png')); plt.close(fig)


def main(res, out):
    os.makedirs(out, exist_ok=True)
    for fn in (pipeline_diagram, schematic):
        fn(out)
    for fn in (parity, leakage, camera, shape, examples, registration_check, unet_compare, transfer_process_plot,
               transfer_unet_plot, wire_seg_plot, wire_unet_plot):
        fn(res, out)
    print(sorted(os.listdir(out)))


if __name__ == '__main__':
    main(*sys.argv[1:3])
