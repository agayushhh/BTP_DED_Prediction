"""Report figures from the result files. Run from the project root (expects data/optomec/...).
Usage: python pipeline/make_figures.py <results_dir> <out_dir>"""
import sys, os, shutil
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def pipeline_diagram(out):
    fig, ax = plt.subplots(figsize=(11, 3.6)); ax.axis('off')
    boxes = [
        (0.0, .62, 'Public powder-DED data (316L)\n1947 sections, 1519 micrographs,\n243 coaxial videos'),
        (0.0, .12, 'Laser-wire DED data\n(IN718 wire on SS304)'),
        (.25, .62, 'Rule-based segmentation\nthree-phase / fusion-line DP\n+ expert QC'),
        (.25, .12, 'Self-trained U-Net\n(validated masks as labels)'),
        (.50, .62, 'Geometry labels\nH, W, D, A_d, A_s, dilution\nshape factors'),
        (.50, .12, 'Process models\nGPR / RF / power law\nphysics features'),
        (.75, .62, 'Few-shot transfer\nimage: fine-tune U-Net\nprocess: Δ-GPR residual'),
        (.75, .12, 'Leakage-free validation\nunseen settings,\nextrapolation in power'),
    ]
    for x, y, t in boxes:
        ax.add_patch(FancyBboxPatch((x, y), .21, .28, boxstyle='round,pad=0.01', fc='#eef2f7', ec='#34495e'))
        ax.text(x + .105, y + .14, t, ha='center', va='center', fontsize=8)
    arr = [((.21, .76), (.25, .76)), ((.355, .62), (.355, .40)), ((.46, .76), (.50, .76)), ((.46, .26), (.50, .26)),
           ((.605, .62), (.605, .40)), ((.71, .76), (.75, .76)), ((.71, .26), (.75, .26)), ((.855, .62), (.855, .40)),
           ((.21, .26), (.25, .26))]
    for a, b in arr:
        ax.annotate('', b, a, arrowprops=dict(arrowstyle='->', color='#34495e'))
    ax.set_xlim(-.01, .97); ax.set_ylim(.05, .95)
    fig.savefig(os.path.join(out, 'fig1_pipeline.png'), dpi=180, bbox_inches='tight'); plt.close(fig)


def parity(res, out):
    d = pd.read_csv(os.path.join(res, 'optomec_seg', 'optomec_segmented.csv'))
    u = pd.read_csv(os.path.join(res, 'unet', 'unet_eval.csv'))
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (k, kr, lab) in zip(axs, [('H', 'H', 'height H'), ('W_fus', 'W', 'width W'), ('D', 'D', 'depth D')]):
        q = d[d.qc_pass]
        ax.scatter(d[f'{kr}_ref'], d[k], s=4, c='#bbb', label='rule-based, all segmented')
        ax.scatter(q[f'{kr}_ref'], q[k], s=5, c='C0', label='rule-based, accepted')
        uu = u[u.set == 'qc_fail']
        ok = ((uu.H_unet - uu.H_ref).abs() < .1) & ((uu.W_unet - uu.W_ref).abs() < .1) & ((uu.D_unet - uu.D_ref).abs() < .1)
        ax.scatter(uu[ok][f'{kr}_ref'], uu[ok][f'{kr}_unet'], s=5, c='C2', label='U-Net, recovered')
        lim = [0, max(d[f'{kr}_ref'].max(), 1.6)]
        ax.plot(lim, lim, 'k-', lw=.8); ax.plot(lim, [l + .1 for l in lim], 'k:', lw=.6); ax.plot(lim, [l - .1 for l in lim], 'k:', lw=.6)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel(f'expert {lab} (mm)'); ax.set_ylabel(f'automatic {lab} (mm)')
        e = (q[k] - q[f'{kr}_ref']).abs()
        ax.set_title(f'{lab}: accepted MAE {e.mean()*1000:.0f} µm (n={len(q)})')
    axs[0].legend(fontsize=7, markerscale=2)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'fig3_parity.png'), dpi=180); plt.close(fig)


def leakage(res, out):
    r = pd.read_csv(os.path.join(res, 'optomec', 'optomec_cv_results.csv'))
    pick = [('D', 'physics', 'GPR'), ('D', 'raw', 'RF'), ('D', 'physics', 'PowerLaw'), ('H', 'physics', 'RF'), ('W', 'physics', 'GPR'), ('dil', 'physics', 'PowerLaw')]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    x = np.arange(len(pick)); w = .27
    for i, (sp, c, lab) in enumerate([('random', '#bbb', 'random split (leaky)'), ('condition', 'C0', 'unseen settings'), ('extrap_P', 'C3', 'higher power (extrapolation)')]):
        v = [r[(r.target == t) & (r.features == f) & (r.model == m) & (r.split == sp)].R2.iloc[0] for t, f, m in pick]
        ax.bar(x + (i - 1) * w, v, w, color=c, label=lab)
    ax.set_xticks(x, [f'{t if t != "dil" else "D/(H+D)"}\n{m} ({f})' for t, f, m in pick])
    ax.set_ylabel('R²'); ax.set_ylim(0, 1); ax.legend(fontsize=7, ncol=3, loc='lower right')
    ax.set_title('Same models, three validation protocols: random splits overstate accuracy')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'fig4_leakage.png'), dpi=180); plt.close(fig)


def camera(res, out):
    p = os.path.join(res, 'optomec', 'coaxial_value.csv')
    if not os.path.exists(p):
        return
    r = pd.read_csv(p).set_index('target')
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(r)); w = .27
    for i, (c, col, lab) in enumerate([('R2_process', '#bbb', 'process only'), ('R2_process_plus_camera', 'C0', 'process + camera'), ('R2_camera_only', 'C1', 'camera only')]):
        ax.bar(x + (i - 1) * w, r[c].clip(lower=0), w, color=col, label=lab)
    ax.set_xticks(x, [{'dil': 'D/(H+D)'}.get(t, t) for t in r.index]); ax.set_ylabel('R² (unseen settings)'); ax.set_ylim(0, 1)
    ax.legend(fontsize=7); ax.set_title('Coaxial camera senses visible geometry (H, W), not penetration')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'fig8_camera_value.png'), dpi=180); plt.close(fig)


def shape(res, out):
    d = pd.read_csv(os.path.join(res, 'optomec_seg', 'optomec_segmented.csv'))
    q = d[d.qc_pass]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
    axs[0].hist(q.Ad / (q.W_fus * q.H), bins=30, alpha=.7, label=f'A_d/(W·H), median {np.median(q.Ad/(q.W_fus*q.H)):.2f}')
    axs[0].hist(q.As / (q.W_fus * q.D), bins=30, alpha=.7, label=f'A_s/(W·D), median {np.median(q.As/(q.W_fus*q.D)):.2f}')
    axs[0].axvline(np.pi / 4, c='k', ls='--', lw=.8); axs[0].axvline(2 / 3, c='k', ls=':', lw=.8)
    axs[0].set_xlabel('shape factor (dashed π/4 ellipse, dotted 2/3 parabola)'); axs[0].legend(fontsize=7)
    prox = q.D_ref / (q.H_ref + q.D_ref)
    axs[1].scatter(q.dilution, prox, s=5)
    axs[1].plot([0, 1], [0, 1], 'k-', lw=.8)
    axs[1].set_xlabel('area dilution A_s/(A_d+A_s) (segmented)'); axs[1].set_ylabel('length proxy D/(H+D) (experts)')
    axs[1].set_title(f'r = {np.corrcoef(q.dilution, prox)[0,1]:.2f}, proxy bias {(prox - q.dilution).mean()*100:+.1f} points')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'fig9_shape_factors.png'), dpi=180); plt.close(fig)


def schematic(out):
    """Cross-section definitions: H, W, D, clad A_d, remelt A_s."""
    fig, ax = plt.subplots(figsize=(6, 3.2)); ax.axis('off')
    x = np.linspace(-1, 1, 200)
    ax.fill_between([-1.6, 1.6], -1.3, 0, color='#d9dde3')
    ax.fill_between(x, 0, .8 * np.sqrt(1 - x**2), color='#7f93a8')
    ax.fill_between(x, -.6 * np.sqrt(1 - x**2), 0, color='#a9b8c7')
    ax.plot([-1.6, 1.6], [0, 0], 'k--', lw=1)
    ax.annotate('', (0, .8), (0, 0), arrowprops=dict(arrowstyle='<->')); ax.text(.05, .38, 'H', fontsize=11)
    ax.annotate('', (0, -.6), (0, 0), arrowprops=dict(arrowstyle='<->')); ax.text(.05, -.35, 'D', fontsize=11)
    ax.annotate('', (1, .05), (-1, .05), arrowprops=dict(arrowstyle='<->')); ax.text(-.9, .1, 'W', fontsize=11)
    ax.text(-.45, .45, r'clad $A_d$', fontsize=10, color='w'); ax.text(-.85, -.3, r'remelt $A_s$', fontsize=10)
    ax.text(1.05, .05, 'original substrate surface', fontsize=8); ax.text(.75, -1.1, 'substrate', fontsize=9)
    ax.text(-1.55, -1.1, r'dilution $= A_s/(A_d+A_s)$', fontsize=10)
    ax.set_xlim(-1.6, 2.3); ax.set_ylim(-1.3, 1.0)
    fig.savefig(os.path.join(out, 'fig0_cross_section.png'), dpi=200, bbox_inches='tight'); plt.close(fig)


def examples(res, out, tif_dir='data/optomec/micro_tif'):
    """Raw micrograph and rule-based overlay for one accepted section of each imaging set."""
    import cv2
    from PIL import Image
    d = pd.read_csv(os.path.join(res, 'optomec_seg', 'optomec_segmented.csv'))
    q = d[d.qc_pass].copy(); q['n'] = q.id.str[-4:].astype(int)
    tiles = []
    for sub in (q[q.n < 100], q[q.n >= 200]):
        r = sub.sample(1, random_state=3).iloc[0]; key = r.id + r.sec
        paths = sorted(f for f in os.listdir(tif_dir) if f.startswith(key))
        if not paths:
            return
        raw = cv2.cvtColor(np.array(Image.open(os.path.join(tif_dir, paths[-1])).convert('L').resize((640, 480))), cv2.COLOR_GRAY2BGR)
        ov = cv2.imread(os.path.join(res, 'optomec_seg', 'overlays', key + '.jpg'))
        tiles.append(np.hstack([raw, ov]))
    cv2.imwrite(os.path.join(out, 'fig2a_examples.jpg'), np.vstack(tiles))


def registration_check(res, coax_dir='data/optomec/coaxial', tid='DED_SS316L_0017'):
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
    fig, ax = plt.subplots(figsize=(8, 3)); ax.plot([f['A30'] if f else np.nan for f in F], '.', ms=2, c='k')
    for x in e0: ax.axvline(x, c='C3', alpha=.4)
    for x in e: ax.axvline(x, c='C2', lw=2)
    for x in np.linspace(0, len(F), len(s) + 1): ax.axvline(x, c='C0', ls='--')
    ax.set_xlabel('frame  | blue dashed: equal bins (original) | faint red: 1/V prior | green: snapped (ours)')
    ax.set_ylabel('glow area A30 (mm²)')
    ax.set_title(f'{tid}  V = {list(s.V)} mm/min', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(res, 'coaxial_registration_check.png'), dpi=130); plt.close(fig)


def unet_compare(res):
    """Bright-mount rule-based rejects: rule-based overlay vs U-Net overlay (3 recovered, 3 still failing)."""
    import cv2
    p = os.path.join(res, 'unet', 'unet_eval.csv')
    if not os.path.exists(p):
        return
    r = pd.read_csv(p); r['n'] = r.key.str[11:15].astype(int)
    f = r[r.set == 'qc_fail'].copy()
    ok = ((f.H_unet - f.H_ref).abs() < .1) & ((f.W_unet - f.W_ref).abs() < .1) & ((f.D_unet - f.D_ref).abs() < .1)
    pick = list(f[ok & (f.n >= 200)].sample(3, random_state=2).key) + list(f[~ok & (f.n >= 200)].sample(3, random_state=2).key)
    ims = []
    for k in pick:
        a = cv2.resize(cv2.imread(os.path.join(res, 'optomec_seg', 'overlays', f'{k}.jpg')), (320, 240))
        b = cv2.imread(os.path.join(res, 'unet', 'overlays', f'{k}.jpg'))
        cv2.putText(b, 'U-Net', (5, 15), 0, .45, (255, 255, 255), 1); ims.append(np.hstack([a, b]))
    cv2.imwrite(os.path.join(res, 'unet', 'compare_grid.jpg'), np.vstack(ims))


def main(res, out):
    os.makedirs(out, exist_ok=True)
    pipeline_diagram(out); parity(res, out); leakage(res, out); camera(res, out); shape(res, out)
    schematic(out); examples(res, out); registration_check(res); unet_compare(res)
    for src, dst in [('transfer_learning/transfer_process_curves.png', 'fig6_transfer_process.png'),
                     ('transfer_learning/unet_transfer_curves.png', 'fig5_transfer_unet.png'),
                     ('coaxial_registration_check.png', 'fig7_camera_registration.png'),
                     ('unet/compare_grid.jpg', 'fig2b_unet_vs_classical.jpg'),
                     ('wire/wire_seg_overlay.png', 'fig10a_target_lab_segmentation.png'),
                     ('transfer_learning/wire_unet_transfer.png', 'fig10c_target_lab_unet_transfer.png')]:
        if os.path.exists(os.path.join(res, src)):
            shutil.copy(os.path.join(res, src), os.path.join(out, dst))
    print(sorted(os.listdir(out)))


if __name__ == '__main__':
    main(*sys.argv[1:3])
