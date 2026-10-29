"""Few-shot transfer learning for cross-section segmentation (simulated new lab).

The public dataset contains two imaging set-ups: dark speckled mount (source) and bright smooth mount with
different lighting/etch (target). Micrographs from another process or lab (e.g. laser-wire IN718/SS304)
are a domain shift of the same kind. Question: how many labelled target images does a source-trained U-Net need?
  source-only : U-Net trained on all validated source masks, applied to target as is
  fine-tune   : source U-Net fine-tuned on k target images (lr 3e-4, 30 epochs)
  scratch     : new U-Net trained on the same k target images only (lr 1e-3, 80 epochs)
Target images split by track: a labelled pool and a fixed held-out test set. k in {5, 10, 20, 40},
3 random draws each. Scores on the held-out target tracks: MAE of H, W, D vs the experts, % of sections
with all three within 0.1 mm, IoU of clad and remelt vs the validated masks.
Usage: python unet_transfer.py <tif_dir> <seg_dir> <out_dir>
"""
import sys, os, glob, copy
import numpy as np, pandas as pd, cv2, torch
from PIL import Image
from unet_seg import UNet, load_img, train, predict, measure, SZ

torch.manual_seed(0); np.random.seed(0)


def bright_mount(tif_dir, key):
    path = sorted(glob.glob(os.path.join(tif_dir, key + '*.tif')))[-1]
    g = np.array(Image.open(path).convert('L').resize((640, 480)), np.uint8)
    return g[:15].mean() > 100


def score(net, X, Y, te):
    rows = []
    for _, r in te.iterrows():
        m = predict(net, X[r.key])
        mm = measure(m) or {}
        iou = [((m == c) & (Y[r.key] == c)).sum() / max(((m == c) | (Y[r.key] == c)).sum(), 1) for c in (1, 2)]
        e = {k: abs(mm[k] - r[f'{k}_ref']) if k in mm else np.nan for k in 'HWD'}
        rows.append(dict(**{f'e{k}': v for k, v in e.items()}, iou_clad=iou[0], iou_remelt=iou[1]))
    s = pd.DataFrame(rows)
    ok = (s[['eH', 'eW', 'eD']] < .1).all(1)
    return dict(MAE_H=s.eH.mean(), MAE_W=s.eW.mean(), MAE_D=s.eD.mean(), all3_within_0p1=ok.mean(),
                IoU_clad=s.iou_clad.mean(), IoU_remelt=s.iou_remelt.mean())


def main(tif_dir, seg_dir, out):
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)
    d = pd.read_csv(os.path.join(seg_dir, 'optomec_segmented.csv'))
    d['key'] = d.id + d.sec
    d = d[d.qc_pass].copy()
    d['bright'] = [bright_mount(tif_dir, k) for k in d.key]
    X = {k: load_img(tif_dir, k) for k in d.key}
    Y = {k: cv2.resize(cv2.imread(os.path.join(seg_dir, 'masks', k + '.png'), 0), SZ, interpolation=cv2.INTER_NEAREST).astype(np.int64) for k in d.key}
    src, tgt = d[~d.bright], d[d.bright]
    tracks = tgt.id.unique(); np.random.default_rng(0).shuffle(tracks)
    test_tracks = set(tracks[: int(len(tracks) * .4)])
    pool, te = tgt[~tgt.id.isin(test_tracks)], tgt[tgt.id.isin(test_tracks)]
    print(f'source {len(src)} images | target pool {len(pool)} | target test {len(te)} ({te.id.nunique()} tracks)', flush=True)

    base = os.path.join(out, 'unet_source.pt')
    net_src = UNet()
    if os.path.exists(base):
        net_src.load_state_dict(torch.load(base))
    else:
        train(net_src, X, Y, src.key, 30)
        torch.save(net_src.state_dict(), base)
    rows = [dict(method='source-only', k=0, rep=0, **score(net_src, X, Y, te))]
    print(rows[-1], flush=True)
    for k in [5, 10, 20, 40]:
        if k > len(pool):
            break
        for rep in range(3):
            keys = np.random.default_rng(rep).choice(pool.key.values, k, replace=False)
            ft = copy.deepcopy(net_src)
            train(ft, X, Y, keys, 30, lr=3e-4, log=False)
            rows.append(dict(method='fine-tune', k=k, rep=rep, **score(ft, X, Y, te)))
            print(rows[-1], flush=True)
            sc = UNet()
            train(sc, X, Y, keys, 80, lr=1e-3, log=False)
            rows.append(dict(method='scratch', k=k, rep=rep, **score(sc, X, Y, te)))
            print(rows[-1], flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(out, 'unet_transfer_runs.csv'), index=False)
    s = r.groupby(['method', 'k']).mean(numeric_only=True).drop(columns='rep').round(3)
    s.to_csv(os.path.join(out, 'unet_transfer_summary.csv'))
    print(s.to_string())

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (col, lab) in zip(axs, [('MAE_D', 'depth D MAE (mm)'), ('all3_within_0p1', 'sections with H, W, D all within 0.1 mm'), ('IoU_remelt', 'remelt-zone IoU')]):
        ax.axhline(r[r.method == 'source-only'][col].iloc[0], c='k', ls='--', lw=1, label='source model, no target labels')
        for m, c in [('fine-tune', 'C2'), ('scratch', 'C3')]:
            q = r[r.method == m].groupby('k')[col]
            ax.errorbar(q.mean().index, q.mean(), yerr=q.std(), fmt='o-', c=c, capsize=3,
                        label={'fine-tune': 'transfer: fine-tune source model', 'scratch': 'target labels only (from scratch)'}[m])
        ax.set_xlabel('labelled target images (k)'); ax.set_title(lab, fontsize=9)
    axs[0].legend(fontsize=7)
    fig.suptitle('Few-shot transfer of the U-Net to a new imaging set-up (dark mount → bright mount)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'unet_transfer_curves.png'), dpi=160)


if __name__ == '__main__':
    main(*sys.argv[1:4])
