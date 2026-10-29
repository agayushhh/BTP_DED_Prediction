"""Demonstration: transfer the 316L-trained U-Net to laser-wire DED micrographs (IN718 wire on SS304).

8 panels from segment_wire.py (1100/450 excluded: an annotation line covers the fusion line). Labels = the
classical three-phase masks (dissimilar metals, so the classical result is reliable here).
Leave-one-out: for each panel, (a) zero-shot source U-Net, (b) source U-Net fine-tuned on the other 7,
(c) U-Net trained from scratch on the other 7. Scores: IoU of clad and remelt, error of area dilution.
Small, low-resolution images: an illustration of the mechanism, not a benchmark.
Usage: python wire_unet_transfer.py <panels_dir> <source_unet.pt> <out_dir>
"""
import sys, os, glob, copy
import numpy as np, pandas as pd, cv2, torch
from unet_seg import UNet, train, predict, SZ

torch.manual_seed(0); np.random.seed(0)


def iou(a, b, c):
    return ((a == c) & (b == c)).sum() / max(((a == c) | (b == c)).sum(), 1)


def dil(m):
    ad, as_ = (m == 1).sum(), (m == 2).sum()
    return as_ / max(ad + as_, 1)


def main(panels, weights, out):
    os.makedirs(out, exist_ok=True)
    keys = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(panels, '*_*.png')) if not p.endswith('_mask.png'))
    keys = [k for k in keys if k != '1100_450']
    X, Y = {}, {}
    for k in keys:
        g = cv2.resize(cv2.imread(os.path.join(panels, k + '.png'), 0), SZ, interpolation=cv2.INTER_AREA).astype(np.float32)
        X[k] = (g - g.mean()) / (g.std() + 1e-6)
        Y[k] = cv2.resize(cv2.imread(os.path.join(panels, k + '_mask.png'), 0), SZ, interpolation=cv2.INTER_NEAREST).astype(np.int64)
    src = UNet(); src.load_state_dict(torch.load(weights))
    rows, preds = [], {}
    for k in keys:
        others = [o for o in keys if o != k]
        ft = train(copy.deepcopy(src), X, Y, others, 30, lr=3e-4, log=False)
        sc = train(UNet(), X, Y, others, 80, lr=1e-3, log=False)
        for name, net in [('zero-shot', src), ('fine-tuned (7 panels)', ft), ('scratch (7 panels)', sc)]:
            m = predict(net, X[k]); preds[(k, name)] = m
            rows.append(dict(panel=k, method=name, IoU_clad=iou(m, Y[k], 1), IoU_remelt=iou(m, Y[k], 2),
                             dil_err=abs(dil(m) - dil(Y[k]))))
        print(k, rows[-3:], flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(out, 'wire_unet_transfer.csv'), index=False)
    s = r.groupby('method')[['IoU_clad', 'IoU_remelt', 'dil_err']].mean().round(3)
    s.to_csv(os.path.join(out, 'wire_unet_transfer_summary.csv'))
    print(s.to_string())

    show = [k for k in ['900_750', '1100_600', '1100_750'] if k in keys]
    tiles = []
    for k in show:
        row = []
        g = cv2.normalize(X[k], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        for name in ['label', 'zero-shot', 'fine-tuned (7 panels)', 'scratch (7 panels)']:
            m = Y[k] if name == 'label' else preds[(k, name)]
            ov = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            for c, col in ((1, (0, 220, 0)), (2, (0, 0, 255))):
                cs, _ = cv2.findContours((m == c).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(ov, cs, -1, col, 2)
            cv2.putText(ov, f'{k} {name}', (5, 16), 0, .45, (255, 255, 255), 1)
            row.append(ov)
        tiles.append(np.hstack(row))
    cv2.imwrite(os.path.join(out, 'wire_unet_transfer.png'), np.vstack(tiles))


if __name__ == '__main__':
    main(*sys.argv[1:4])
