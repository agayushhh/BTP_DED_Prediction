"""Example masks for the dark -> bright U-Net transfer (one draw, k = 5): input, label, source-only,
fine-tuned, from scratch. Same split and settings as unet_transfer.py; illustrative only (the table and
curves in unet_transfer.py average three draws).
Usage: python -I pipeline/unet_transfer_examples.py <tif_dir> <seg_dir> <source_unet.pt> <out.png>"""
import sys, os, copy
import numpy as np, pandas as pd, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet_seg import UNet, load_img, train, predict, SZ
from unet_transfer import bright_mount
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

tif_dir, seg_dir, src_pt, out = sys.argv[1:5]
torch.manual_seed(0)
torch.set_num_threads(os.cpu_count() or 4)
d = pd.read_csv(os.path.join(seg_dir, 'optomec_segmented.csv'))
d['key'] = d.id + d.sec
d = d[d.qc_pass].copy()
d['bright'] = [bright_mount(tif_dir, k) for k in d.key]
tgt = d[d.bright]
tracks = tgt.id.unique(); np.random.default_rng(0).shuffle(tracks)
test_tracks = set(tracks[: int(len(tracks) * .4)])
pool, te = tgt[~tgt.id.isin(test_tracks)], tgt[tgt.id.isin(test_tracks)]
X = {k: load_img(tif_dir, k) for k in tgt.key}
Y = {k: cv2.resize(cv2.imread(os.path.join(seg_dir, 'masks', k + '.png'), 0), SZ, interpolation=cv2.INTER_NEAREST).astype(np.int64) for k in tgt.key}

net_src = UNet(); net_src.load_state_dict(torch.load(src_pt))
keys = np.random.default_rng(0).choice(pool.key.values, 5, replace=False)
ft = copy.deepcopy(net_src); train(ft, X, Y, keys, 30, lr=3e-4, log=False)
sc = UNet(); train(sc, X, Y, keys, 80, lr=1e-3, log=False)

def iou(m, y, c):
    return ((m == c) & (y == c)).sum() / max(((m == c) | (y == c)).sum(), 1)

def show(ax, g, m):
    rgb = np.stack([g] * 3, -1) * 0.6
    rgb[m == 1] = rgb[m == 1] * 0.3 + np.array([0.2, 0.8, 0.2]) * 0.7
    rgb[m == 2] = rgb[m == 2] * 0.3 + np.array([0.9, 0.2, 0.2]) * 0.7
    ax.imshow(np.clip(rgb, 0, 1)); ax.axis('off')

show_keys = te.key.values[np.linspace(0, len(te) - 1, 4).astype(int)]
nets = [('source only (k = 0)', net_src), ('fine-tuned (k = 5)', ft), ('from scratch (k = 5)', sc)]
fig, axs = plt.subplots(len(show_keys), 5, figsize=(14, 2.55 * len(show_keys)))
for i, k in enumerate(show_keys):
    g = X[k]; g = (g - g.min()) / (g.max() - g.min())
    axs[i, 0].imshow(g, cmap='gray'); axs[i, 0].axis('off')
    axs[i, 0].set_ylabel(k[-5:])
    axs[i, 0].text(4, 20, k[-5:], color='w', fontsize=9, bbox=dict(fc='k', alpha=0.5, lw=0))
    show(axs[i, 1], g, Y[k])
    for j, (name, net) in enumerate(nets):
        m = predict(net, X[k])
        show(axs[i, 2 + j], g, m)
        axs[i, 2 + j].text(0.5, -0.04, f'IoU clad {iou(m, Y[k], 1):.2f}, remelt {iou(m, Y[k], 2):.2f}',
                           transform=axs[i, 2 + j].transAxes, ha='center', va='top', fontsize=8.5)
for j, t in enumerate(['input (bright mount, held-out track)', 'label (accepted rule-based mask)'] + [n for n, _ in nets]):
    axs[0, j].set_title(t, fontsize=10)
fig.tight_layout(h_pad=1.6)
fig.savefig(out, dpi=120)
print('train keys', list(keys), 'shown', list(show_keys))
