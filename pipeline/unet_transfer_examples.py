"""Example masks for the dark -> bright U-Net transfer (one draw, k = 5): input, label, source-only,
fine-tuned, from scratch. Same split and settings as unet_transfer.py; illustrative only (the table and
curves in unet_transfer.py average three draws).
Usage: python -I pipeline/unet_transfer_examples.py <tif_dir> <seg_dir> <source_unet.pt> <out.png>"""
import sys, os, copy
import numpy as np, pandas as pd, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet_seg import UNet, load_img, train, predict, SZ
from unet_transfer import bright_mount
from figstyle import plt, FULL, fill, crop_box, mask_legend

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

show_keys = te.key.values[np.linspace(0, len(te) - 1, 4).astype(int)]
nets = [('source only\n(k = 0)', net_src), ('fine-tuned\n(k = 5)', ft), ('from scratch\n(k = 5)', sc)]
fig, axs = plt.subplots(len(show_keys), 5, figsize=(FULL, 1.3 * len(show_keys) + 0.7))
for i, k in enumerate(show_keys):
    g = X[k]; g = (g - g.min()) / (g.max() - g.min())
    preds = [predict(net, X[k]) for _, net in nets]
    box = crop_box([Y[k]] + preds, g.shape, pad=.15)
    fill(axs[i, 0], g, np.zeros_like(Y[k]), box)
    axs[i, 0].axis('on'); axs[i, 0].set_xticks([]); axs[i, 0].set_yticks([]); axs[i, 0].set_ylabel(k[-5:], fontsize=10.5)
    [s_.set_visible(False) for s_ in axs[i, 0].spines.values()]
    fill(axs[i, 1], g, Y[k], box)
    for j, m in enumerate(preds):
        fill(axs[i, 2 + j], g, m, box)
        axs[i, 2 + j].text(0.5, -0.03, f'IoU {iou(m, Y[k], 1):.2f} / {iou(m, Y[k], 2):.2f}',
                           transform=axs[i, 2 + j].transAxes, ha='center', va='top', fontsize=9)
for j, t in enumerate(['input\n(held-out track)', 'label\n(rule-based)'] + [n for n, _ in nets]):
    axs[0, j].set_title(t, fontsize=10)
fig.tight_layout(h_pad=1.3, w_pad=.3)
mask_legend(fig, filled=True, loc='lower center', bbox_to_anchor=(.5, -.06))
fig.savefig(out)
print('train keys', list(keys), 'shown', list(show_keys))
