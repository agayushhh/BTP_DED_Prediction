"""Process models for three dilution targets on the same validated sections: segmented area dilution,
geometric proxy D/(H+D) and shape-factor estimate Ks*D/(Kd*H+Ks*D). Same models and splits as
process_model.py, so the targets can be compared directly.
Usage: python -I pipeline/dilution_targets.py <results_dir>"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from process_model import models, evaluate

res = sys.argv[1]
sec = pd.read_csv(os.path.join(res, 'optomec', 'optomec_sections_mm.csv'))
sec['key'] = sec.id + sec.sec
seg = pd.read_csv(os.path.join(res, 'optomec_seg', 'optomec_segmented.csv'))
seg['key'] = seg.id + seg.sec
acc = seg[seg.qc_pass == True][['key', 'dilution']]
u = pd.read_csv(os.path.join(res, 'unet', 'unet_eval.csv'))
u = u[u.set == 'qc_fail']
ok = ((u.H_unet - u.H_ref).abs() < .1) & ((u.W_unet - u.W_ref).abs() < .1) & ((u.D_unet - u.D_ref).abs() < .1)
rec = u[ok][['key', 'dilution_unet']].rename(columns={'dilution_unet': 'dilution'})
d = sec.merge(pd.concat([acc, rec]), on='key').reset_index(drop=True)
KD, KS = 0.803, 0.715  # measured shape factors (accepted rule-based sections)
d['dil_area'] = d.dilution
d['dil_proxy'] = d.D / (d.H + d.D)
d['dil_shape'] = KS * d.D / (KD * d.H + KS * d.D)
print(f'{len(d)} validated sections, {d.cond.nunique()} conditions, {int((d.P >= 450).sum())} at P >= 450 W')
rep = d.groupby('cond').filter(lambda g: len(g) >= 3)
rows = []
for t in ['dil_area', 'dil_proxy', 'dil_shape']:
    noise = rep.groupby('cond')[t].std().mean()
    for fs, feats in {'raw': ['P', 'V', 'F'], 'grouped': ['logEl', 'logAf', 'logP']}.items():
        for mn, mk in models().items():
            if mn == 'PowerLaw' and fs == 'raw':
                continue
            for sp in ['random', 'condition', 'extrap_P']:
                r2, mae, _ = evaluate(d, t, feats, mk, sp)
                rows.append(dict(target=t, inputs=fs, model={'PowerLaw': 'LogLinear'}.get(mn, mn), split=sp,
                                 R2=round(r2, 3), MAE_points=round(mae * 100, 1), noise_points=round(noise * 100, 1)))
r = pd.DataFrame(rows)
r.to_csv(os.path.join(res, 'optomec', 'dilution_targets.csv'), index=False)
print(r.pivot_table(index=['target', 'inputs', 'model'], columns='split', values=['R2', 'MAE_points']).to_string())
print('noise floor (points):', r.groupby('target').noise_points.first().to_dict())
print('area vs proxy: mean diff %+.1f, MAE %.1f points' % ((d.dil_proxy - d.dil_area).mean() * 100, (d.dil_proxy - d.dil_area).abs().mean() * 100))
print('area vs shape estimate: mean diff %+.1f, MAE %.1f points' % ((d.dil_shape - d.dil_area).mean() * 100, (d.dil_shape - d.dil_area).abs().mean() * 100))
