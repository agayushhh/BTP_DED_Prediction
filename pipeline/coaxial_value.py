"""Does the coaxial camera add information about the hidden geometry (D, dilution) beyond P, V, F?

Test 1 (unseen settings): RF on process features vs process + camera, GroupKFold by condition.
Test 2 (repeats): within each condition with >=3 sections, subtract the condition mean from target and
camera features; ridge regression on these deviations, GroupKFold by condition -> R2 of the deviation
explained. Process features are constant inside a condition, so whatever is explained here comes only
from the camera (bead-to-bead variability: powder flow, plate heating, ...).
Usage: python coaxial_value.py <sections_mm.csv> <coaxial_section_features.csv> <out_dir>
"""
import sys, os
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score

PROC = ['P', 'V', 'F']


def main(sec_csv, cx_csv, out):
    os.makedirs(out, exist_ok=True)
    d = pd.read_csv(sec_csv).merge(pd.read_csv(cx_csv), on=['id', 'sec'])
    cam = [c for c in d.columns if c.startswith('cx_') and not c.endswith('_sd')]
    print(f'{len(d)} sections with camera features, {d.id.nunique()} tracks, {d.cond.nunique()} conditions')
    rows = []
    for t in ['H', 'W', 'D', 'dil']:
        g = GroupKFold(5)
        rf = lambda: RandomForestRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=-1)
        p0 = cross_val_predict(rf(), d[PROC], d[t], groups=d.cond, cv=g)
        p1 = cross_val_predict(rf(), d[PROC + cam], d[t], groups=d.cond, cv=g)
        p2 = cross_val_predict(rf(), d[cam], d[t], groups=d.cond, cv=g)

        rep = d.groupby('cond').filter(lambda x: len(x) >= 3)
        dev = lambda c: rep[c] - rep.groupby('cond')[c].transform('mean')
        X = pd.concat([dev(c) for c in cam], axis=1).values
        y = dev(t).values
        X = (X - X.mean(0)) / (X.std(0) + 1e-12)
        pr = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 3, 20)), X, y, groups=rep.cond, cv=GroupKFold(5))
        rows.append(dict(target=t, R2_process=r2_score(d[t], p0), R2_process_plus_camera=r2_score(d[t], p1),
                         R2_camera_only=r2_score(d[t], p2), R2_repeat_deviation=r2_score(y, pr), n_repeat=len(rep)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).round(3).to_csv(os.path.join(out, 'coaxial_value.csv'), index=False)


if __name__ == '__main__':
    main(*sys.argv[1:4])
