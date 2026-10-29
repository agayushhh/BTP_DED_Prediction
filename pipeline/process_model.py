"""Process -> cross-section geometry models on the Optomec 316L table, with honest (leak-free) validation.

Targets: H, W, D (mm) and depth-dilution D/(H+D).
Feature sets: raw (P, V, F) vs physics (log P/V, log fed area A_f, log energy ratio Pi).
Splits: random rows (leaky: sister sections of one track and repeat conditions sit in train and test),
GroupKFold by condition (P,V,F unseen), and power extrapolation (train P<=400 W, test P>=450 W).
Usage: python -I process_model.py <Final_total_measurement.xlsx> <out_dir>
"""
import sys, os, json
import numpy as np, pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error

PX_PER_MM = 1202.0  # microscope calibration, Akhavan et al. 2023
RPM = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15]  # their Table 4
GMIN = [0, 0.7, 1.7, 2.7, 4.0, 5.7, 6.6, 7.9, 9.2, 10.5, 14.1, 17.0, 18.9, 20.1]
MAT = {  # density g/mm3, melt enthalpy from 25 C incl. latent heat J/g (handbook, ~+-10%)
    '316L': dict(rho=7.95e-3, hm=0.50 * 1375 + 270),
    'IN718': dict(rho=8.19e-3, hm=0.48 * 1310 + 210),
}


def load(xlsx):
    d = pd.read_excel(xlsx)
    d['Sample ID'] = d['Sample ID'].ffill()
    d.columns = ['id', 'sec', 'P', 'V', 'rpm', 'H', 'W', 'D']
    for c in 'HWD':
        d[c] = pd.to_numeric(d[c], errors='coerce') / PX_PER_MM
    d = d.dropna().reset_index(drop=True)
    d['F'] = np.interp(d.rpm, RPM, GMIN)  # g/min
    return d


def physics(P, V_mm_min, A_dep_mm2, mat):
    """Pi = laser energy per mm / energy to melt the deposited metal per mm."""
    El = P / (V_mm_min / 60.0)  # J/mm
    m = MAT[mat]
    return El / (m['rho'] * m['hm'] * A_dep_mm2)


def add_features(d):
    d['El'] = d.P / (d.V / 60)
    d['Af'] = d.F / (MAT['316L']['rho'] * d.V)  # fed volume per mm: (g/min)/(g/mm3)/(mm/min) = mm2
    d['Pi_feed'] = physics(d.P, d.V, d.Af, '316L')
    d['cond'] = d.P.astype(str) + '_' + d.V.astype(str) + '_' + d.rpm.astype(str)
    d['dil'] = d.D / (d.H.clip(lower=0) + d.D)
    return d


FEATS = {
    'raw': ['P', 'V', 'F'],
    'physics': ['logEl', 'logAf', 'logP'],  # log Pi = logEl - logAf + const, so it is implied
}


def models():
    k = C(1.0) * RBF(length_scale=[1.0] * 3, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(0.1, (1e-4, 10))
    return {
        'GPR': lambda: make_pipeline(StandardScaler(), GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=0, random_state=0)),
        'RF': lambda: RandomForestRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=-1),
        'PowerLaw': lambda: make_pipeline(StandardScaler(), LinearRegression()),  # linear in logs = power law
    }


def evaluate(d, target, feats, model, split):
    X = d[feats].values
    y = d[target].values
    if split == 'random':
        folds = KFold(5, shuffle=True, random_state=0).split(X)
    elif split == 'condition':
        folds = GroupKFold(5).split(X, y, d.cond)
    elif split == 'extrap_P':
        folds = [(np.flatnonzero(d.P <= 400), np.flatnonzero(d.P >= 450))]
    pred = np.full(len(y), np.nan)
    for tr, te in folds:
        m = model()
        if isinstance(m.steps[-1][1] if hasattr(m, 'steps') else m, GaussianProcessRegressor) and len(tr) > 800:
            tr = np.random.default_rng(0).choice(tr, 800, replace=False)  # ponytail: GPR is O(n^3); sparse GP if all rows needed
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    ok = ~np.isnan(pred)
    return r2_score(y[ok], pred[ok]), mean_absolute_error(y[ok], pred[ok]), pred


def main(xlsx, out):
    os.makedirs(out, exist_ok=True)
    d = add_features(load(xlsx))
    d = d[(d.H > 0)].reset_index(drop=True)  # ponytail: 12 sunk tracks (H<=0) dropped; separate regime
    d['logEl'], d['logAf'], d['logP'] = np.log(d.El), np.log(d.Af), np.log(d.P)
    print(f'{len(d)} sections, {d.id.nunique()} tracks, {d.cond.nunique()} conditions')
    # noise floor: spread of repeats at identical nominal conditions
    rep = d.groupby('cond').filter(lambda g: len(g) >= 3)
    for t in ['H', 'W', 'D', 'dil']:
        within = rep.groupby('cond')[t].std().mean()
        print(f'  repeat std {t}: {within:.3f}  (total std {d[t].std():.3f})')

    rows = []
    for t in ['H', 'W', 'D', 'dil']:
        for fs, feats in FEATS.items():
            for mn, mk in models().items():
                if mn == 'PowerLaw' and fs == 'raw':
                    continue
                for sp in ['random', 'condition', 'extrap_P']:
                    r2, mae, _ = evaluate(d, t, feats, mk, sp)
                    rows.append(dict(target=t, features=fs, model=mn, split=sp, R2=round(r2, 3), MAE=round(mae, 4)))
                    print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, 'optomec_cv_results.csv'), index=False)
    d.to_csv(os.path.join(out, 'optomec_sections_mm.csv'), index=False)
    print(res.pivot_table(index=['target', 'features', 'model'], columns='split', values='R2').to_string())


if __name__ == '__main__':
    main(*sys.argv[1:3])
