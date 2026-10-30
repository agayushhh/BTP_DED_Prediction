"""Few-shot transfer of the process -> geometry model to a shifted domain (simulated new lab/setting).

Source: all sections with P <= 400 W. Target: P >= 450 W (24 conditions). A "target point" is one condition
(mean of its sections), like one printed bead per setting in a new lab. For k target conditions we compare:
  zero-shot  : source GPR as is
  scratch    : GPR trained on the k target points only (mean if k < 3)
  recal      : source prediction recalibrated, y = a*f + b (bias only if k < 3)
  delta      : source prediction + GPR fitted on the k residuals (prior mean 0 -> falls back to source)
Test on all sections of the remaining (unseen) target conditions; 30 random draws per k.
Targets: H, W, D (all sections) and the segmented area dilution (validated sections only).
Usage: python transfer_process.py <sections_mm.csv> <out_dir> <results_dir>
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dilution_targets import area_dilution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score

FEATS = ['logEl', 'logAf', 'logP']
KS_LIST = [0, 1, 2, 3, 5, 8, 12]
REPS = 30


def gpr(n_restarts=0):
    k = C(1.0) * RBF([1.0] * len(FEATS), (1e-2, 1e3)) + WhiteKernel(.1, (1e-4, 10))
    return make_pipeline(StandardScaler(), GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=n_restarts, random_state=0))


def main(sec_csv, out, res):
    os.makedirs(out, exist_ok=True)
    d = pd.read_csv(sec_csv)
    d['logEl'], d['logAf'], d['logP'] = np.log(d.El), np.log(d.Af), np.log(d.P)
    d['key'] = d.id + d.sec
    d = d.merge(area_dilution(res).rename(columns={'dilution': 'dil_area'}), on='key', how='left')
    rows = []
    for t in ['H', 'W', 'D', 'dil_area']:
        dt = d[d[t].notna()]  # area dilution exists only for the validated sections
        src, tgt = dt[dt.P <= 400], dt[dt.P >= 450]
        conds = tgt.cond.unique()
        print(t, len(src), 'source sections,', len(conds), 'target conditions', flush=True)
        s = src.sample(min(800, len(src)), random_state=0)
        f_src = gpr(2).fit(s[FEATS].values, s[t].values)
        tgt_pred = pd.Series(f_src.predict(tgt[FEATS].values), index=tgt.index)
        cm = tgt.groupby('cond').agg({**{c: 'first' for c in FEATS}, t: 'mean'})
        cm['f'] = f_src.predict(cm[FEATS].values)
        for k in KS_LIST:
            for rep in range(REPS if k else 1):
                rng = np.random.default_rng(rep)
                tr_c = rng.choice(conds, k, replace=False) if k else np.array([])
                te = tgt[~tgt.cond.isin(tr_c)]
                y, f = te[t].values, tgt_pred[te.index].values
                tr = cm.loc[tr_c] if k else cm.iloc[:0]
                preds = {'zero-shot': f}
                if k:
                    Xk, yk, fk = tr[FEATS].values, tr[t].values, tr.f.values
                    preds['scratch'] = gpr().fit(Xk, yk).predict(te[FEATS].values) if k >= 3 else np.full(len(y), yk.mean())
                    if k >= 3:
                        a, b = np.polyfit(fk, yk, 1)
                    else:
                        a, b = 1.0, (yk - fk).mean()
                    preds['recal'] = a * f + b
                    preds['delta'] = f + gpr().fit(Xk, yk - fk).predict(te[FEATS].values)
                for m, p in preds.items():
                    rows.append(dict(target=t, k=k, rep=rep, method=m, MAE=mean_absolute_error(y, p), R2=r2_score(y, p)))
            print(t, k, flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(out, 'transfer_process_runs.csv'), index=False)
    s = r.groupby(['target', 'method', 'k'])[['MAE', 'R2']].median().round(3).reset_index()
    s.to_csv(os.path.join(out, 'transfer_process_summary.csv'), index=False)
    print(s.pivot_table(index=['target', 'method'], columns='k', values='MAE').to_string())
    print(s.pivot_table(index=['target', 'method'], columns='k', values='R2').to_string())

    names = {'H': 'height H (mm)', 'W': 'width W (mm)', 'D': 'depth D (mm)', 'dil_area': 'area dilution'}
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, t in zip(axs, names):
        z = s[(s.target == t) & (s.method == 'zero-shot')].MAE.iloc[0]
        ax.axhline(z, c='k', ls='--', lw=1, label='source model, no target data')
        for m, c in [('scratch', 'C3'), ('recal', 'C0'), ('delta', 'C2')]:
            q = r[(r.target == t) & (r.method == m)].groupby('k').MAE
            ax.plot(q.median().index, q.median(), 'o-', c=c, label={'scratch': 'target data only', 'recal': 'transfer: recalibrate a·f+b', 'delta': 'transfer: + GPR residual'}[m])
            ax.fill_between(q.median().index, q.quantile(.25), q.quantile(.75), color=c, alpha=.15)
        ax.set_title(names[t], fontsize=9); ax.set_xlabel('target conditions used (k)'); ax.set_ylabel('MAE on unseen target conditions')
    axs[0].legend(fontsize=7)
    fig.suptitle('Few-shot transfer: source P ≤ 400 W  →  target P ≥ 450 W (median and IQR over 30 draws)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'transfer_process_curves.png'), dpi=160)


if __name__ == '__main__':
    main(*sys.argv[1:4])
