"""Export pinned Git data; independently check objective/constraint math on a grid.

This is NOT a MATLAB or NSGA-II run. Requires only numpy and pandas.
Run with --repo pointing at the containing repository if moved elsewhere.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import time
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REF = '3e41cdf5380ea05ba9c37acbdd4992a062fc8e14'
DEFECTS = ['DMS', 'X2.3.butanedione', 'X4VG', 'X.E..2.nonenal']
LOG_DEFECT = np.array([True, False, True, False])


def concentration(v):
    c = np.array(v, copy=True)
    c[..., LOG_DEFECT] = 10**c[..., LOG_DEFECT]
    return c


def fit_model(x, y):
    med = np.nanmedian(x, axis=0)
    x = np.where(np.isfinite(x), x, med)
    lo, hi = np.quantile(x, [.01, .99], axis=0)
    x = np.clip(x, lo, hi)
    mu, sd = x.mean(0), x.std(0, ddof=1)
    sd[sd < 1e-12] = 1
    z = (x-mu)/sd
    ym = y.mean(0)
    b = np.linalg.solve(z.T@z + 1000*np.eye(z.shape[1]), z.T@(y-ym))
    return med, mu, sd, ym, b, lo, hi


def nondominated(f):
    """Exact pairwise dominance with a sorted incremental archive (all minimized)."""
    archive = []
    for i in np.argsort(f[:, 0], kind='stable'):
        if archive:
            af = f[archive]
            if np.any(np.all(af <= f[i], axis=1) & np.any(af < f[i], axis=1)):
                continue
            remove = np.all(f[i] <= af, axis=1) & np.any(f[i] < af, axis=1)
            archive = [a for a, bad in zip(archive, remove) if not bad]
        archive.append(int(i))
    return np.array(archive, dtype=int)


def evaluate(ab, model):
    med, mu, sd, ym, b, lo, hi, src, tgt, target_y, di, cap, scale, stale, control, bound = model
    v = src + ab[:, :1]*(tgt-src)
    conc = np.minimum(concentration(v[:, di]), cap) * 2**(-ab[:, 1:2])
    v[:, di] = conc
    v[:, di[LOG_DEFECT]] = np.log10(conc[:, LOG_DEFECT])
    v[:, control] += ab[:, 2]*(bound-v[:, control])
    yh = ((np.clip(v, lo, hi)-mu)/sd)@b+ym
    f = np.column_stack([
        np.mean(((yh-target_y)/scale)**2, axis=1),
        np.sqrt(np.mean(((v-src)/sd)**2, axis=1)),
        np.mean(conc/cap, axis=1),
    ])
    src_pred = ((np.clip(src, lo, hi)-mu)/sd)@b+ym
    c = (yh[:, stale]-src_pred[stale])/scale[stale]
    return f, c, v, yh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=HERE.parents[1])
    args = ap.parse_args()
    started = time.time()
    paths = ['资料/附件/1.csv', '资料/附件/4.csv',
             '成果/results/00_数值变换核验.csv']
    provenance = {'source_commit': REF, 'files': [], 'date': '2026-10-29 Asia/Shanghai'}
    for p in paths:
        data = subprocess.check_output(['git', '-C', str(args.repo), 'show', f'{REF}:{p}'])
        out = HERE/'inputs'/Path(p).name
        out.write_bytes(data)
        provenance['files'].append({'path': p, 'saved_as': out.name,
                                    'sha256': hashlib.sha256(data).hexdigest()})
    a = pd.read_csv(HERE/'inputs/1.csv')
    d = pd.read_csv(HERE/'inputs/4.csv')
    assert np.array_equal(a.beer, d.beer), 'beer keys/order mismatch'
    assert a.beer.is_unique and d.beer.is_unique
    assert np.array_equal(a.tasting_category_fine, d.tasting_category_fine)
    cols, ys = list(a.columns[3:]), list(d.columns[3:])
    x, y = a[cols].to_numpy(float), d[ys].to_numpy(float)
    assert np.isfinite(y).all()
    transforms = pd.read_csv(HERE/'inputs/00_数值变换核验.csv').set_index('compound')
    assert [transforms.loc[c, 'transform'] == 'log10' for c in DEFECTS] == LOG_DEFECT.tolist()
    folds = np.empty(len(x), int)
    order = np.random.default_rng(42).permutation(len(x))
    folds[order] = np.arange(len(x)) % 5 + 1
    np.savetxt(HERE/'inputs/folds.csv', folds, fmt='%d')
    pred = np.empty_like(y)
    for k in range(1, 6):
        tr, va = folds != k, folds == k
        med, mu, sd, ym, b, lo, hi = fit_model(x[tr], y[tr])
        xv = np.where(np.isfinite(x[va]), x[va], med)
        pred[va] = ((np.clip(xv, lo, hi)-mu)/sd)@b+ym
    r2 = 1-np.sum((pred-y)**2, axis=0)/np.sum((y-y.mean(0))**2, axis=0)
    pd.DataFrame({'target': ys, 'oof_R2': r2,
                  'oof_MAE': np.mean(np.abs(pred-y), axis=0)}).to_csv(
        HERE/'reference_grid/surrogate_oof.csv', index=False)
    med, mu, sd, ym, b, lo, hi = fit_model(x, y)
    xi = np.where(np.isfinite(x), x, med)
    source_indices = np.where(a.tasting_category_fine == 'Pils/Lager')[0]
    target_indices = np.where(a.tasting_category_fine == 'Tripel')[0]
    source_data = xi[source_indices]
    center = source_data.mean(0)
    si = source_indices[np.argmin(np.mean(((source_data-center)/sd)**2, axis=1))]
    src = xi[si]
    tgt = np.median(xi[target_indices], axis=0)
    target_y = y[target_indices].mean(0)
    di = np.array([cols.index(c) for c in DEFECTS])
    src_c = concentration(src[di])
    thresholds = np.array([17.25, .29925, .206333333, np.inf])
    cap = .95*np.minimum(src_c, thresholds)
    scale = y.std(0, ddof=1)
    scale[scale < 1e-12] = 1
    stale = ys.index('stale_hops')
    # A third control expands the family when alpha/beta alone violate stale_hops.
    # Select a non-defect, non-derived chemical by maximum attainable proxy reduction.
    lo, hi = np.quantile(xi, [.01, .99], axis=0)
    bounds = np.where(b[:, stale] >= 0, lo, hi)
    benefit = (src-bounds)/sd*b[:, stale]
    benefit[di] = -np.inf
    benefit[[i for i, col in enumerate(cols) if col.endswith('_sum')]] = -np.inf
    control = int(np.argmax(benefit))
    bound = float(bounds[control])
    model = med, mu, sd, ym, b, lo, hi, src, tgt, target_y, di, cap, scale, stale, control, bound
    grid = np.array(np.meshgrid(*[np.linspace(0, 1, 21)]*3, indexing='ij')).reshape(3, -1).T
    f, c, v, yh = evaluate(grid, model)
    assert np.isfinite(f).all() and np.isfinite(c).all()
    assert np.all(concentration(v[:, di]) <= cap*(1+1e-12))
    assert np.all(concentration(v[:, di]) <= src_c*(1+1e-12))
    feasible = c <= 1e-8
    if not feasible.any():
        raise RuntimeError('No feasible grid point; do not weaken stale constraint silently.')
    indices = np.where(feasible)[0]
    front_idx = indices[nondominated(f[feasible])]
    front = f[front_idx]
    span = np.ptp(front, axis=0)
    active = span > 1e-12
    norm = np.zeros_like(front)
    norm[:, active] = (front[:, active]-front[:, active].min(0))/span[active]
    scenarios = {'balanced': [1, 1, 1], 'sensory_first': [.6, .2, .2],
                 'small_change': [.2, .6, .2], 'low_defect': [.2, .2, .6]}
    decisions = []
    for name, w in scenarios.items():
        w = np.array(w, float)*active
        w /= w.sum()
        dp = np.sqrt(np.sum(norm**2*w, axis=1))
        dm = np.sqrt(np.sum((norm-active)**2*w, axis=1))
        closeness = np.divide(dm, dp+dm, out=np.ones_like(dm)*.5, where=(dp+dm)>0)
        k = int(np.argmax(closeness))
        i = front_idx[k]
        decisions.append({'scenario': name, 'front_id': k+1, 'alpha': grid[i, 0],
                          'beta': grid[i, 1], 'gamma': grid[i, 2], 'f1': f[i, 0], 'f2': f[i, 1],
                          'f3': f[i, 2], 'topsis': closeness[k], 'stale_constraint': c[i]})
    table = pd.DataFrame(np.column_stack([grid[front_idx], front, c[front_idx]]),
                         columns=['alpha', 'beta', 'gamma', 'f1', 'f2', 'f3', 'stale_constraint'])
    table.insert(0, 'front_id', np.arange(1, len(table)+1))
    table.to_csv(HERE/'reference_grid/pareto_grid.csv', index=False)
    pd.DataFrame(decisions).to_csv(HERE/'reference_grid/decisions_grid.csv', index=False)
    # Deterministic fixtures for a future MATLAB parity check; no random GA comparison.
    probes = np.array([[0, 0, 0], [.5, .5, .5], [1, 1, 1]])
    pf, pc, _, _ = evaluate(probes, model)
    pd.DataFrame(np.column_stack([probes, pf, pc]),
                 columns=['alpha', 'beta', 'gamma', 'f1', 'f2', 'f3', 'stale_constraint']).to_csv(
                     HERE/'inputs/objective_fixtures.csv', index=False)
    # Meaningful dominance and decision checks, including constant objective safety.
    assert set(nondominated(np.array([[1, 2], [2, 1], [3, 3], [1, 2]]))) == {0, 1, 3}
    for start in range(0, len(front), 128):
        block = front[start:start+128]
        dominated = np.all(front[:, None, :] <= block[None, :, :], axis=2) & np.any(
            front[:, None, :] < block[None, :, :], axis=2)
        assert not dominated.any()
    summary = {'engine': 'Python numpy reference grid; NOT NSGA-II; NOT MATLAB',
               'n_samples': len(x), 'n_features': len(cols), 'n_sensory': len(ys),
               'source_style': 'Pils/Lager', 'target_style': 'Tripel',
               'source_beer': int(a.iloc[si].beer), 'source_row_1based': int(si+1),
               'stale_control_feature': cols[control], 'stale_control_bound': bound,
               'n_source': len(source_indices), 'n_target': len(target_indices),
               'ridge_lambda': 1000, 'oof_mean_R2': float(r2.mean()),
               'cv_scope': 'exploratory CV; lambda screened on same folds, not an unbiased test estimate',
               'oof_stale_R2': float(r2[stale]), 'grid_points': len(grid),
               'feasible_points': int(feasible.sum()), 'pareto_points': len(front),
               'max_front_stale_constraint': float(c[front_idx].max()),
               'chemical_defect_cap': cap.tolist(), 'decision_scenarios': decisions,
               'checks': ['row-key alignment', 'verified mixed raw/log10 transforms for four defects',
                          'fold-local imputation and scaling', 'finite objectives',
                          'chemical caps and non-worsening', 'grid front nondominance',
                          'known dominance fixture'],
               'runtime_seconds': round(time.time()-started, 2),
               'MATLAB_execution': 'not performed; executable not found'}
    (HERE/'inputs/provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    (HERE/'reference_grid/validation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
