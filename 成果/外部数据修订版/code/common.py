"""Shared paths, aligned data, numerical and reproducibility helpers."""
from pathlib import Path
import os, sys, json, hashlib
ROOT = Path(__file__).resolve().parents[1]
# Optional task-local dependency folder; normal pip installations work without it.
local = ROOT.parent / '.beer_runtime'
if local.exists(): sys.path.insert(0, str(local))
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True)
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
DATA = ROOT/'data'/'official'
EXT = ROOT/'data'/'external'
OUT = ROOT/'results'
FIG = ROOT/'figures'
for p in [OUT, FIG]: p.mkdir(parents=True, exist_ok=True)
META = ['beer','beer_id','tasting_category_fine']
SEED = 42

def load():
    x = pd.read_csv(DATA/'1.csv'); y = pd.read_csv(DATA/'4.csv')
    assert x.beer.is_unique and y.beer.is_unique
    y = y.set_index('beer').loc[x.beer].reset_index()
    assert np.array_equal(x.tasting_category_fine, y.tasting_category_fine)
    return x.drop(columns=META), y.drop(columns=META), x.tasting_category_fine, x[META]

def dump(obj, name):
    def conv(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        return str(x)
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=conv), encoding='utf-8')

def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if y.ndim == 1: y,p=y[:,None],p[:,None]
    rho=[spearmanr(y[:,j],p[:,j]).statistic for j in range(y.shape[1]) if np.std(p[:,j])>1e-12 and np.std(y[:,j])>1e-12]
    return dict(r2=float(r2_score(y,p)),mae=float(mean_absolute_error(y,p)),spearman=float(np.mean(rho)) if rho else None)

def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'figure.dpi':140,'savefig.dpi':180})
    return plt

def audit():
    files=[]
    for p in sorted((ROOT/'data').rglob('*')):
        if p.is_file(): files.append({'file':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    pd.DataFrame(files).to_csv(OUT/'input_manifest.csv',index=False,encoding='utf-8-sig')
    x,y,s,_=load()
    dump({'n':len(x),'chemical_features':len(x.columns),'sensory_targets':len(y.columns),'missing_chemical_cells':int(x.isna().sum().sum()),'missing_sensory_cells':int(y.isna().sum().sum()),'style_counts':s.value_counts().to_dict()},'data_audit.json')
