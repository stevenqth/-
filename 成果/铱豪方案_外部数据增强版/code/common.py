"""Portable paths and aligned official data. No network access at runtime."""
from pathlib import Path
import sys,os,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
runtime=ROOT.parent/'.beer_runtime'
if runtime.exists():sys.path.insert(0,str(runtime))
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True)
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error
OUT=ROOT/'results';DATA=ROOT/'data/official';EXT=ROOT/'data/external';MODELS=ROOT/'models';FIG=ROOT/'figures'
META=['beer','beer_id','tasting_category_fine']
for p in [OUT,MODELS,FIG]:p.mkdir(exist_ok=True)

def save(obj,name):
    def default(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        return str(x)
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=default),encoding='utf-8')

def load():
    x=pd.read_csv(DATA/'1.csv');y=pd.read_csv(DATA/'4.csv')
    assert x.beer.is_unique and y.beer.is_unique
    y=y.set_index('beer').loc[x.beer].reset_index()
    assert np.array_equal(x.tasting_category_fine,y.tasting_category_fine)
    return x.drop(columns=META),y.drop(columns=META),x[META]

def plotter():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'savefig.dpi':180})
    return plt
