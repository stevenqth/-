"""Independently re-evaluate exported fronts, constraints and heldout isolation."""
from common import *
import csv,joblib
from optimize import Problem,DEFECTS,VARIABLES,OBJECTIVES,concentration
from nsga2 import fronts,self_test

def main():
    checks=[]
    def check(name,condition):
        checks.append({'check':name,'pass':bool(condition)})
        if not condition:raise AssertionError(name)
    split=pd.read_csv(OUT/'external_split.csv');a=set(split[split.split=='optimization'].brewery);b=set(split[split.split=='heldout'].brewery)
    check('external brewery groups disjoint',not a&b)
    check('all external products assigned once',len(split)==5558 and split.key.is_unique)
    check('optimizer analytical benchmark',self_test()['zdt1_mean_front_error']<.06)
    x,_,ids=load();cal=pd.read_excel(DATA/'8.xlsx',sheet_name='Calibrated chemical dataset').set_index('beer').loc[ids.beer]
    transform=pd.read_csv(ROOT/'baseline/inputs/00_数值变换核验.csv').set_index('compound');scale_checks=[]
    for column,is_log in zip(DEFECTS,[True,False,True,False]):
        check('recorded transform '+column,(transform.loc[column,'transform']=='log10')==is_log)
        values=cal[column].to_numpy();valid=np.isfinite(values)&(values>0)&np.isfinite(x[column].to_numpy())
        ref=np.log10(values[valid]) if is_log else values[valid];delta=abs(x[column].to_numpy()[valid]-ref)
        check('majority calibration supports scale '+column,np.mean(delta<1e-6)>.95)
        scale_checks.append({'column':column,'transform':'log10' if is_log else 'raw','compared':len(delta),'discrepancies_above_1e_6':int((delta>=1e-6).sum()),'max_difference':delta.max()})
    pd.DataFrame(scale_checks).to_csv(OUT/'defect_scale_audit.csv',index=False)
    summary=pd.read_csv(OUT/'optimization_summary.csv');n=0
    for case in summary['case'].unique():
        root=OUT/case;config=json.loads((root/'case_config.json').read_text(encoding='utf-8'));p=Problem(config['source_beer'],config['target_style'])
        for file in root.glob('*_pareto.csv'):
            frame=pd.read_csv(file);u=frame[VARIABLES].to_numpy();f,cv,d=p.evaluate(u,True)
            check(case+'/'+file.stem+' recomputes exactly',np.allclose(f,frame[OBJECTIVES],atol=1e-9))
            check(case+'/'+file.stem+' variable box',np.all((u>=0)&(u<=1)))
            check(case+'/'+file.stem+' chemical cap',np.all(concentration(d['chemistry'][:,p.di])<=p.cap*(1+1e-10)))
            check(case+'/'+file.stem+' model-wise stale guard',np.max(d['stale_worst_delta'])<=1e-8)
            check(case+'/'+file.stem+' support guard',np.max(d['support_distance'])<=p.support_limit*(1+1e-8))
            check(case+'/'+file.stem+' internally nondominated',len(fronts(f[:,:3] if file.stem=='no_external_pareto' else f)[0])==len(f))
            old=p.external_holdout.copy();p.external_holdout=np.ones_like(old)*.123;ff,cc=p.evaluate(u[:5]);p.external_holdout=old
            check(case+'/'+file.stem+' objective independent of heldout',np.allclose(ff,f[:5],atol=1e-12) and np.allclose(cc,cv[:5],atol=1e-12))
            n+=len(frame)
        recommendations=pd.read_csv(root/'recommendations.csv')
        for row in recommendations.itertuples():
            path=root/f'{row.variant}_{row.scenario}_chemistry.csv';saved=pd.read_csv(path).to_numpy();u=np.array([[getattr(row,c) for c in VARIABLES]])
            _,cv,d=p.evaluate(u,True)
            check(case+'/'+row.variant+'/'+row.scenario+' exported recommendation',np.allclose(saved,d['chemistry'],atol=1e-9) and cv[0]<=1e-8)
    checks_path=OUT/'validation_checks.csv';pd.DataFrame(checks).to_csv(checks_path,index=False)
    save({'checks':len(checks),'passed':sum(x['pass'] for x in checks),'rechecked_front_points':n,'model_pickle_loaded_in_fresh_process':True},'validation_summary.json')
    print('PASS',len(checks),'checks,',n,'front points')

if __name__=='__main__':main()
