"""Incremental completion of the compact-family ablation; no refitting of proxies.
Normal full reproduction uses optimize.py, which already includes these runs.
"""
from common import *
from optimize import Problem,solve_case

def main():
    summary=pd.read_csv(OUT/'optimization_summary.csv');rec=pd.read_csv(OUT/'all_recommendations.csv');extra=[];recs=[]
    for case in summary['case'].unique():
        if ((summary['case']==case)&(summary.variant=='shared_defect_knob')).any():continue
        config=json.loads((OUT/case/'case_config.json').read_text(encoding='utf-8'))
        p=Problem(config['source_beer'],config['target_style'])
        a,b=solve_case(p,64,50,[42,43],only_variants=['shared_defect_knob']);extra+=a;recs+=b
        old=json.loads((OUT/(case+'_summary.json')).read_text(encoding='utf-8'));old['variants']=summary[summary['case']==case].to_dict('records')+a;save(old,case+'_summary.json')
    if extra:pd.concat([summary,pd.DataFrame(extra)],ignore_index=True).to_csv(OUT/'optimization_summary.csv',index=False)
    if recs:pd.concat([rec,pd.DataFrame(recs)],ignore_index=True).to_csv(OUT/'all_recommendations.csv',index=False)

if __name__=='__main__':main()
