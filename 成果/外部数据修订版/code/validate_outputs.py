"""Meaningful data and delivery checks; fails on a violated invariant."""
from common import *
import joblib
from pypdf import PdfReader

def main():
    checks=[]
    def check(name,condition):
        checks.append({'check':name,'pass':bool(condition)})
        if not condition:raise AssertionError(name)
    X,Y,s,ids=load()
    grade=pd.read_csv(OUT/'q1_grades_0_6.csv');vals=grade[Y.columns].to_numpy()
    check('250 samples and 50 grade targets',vals.shape==(250,50))
    check('all grades finite integers in 0..6',np.isfinite(vals).all() and ((vals>=0)&(vals<=6)).all() and np.equal(vals,np.rint(vals)).all())
    check('fold-local OOF outputs aligned',pd.read_csv(OUT/'q1_oof_predictions.csv').beer.equals(ids.beer))
    top=pd.read_csv(OUT/'q2_top3.csv');check('exactly 3 distinct factors per target',len(top)==150 and top.groupby('target').compound.nunique().eq(3).all())
    check('distinct graph modules per target',top.groupby('target').module.nunique().eq(3).all())
    check('bootstrap frequencies within [0,1]',top.bootstrap_frequency.between(0,1).all())
    g=pd.read_csv(OUT/'q3_generated_chemistry.csv');check('22 styles with 60 generated samples',len(g)==1320 and g.groupby('style').size().eq(60).all())
    check('generated values finite',np.isfinite(g[list(X)].to_numpy()).all())
    t=pd.read_csv(OUT/'q3_transfer_checks.csv');check('no reported non-worsening violations',t.nonworsening_violations.sum()==0)
    rules=json.loads((OUT/'q3_defect_rules.json').read_text(encoding='utf-8'))
    actual=pd.read_csv(OUT/'q3_transfer_chemistry.csv');sources=X.copy();sources.index=ids.beer
    for col,r in rules.items():
        source_values=sources.loc[actual.beer,col].to_numpy();known=np.isfinite(source_values)
        check('observed-source transfer non-worsening independently: '+col,np.all(actual[col].to_numpy()[known]<=source_values[known]+1e-10))
        check('imputed-source transfer non-worsening independently: '+col,np.all(actual[col].to_numpy()<=np.where(known,source_values,X[col].median())+1e-10))
        if r['threshold'] is not None:
            c=10**g[col] if r['transform']=='log10' else g[col]
            check('generated threshold independently: '+col,np.all(c<=r['threshold']+1e-10))
    m=joblib.load(OUT/'q1_model.joblib');stale=Y.columns.get_loc('stale_hops')
    source_x=sources.loc[actual.beer].copy();source_x=source_x.fillna(X.median())
    src=m.predict(source_x)[:,stale];dst=m.predict(actual[list(X)])[:,stale]
    check('stale-hop proxy non-worsening independently',np.all(dst<=src+1e-10))
    check('identity fallback not counted successful',t.loc[t.fallback_identity,'target_class_success'].sum()==0)
    e=pd.read_csv(OUT/'external_style_mapping.csv');check('unmapped styles retained explicitly',e.confidence.eq('unmapped').any())
    v=pd.read_csv(OUT/'external_vocabulary.csv');check('all three vocabulary sheets parsed',v.sheet.nunique()==3)
    a=pd.read_csv(OUT/'external_agreement.csv');check('both broad and conservative external comparisons',set(a.subset)=={'all_mapped','closer_only'})
    pdf=ROOT/'output/pdf/啤酒风味建模与外部数据验证论文.pdf'
    if pdf.exists():
        reader=PdfReader(pdf);txt='\n'.join(p.extract_text() for p in reader.pages)
        check('paper has all three model chapters',all(h in txt for h in ['问题一的模型','问题二的模型','问题三的模型']))
        check('paper has external analysis and limitations','外部数据的补充分析' in txt and '不足' in txt)
        check('no obsolete drive paths in paper','D:\\github mama' not in txt)
        dump({'pages':len(reader.pages),'characters':len(txt)},'pdf_validation.json')
    pd.DataFrame(checks).to_csv(OUT/'validation_checks.csv',index=False,encoding='utf-8-sig');print('PASS',len(checks))

if __name__=='__main__':main()
