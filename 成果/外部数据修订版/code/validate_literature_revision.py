"""Independent invariants for the revised model outputs and final paper."""
from common import *
import ast,joblib
from literature_models import FoldFeatures
from literature_nsga2 import ranks_and_crowding

REV=OUT/'literature_revision'


def main():
    checks=[]
    def check(label,condition):
        checks.append({'check':label,'pass':bool(condition)})
        if not condition:raise AssertionError(label)
    X,Y,s,ids=load()
    # Fit rules must remain frozen even when validation values shift markedly.
    train=np.array([[0.,np.nan],[0.,2.],[0.,4.],[8.,6.]])
    p=FoldFeatures(rank=True).fit(train);floor=p.floor_.copy();med=p.median_.copy()
    transformed=p.transform(np.array([[100.,np.nan],[-50.,3.]]))
    check('fold rules frozen on unseen extreme validation values',np.array_equal(floor,p.floor_) and np.array_equal(med,p.median_))
    check('unseen extreme is not a newly invented repeated floor',transformed[1,4]==0)
    check('missing state separate from repeated floor state',transformed[0,3]==1 and transformed[0,5]==0)
    check('fold transforms finite',np.isfinite(transformed).all())
    choices=json.loads((REV/'q1_nested_choices.json').read_text(encoding='utf-8'))
    check('nine independent outer fit selections',len(choices)==9 and len({(r['seed'],r['outer_fold']) for r in choices})==9)
    check('all inner choices contain all six declared candidates',all(len(r['inner_scores'])==6 for r in choices))
    oof=pd.read_csv(REV/'q1_oof_predictions.csv');check('OOF rows aligned',oof.beer.equals(ids.beer))
    grade=pd.read_csv(REV/'q1_relative_grades.csv')[list(Y)].to_numpy()
    check('250 by 50 relative grades, finite integers 0 to 6',grade.shape==Y.shape and np.isfinite(grade).all() and np.equal(grade,np.rint(grade)).all() and ((grade>=0)&(grade<=6)).all())
    source=json.loads((REV/'audit_summary.json').read_text(encoding='utf-8'))
    check('input and checkpoint hashes unchanged',all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h for path,h in source['input_sha256'].items()))
    scale=pd.read_csv(REV/'scale_evidence.csv').set_index('compound')
    check('no blanket raw concentration verification',scale.loc['X2.3.butanedione','status']=='mixed_or_unresolved' and scale.loc['X.E..2.nonenal','status']=='mixed_or_unresolved')
    top=pd.read_csv(REV/'q2_top3.csv')
    check('Top3 chemically grouped and unique',len(top)==150 and top.groupby('target').compound.nunique().eq(3).all())
    check('fixed graph diversity constraints retained',top.groupby('target').module.nunique().eq(3).all())
    from literature_revision import NON_MOLECULAR
    check('single-chemical Top3 excludes all 13 macro and derived fields',not top.compound.isin(NON_MOLECULAR).any())
    check('half-sample probabilities valid',top.half_sample_frequency.between(0,1).all())
    conditional=pd.read_csv(REV/'q2_conditional_importance.csv')
    check('all three heldout explanation folds and three repeats present',set(conditional.fold)=={0,1,2} and set(conditional.repeat)=={0,1,2})
    check('negative conditional effects not silently clipped',(conditional.conditional_delta_mse<0).any())
    ns=json.loads((REV/'q3_nsga2_summary.json').read_text(encoding='utf-8'))
    a=pd.read_csv(REV/'q3_pareto.csv');chem=pd.read_csv(REV/'q3_pareto_chemistry.csv');pred=pd.read_csv(REV/'q3_pareto_sensory.csv')
    check('all Pareto files have matching rows',len(a)==len(chem)==len(pred)==ns['n_pareto'])
    check('numeric Pareto outputs finite',np.isfinite(a.to_numpy()).all() and np.isfinite(chem.to_numpy()).all())
    m=joblib.load(REV/'q1_model.joblib');fresh=m.predict(chem[list(X)])
    check('stored Pareto sensory predictions match independent proxy evaluation',np.allclose(fresh,pred[list(Y)],atol=1e-9))
    src=X.loc[ids.beer==ns['source_beer']].fillna(X.median());srcpred=m.predict(src)[0]
    stale=Y.columns.get_loc('stale_hops')
    check('independent final stale proxy non-worsening',np.all(fresh[:,stale]<=srcpred[stale]+1e-8))
    for col in ['DMS','X2.3.butanedione','X4VG','X.E..2.nonenal']:
        check('independent analysis scale non-worsening '+col,np.all(chem[col]<=src.iloc[0][col]+1e-10))
    check('DMS assumed concentration threshold',np.all(10**chem.DMS<=17.25+1e-9))
    check('zero-calibrated source 4VG explicitly not physical non-worsening',ns['source_calibrated_4vg']==0 and 'X4VG' in ns['analysis_scale_only'] and 'X4VG' not in ns['verified_physical_caps'])
    ranks,_=ranks_and_crowding(a[['f1','f2','f3']].to_numpy(),np.zeros(len(a)))
    check('independent combined Pareto nondominance',np.all(ranks==0))
    target=Y.loc[s=='Tripel'].mean().to_numpy();f1=np.mean(((fresh-target)/Y.std().to_numpy())**2,axis=1)
    check('independent objective1 recomputation',np.allclose(f1,a.f1,atol=1e-9))
    d=pd.read_csv(REV/'q3_decisions.csv');check('all four decision preferences refer to real Pareto rows',len(d)==4 and d.front_id.between(1,len(a)).all())
    check('known analytical NSGA-II test performed',ns['tests']['zdt1_median_front_residual']<.03 and ns['tests']['feasibility_priority_fixture'])
    refs=json.loads((ROOT/'literature/sources.json').read_text(encoding='utf-8'))
    check('two provided papers and exactly five additional papers',sum(r['provided'] for r in refs)==2 and sum(not r['provided'] for r in refs)==5)
    for file in (ROOT/'code').glob('literature*.py'):ast.parse(file.read_text(encoding='utf-8'))
    check('all new experimental modules parse',True)
    pdf=ROOT/'output/literature_revision/啤酒风味论文_七篇文献优化版.pdf'
    if pdf.exists():
        from pypdf import PdfReader
        reader=PdfReader(pdf);text='\n'.join(p.extract_text() or '' for p in reader.pages)
        check('revised paper includes all model chapters and references',all(word in text for word in ['问题一','问题二','问题三','Pareto','参考文献']))
        check('paper preserves important limitations',all(word in text for word in ['不是精确','未安装MATLAB','零','原始绝对评分']))
        check('paper pages contain no replacement glyph', '\ufffd' not in text)
        source['pdf_pages']=len(reader.pages)
    pd.DataFrame(checks).to_csv(REV/'validation_checks.csv',index=False)
    (REV/'validation_summary.json').write_text(json.dumps({'checks':len(checks),'all_passed':True,'pdf_pages':source.get('pdf_pages'),'visual_review':'separate PNG review required'},indent=2),encoding='utf-8')
    print('PASS',len(checks))


if __name__=='__main__':main()
