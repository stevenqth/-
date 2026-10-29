"""Example: predict all original beers with the fitted 60-target model."""
from common import *
import train_surrogates,joblib

def main():
    X,Y,ids=load();m=joblib.load(MODELS/'surrogates.joblib');meta=json.loads((OUT/'surrogate_summary.json').read_text(encoding='utf-8'))
    pred=m.predict(X.to_numpy());pd.concat([ids,pd.DataFrame(pred,columns=meta['target_columns'])],axis=1).to_csv(OUT/'full_fit_predictions.csv',index=False)
    print('Wrote fitted predictions, not validation predictions. Use nested_oof.csv for validation.')

if __name__=='__main__':main()
