"""Create a portable delivery bundle, excluding caches and legacy experiments."""
from pathlib import Path
import hashlib,csv,zipfile
ROOT=Path(__file__).resolve().parents[1]

def main():
    allowed={'code','data','results','figures','output'}
    files=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if any(part in {'__pycache__','.cache','tmp','legacy'} for part in rel.parts):continue
        if p.suffix in {'.pyc','.log','.zip'} or p.name=='交付文件清单.csv':continue
        if rel.parts[0] in allowed or p.parent==ROOT:files.append(p)
    manifest=ROOT/'交付文件清单.csv'
    with manifest.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['path','bytes','sha256'])
        for p in files:writer.writerow([p.relative_to(ROOT).as_posix(),p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest()])
    bundle=ROOT.parent/'啤酒建模修订版_论文与三问代码.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[manifest]:z.write(p,'beer_revision/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None
        for name in ['code/question1.py','code/question2.py','code/question3.py','code/common.py','code/external_analysis.py','data/official/1.csv','data/external/beer_data_set.csv','output/pdf/啤酒风味建模与外部数据验证论文.pdf']:
            assert 'beer_revision/'+name in z.namelist()
    print(bundle, 'files',len(files)+1,'bytes',bundle.stat().st_size)

if __name__=='__main__':main()
