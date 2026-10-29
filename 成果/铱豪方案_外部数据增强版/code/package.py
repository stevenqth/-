"""Package code, fitted model, data, results and improvement advice; no paper."""
from pathlib import Path
import csv,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1]

def main():
    files=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        if any(part in {'__pycache__','.cache'} for part in p.relative_to(ROOT).parts):continue
        if p.suffix in {'.log','.pyc','.zip'} or p.name=='交付文件清单.csv':continue
        assert p.suffix not in {'.pdf','.docx'},'No paper requested'
        files.append(p)
    manifest=ROOT/'交付文件清单.csv'
    with manifest.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['path','bytes','sha256'])
        for p in files:w.writerow([p.relative_to(ROOT).as_posix(),p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest()])
    output=ROOT.parent/'铱豪方案_外部数据增强模型与代码.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[manifest]:z.write(p,'yihao_external_model/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        for n in ['models/surrogates.joblib','code/optimize.py','code/nsga2.py','基于模型的论文改进意见.md','模型说明与结果.md']:
            assert 'yihao_external_model/'+n in z.namelist()
        for p in files:
            assert hashlib.sha256(z.read('yihao_external_model/'+p.relative_to(ROOT).as_posix())).digest()==hashlib.sha256(p.read_bytes()).digest()
    print(output,'bytes',output.stat().st_size,'files',len(files)+1)

if __name__=='__main__':main()
