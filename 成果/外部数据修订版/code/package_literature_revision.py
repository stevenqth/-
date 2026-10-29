"""Package the complete runnable revision; omit caches and self-hash recursion."""
from pathlib import Path
import json,hashlib,zipfile,importlib.metadata,platform


def main():
    base=Path(__file__).resolve().parents[1]
    rev=base/'results/literature_revision'
    names=['numpy','pandas','scipy','scikit-learn','matplotlib','networkx','joblib','openpyxl','reportlab','pypdf','pypdfium2']
    (rev/'runtime_environment.json').write_text(json.dumps({'python':platform.python_version(),
        'packages':{name:importlib.metadata.version(name) for name in names}},indent=2),encoding='utf-8')
    qa=json.loads((rev/'delivery_verification.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((base/qa['pdf']).read_bytes()).hexdigest()==qa['sha256'],'PDF changed after visual review'
    rows=[];manifest=base/'交付文件清单_文献优化版.json'
    for p in sorted(base.rglob('*')):
        if p.is_file() and '.cache' not in p.parts and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.log'] and p!=manifest:
            rows.append({'file':p.relative_to(base).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    output=base.parent/'文献优化版_论文代码与证据.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for row in rows:archive.write(base/row['file'],arcname='外部数据修订版/'+row['file'])
        archive.write(manifest,arcname='外部数据修订版/'+manifest.name)
    with zipfile.ZipFile(output) as archive:assert archive.testzip() is None
    print('Verified ZIP:',output,'bytes:',output.stat().st_size,'files:',len(rows)+1)


if __name__=='__main__':main()
