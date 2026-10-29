"""Refresh the readable source and runnable source/data bundle for RYH优化1."""
from pathlib import Path
import ast,hashlib,json,zipfile


def main():
    base=Path(__file__).resolve().parents[1];dest=base/'output/literature_revision';repo=base.parents[1]
    first=['literature_revision.py','literature_models.py','literature_nsga2.py','literature_generation.py','literature_inference.py',
           'validate_literature_revision.py','build_literature_paper.py','format_ryh1_template.py']
    names=first+sorted(p.name for p in (base/'code').glob('*.py') if p.name not in first)
    overview=['# RYH优化1 完整源码','按模块汇总当前代码，供浏览。执行时使用独立.py文件。正文各级标题按用户要求左对齐。','## 模块索引']
    overview += [f'{i}. {name}' for i,name in enumerate(names,1)]
    for name in names:
        source=(base/'code'/name).read_text(encoding='utf-8');ast.parse(source)
        overview += ['## '+name,'```python',source.rstrip(),'```']
    (dest/'RYH优化1_完整源码.md').write_text('\n\n'.join(overview),encoding='utf-8')
    readme='''# RYH优化1 完整代码与运行材料

保留仓库相对目录结构。进入“成果/外部数据修订版”，依次运行：

    python -m pip install -r requirements_literature.txt
    python code/literature_revision.py --stage all --subsamples 100
    python code/build_literature_paper.py
    python code/validate_literature_revision.py
    python code/format_ryh1_template.py

新实验结果位于results/literature_revision，原版结果保留。
本次复用已有cVAE权重做NumPy推理，无需PyTorch；重训原版生成器时另安装原requirements.txt。
模板版论文是RYH优化1.docx，正文各级标题及关键词左对齐，过程性附录已移除。
使用Microsoft Word更新目录和页码后导出PDF；包内已有最新Word与PDF成稿。
完整源码.md用于通读，运行时使用各独立.py模块。主实验、数据、现有权重和元数据已随包提供。
'''
    (dest/'RYH优化1_代码运行说明.md').write_text(readme,encoding='utf-8')
    files=list((base/'code').glob('*.py'))+[p for p in (base/'data').rglob('*') if p.is_file()]
    files += [base/n for n in ['requirements.txt','requirements_literature.txt','README_文献优化版.md','literature/sources.json']]
    files += [base/'results'/n for n in ['q3_cvae.pt','q3_preprocessing.joblib','q1_summary.json','q2_top3.csv','external_audit.json','external_rating_summary.json']]
    files += [dest/n for n in ['RYH优化1_完整源码.md','RYH优化1_代码运行说明.md','RYH优化1.docx','RYH优化1.pdf']]
    files += list((base/'results/literature_revision').glob('*.json'))+list((base/'results/literature_revision').glob('*.csv'))
    files += [repo/'资料/Mathorcup论文标准模板(1).docx']
    unique={p.resolve():p for p in files if p.is_file()};manifest=[]
    output=dest/'RYH优化1_完整代码与运行材料.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in unique.values():
            name=p.relative_to(repo).as_posix();z.write(p,arcname=name)
            manifest.append({'file':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        z.writestr('源码包文件校验.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(output) as z:assert z.testzip() is None
    print('Source modules:',len(names),'bundle bytes:',output.stat().st_size)


if __name__=='__main__':main()
