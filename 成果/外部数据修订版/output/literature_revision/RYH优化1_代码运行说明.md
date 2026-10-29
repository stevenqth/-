# RYH优化1 完整代码与运行材料

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
