"""Build the Chinese paper directly from actual experiment outputs (PDF + text).
Uses ReportLab; no manually entered experimental results. Windows CJK fonts are
preferred, otherwise supply BEER_CJK_FONT pointing to a TrueType Chinese font.
"""
from common import *
import html,platform,importlib.metadata
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER,TA_JUSTIFY
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def main():
    out=ROOT/'output/pdf';out.mkdir(parents=True,exist_ok=True)
    font=Path(os.environ.get('BEER_CJK_FONT',r'C:\Windows\Fonts\simsun.ttc'))
    if not font.exists():raise FileNotFoundError('Set BEER_CJK_FONT to a Chinese TrueType font')
    pdfmetrics.registerFont(TTFont('CJK',str(font),subfontIndex=0))
    bold=Path(r'C:\Windows\Fonts\simhei.ttf');pdfmetrics.registerFont(TTFont('CJKbold',str(bold if bold.exists() else font)))
    styles={
      'body':ParagraphStyle('body',fontName='CJK',fontSize=10.5,leading=18,spaceAfter=7,firstLineIndent=21,wordWrap='CJK',alignment=TA_JUSTIFY),
      'h1':ParagraphStyle('h1',fontName='CJKbold',fontSize=15,leading=22,spaceBefore=16,spaceAfter=10,keepWithNext=True),
      'h2':ParagraphStyle('h2',fontName='CJKbold',fontSize=12,leading=19,spaceBefore=12,spaceAfter=7,keepWithNext=True),
      'title':ParagraphStyle('title',fontName='CJKbold',fontSize=22,leading=34,alignment=TA_CENTER,spaceAfter=18),
      'caption':ParagraphStyle('caption',fontName='CJK',fontSize=9,leading=14,alignment=TA_CENTER,spaceAfter=8),
      'tablecaption':ParagraphStyle('tablecaption',fontName='CJK',fontSize=9,leading=14,alignment=TA_CENTER,spaceAfter=8,keepWithNext=True),
      'cell':ParagraphStyle('cell',fontName='CJK',fontSize=8.5,leading=12,wordWrap='CJK'),
      'eq':ParagraphStyle('eq',fontName='CJK',fontSize=10,leading=17,alignment=TA_CENTER,spaceBefore=6,spaceAfter=9),
    }
    story=[];text=[]
    def p(t,style='body'):
        text.append(t);story.append(Paragraph(html.escape(str(t)),styles[style]))
    def h(t):p(t,'h1')
    def sub(t):p(t,'h2')
    def table(title,headers,rows,widths=None):
        p(title,'tablecaption');data=[[Paragraph(html.escape(str(v)),styles['cell']) for v in row] for row in [headers]+rows]
        tab=Table(data,colWidths=widths or [475/len(headers)]*len(headers),repeatRows=1,hAlign='CENTER')
        tab.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.8,colors.black),('LINEBELOW',(0,0),(-1,0),.5,colors.black),('LINEBELOW',(0,-1),(-1,-1),.8,colors.black),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf2f4')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.append(tab);story.append(Spacer(1,9));text.extend([' | '.join(map(str,r)) for r in [headers]+rows])
    def fig(name,caption):
        from PIL import Image as PI
        path=FIG/name;im=PI.open(path);w,h0=im.size;story.append(KeepTogether([Image(str(path),width=475,height=475*h0/w),Paragraph(html.escape(caption),styles['caption'])]));text.append(caption)
    def f(x,n=3):return '未定义' if x is None or pd.isna(x) else f'{float(x):.{n}f}'
    def js(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
    q1=js('q1_summary.json');q2=js('q2_summary.json');q3=js('q3_summary.json');ex=js('external_audit.json');rating=js('external_rating_summary.json');audit=js('data_audit.json')
    folds=pd.read_csv(OUT/'q1_cv_folds.csv');grades=pd.read_csv(OUT/'q1_grade_sensitivity.csv');top=pd.read_csv(OUT/'q2_top3.csv');agreement=pd.read_csv(OUT/'external_agreement.csv');transfer=pd.read_csv(OUT/'q3_transfer_checks.csv');best=q1['selected_model'];bestscore=q1['scores'][best]
    # Cover contains no invented team number or person name.
    story.append(Spacer(1,70));p('基于化学测量与外部风味描述的\n啤酒感官预测及风格生成研究'.replace('\n',' '),'title');p('2026 年 MathorCup 数学应用挑战赛 大数据竞赛赛道 A','caption');story.append(Spacer(1,24));p('摘要','h1')
    p('啤酒化学组成、专业品评与消费者语言描述之间存在尺度和采样机制差异。本文以赛题提供的250款啤酒为主数据，构建感官预测、关键因素识别及风格条件生成的递进框架，并引入5,558款外部啤酒的评论风味特征进行独立的风格趋势对照。外部数据不具备完整化合物浓度，因此不与主数据直接拼接训练。')
    p(f'针对问题一，采用训练折内缺失插补与三次三折风格分层验证，比较均值、Ridge、随机森林和极端随机树。{best}的折均R²为{f(bestscore["r2"])}，MAE为{f(bestscore["mae"])}。通过训练折内极差和分位数映射输出0至6级结果，并明确区分假设分级与真实原始评分。')
    p(f'针对问题二，以附件相关矩阵构建{q2["nodes"]}节点、{q2["edges"]}边的共表达图，融合风格控制偏相关、Ridge系数和极端随机树重要性，在社区多样性约束下提取Top-3。{q2["bootstrap_repeats"]}次分层有放回重采样所得Top-1与Top-3平均入选频率分别为{f(q2["top1_frequency"])}、{f(q2["top3_frequency"])}，反映候选因素的稳定性差异。')
    p(f'针对问题三，在给定数据尺度标准化后训练条件变分自编码器，生成{q3["generated_n"]}条化合物谱；通过浓度阈值投影和陈旧酒花代理约束完成{q3["transfer_n"]}次迁移尝试，出现{q3["identity_fallback_n"]}次原样回退，非恶化违例计数为{q3["nonworsening_violations"]}。外部分析建立{q3["external"]["mapped_style_count"]}类风格与9个粗粒度感官维度的对应关系，检验风格趋势的一致与偏离。消费者评分模型采用酒厂分组验证，避免同酒厂产品同时进入训练和测试。')
    p('模型将化合物实测、品评标签和评论词频的证据边界分别处理。结果可用于候选因素筛选和计算机辅助探索；原始评分校准、独立化学感官配对数据及酿造盲评仍是提高预测可信度和实际应用能力的关键。')
    p('关键词  感官预测  共表达图  条件变分自编码器  风格迁移  外部验证','caption');story.append(PageBreak())
    h('一 问题重述')
    p('赛题围绕啤酒酿造化合物与风味形成提出三个任务：由存在缺失的化学数据预测Aroma和Flavour评分；依据化合物相关网络识别每个感官维度的三个关键因素并检验稳定性；以啤酒风格为条件生成化合物谱、给出感官画像并实现缺陷不恶化的跨风格迁移[1]。')
    p('参考研究通过化学测量、训练品评组及消费者评论建立啤酒风味预测模型，为本题提供方法与数据背景[2]。本文继承该研究及已有项目中的树模型、网络结构和条件生成思路，对验证划分、重采样、约束结果及外部证据的表达进行重新实现。')
    h('二 问题分析')
    sub('2.1 三个问题的递进关系')
    p('第一问建立化学向量到感官画像的预测函数，其输出是第三问的代理评价；第二问从相关结构和监督证据中给出可解释候选因素；第三问在风格条件下生成向量，并检查化学约束和代理画像。第一问误差会传播到第三问，因此代理评价不能代替独立品评。第二问的关联不直接成为可操作的因果配方。')
    sub('2.2 外部数据在研究中的位置')
    p('Kaggle数据来自BeerAdvocate评论汇总，主要由高评分啤酒构成[3]。其风味列是描述词频，既不是化学浓度，也不是0至6专业评分。本文将它用于风格级画像、描述词词典对照和消费者评分关联分析，并保持其独立于前三问主模型训练和参数选择。')
    p('若外部画像与主数据不一致，可能来源于产品构成、风格分类、语言表达和采样偏差，不能直接判断某一方错误。跨域对照只检验粗粒度方向与相对排序，不主张数值尺度相等。')
    h('三 模型假设与符号说明')
    table('表1 主要符号',['符号','含义','维度或尺度'],[['X','化学及理化特征','250 × 231，沿用题给尺度'],['Y','专业品评结果','250 × 50，标准化连续值'],['s','啤酒风格标签','22类'],['g(Y)','假设的有序等级映射','0至6整数'],['G','化合物共表达图','无向加权图'],['z','条件VAE潜变量','8维'],['p外部','11维风味词频份额','每行归一化后和为1'],['T','附件参考阈值','保留各化合物单位']],[60,205,210])
    p('假设一：附件中相同beer编号指向同一款测量样品，按该标识对齐化学与感官数据。假设二：在缺少原始品评员标准化参数时，0至6级只能采用单调映射近似，不能唯一还原。假设三：共表达社区可帮助分散高度共线候选，但不保证社区内变量可相互替代。')
    p('假设四：风格标签能够提供条件生成的统计信息，但小样本风格的分布估计不稳定。假设五：缺陷参考阈值适用于本题的约束演示，阈值属于风味参考而非法定安全限值。假设六：显式匹配的风格与粗粒度词义可作趋势比较，其差异需要通过敏感性分析体现。')
    h('四 数据预处理')
    sub('4.1 数据规模与来源审计')
    table('表2 输入数据与用途',['数据','规模','用途'],[['附件1.csv',f'{audit["n"]}行，{audit["chemical_features"]}个特征','三问化学与理化输入'],['附件2.csv','231 × 231','化合物网络结构先验'],['附件3.xlsx','品评卡及汇总表','维度定义与分级刻度'],['附件4.csv','250 × 50个感官目标','第一问标签与第三问代理'],['附件5和6.csv','感官及化学感官相关摘要','同源背景核查，不作独立验证'],['附件7.csv','250个样品，186个描述符','消费者描述词语义对照'],['附件8.xlsx','校准浓度、阈值及参考模型表','单位核验、来源与阈值注释'],['外部CSV',f'{ex["rows"]}行，{ex["styles"]}类风格，21列','外部画像及评分关联'],['外部Excel','3个词典工作表','描述词归类']],[105,150,220])
    p(f'化学矩阵共有{audit["missing_chemical_cells"]}个缺失单元，感官矩阵有{audit["missing_sensory_cells"]}个缺失单元。化学表beer_id唯一，但感官表存在精度损失造成的重复；本版以唯一beer编号重排并核对风格标签。输入文件的字节数和SHA256写入input_manifest.csv，避免因文件版本不同导致结果无法复现。')
    p(f'外部数据有{ex["breweries"]}个酒厂，key无重复；缺失名称有{ex["missing"].get("Name",0)}条，11项风味词频总和为零的样本有{ex["zero_flavor_totals"]}条。名称缺失不影响按风格汇总，但不能用于产品身份匹配。外部key与题目beer_id含义不同，本文未假定样本一一对应。')
    sub('4.2 缺失与数值尺度')
    p('回归验证中，插补器仅在训练折拟合中位数，并添加缺失指示列；Ridge额外执行训练折标准化。树模型沿用输入尺度，不对未经确认的化学列强行反变换。生成模型同样在题给尺度逐列标准化，既避免将负数直接当作负浓度，也不宣称全部231列已经获得统一的物理量纲。')
    p('缺陷约束只对经校准表核验的列转换浓度：DMS和4VG采用log10还原，二乙酰与反式2壬烯醛保留原始浓度。约束后按原列尺度写回。其他列的经验上下界仅作为防止极端外推的统计限制，不代替质量守恒或工艺可行性。')
    calibration=pd.read_csv(OUT/'q3_calibration_checks.csv')
    p('按beer编号逐元素核对校准浓度，DMS和4VG的正浓度条目与log10转换一致；二乙酰及反式2壬烯醛的大多数条目支持原始浓度口径，但仍有少量数值差异。模型沿用题目1.csv，不静默覆盖这些观测；完整差异统计见q3_calibration_checks.csv。')
    table('表2a 缺陷列校准一致性',['化学字段','可比条目','差异超过1e-6','最大绝对差异'],[[r.compound,r.n_compared,r.n_discrepancy_above_1e_6,f(r.max_abs_difference,6)] for r in calibration.itertuples()],[180,85,105,105])
    sub('4.3 外部词频标准化')
    p('对每款外部啤酒，将11项词频除以该行词频总和，得到组成份额，以减少总描述量差异。该变换仍不能完全校正评论数、文本长度、语言或词典重叠；增加某项份额也会机械改变其他份额，因此不作独立因果解释。')
    p('132条风味词频总和为零的记录无法计算份额，风格画像统计将其排除；消费者评分模型则用全零向量表示未获得风味信息并保留样本。该向量不能解释为完全没有风味，未来取得原始评论数后应进一步区分无评论与无词典命中。')
    p('p(i,k) = count(i,k) / sum_j count(i,j)                                      （1）','eq')
    p('跨域比较时，分别在专业感官数据和外部份额数据中按维度计算经验百分位，再按风格汇总。生成样品的百分位以真实专业感官样本为参考，不在生成样本自身重新排序。比较指标包括跨风格Spearman相关及平均绝对百分位差。')
    h('五 问题一的模型建立与求解')
    sub('5.1 求解思路与模型')
    p('以50维标准化感官结果为联合回归目标，比较均值预测、Ridge、随机森林与Extra Trees[4,5]。均值预测检验模型是否优于不使用化学信息的基准。两个树模型均为300棵树，随机森林叶节点最小样本数为3，Extra Trees为2，特征子采样采用平方根；Ridge正则强度固定为100。参数预先固定，避免本轮对外部数据调参。')
    p('最小风格只有3个样品，因此采用3折分层而非5折。重复种子42、1024和2024，共9个验证折。插补及线性标准化在每个训练折独立拟合。报告各折平均与标准差，标准差仅反映划分波动，不是独立样本的置信区间。')
    p('R² = 1 - sum_i (y_i - yhat_i)² / sum_i (y_i - mean(y))²                     （2）','eq')
    p('综合R²为各目标R²的均值，MAE为标准化目标的平均绝对误差。候选模型按本轮平均R²选择并在全部样品上重训；同一交叉验证用于选择和报告，存在选择乐观性，不能等同于独立外部测试。')
    sub('5.2 有序等级的假设映射')
    p('主口径将训练折每个目标的最小值、最大值映射到0和6，验证预测值先截断到区间再取整；敏感性口径使用训练折经验分位数切成7级。所有边界仅用训练折估计，避免验证标签参与边界确定。')
    p('g(y) = round[6 × clip((y - min_train)/(max_train - min_train), 0, 1)]      （3）','eq')
    p('分级精确率、相差不超过一级的比例和加权Kappa均相对于同一映射得到的标签计算，不能证明恢复真实0至6评分。导出全部50目标预测与品评卡维度映射；没有对应标签的乙醛维度标为不可评分，不以零值填充。')
    sub('5.3 求解结果与误差分析')
    rows=[]
    for name,g in folds.groupby('model'):
        rows.append([name,f'{f(g.r2.mean())} ± {f(g.r2.std())}',f(g.mae.mean()),f(g.spearman.mean())])
    table('表3 问题一三次三折验证结果',['模型','R² 均值±标准差','MAE','Spearman'],rows,[115,145,100,115]);fig('q1_models.png','图1 不同回归模型的验证R²及折间波动')
    per=pd.read_csv(OUT/'q1_per_target.csv');p(f'选择的模型为{best}。逐目标结果中有{int((per.r2>0).sum())}/{len(per)}个维度的平均重复OOF预测R²大于零。不同目标难度不一致，因此总平均不能代表每个香气维度均可可靠预测；完整逐目标结果随代码交付。')
    gg=grades[grades.model==best].groupby('mapping')[['exact','within1','kappa']].mean()
    table('表4 选定模型的假设分级敏感性',['映射','精确一致率','相差至多一级','加权Kappa'],[[i,f(r.exact),f(r.within1),f(r.kappa)] for i,r in gg.iterrows()])
    p('发布的0至6答案表由全数据重训模型生成，是已知样本的拟合答案；另提供折外答案表用于评估。两者用途不同，不用全数据拟合误差替代泛化误差。')
    h('六 问题二的模型建立与求解')
    sub('6.1 共表达图与候选证据')
    p('附件2.csv中的标签经已有字段映射表对齐化学列，建立无向加权图。若两化合物相关系数绝对值不低于0.5则连边，边权为绝对相关系数。采用贪心模块度社区划分，各感官目标的三个候选尽量来自不同社区。图来自全样本摘要，作为固定结构先验使用。')
    p('监督证据包含三类：先对化学和感官变量分别回归风格哑变量，再计算残差相关；标准化化学特征上的Ridge系数绝对值；逐感官目标拟合的Extra Trees特征重要性。每类证据转为231个特征内的百分位后等权平均。树重要性用于候选排序，不能解释为因果效应大小。')
    p('Score(c,t) = [rank(|partial(c,t)|) + rank(|ridge(c,t)|) + rank(tree(c,t))] / 3  （4）','eq')
    p('附件6.csv由同源样本产生，不作为新增独立证据参与评分，避免将已有相关性重复计权。按综合分数降序依次选取不同社区候选，每个目标输出三个因素，并保留原始偏相关方向、来源类别、阈值与单位。')
    sub('6.2 稳定性估计')
    p(f'本版在每个风格内部有放回抽取与原风格同数量的样本，重复{q2["bootstrap_repeats"]}次；每次重新插补并计算三种监督证据，然后重新选择Top-3。此过程是真正的分层Bootstrap，而不是将重复交叉验证折计为Bootstrap。固定相关图不随抽样变化，因此频率表示条件于当前图的选择稳定性。')
    p('f(c,t) = 被选入目标t的Top-3的次数 / Bootstrap次数                         （5）','eq')
    p(f'在当前重复次数下，二项近似的最大Monte Carlo标准误为{f(q2["maximum_mc_se"])}。该误差仅衡量有限重采样次数造成的数值波动；若要精确区分0.5与0.6的入选频率，应增加到200次或更多。')
    sub('6.3 网络与关键因素结果')
    table('表5 网络与稳定性统计',['指标','结果'],[['节点 / 边 / 社区',f'{q2["nodes"]} / {q2["edges"]} / {q2["modules"]}'],['Top-1平均入选频率',f(q2['top1_frequency'])],['Top-3平均入选频率',f(q2['top3_frequency'])],['三个候选均达到0.8的目标数',str(q2['all3_frequency_at_least_0_8'])],['三种方法平均Top-3 Jaccard',f(q2['mean_method_jaccard'])],['具有来源类别注释的候选数',str(q2['source_annotated'])+' / 150']],[285,190])
    selected=['acidity','bitternes','sweetness','alcohol','X4vg','dms','overall'];r=[]
    for target in selected:
        t=top[top.target==target].sort_values('rank');r.append([target,'；'.join(f'{a.compound} ({a.bootstrap_frequency:.2f})' for a in t.itertuples())])
    table('表6 代表性感官维度的Top-3及入选频率',['维度','候选化合物与频率'],r,[85,390]);fig('q2_stability.png','图2 Top-3入选频率分布及网络阈值和证据消融敏感性')
    p('化学合理性讨论分成两个层次：来源注释说明候选与酵母、麦芽或酒花的已知类别关系；相关方向与阈值帮助检查明显不合理的解释。来源匹配只能提供解释线索，不是机制验证。同名缺陷化合物若未进入Top-3，可能与浓度范围、共线性或测量噪声有关，不能据此否认其感官作用。')
    p('当第二、第三名频率较低或方法间Jaccard较低时，论文保留不稳定标记，推荐把候选看作实验筛选集合。若需要判断单一化合物的真实贡献，应进行控制其他条件的加标或干预实验。')
    h('七 问题三的模型建立与求解')
    sub('7.1 条件生成模型')
    p('以22类风格的独热向量作为条件，使用条件VAE[6]。编码器输入231维标准化化学向量与风格条件，隐藏层为128和64维，输出8维潜变量均值和对数方差；解码器将潜变量与风格拼接，经64、128维隐藏层重建231维特征。激活函数为ReLU，优化器Adam，学习率0.001，KL权重0.005，训练1200轮。')
    p('L = mean[(x - decoder(z,s))²] + 0.005 × KL(q(z|x,s) || N(0,I))             （6）','eq')
    p('生成时从目标风格真实样本的后验均值中有放回抽样，再加入标准差0.2的高斯扰动。相较只围绕一个风格中心采样，这种后验混合保留更多组内差异，但仍可能记忆训练样本。每种风格生成60条谱，反标准化后截断到训练特征经验范围，再实施缺陷约束。')
    sub('7.2 留出重构与基准')
    p('三折验证中，插补和标准化均在训练折拟合，VAE仅使用训练折样本学习。将验证样本编码后重建，比较标准化MSE及逐特征平均R²，并与8维PCA重构及训练集风格均值基准比较。该测试衡量重构泛化，不等同于无条件生成质量。全数据训练的风格分类器仅用于内部一致性检查，其真实样本折外准确率另行报告。')
    table('表7 三折留出重构结果',['模型','标准化MSE','逐特征平均R²'],[[k,f(v['standardized_mse']),f(v['r2'])] for k,v in q3['heldout_reconstruction'].items()],[175,150,150])
    p('本次留出检验暴露了生成模型的泛化不足：训练折中接近常量的特征可能在验证折出现极端取值，放大标准化误差；验证目标方差很小也会使逐特征R²极负。因此不能沿用旧版训练集高重构R²来证明泛化。应结合PCA基准、绝对误差与逐列尺度核查解读，当前生成结果只作受约束探索，不应称为高保真化学谱。')
    p(f'真实样本的分类器折外准确率为{f(q3["real_classifier_oof_accuracy"])}，生成谱在全数据分类器上的风格一致率为{f(q3["generated_classifier_agreement"])}。两者测试对象和训练条件不同，不能以生成一致率高于真实折外准确率证明生成更真实。')
    sub('7.3 缺陷约束与跨风格迁移')
    rules=js('q3_defect_rules.json');table('表8 缺陷尺度与参考阈值',['化学字段','输入尺度','阈值','单位'],[[k,v['transform'],f(v['threshold'],5),str(v['unit'])] for k,v in rules.items()],[180,85,100,110])
    p('对有参考阈值的生成缺陷浓度采用0.95倍阈值上限。迁移时进一步要求不超过源样品浓度的0.95倍；反式2壬烯醛没有可用绝对阈值时，只采用相对非恶化约束，不编造阈值。陈旧酒花没有单一对应化学列，用第一问模型预测的stale_hops作为代理约束。')
    p('C_new = min(C_candidate, 0.95 × T, 0.95 × C_source)                       （7）','eq')
    p('对每个源样本生成24个目标条件候选，先进行化学投影，再筛选陈旧酒花代理不高于源样品的候选，最后选目标分类概率最高者。若无可行候选，原样返回源样品并明确记为迁移失败。身份回退只保证非恶化，不保证低于绝对阈值，也不作为成功生成。')
    original_x,_,_,original_ids=load();original_x.index=original_ids.beer
    missing_source=int(original_x.loc[transfer.beer,list(rules)].isna().any(axis=1).sum())
    p(f'迁移源样品中有{missing_source}次尝试涉及至少一个缺失缺陷浓度，程序采用训练数据中位数作为约束基准。这些尝试的非恶化仅相对于插补值成立，不能证明相对于未知真实浓度也不恶化；其余源浓度可由输出独立核查。')
    t=transfer.groupby(['source_style','target_style']).agg(n=('beer','size'),fallback=('fallback_identity','sum'),success=('target_class_success','mean'),violations=('nonworsening_violations','sum')).reset_index()
    table('表9 跨风格迁移结果',['方向','尝试数','原样回退','目标识别率','非恶化违例'],[[r.source_style+' → '+r.target_style,r.n,r.fallback,f(r.success),r.violations] for r in t.itertuples()],[205,60,65,75,70]);fig('q3_validation.png','图3 留出重构误差和跨风格迁移的内部识别结果')
    checks=pd.read_csv(OUT/'q3_defect_checks.csv');table('表10 按风格生成的阈值检查',['字段','投影前超阈值','投影后超阈值','被投影比例'],[[r.compound,f(r.raw_exceedance),f(r.projected_exceedance),f(r.projection_fraction)] for r in checks.itertuples()],[175,100,100,100])
    p('投影后零超限由约束算子保证，并不意味着VAE已自主学会缺陷控制；若投影比例较高，生成分布会被集中到边界附近。实际产品验证还需结合基质效应、风格适用性及测量误差，尤其4VG在部分风格中具有特征风味而非普遍缺陷。')
    h('八 外部数据的补充分析')
    sub('8.1 风格和描述词映射')
    mapping=pd.read_csv(OUT/'external_style_mapping.csv');table('表11 外部风格对应与样本数',['题目风格','题目样本','外部样本','对应程度'],[[r.official_style,r.official_n,r.external_n,{'closer':'较接近','approximate':'近似','unmapped':'未对应'}[r.confidence]] for r in mapping.itertuples()],[190,85,85,115])
    p('对应程度为较接近仍不代表类别定义完全一致。Tripel、Dubbel、Faro等有较明确名称对应；Pils/Lager、Wheat和Stout/Porter需要合并外部子类，标为近似。Kriek与Lambic - Fruit的关系并非一一对应，本版不强行合并。全部16类及8类较接近风格分别计算结果，避免只展示一致性最好的类别。')
    p(f'词典共提取{ex["vocabulary"]["vocabulary_entries"]}个类别词条，按去掉aroma、taste、palate前缀后的精确词匹配及少量显式类别名别名，匹配原附件{ex["vocabulary"]["mapped_original_columns"]}/186个消费者描述列。一个词可能属于多个词典类别，保留重叠标记；未匹配字段不做模糊补全。附件7已标准化，聚合结果只作相对强弱对照，不能解释为原始词频。')
    p('专业品评与外部数据对齐酒体、酒精感、苦、甜、酸、果香、酒花、香料和麦芽9个维度；涩感与咸味缺少直接专业标签，留作未对应。果香等合并多个专业感官列，是显式语义假设，完整映射写入external_sensory_mapping.csv。')
    sub('8.2 风格画像的外部一致性')
    a=agreement[(agreement.subset=='all_mapped')&(agreement.source=='official')].set_index('dimension');b=agreement[(agreement.subset=='all_mapped')&(agreement.source=='generated')].set_index('dimension')
    table('表12 16类风格的跨域排序相关',['维度','实测与外部rho','95%重采样区间','生成代理与外部rho'],[[k,f(r.spearman),f'[{f(r.ci_low)}, {f(r.ci_high)}]',f(b.loc[k,'spearman'])] for k,r in a.iterrows()],[90,105,165,115]);fig('external_profiles.png','图4 Tripel与Lambic的专业实测 外部评论和生成代理百分位画像')
    p('95%区间对匹配风格进行500次有放回重采样，反映有限风格数量带来的不确定性；并非每个产品的预测区间。不同风格共享来源平台和词典，重采样独立性假设仍有限。9维结果为探索性描述，不以多重显著性筛选作为结论。')
    p(f'跨维度描述性平均相关为：专业实测对外部{f(q3["external"]["mean_official_rho"])}，生成代理对外部{f(q3["external"]["mean_generated_rho"])}。平均相关只用于概览，具体判断应结合表12的逐维方向、区间与图4的偏差。生成画像经第一问模型转换，仍携带其预测误差，不能作为独立化学验证。')
    closer=agreement.query("subset=='closer_only' and source=='official'");p(f'仅保留8类较接近风格时，专业实测对外部的跨维度平均rho为{f(closer.spearman.mean())}。子集减少使估计更不稳定；若与全部风格结果差异明显，应优先核查类别映射而非声称模型效果改变。')
    cons=pd.read_csv(OUT/'external_consumer_agreement.csv');p(f'对原附件消费者描述符按词典聚合后，{len(cons)}个可匹配维度与外部风格排序的平均rho为{f(cons.spearman.mean())}。这项比较补充了专业品评对照，但两个评论来源的语言、评分人群及聚合方式仍不同。')
    sub('8.3 消费者评分的探索性预测')
    p('以Ave Rating为目标，比较“风格与ABV”基线和“风格与ABV加11项词频份额”模型。采用按酒厂分组的5折验证，同一酒厂不会同时进入训练和测试；风格独热编码器只在训练折拟合，未知类别忽略。两个模型均使用160棵树、叶节点至少5个样本的随机森林。')
    table('表13 酒厂分组验证的消费者评分结果',['模型','OOF R²','OOF MAE','OOF Spearman'],[[k,f(rating[k]['r2']),f(rating[k]['mae']),f(rating[k]['spearman'])] for k in ['Style_ABV','Style_ABV_Flavors']],[190,95,95,95]);fig('external_rating.png','图5 按酒厂验证的风味特征置换重要性')
    gain=rating['equal_brewery_mae_gain'];p(f'先计算每个酒厂的平均绝对误差改善，再对酒厂进行1000次重采样，得到等酒厂加权MAE改善{f(gain["estimate"],4)}，95%区间[{f(gain["ci_low"],4)}, {f(gain["ci_high"],4)}]。该估计与按所有产品等权的总体MAE差值不同，应避免混用。')
    p('风味词频和平均评分来自同一评论体系，可能共享评价语境；外部样本又偏向高评分产品。因此该模型是描述性关联分析，不代表增加某种风味就能提升评分，也不用于直接优化题目中的化合物浓度。置换单个份额会暂时破坏组成约束，重要性仅作敏感度参考。')
    h('九 模型分析与检验')
    sub('9.1 缺失和等级映射敏感性')
    missing=pd.read_csv(OUT/'q1_missing_sensitivity.csv').groupby('extra_missing_rate')[['r2','mae']].mean();table('表14 只对验证特征增加随机缺失',['新增缺失比例','平均R²','平均MAE'],[[f(i,2),f(r.r2),f(r.mae)] for i,r in missing.iterrows()])
    p('扰动实验保持训练集不变，仅在验证集随机遮蔽0%、5%和10%的单元。该实验评估随机缺失鲁棒性，不代表检测限或特定仪器造成的非随机缺失。等级映射的差异已由表4给出；若结果随映射明显改变，分级结论必须保留假设限定。')
    sub('9.2 网络与生成敏感性')
    noise=pd.read_csv(OUT/'q3_noise_sensitivity.csv');table('表15 固定生成器下潜变量噪声敏感性',['噪声标准差','风格分类一致率','标准化总体方差'],[[f(r.latent_noise,1),f(r.classifier_agreement),f(r.mean_standardized_variance)] for r in noise.itertuples()])
    p('第二问分别使用0.4、0.5和0.6相关阈值，并逐一移除三种证据，计算相对主方案的Jaccard。第三问改变潜变量扰动强度，只检验已拟合模型的采样敏感性，不将其表述为重新训练的超参数稳健性。总体方差包括风格间差异，不能单独证明组内多样性。')
    norm=pd.read_csv(OUT/'external_normalization_sensitivity.csv');p(f'外部词频原值与份额归一化得到的风格排序相关，在9个对应维度中的范围为{f(norm.raw_vs_share_style_spearman.min())}至{f(norm.raw_vs_share_style_spearman.max())}。该差异提示文本量处理会影响外部对照，归一化选择需随论文公开。')
    sub('9.3 验证层次与误差传播')
    p('第一问的折外预测评价泛化误差；第二问的Bootstrap评价固定图条件下的候选稳定性；第三问的留出重构评价编码解码能力，生成分类评价内部风格一致性；外部评论对照评价粗粒度语义趋势。四类指标不能互相替代。特别是使用第一问代理验证第三问，存在共享模型误差，须由真实品评补充。')
    h('十 模型评价与推广')
    sub('10.1 优点')
    p('本研究将各类数据来源与量纲分开处理，使用折内预处理、适合小风格样本数的分层划分和酒厂分组验证；关键因素选择同时提供频率、方法差异和来源注释；生成过程显式记录投影比例及迁移失败；全部表格由实际实验文件自动生成，减少人工抄录错误。')
    sub('10.2 不足')
    p('主数据样本量小而特征维度高，模型选择尚无独立测试集；0至6原始评分无法唯一还原；固定全样本网络使稳定性结果条件于已有结构；Bootstrap次数有限。外部样本具有高评分选择偏差，粗粒度映射并不精确。生成谱缺少物理守恒与工艺变量约束，不能证明可酿造或可销售。')
    p('条件VAE的经验后验采样依赖已观测风格，不支持对未见风格作可靠外推。阈值投影可能造成边界堆积，陈旧酒花仅靠代理模型，反式2壬烯醛缺少绝对阈值；原样回退满足非恶化却没有实现目标风格，这类情况已如实计入失败。')
    sub('10.3 提高内容质量所需的证据')
    table('表16 后续补充材料及用途',['优先级','所需材料','可解决的问题'],[['高','原始逐品评员0至6评分及标准化参数','校准真实等级、分解评委差异'],['高','完整化学单位与变换、检出限数据字典','核实所有列及缺失机制'],['高','独立批次化学与感官配对样本','真正检验泛化与候选因素'],['高','盲评及小规模酿造或加标实验','验证生成风味及因果机制'],['中','原附件商品身份、酒厂、年份与外部key对照','建立产品级外部验证'],['中','逐条评论、有效评论数和文本长度','校正词频曝光及同源偏差'],['中','普通及低评分产品的外部样本','降低高分筛选偏差'],['中','缺陷阈值文献与风格适用条件','完善非恶化与可接受性约束'],['中','配方、温度、发酵时间和质量守恒信息','将谱空间迁移转化为可实施工艺']],[45,220,210])
    p('推广应先在同风格、同测量口径下开展外部批次验证，再引入原料和工艺条件形成受约束的生成模型。消费者喜好目标应在独立评分样本上检验，并以多个感官维度及缺陷风险共同评价，避免将单一总体评分作为唯一优化目标。')
    h('十一 参考文献')
    refs=[
    '[1] 2026年MathorCup数学应用挑战赛大数据竞赛赛道A初赛题面及附件1至8. 用户提供的本地赛题材料, 2026.',
    '[2] Schreurs M, Piampongsant S, Roncoroni M, et al. Predicting and improving complex beer flavor through machine learning. Nature Communications, 15:2368, 2024. doi:10.1038/s41467-024-46346-0.',
    '[3] sp1222 / stephenpolozoff. Beer Information - Tasting Profiles. Kaggle. https://www.kaggle.com/datasets/stephenpolozoff/top-beer-information . 用户提供文件及页面说明, 访问日期2026-10-28.',
    '[4] Breiman L. Random Forests. Machine Learning, 45:5-32, 2001.',
    '[5] Geurts P, Ernst D, Wehenkel L. Extremely randomized trees. Machine Learning, 63:3-42, 2006.',
    '[6] Kingma D P, Welling M. Auto-Encoding Variational Bayes. ICLR, 2014. 条件形式在本文中通过拼接风格向量实现.',
    '[7] Schreurs M, et al. Data from Predicting and improving complex beer flavor through machine learning. Zenodo, 2024. doi:10.5281/zenodo.10653704.',
    ]
    for r in refs:p(r)
    h('附录 代码与复现说明')
    p('本论文配套三问独立入口：question1.py、question2.py、question3.py；共享common.py，第三问调用external_analysis.py。先运行问题一生成感官代理，再运行问题二和问题三。所有输入按脚本位置定位，不依赖旧版D盘路径。完整代码与结果随交付包提供，正文未复制数百行程序以免掩盖建模内容。')
    table('附表A 三问输入输出',['入口','主要输入','主要输出'],[['question1.py','1.csv、4.csv、评分卡映射','折外预测、0至6等级、模型比较'],['question2.py','1.csv、2.csv、4.csv、8.xlsx、字段映射','150条Top-3、Bootstrap频率、图边及社区'],['question3.py','主数据、问题一模型、两份外部文件','生成谱、迁移谱、约束检查、外部画像与评分实验']],[110,180,185])
    p('复现命令：python code/question1.py；python code/question2.py --bootstrap 30；python code/question3.py --epochs 1200；python code/build_paper.py。Python及依赖版本见requirements.txt和results/environment.json。文件哈希见input_manifest.csv及交付清单。')
    p('已验证结论对应本次脚本的实际运行输出；风格语义对应、单调分级和代理阈值适用性属于明确假设；真实品评校准、产品级身份匹配和工艺可行性属于待验证问题。')
    def footer(c,doc):
        c.setFont('CJK',8);c.setFillColor(colors.HexColor('#555555'));c.drawCentredString(A4[0]/2,25,str(doc.page))
    pdf=out/'啤酒风味建模与外部数据验证论文.pdf'
    doc=SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=60,leftMargin=60,topMargin=48,bottomMargin=48,title='基于化学测量与外部风味描述的啤酒感官预测及风格生成研究',author='')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    (ROOT/'output/论文正文.txt').write_text('\n\n'.join(text),encoding='utf-8')
    versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','matplotlib','networkx','torch','openpyxl','reportlab']}
    dump({'python':platform.python_version(),'packages':versions},'environment.json')
    print(pdf)

if __name__=='__main__': main()

