"""Build a complete revised paper and per-source change ledger from actual runs.
PDF rendering and visual inspection are a separate required delivery step.
"""
from common import *
import html
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER,TA_JUSTIFY
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage

REV=OUT/'literature_revision';RFIG=FIG/'literature_revision'
OUTDOC=ROOT/'output/literature_revision'


def main():
    OUTDOC.mkdir(parents=True,exist_ok=True)
    def js(n):return json.loads((REV/n).read_text(encoding='utf-8'))
    audit=js('audit_summary.json');q1=js('q1_summary.json');q2=js('q2_summary.json');q3=js('q3_summary.json');nsga=js('q3_nsga2_summary.json')
    refs=json.loads((ROOT/'literature/sources.json').read_text(encoding='utf-8'))
    old1=json.loads((OUT/'q1_summary.json').read_text(encoding='utf-8'))
    external=json.loads((OUT/'external_audit.json').read_text(encoding='utf-8'))
    ratings=json.loads((OUT/'external_rating_summary.json').read_text(encoding='utf-8'))
    v1=q1['pooled_oof_scores']['NestedSelected'];vbase=q1['pooled_oof_scores']['FixedRandomForest']
    top=pd.read_csv(REV/'q2_top3.csv');ci=pd.read_csv(REV/'q2_conditional_importance.csv')
    decisions=pd.read_csv(REV/'q3_decisions.csv');scale=pd.read_csv(REV/'scale_evidence.csv')
    compare=pd.read_csv(REV/'q2_selector_comparison.csv')
    pdfmetrics.registerFont(TTFont('Song',r'C:\Windows\Fonts\simsun.ttc',subfontIndex=0))
    pdfmetrics.registerFont(TTFont('Hei',r'C:\Windows\Fonts\simhei.ttf'))
    sty={
      'body':ParagraphStyle('body',fontName='Song',fontSize=10.5,leading=18,spaceAfter=7,firstLineIndent=21,wordWrap='CJK',alignment=TA_JUSTIFY),
      'h1':ParagraphStyle('h1',fontName='Hei',fontSize=15,leading=23,spaceBefore=15,spaceAfter=9,keepWithNext=True),
      'h2':ParagraphStyle('h2',fontName='Hei',fontSize=12,leading=20,spaceBefore=10,spaceAfter=6,keepWithNext=True),
      'title':ParagraphStyle('title',fontName='Hei',fontSize=22,leading=33,spaceAfter=20,alignment=TA_CENTER),
      'cell':ParagraphStyle('cell',fontName='Song',fontSize=8.5,leading=13,wordWrap='CJK'),
      'caption':ParagraphStyle('caption',fontName='Song',fontSize=9,leading=14,alignment=TA_CENTER,spaceAfter=8,keepWithNext=True)}
    story=[];md=[]
    def p(text,style='body'):
        story.append(Paragraph(html.escape(str(text)),sty[style]));md.append(str(text))
    def h(text):p(text,'h1');md[-1]='## '+md[-1]
    def sub(text):p(text,'h2');md[-1]='### '+md[-1]
    def table(title,headers,rows,widths=None):
        p(title,'caption');data=[[Paragraph(html.escape(str(c)),sty['cell']) for c in r] for r in [headers]+rows]
        t=Table(data,colWidths=widths or [475/len(headers)]*len(headers),repeatRows=1,hAlign='CENTER')
        t.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.7,colors.black),('LINEBELOW',(0,0),(-1,0),.5,colors.black),('LINEBELOW',(0,-1),(-1,-1),.7,colors.black),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf2f4')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.extend([t,Spacer(1,8)]);md.extend(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
    def fig(name,caption):
        path=RFIG/name;w,hh=PILImage.open(path).size
        story.append(KeepTogether([Image(str(path),width=460,height=460*hh/w),Paragraph(html.escape(caption),sty['caption'])]));md.append(f'![{caption}](../../figures/literature_revision/{name})')
    def f(x,n=4):return f'{float(x):.{n}f}'
    p('啤酒感官预测与受约束风格迁移','title')
    p('结合外部数据与文献证据的多目标决策研究','caption')
    h('摘要')
    p('针对化学测量、专业感官评分与消费者评论具有不同数据尺度和采样机制的问题，本文在250款啤酒、231个化学与理化特征及50个感官响应上，构建带有证据审计的预测、归因、生成和多目标决策流程。外部啤酒评论数据只作风格趋势对照，不充当化学测量或专业品评标签。')
    p(f'针对问题一，采用折内缺失及重复底值状态处理，引入秩正态偏最小二乘与树模型候选。外层三折重复三次评价完整选型过程，内层三折选择模型。嵌套程序的完整折外预测平均R²为{f(v1["r2"])}、MAE为{f(v1["mae"])}；同划分固定随机森林的R²为{f(vbase["r2"])}。输出0至6相对等级，但不宣称恢复原始绝对评分。')
    p(f'针对问题二，保留固定共表达网络的原三方法结果作为对照，新增风格残差秩关联选择器及{q2["subsamples"]}次无放回近似分层半样本诊断。150个候选中有{q2["stable_entries_at_0_8"]}项频率达到0.8；通过未见标签的验证特征近邻供体估计条件置换重要性，明确相关变量、候选选择规则及固定图的局限。')
    p(f'针对问题三，复用已训练条件VAE比较三种采样方式，另对条件收缩高斯及两个经验基准进行分层留出分布检验。尺度审计发现70列有标定对照，其中{audit["scale_status_counts"]["verified_positive_rows"]}列在可比正值行中一致、32列存在局部不一致。迁移将已核实正浓度约束和分析尺度非恶化分开，实际运行三次8维潜空间NSGA-II搜索，获得{nsga["n_pareto"]}个近似Pareto候选。')
    p('不同偏好下的决策体现目标感官接近度、化学画像改动与相对缺陷指标之间的取舍。数值约束及代理模型结果仅用于计算机筛选，不替代加标品评或酿造验证。本文的改进首先在于验证隔离、证据标注与可追溯决策，不以增加算法数量代替性能和实验依据。')
    p('关键词  嵌套验证  条件重要性  稳定性选择  条件生成  Pareto决策','caption');story.append(PageBreak())
    h('一 问题与研究范围')
    p('赛题要求由化学谱预测感官评分、识别各感官属性Top-3因素，以及在风格条件下生成化学谱并完成缺陷不恶化的迁移。本文保留外部数据修订版的数据对齐和风格语义映射，并在其基础上重新实现可被文献支持的验证与优化模块。三问之间传递的是预测模型及候选证据，不是已经验证的因果关系。')
    p('用户提供方法稿[1]强调折内尺度处理、重复底值状态与约束证据分层；Schreurs等[2]在真实商业啤酒上比较多类模型并开展加标品评。本文进一步参考模型选择偏差[3]、条件重要性[4]、稳定性选择[5]、条件生成[6]和NSGA-II[7]，分别改善统计验证、解释可靠性及决策透明度。')
    p('原稿[2]的175/75分层训练测试及五折选参，与本文重复嵌套验证不同；其盲评加标结果是该研究自身的实验，不能作为本文候选产品已经改善的证据。[1]为用户提供的方法稿，未核实其作者和公开发表状态，本研究不把它等同于已同行评审成果，也不搬用其数值。')
    h('二 数据尺度与证据')
    sub('2.1 样本连接与来源')
    table('表1 数据与分析角色',['来源','规模','角色'],[['附件1与4','250款、231特征、50响应','按唯一beer连接并核对风格'],['附件2','231维相关结构摘要','固定网络先验，非独立新样本'],['附件3','空白评分卡及汇总','定义评分维度，不能从缓存零造标签'],['附件8','70列有可比标定','逐元素尺度与单位证据'],['外部数据',f'{external["rows"]}款、{external["breweries"]}个酒厂','评论词频和风格背景，不与专业标签拼接']],[105,155,215])
    p(f'主数据有{audit["complete_cases"]}个完整案例，其他样本保留并在训练折内插补。输入以SHA256固定。未把同一啤酒的50个感官维度视为50个独立样本，也未以发生精度损失的长商品编号强行匹配外部产品。')
    sub('2.2 尺度账本与局部不一致')
    p('对每个有标定的字段，在参考浓度为正且双方非缺失的行上，逐一比较恒等、常用对数及自然对数关系。只有至少五个可比行且全部误差不超过1e-6，才记为正值行已核实；中位数误差接近零不能掩盖少量不一致。该证据不覆盖零值替代机制，也不证明不同版本数据完全一致。')
    defects=['DMS','X2.3.butanedione','X4VG','X.E..2.nonenal'];r=scale.set_index('compound')
    table('表2 四类缺陷的尺度证据',['字段','候选变换','不一致条数','本次约束口径'],[[c,r.loc[c,'suggested_transform'],int(r.loc[c,'n_discrepancy']),'正值域浓度及阈值' if r.loc[c,'status']=='verified_positive_rows' else '仅附件分析值非恶化'] for c in defects],[155,75,75,170])
    p('DMS和4VG的可比正值支持log10转换；双乙酰和反式2壬烯醛虽多数支持原值口径，却存在局部不一致，本次不以“多数一致”宣称全部物理浓度已验证。后两项继续采用附件值非增加约束，不对其未知或未完全核实尺度作完整化学阈值保证。DMS的17.25 µg/L和4VG的0.206333333 mg/L来自附件参考表，属于风味参考值，不是法定安全限值。')
    sub('2.3 缺失与重复底值')
    p('参考[1]第5页，显式NA以训练中位数插补并附缺失标记；当某列训练最小值至少重复三次时，另构造底值状态标记。重复最低值不是已知仪器检出限，不随机抖动也不直接当作真实浓度。验证样本只按训练规则变换，不能重新估计中位数、底值或经验分布。')
    p('秩正态候选使用训练经验中秩u=(小于当前值的个数+等于值的个数/2)/训练非缺失数，并以1/(2n)截断概率后映射到标准正态分位。它是一种稳健预测尺度，不是对全部字段重复施加化学对数。')
    h('三 问题一 感官预测与验证隔离')
    sub('3.1 候选模型与完整选型程序')
    p('候选预先固定为均值、3或8成分的秩正态PLS、秩正态Ridge、随机森林和极端随机树。PLS在相关响应间共享低维结构；两个树模型各160棵，分别使用最小叶样本3和2、平方根特征抽样。Ridge强度固定1000。所有候选均保留NA和重复底值状态，PLS与Ridge另执行训练内标准化。')
    p('根据[3]，选择模型本身属于训练。外层按风格分三折、使用42、1024和2024三个种子重复，内层在外层训练集上打乱三折比较多响应均方误差。内层采用普通K折，是因为稀有风格进一步划分后可能不足三例。每次内层拟合都独立重建预处理；外层响应不参与选型。最终评价对象是这套预声明选型程序。')
    p('每个重复拼接完整的250行折外预测后，按各感官目标计算R²、MAE和Spearman，再跨目标与重复汇总。折间结果、逐目标结果及各外层内层选择明细均保存。跨重复标准差只描述划分波动，不是九个独立试验的置信区间。')
    table('表3 同划分外层折外预测结果',['程序','R²','MAE','Spearman'],[[k,f(v['r2']),f(v['mae']),f(v['spearman'])] for k,v in q1['pooled_oof_scores'].items()],[180,95,95,105]);fig('q1_nested.png','图1 选型程序与固定随机森林的重复外层折外表现')
    gain=v1['r2']-vbase['r2']
    p(f'本次嵌套程序相对固定随机森林的平均R²差为{f(gain)}。该差异来自同划分的两套程序，但本研究没有据此进行未经设计的显著性宣称。原修订版固定模型比较中随机森林折均R²为{f(old1["scores"]["RandomForest"]["r2"])}，其汇总尺度和本次完整OOF汇总不同，不能直接把两者之差全归因于算法提高。')
    table('表4 外层训练内部选择次数',['候选','次数'],[[k,v] for k,v in q1['selected_counts'].items()],[300,175])
    sub('3.2 相对等级与误差传播')
    p('附件4是标准化连续响应，评分卡不提供原始逐人0至6校准锚点。因此论文首先报告连续响应的可检验误差，再输出训练响应经验分位划成七档的相对等级。全样本拟合表只用于描述已知样品，不用拟合等级一致率证明绝对评分恢复。后续真实品评应记录原始逐人评分及标准化参数。')
    per=pd.read_csv(REV/'q1_nested_per_target.csv').groupby('target').r2.mean()
    p(f'50个属性中有{int((per>0).sum())}个的重复外层平均R²大于零，stale_hops为{f(per.loc["stale_hops"])}。尤其第三问使用感官代理作为约束时，必须同时公开该属性的泛化能力；数值不恶化不等于真实品评不恶化。')
    h('四 问题二 相关化合物与稳定性')
    sub('4.1 保留原结果并引入可核查对照')
    p('原修订版融合风格控制残差相关、Ridge系数与逐属性树重要性，且Top-3来自不同图社区。本次保留该表及原Bootstrap结果；参考[1]排除13项宏观与派生汇总量，在218项单项化学候选内新增秩正态风格残差关联选择器。第一问预测仍可使用全部231项信息，但不能把糖类总量或派生酯类汇总冒充可直接干预的单一分子。新选择器用于诊断对照，不宣称自动优于原三证据融合。')
    p(f'新选择器与原三方法Top-3的逐属性平均Jaccard为{f(compare.jaccard_vs_original_three_method.mean())}。名单变化提示选择规则敏感，不能仅因更换选择器后频率提高便称化学机制更可靠。固定附件网络仍来自全样本摘要，所以所有结果均条件于该图，未构成独立网络验证。')
    sub('4.2 条件重要性')
    p('Strobl等[4]指出，在相关预测变量上，普通随机置换可能破坏协变量依赖并扭曲重要性。本次只用每个训练折选择Top-3及至多五个绝对相关大于等于0.3的上下文变量，再在验证特征的上下文空间寻找八个近邻供体，整体替换该分子的原值，由管线重算其NA和底值状态。')
    p('分别进行三次局部供体置换及普通置换，计算逐目标MSE相对未置换预测的增量，并保留负值。供体选择不读取验证响应。该近邻方法是本地条件分布的近似，不是[4]按树切点分组的精确实现，不能当作条件随机化检验或因果效应。')
    group=ci.groupby(['target','compound'])[['conditional_delta_mse','marginal_delta_mse']].mean()
    table('表5 条件与普通置换的描述性结果',['指标','结果'],[['目标化合物组合数',len(group)],['条件置换平均MSE增量',f(group.conditional_delta_mse.mean())],['普通置换平均MSE增量',f(group.marginal_delta_mse.mean())],['条件增量为负的组合数',int((group.conditional_delta_mse<0).sum())]],[320,155])
    sub('4.3 无放回半样本稳定性')
    p(f'参考[5]，重复{q2["subsamples"]}次在各风格内无放回抽取约一半样本，稀有风格至少保留一个，每次重估插补、经验秩、风格残差与Top-3。它与原有有放回Bootstrap不同。由于逐风格取整，样本数并非严格总样本的一半；故这里只借鉴子样本选择概率思想，不套用其原理论的误选数界。')
    p(f'新诊断Top-3平均频率为{f(q2["top3_mean_frequency"])}，达到0.8的条目为{q2["stable_entries_at_0_8"]}/150；有限重复的最坏Monte Carlo标准误约{f(q2["maximum_mc_se"])}。频率只反映当前样本、固定图和选型规则的稳定性，不是化合物为真因子的概率。')
    selected=['A_esters_all','F_hops_all','dms','stale_hops','overall'];rows=[]
    for target in selected:
        a=top[top.target==target];rows.append([target,'；'.join(f'{r.compound} ({r.half_sample_frequency:.2f})' for r in a.itertuples())])
    table('表6 代表属性的诊断Top-3及半样本频率',['属性','候选及频率'],rows,[105,370]);fig('q2_stability.png','图2 无放回半样本诊断的入选频率')
    h('五 问题三 条件生成与分布检验')
    sub('5.1 条件变分模型的继承范围')
    p('复用外部数据修订版已保存的8维条件VAE和其插补、标准化参数。编码器以化学谱和风格输出mu及logvar，解码器以潜变量和风格输出化学画像。与[6]一般的条件先验网络不同，原模型KL项使用标准正态先验；拼接风格本身不能被写成已学习条件先验。')
    p('本次不重新训练cVAE，使用限制可加载类型的NumPy推理器读取项目自身的float32权重，计算确定性线性层和ReLU。原模型完整数据重构仅为训练内诊断；没有把它称为留出验证。原修订版三折cVAE留出误差较大，仍保留作为当前生成器泛化不足的证据。')
    sub('5.2 后验不确定性与三种采样')
    p('根据[6]的重参数化形式，新增z=mu+exp(logvar/2)*epsilon，与原有后验均值加固定0.2噪声及标准正态采样并列。前者反映已学习编码后验的不确定性，后两者分别是经验扰动及先验对照；任一方式都不能仅凭高风格识别率证明分布真实。')
    table('表7 固定cVAE的内部采样诊断',['采样','风格一致率','秩空间方差','最近真实谱RMS'],[[k,f(v['classifier_agreement']),f(v['mean_rank_variance']),f(v['nearest_real_rms'])] for k,v in q3['sampler_averages'].items()],[185,95,95,100])
    p('每种风格各生成60条并记录最近真实样品距离与近重复率。方差为组内诊断平均，不能与包含风格间差异的总体方差混称。生成谱明确带generated_style标识；它们是模型输出，不是新增观测，未被回灌为第一问监督标签。')
    sub('5.3 分层留出分布基准')
    p('借鉴[1]第21至23页，在每个外层训练集用共同经验中秩构造正态得分，先减去各风格均值再用LedoitWolf估计共享组内协方差。风格中心按n/(n+3)向全局中心收缩，参数3与共享协方差事先固定，本次没有复现[1]完整超参数网格。')
    p('将Tripel、Kriek和Lambic的留出样本与条件高斯生成、无条件高斯及风格内训练谱重采样比较。评价为231个正态得分边缘的平均Wasserstein距离；它衡量边缘而非联合分布，不能证明消费者喜好或物理可酿造性。所有边缘和协方差均由对应训练折拟合。')
    table('表8 三折留出平均边缘距离',['模型','W1 越小越好'],[[k,f(v)] for k,v in q3['heldout_rank_gaussian_w1'].items()],[330,145])
    p('无论哪一种方法数值最优，均完整保留经验重采样基准。复制训练样品可能在边缘距离上表现较好，但新颖性有限；模型需要同时报告分布接近程度和与真实样品的距离，不能删除强基准突出新方法。')
    h('六 多目标迁移与Pareto决策')
    sub('6.1 决策变量与约束证据')
    p(f'以真实源样品beer={nsga["source_beer"]}从Pils/Lager迁移到Tripel为演示，在固定解码器的8维潜空间搜索。潜变量边界覆盖源后验与目标风格中心附近两个经验标准差；解码输出限制到训练各列经验范围，再施加缺陷修正。该候选族是化学分析谱，不是配方，经验边界也不证明联合物理可行。')
    p('DMS采用正浓度口径，数值上限制为不超过参考阈值和源值的0.95倍。4VG虽在正标定行支持log10变换，但本次源样品的参考标定为零，附件记录的是底值，不能把其指数值当作已核实源浓度；因此4VG源非恶化与双乙酰、反式2壬烯醛一样只解释为分析尺度限制。4VG阈值比较是变换假设下的数值检查，不是源浓度或真实风味保证。stale_hops由第一问新代理预测，候选不得高于同一代理对源样品的预测。')
    sub('6.2 三个目标与求解')
    table('表9 最小化目标',['目标','定义','限定'],[['f1','50维标准化感官偏差平方的均值','与目标风格实测均值比较，包含代理误差'],['f2','相对源画像的标准化RMS改变量','改动代理，非财务或工艺成本'],['f3','四项缺陷指标相对源值比的均值','DMS浓度比加三项分析指标比，不是毒理风险']],[50,230,195])
    p('依据[7]实现非支配排序、目标空间拥挤距离、父子代精英保留、模拟二进制交叉和多项式变异。约束支配使可行解先于不可行解，不可行者按违例大小比较。种群80，每种种子42、43、44固定运行80代，合并三次可行解后重新计算非支配关系。固定代数结束不等于证明收敛或全球最优。')
    p('本次实际执行的是Python实现，未安装MATLAB，不能把这些数值称作MATLAB gamultiobj结果。算法先通过具有已知解析前沿的ZDT1收敛检查及可行性优先的构造样例，再逐项复核本题候选的原始缺陷指标与感官代理非恶化。')
    table('表10 三次求解记录',['随机种子','可行前沿数','停止原因'],[[r['seed'],r['n_feasible_front'],'80代预算结束'] for r in nsga['runs']],[110,110,255]);fig('q3_pareto.png','图3 三次潜空间NSGA-II合并后的近似Pareto集合')
    sub('6.3 偏好后置的决策分析')
    p('先保留Pareto集合，再按其各目标范围归一化，恒定目标不参与距离。采用加权TOPSIS比较到理想点与最差点的距离；均衡、感官优先、少改动和低缺陷场景的权重分别为等权、0.6/0.2/0.2、0.2/0.6/0.2和0.2/0.2/0.6。权重为演示偏好，尚不是团队正式决策。')
    table('表11 四种偏好下的候选选择',['偏好','前沿编号','f1','f2','f3'],[[r.scenario,r.front_id,f(r.f1),f(r.f2),f(r.f3)] for r in decisions.itertuples()],[135,70,90,90,90])
    p('不同偏好可以指向同一候选，也可以产生不同取舍；单一推荐不是客观唯一最优。推荐表需要连同化学画像、感官代理、源编号和证据等级一起审阅。严禁把没有满足代理约束的候选通过放宽条件伪装成合格；若没有可行前沿，程序停止并报告失败。')
    h('七 外部数据对照与验证边界')
    p(f'保留外部修订版的{external["rows"]}款评论汇总与三张词典工作表。外部key不是题目beer编号，未建立产品一一对应；消费者词频和专业感官z分数通过显式风格及九个粗粒度属性对齐，只能用于趋势一致性。本文不以外部数据调节第一问选型或迁移权重。')
    p('原外部评分关联实验按酒厂分五折验证，以避免同酒厂产品同时进入训练和测试。该部分未因本次文献修订重新训练，保留原结果与原版本标识，不把旧生成谱的外部一致性冒充新后验采样表现。评论词频与消费者评分来自同一评论环境，仍可能存在同源与高分样本选择偏差。')
    rate_rows=[]
    for name in ['Style_ABV','Style_ABV_Flavors']:
        if name in ratings:rate_rows.append([name,f(ratings[name]['r2']),f(ratings[name]['mae'])])
    table('表12 沿用原版的分酒厂消费者评分关联',['模型','折外R²','MAE'],rate_rows,[285,95,95])
    if (REV/'external_agreement.csv').exists():
        ea=pd.read_csv(REV/'external_agreement.csv')
        table('表13 本次生成代理与外部画像的风格级相关',['维度','实测对外部','新生成代理对外部'],[[r.dimension,f(r.official_rho),f(r.generated_rho)] for r in ea.itertuples()],[155,160,160])
        p('此处新生成画像由本次后验重参数化样品和新感官代理重新计算。相关系数是按显式对应风格聚合后的生态层面描述，不是产品级化学验证；没有对这九个维度进行选择性显著性报告。')
    h('八 局限与验证方案')
    sub('8.1 哪些改动已经验证')
    p('已验证内容包括键对齐、折内预处理、外层标签与内层选型隔离、半样本不重复、构造支配关系、解析测试前沿以及本题推荐约束。七篇来源逐条对应代码函数、数据表及论文章节，完整对照另附。程序运行成功不等于预测准确或真实风味改善。')
    sub('8.2 尚不能推出的结论')
    p('样本仍小且缺少新酒厂或新批次独立测试；重复嵌套验证只评价当前附件总体。半样本选择不满足原稳定性选择理论全部假设，不能给出FDR/PFER保证。近邻置换不是精确条件检验。固定VAE来自完整数据，其内部采样质量不能当成新的留出泛化。单一源样品和一个目标风格的Pareto结果不能外推到所有迁移对。')
    p('双乙酰和反式2壬烯醛的局部标定差异、重复底值机制、未知尺度及缺少工艺变量均限制物理解释。派生总量和非线性基质作用没有完全纳入约束；陈旧酒花约束依赖预测较弱的代理。当前结果适合筛选实验候选，不支持可酿造、可销售或盲评偏好已改善的声明。')
    sub('8.3 真实品评与工艺验证设计')
    p('参考[2]的加标对照思路，后续应先核实化合物单位、批次和完整浓度，选择有限的Pareto候选，设置同批次未加标对照；随机化顺序、盲法编码、暗色容器及一致开瓶操作，记录逐位品评员原始0至6评分、方向差异与偏好。该段是待执行实验方案，不列为本研究已完成实验。')
    p('对候选的工艺可实现性，应补充原料、发酵条件、组分耦合、总量关系与干预成本，先完成小规模加标或试酿，再对真实感官和缺陷响应作独立检验。若实测与代理冲突，应更新代理与约束，而不是只保留符合模型预期的样品。')
    h('九 结论')
    p(f'本研究以七篇来源为依据，对外部数据修订版的验证、解释与决策流程进行了可复现修改。第一问完整嵌套选型的折外R²为{f(v1["r2"])}；第二问以条件置换和半样本频率揭示名单敏感性；第三问分别报告固定cVAE采样、条件高斯留出分布以及潜空间Pareto搜索，不混同三者的验证层次。')
    p('论文的可交付价值是可追溯的模型比较、证据分级和候选选择，而不是未经品评证实的产品改良。后续应优先取得独立化学感官配对样本、原始评分与标定数据，再扩大迁移对象和工艺变量范围。')
    h('参考文献')
    for index,r in enumerate(refs,1):
        bibli=f'[{index}] {r["authors"]}. {r["title"]}. {r.get("publication",r.get("location",""))}, {r["year"]}.'
        if r.get('doi'):bibli+=' DOI: '+r['doi']+'.'
        elif r.get('url'):bibli+=' '+r['url']+'.'
        p(bibli)
    p('[8] 2026年MathorCup大数据竞赛赛道A题面及附件1至8. 用户提供的本地材料, 2026.')
    p('[9] sp1222 / stephenpolozoff. Beer Information Tasting Profiles. Kaggle. https://www.kaggle.com/datasets/stephenpolozoff/top-beer-information . 外部文件由用户提供，平台资料访问日期2026-10-28.')
    h('附录 复现与逐篇修改记录')
    p('执行code/literature_revision.py --stage all --subsamples 100生成独立结果；执行code/build_literature_paper.py生成本论文及Markdown；执行code/validate_literature_revision.py复核数据、模型和推荐。不覆盖原results/下的实验表，新增结果位于results/literature_revision/，输入及cVAE快照哈希保存于audit_summary.json。')
    table('附表A 七篇来源对应的实际改动',['文献','论文章节','代码与验证'],[[r['id'],'、'.join(r['paper_sections']),r['change']] for r in refs],[45,130,300])
    def footer(c,doc):c.setFont('Helvetica',8);c.drawCentredString(A4[0]/2,25,str(doc.page))
    pdf=OUTDOC/'啤酒风味论文_七篇文献优化版.pdf'
    SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=60,rightMargin=60,topMargin=45,bottomMargin=45,
                      title='啤酒感官预测与受约束风格迁移',author='铱豪').build(story,onFirstPage=footer,onLaterPages=footer)
    (OUTDOC/'论文正文_七篇文献优化版.md').write_text('\n\n'.join(md),encoding='utf-8')
    ledger=['# 七篇文献的具体内容与修改对照','日期 2026-10-29；操作者 铱豪；本表区分直接采用、近似实现和待验证内容。']
    for r in refs:
        ledger += ['## '+r['id']+' '+r['title'], '**来源**：'+r['authors']+'；'+str(r['year'])+'；'+r.get('url',r.get('location','用户本地稿件')),
                   '**具体文献内容与位置**：'+r['evidence'], '**修改了什么**：'+r['change'],
                   '**论文章节**：'+'；'.join(r['paper_sections']), '**对应代码**：'+'；'.join(r['code'])]
        if r['id'] in ['U1','U2','L1']:ledger+=['**验证证据**：results/literature_revision/scale_evidence.csv、q1_nested_choices.json、q1_nested_metrics.csv。']
        if r['id'] in ['U1','L2','L3']:ledger+=['**验证证据**：q2_conditional_importance.csv、q2_top3.csv、q2_selector_comparison.csv；频率不能冒充因果概率或FDR保证。']
        if r['id'] in ['U1','U2','L4','L5']:ledger+=['**验证证据**：q3_sampler_diagnostics.csv、q3_heldout_distribution.csv、q3_nsga2_summary.json、q3_decisions.csv；均在results/literature_revision/。']
    ledger += ['## 与原版比较','原版输出保留，新结果另存。第一问使用同划分固定随机森林对照；第二问保留原三证据名单并报告新诊断选择器的Jaccard；第三问复用原cVAE，只把重新运行的采样与NSGA-II标为新结果。',
               'MATLAB未安装，本轮NSGA-II实际运行于Python。未完成真实盲评或试酿，不作生产保证。']
    (OUTDOC/'逐篇文献内容与修改对照.md').write_text('\n\n'.join(ledger),encoding='utf-8')
    print(pdf)


if __name__=='__main__':main()
