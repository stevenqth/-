"""Generate model documentation and paper-improvement advice, NOT a paper."""
from common import *
from optimize import Problem,VARIABLES,OBJECTIVES
import platform,importlib.metadata

def md_table(frame):return frame.to_markdown(index=False,floatfmt='.4f')

def simple_table(headers,rows):
    text=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    text+=['| '+' | '.join(str(v) for v in row)+' |' for row in rows];return '\n'.join(text)

def main():
    proxy=json.loads((OUT/'surrogate_summary.json').read_text(encoding='utf-8'));summary=pd.read_csv(OUT/'optimization_summary.csv');rec=pd.read_csv(OUT/'all_recommendations.csv')
    external=json.loads((OUT/'external_audit.json').read_text(encoding='utf-8'));checks=json.loads((OUT/'validation_summary.json').read_text(encoding='utf-8'))
    primary='Pils_Lager_beer218_to_Tripel';mainrec=rec[(rec['case']==primary)&(rec.scenario=='balanced')].set_index('variant')
    maincase=summary[summary['case']==primary]
    full=mainrec.loc['four_objective'];no=mainrec.loc['no_external'];compact=mainrec.loc['shared_defect_knob']
    gain=1-full.heldout_external_error/no.heldout_external_error
    expertgain=proxy['baseline']['expert_mae']-proxy['optimized']['expert_mae']
    # Re-score the four original Yihao grid recommendations under the common NEW model.
    p=Problem(218,'Tripel');original=pd.read_csv(ROOT/'baseline/reference_grid/decisions_grid.csv')
    u=np.c_[original.alpha,np.repeat(original.beta.to_numpy()[:,None],4,axis=1),original.gamma]
    f,cv,d=p.evaluate(u,True)
    comparison=pd.concat([original[['scenario']].reset_index(drop=True),pd.DataFrame(f,columns=OBJECTIVES)],axis=1)
    comparison['new_guard_violation']=cv;comparison['heldout_external_error']=d['holdout_external_error'];comparison.to_csv(OUT/'yihao_original_recommendations_rescored.csv',index=False)
    matched=simple_table(['代理','专业感官OOF R²','专业感官MAE','消费者描述OOF R²','陈旧酒花OOF R²'],[[name,f'{proxy[name]["expert_oof_r2"]:.4f}',f'{proxy[name]["expert_mae"]:.4f}',f'{proxy[name]["consumer_oof_r2"]:.4f}',f'{proxy[name]["stale_oof_r2"]:.4f}'] for name in ['baseline','optimized']])
    choices=simple_table(['主案例均衡推荐','感官偏差↓','调整幅度↓','缺陷负荷↓','优化集外部差距↓','留出外部差距↓'],[[name]+[f'{mainrec.loc[name,c]:.5f}' for c in OBJECTIVES+['heldout_external_error']] for name in mainrec.index])
    percase=rec[(rec.scenario=='balanced')&rec.variant.isin(['four_objective','shared_defect_knob'])].copy()
    case_table=simple_table(['案例','参数族','感官偏差','调整幅度','留出外部差距'],[[r['case'],r['variant'],f'{r["expert_error"]:.4f}',f'{r["adjustment"]:.4f}',f'{r["heldout_external_error"]:.4f}'] for r in percase.to_dict('records')])
    hv=simple_table(['主案例方法','近似前沿点数','共同归一化盒内MC超体积↑'],[[r.variant,int(r.pareto_points),f'{r.common_box_mc_hypervolume:.4f}'] for r in maincase.itertuples()])
    ci=proxy['expert_mae_gain_style_bootstrap_ci']
    doc=f'''# 外部数据增强多目标模型说明与实际结果

## 1 结论与当前可用版本

本模型继承铱豪的缺陷投影、感官代理、Pareto筛选和TOPSIS决策。新增外部消费者画像目标，并在Python中实际运行NSGA-II。推荐优先审查“三参数候选族加第四个外部目标”的简洁版本；六参数扩展保留为实验选项，当前没有稳定优于简洁版本的证据。

代理预测有小幅改善，但陈旧酒花预测依然弱。所有推荐属于统计模型中的候选化学画像，没有酿造或真实盲评验证。

## 2 新数据怎样真正进入计算

- 新CSV共有{external['rows']}条记录；按酒厂分组得到{external['optimization_rows']}条优化用数据、{external['heldout_rows']}条留出数据，酒厂集合不重叠。
- Excel解析出{external['vocabulary_entries']}条词典条目，匹配原附件7.csv的{external['mapped_original_columns']}个描述列，形成10维聚合消费者目标。其列之间存在词典重叠，不宣称独立。
- 训练231维化学输入到50维专业感官及10维消费者目标的代理；使用同一模型将候选转成预测消费者画像。
- 外部词频先除以11维总词频，再使用优化集参考经验分布求百分位。{external['zero_word_count_rows']}条零词频记录不参与画像；不伪造其风味。
- 第四目标为“预测消费者画像百分位与目标风格外部画像百分位的加权均方差”。维度权重依据专业数据嵌套OOF中消费者目标的可预测性设置，最低可靠性权重为0.05，再归一化。
- 16类风格可建立显式对应；本次只运行Tripel、Saison和Dubbel目标。留出画像仅在推荐确定后用于检查，没有参与NSGA-II目标和TOPSIS选择。

## 3 代理模型的匹配验证结果

{matched}

两行使用相同外层三折划分。每个外层训练集内另作分层验证选择各目标的模型，内层折数依最小风格样本数调整为2，最终全数据拟合使用3折。所有插补、1%至99%截尾、特征和目标标准化均在相应训练折拟合。候选为Ridge正则1000、Extra Trees叶节点3/8、随机森林叶节点5。

专业MAE下降{expertgain:.5f}，按风格成组重采样的95%区间为[{ci[0]:.5f}, {ci[1]:.5f}]。这描述单次嵌套划分结果，不是大量独立外部实验。陈旧酒花R²只有{proxy['optimized']['stale_oof_r2']:.4f}，无法据此保证真实陈旧味不恶化。

## 4 优化结构

保留原三个目标：目标专业感官画像偏差、标准化化学改变量、相对缺陷负荷；新增外部画像差距。目标均最小化，不提前压成一个加权总分。

三参数版本为alpha迁移强度、beta共用缺陷削减强度、gamma代理相关化合物调节强度。六参数版本仅把beta拆成四种缺陷各自的参数，其余相同。不同参数族都输出六列参数表示，简洁版本四个beta数值相等。

硬约束包括四种缺陷不高于0.95倍源浓度及可用参考阈值、三个低内层误差代理均预测陈旧酒花不增加、候选到真实训练样本的标准化最近邻距离不超过训练样本留一距离95%分位数。支持约束只是经验保护，不证明物理可实现性。三个代理一致也是模型敏感性保护，不是校准置信上界。

缺陷DMS和4VG按log10还原；二乙酰和反式2壬烯醛保留原浓度。保留附件混合尺度，其他列不强制统一取对数。源样品缺失浓度会标为插补约束，不能声称相对于未知真实值严格不恶化。

NSGA-II使用可行性优先、非支配排序、拥挤距离、边界SBX交叉及多项式变异，种群64、50代、种子42/43。ZDT1解析前沿检验通过。所有输出经过最终候选重算。MATLAB原文件保留，但本次数值来自Python，不是MATLAB gamultiobj。

## 5 主案例对照

beer218 Pils/Lager→Tripel；各推荐按各自前沿中的预设均衡权重选择，TOPSIS归一化范围依赖前沿，因此不是严格控制所有其他因素的因果对照。

{choices}

四目标六参数版本相对无外部目标版本，均衡推荐的留出画像差距下降约{gain*100:.1f}%；同时调整幅度由{no.adjustment:.4f}变为{full.adjustment:.4f}。这说明外部一致性和少改动之间存在取舍，不能只展示外部误差下降。

三参数四目标版本的留出差距为{compact.heldout_external_error:.5f}，六参数版本为{full.heldout_external_error:.5f}。结合下表，增加参数并未取得明确搜索优势；因此保留简洁版本作为优先方案。该判断主要基于优化集目标及复杂度，留出差距只作一次事后核查，不继续根据它调参。

{hv}

超体积使用全部方法共用的积分盒与12000个固定蒙特卡洛点；最大二项标准误约0.0046。小差异不足以断言谁稳定更优。等预算随机搜索未使用NSGA-II，但其非支配候选同样保留，表明当前低维问题不应预设遗传算法一定优于随机搜索。

铱豪原始四种推荐已在新模型上重新打分，见results/yihao_original_recommendations_rescored.csv。原始Ridge目标值与新代理目标值不直接比较。重新评分只作诊断，不能把原推荐事后淘汰当作真实品评验证。

## 6 多案例结果

{case_table}

这些源样品均来自已有250款啤酒，只扩展了案例覆盖，不能称为独立化学测试集。每例两种种子仅检验有限搜索重复性，不足以证明全局Pareto最优。

## 7 已验证结论 假设与待验证问题

已验证结论：新数据与词典确实进入训练目标和第四优化目标；外部酒厂划分互斥；代理经过嵌套验证；实际运行Python NSGA-II；{checks['checks']}项检查通过，独立重算{checks['rechecked_front_points']}个输出前沿点的目标和约束。

假设：跨评论平台的风格及词义可作百分位趋势对应；画像插值能用于候选探索；经验邻域可减轻外推；TOPSIS演示权重有比较意义。

待验证问题：真实0至6品评、独立化学感官批次、缺陷阈值的基质和风格适用性、工艺成本与物料守恒、代理被优化器利用的风险。消费者评论高评分选择偏差和词频分母缺失也未消除。
'''
    (ROOT/'模型说明与结果.md').write_text(doc,encoding='utf-8')
    advice=f'''# 基于本模型的论文改进意见

本文件是改进建议与实验依据，不是论文正文。建议在段睿三问主线的第三问中增加铱豪的多目标决策扩展，并使用本次数据增强实验更新证据。

## 建议一 把第三问写成生成与决策两个层次

原cVAE负责候选生成；Pareto优化负责在感官、改动、缺陷和消费者风味之间取舍。本次求解的是铱豪的参数化画像族，没有把cVAE权重接入求解器。正文应明确“现阶段是画像族上的优化”，未来才能比较VAE潜变量候选族，不可写成已完成VAE与NSGA-II联合训练。

## 建议二 增加外部数据接入机制图和可审计映射表

建议图示：5558条啤酒评论汇总→按酒厂拆分→词频份额与优化集百分位→目标风格画像；Excel词典→附件7的描述词归类→化学到消费者画像代理；最终在第四目标汇合。附录放dictionary_mapping.csv和external_profiles.csv。

需解释10维消费者聚合指标来自已标准化的原描述符，外部词频是另一尺度；百分位只能建立语义趋势，不是精确分值校准。不要把外部5558条数据说成新增5558条化学-品评配对样本。

## 建议三 用匹配验证协议替换跨版本数字比较

建议主表引用results/surrogate_summary.json：基线与新模型专业OOF R²分别{proxy['baseline']['expert_oof_r2']:.4f}和{proxy['optimized']['expert_oof_r2']:.4f}，消费者目标分别{proxy['baseline']['consumer_oof_r2']:.4f}和{proxy['optimized']['consumer_oof_r2']:.4f}。对照共用外层划分，模型选择在内层进行。不要把铱豪旧5折0.20538、段睿旧15折0.2125及本版0.2084排列成单调进步曲线。

说明MAE改善很小，给出风格成组重采样区间[{ci[0]:.5f}, {ci[1]:.5f}]，保留单次划分与小样本局限。避免写“显著大幅提升”。

## 建议四 正式定义四个目标和三类约束

三个原目标加外部画像误差，分别列出单位、归一化方式和优化方向。把缺陷浓度、陈旧酒花多模型一致性、真实样本支持距离写成独立约束，并在附录提供重算接口Problem.evaluate。

不得把标准化化学调整幅度称为真实成本；不得把相对缺陷负荷称为毒理风险；不得把多模型一致称为95%置信保证。4VG具有风格依赖，阈值来自附件参考而非法定限制。

## 建议五 诚实保留简洁模型与随机搜索对照

建议展示四目标六参数、无外部目标、四目标三参数、等预算随机搜索四行。当前六参数没有明确优势，应保留三参数的可解释性作为主要论点，六参数作为消融实验。随机搜索结果不能省略，也不能声称NSGA-II必然更优。

建议用figures/pareto_tradeoffs.png展示目标取舍，用figures/ablation_comparison.png展示均衡推荐的完整四目标。当前主案例外部留出差距下降约{gain*100:.1f}%，但调整幅度也上升，应两者同时呈现。这个百分比仅属于该案例，不是全体风格的平均提升。

## 建议六 明确优化集和外部留出的使用次序

推荐先按优化集目标与预设偏好确定，再计算留出差距。禁止从多个推荐中挑留出效果最好者并当作预先选择的结果；禁止将本次查看留出后的设计取舍宣称为全新无偏验证。后续改模型若继续参考此留出集，需要另设新外部测试数据。

results/optimization_summary.csv中的best_heldout_external_error是事后前沿描述，正文不应把它作为正式推荐性能。正式推荐应引用all_recommendations.csv的固定scenario行。

## 建议七 用多案例支持适用范围 不夸大为泛化

目前共5个源案例、3个迁移方向，其中两种参数族均有结果。案例都来自训练数据，外部独立性仅存在于评论酒厂划分。建议正文选主案例解释机制，其余放稳健性附表；需独立化学-感官配对测试才能谈跨批次泛化。

## 建议八 把陈旧酒花弱代理列为关键局限

当前陈旧酒花OOF R²约{proxy['optimized']['stale_oof_r2']:.3f}，并未因新模型改善。优化器能保证的是这些代理的数值不增加，真实品评仍未知。控制化合物的方向来自模型相关，不能直接给出“改变某化合物就改善陈旧酒花”的因果结论。

## 建议九 删除或限定旧版过强表述

- “已得到最优酿造配方”改为“得到参数化候选族中的近似Pareto画像”。
- “缺陷完全不恶化”限定为已测浓度及指定代理约束，注明插补源浓度情况。
- “高训练重构R²证明生成有效”改为分别报告留出重构、生成分布和真实品评证据。
- “MATLAB NSGA-II已验证”改为“本次实际执行Python NSGA-II；原MATLAB实现仍待单独验证”。
- “外部数据证明化学谱真实”改为“外部消费者描述提供语义趋势对照”。

## 最优先补充材料

1. 原始逐品评员0至6评分及标准化参数，用于真正校准评分。
2. 独立批次的化学浓度与感官配对数据，用于验证代理和优化推荐。
3. 完整单位、变换、检出限及缺失编码，尤其原表与校准表不一致的列。
4. 逐条评论、评论数量和文本长度，以及未按高分筛选的样本，改善消费者画像偏差。
5. 真实配方、工艺条件和控制变量，以及盲评或加标实验，使统计画像逐步转为可实施建议。

## 当前适合写入论文的结论

可以写：外部消费者描述已作为第四目标参与求解；在小样本条件下采用嵌套验证与模型一致性保护；单案例展示外部一致性与改动量的取舍；多参数扩展和遗传算法并不自动带来优势。

不应写：已证明优于所有原模型、已得到真实生产最优配方、已验证所有风格迁移或真实缺陷不恶化。
'''
    (ROOT/'基于模型的论文改进意见.md').write_text(advice,encoding='utf-8')
    # Diagnostic scientific plots, exported as standalone PNG assets.
    plt=plotter();r=OUT/primary
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for variant,color in [('four_objective','#26768c'),('shared_defect_knob','#ce8b3f'),('no_external','#777777')]:
        a=pd.read_csv(r/f'{variant}_pareto.csv');axes[0].scatter(a.adjustment,a.expert_error,s=12,alpha=.65,label=variant,color=color);axes[1].scatter(a.external_error,a.expert_error,s=12,alpha=.65,label=variant,color=color)
    axes[0].set(xlabel='化学调整幅度',ylabel='专业感官偏差');axes[1].set(xlabel='优化集外部画像差距',ylabel='专业感官偏差');axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(FIG/'pareto_tradeoffs.png');plt.close(fig)
    f=mainrec[OBJECTIVES];norm=f/f.max(axis=0);fig,ax=plt.subplots(figsize=(10,4));norm.plot.bar(ax=ax);ax.set_ylabel('相对各列最大值  越低越好');ax.tick_params(axis='x',rotation=15);ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.01,1));fig.tight_layout();fig.savefig(FIG/'ablation_comparison.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));row=mainrec.loc['shared_defect_knob'];u=np.array([[row[c] for c in VARIABLES]]);_,_,det=p.evaluate(u,True)
    frame=pd.DataFrame({'推荐代理':det['consumer_percentile'][0],'外部优化集':p.external_target,'外部留出':p.external_holdout},index=p.dimensions);frame.to_csv(OUT/'primary_recommended_external_profile.csv');frame.plot.bar(ax=ax);ax.set_ylim(0,1);ax.set_ylabel('各域经验百分位');ax.tick_params(axis='x',rotation=25);fig.tight_layout();fig.savefig(FIG/'external_profile.png');plt.close(fig)
    env={'python':platform.python_version(),'packages':{m:importlib.metadata.version(m) for m in ['numpy','pandas','scipy','scikit-learn','matplotlib','openpyxl','joblib']}}
    save(env,'environment.json');print('Generated model report, paper improvement advice, and 3 charts. No paper created.')

if __name__=='__main__':main()
