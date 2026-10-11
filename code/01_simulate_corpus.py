# -*- coding: utf-8 -*-
"""
步骤 1：语料与面板数据的构造
-------------------------------------------------
真实研究中，本步骤对应：
  (a) 从巨潮资讯网 / CSMAR / Wind 下载上市公司年度报告 PDF；
  (b) 用版面分析抽取"管理层讨论与分析（MD&A）"章节（pdfplumber / PyMuPDF）；
  (c) 与财务面板按 股票代码-年度 匹配，剔除金融业、ST 与关键变量缺失样本。

本仓库为教学演示，使用**参数化模拟器**生成一份"已知数据生成过程
（Data Generating Process, DGP）"的语料：每家企业被赋予一个不可观测的
数字化转型强度 dt_latent，它同时决定 (1) 年报 MD&A 中数字化术语的密度、
(2) 企业的创新产出。这样做的好处是：我们可以检验 NLP 测度能否把
dt_latent 还原出来，以及还原出的测度在回归中能否给出与真实设定一致的系数。

替换为真实数据时，只需保证产出字段一致，下游脚本无需改动：
  data/raw/panel_raw.csv   : firm_id, year, industry, province, soe, size,
                             lev, roa, cash, tangibility, top1, rd_ratio, age,
                             patents, digital_invest, software_copyright,
                             analyst, cost_ratio, opcost, dt_latent
  data/raw/mda_corpus.jsonl: {"firm_id":.., "year":.., "industry":.., "text":".."}
"""
import json
import numpy as np
import pandas as pd
from config import (DATA_RAW, N_FIRMS, YEAR_START, YEAR_END,
                    INDUSTRIES, PROVINCES, MIN_DOC_CHARS, MAX_DOC_CHARS, SEED)

rng = np.random.default_rng(SEED)

# ============ 1. 词池 ============
# (A) 数字技术词池：前 10 个为「种子词」，其余为「待自动发现的候选词」
POOL_TECH = [
    "人工智能", "大数据", "云计算", "区块链", "物联网", "机器学习",
    "深度学习", "自然语言处理", "算力", "边缘计算",
    # —— 候选扩展词（不进种子表，期望由 Word2Vec 扩展自动召回）——
    "计算机视觉", "智能语音", "知识图谱", "大模型", "算法模型", "智能算法",
    "分布式存储", "公有云", "私有云", "混合云", "云原生", "容器化",
    "智能合约", "共识机制", "数字人民币", "传感器", "射频识别", "5G网络",
    "联邦学习", "迁移学习", "强化学习", "语音识别", "图像识别", "文本挖掘",
    "数据湖", "数据仓", "算力中心", "超算", "隐私计算", "图计算",
]
# (B) 数字化应用场景词池
POOL_APPLY = [
    "智能制造", "工业互联网", "数字孪生", "智能工厂", "智慧物流",
    "精准营销", "线上化", "供应链协同", "远程运维", "数字营销",
    # —— 候选扩展词 ——
    "柔性制造", "无人工厂", "自动化产线", "工业机器人", "预测性维护",
    "可视化大屏", "数据中台", "业务中台", "用户画像", "标签体系",
    "私域流量", "全渠道", "直播电商", "智慧仓储", "智能客服", "智慧园区",
    "数字工厂", "智能排产", "质量追溯", "智能质检", "无人配送", "自动分拣",
    "能耗监测", "智能电网", "在线监测", "远程诊断", "协同设计", "虚拟仿真",
    "智能投顾", "智慧门店", "电子签章", "无纸化",
]
# (C) 组织与商业模式词池
POOL_ORG = [
    "数字化转型", "数字化战略", "数据驱动", "平台化", "数字化运营",
    "智能化升级", "数字生态", "数字化人才",
    # —— 候选扩展词 ——
    "数字化组织", "敏捷迭代", "中台战略", "数字治理", "数据要素",
    "数据资产化", "流程再造", "组织扁平化", "数字化考核", "生态协同",
    "数字文化", "业财一体", "数字化赋能", "数字竞争力",
]
# (D) 泛化干扰词：既出现在数字化语境、也大量出现在普通经营描述中，
#     用于检验「领域特异性过滤（keyness）」能否把它们挡在词典之外
GENERIC_TECHY = ["信息化", "自动化", "互联网", "技术创新", "技术改造",
                 "系统建设", "平台建设", "网络建设", "信息技术", "管理系统"]

SCENES = ["生产制造", "供应链管理", "客户服务", "产品研发", "质量控制", "仓储配送",
          "门店运营", "财务核算", "风险管控", "能源管理", "设备运维", "营销推广",
          "采购管理", "订单处理", "售后服务", "物流调度", "排产计划", "安全管理"]
EFFECTS = ["效率显著提升", "成本稳步下降", "良品率提高", "交付周期缩短",
           "客户满意度改善", "库存周转加快", "能耗明显降低", "决策效率提升",
           "差错率下降", "响应速度加快", "人均产值提升", "资金占用减少",
           "协同效率改善", "数据口径统一", "管理颗粒度更细", "运营透明度提高"]
PLATFORMS = ["平台", "系统", "中台", "底座", "体系", "解决方案", "架构", "矩阵", "能力中心"]
UNITS = ["生产部", "供应链中心", "营销中心", "财务部", "研究院", "客户服务部",
         "子公司", "事业部", "工厂车间", "区域公司"]

# ============ 2. 通用（非数字化）经营描述句库 ============
# 槽位：{ind} 行业词 / {prod} 产品词 / {reg} 区域 / {gt} 泛化技术词 / {u} 单位
GENERIC = [
    "报告期内，公司紧紧围绕年度经营目标，稳步推进{ind}业务，主营业务收入实现平稳增长。",
    "受宏观经济环境及行业周期影响，公司{prod}产品毛利率出现波动，公司通过优化产品结构加以应对。",
    "公司持续加强应收账款管理，强化信用风险控制，期末应收账款周转率较上年有所改善。",
    "报告期内公司完成对若干子公司的整合，组织架构进一步优化，管理效率得到提升。",
    "公司坚持稳健经营原则，合理安排长短期债务结构，资产负债率保持在合理区间。",
    "面对原材料价格波动，公司通过集中采购与期货套期保值等方式平抑成本压力。",
    "公司进一步完善内部控制体系，修订了采购、销售及资金管理相关制度。",
    "报告期内公司加大市场开拓力度，在{reg}区域新增多家经销商与服务商。",
    "公司重视安全生产与环境保护，全年未发生重大安全与环保事故。",
    "公司持续优化人才结构，完善绩效考核与激励机制，核心技术团队保持稳定。",
    "受汇率波动影响，公司汇兑损益有所变化，已通过外汇衍生工具进行部分对冲。",
    "公司本年度完成一项重大资产收购，相关标的已完成工商变更登记手续。",
    "报告期内公司研发投入持续增加，多个{prod}新产品进入试制或小批量生产阶段。",
    "公司积极响应国家产业政策导向，推进绿色低碳生产方式改造与节能技改。",
    "现金流方面，公司经营活动产生的现金流量净额同比改善，资金状况总体稳健。",
    "公司进一步完善法人治理结构，股东大会、董事会、监事会规范运作。",
    "报告期内行业竞争加剧，公司通过调整定价策略与加强成本控制维持市场份额。",
    "公司持续推进精益管理，在{u}开展了多轮降本增效专项工作。",
    "本年度公司获得多项政府补助与税收优惠，对当期损益产生一定影响。",
    "公司与客户、供应商保持长期稳定合作关系，前五大客户结构总体平稳。",
    "受下游需求结构调整影响，公司部分{prod}传统产品订单量有所回落。",
    "公司加强存货管理，对长库龄物料进行专项清理，存货跌价准备计提充分。",
    "报告期内公司完成新一轮股权激励计划的授予登记工作，覆盖核心管理与技术人员。",
    "公司注重品牌与渠道建设，积极参加行业展会并开展多种形式的推广活动。",
    "公司在境外市场业务稳步推进，出口收入占比保持在合理水平。",
    "本年度公司未发生重大诉讼、仲裁事项，亦未发生控股股东资金占用情形。",
    "公司严格执行信息披露制度，确保披露内容真实、准确、完整、及时。",
    "面对用工成本上升，公司通过优化排班与岗位合并控制人工成本。",
    "报告期内公司若干募投项目按计划推进，{prod}相关产能已陆续释放。",
    "公司将继续聚焦主业，坚持创新驱动，努力提升经营质量与盈利能力。",
    "公司{gt}工作按计划推进，{u}相关{gt}项目完成验收并投入使用。",
    "报告期内，公司持续推进{gt}建设，为后续业务扩张奠定基础。",
    "公司依托{gt}手段优化{ind}业务流程，管理规范化水平进一步提高。",
    "本年度公司在{gt}方面投入保持稳定，相关投入主要用于现有系统维护。",
    "公司组织{u}开展{gt}培训，员工操作规范性有所提升。",
    "报告期内公司{gt}项目通过外部审计与验收，相关支出已按会计准则资本化。",
    "公司结合{ind}行业特点，稳步实施{gt}改造，生产连续性得到保障。",
    "受行业景气度影响，公司{prod}板块收入增速放缓，公司已相应调整排产节奏。",
    "公司加强合同管理，对重大合同实行分级审批与履约跟踪。",
    "报告期内公司完成对{u}的绩效考核体系调整，考核结果与薪酬挂钩更加紧密。",
    "公司参与多项行业标准制定，行业影响力有所提升。",
    "本年度公司计提商誉减值准备，对部分收购标的的业绩承诺完成情况进行了复核。",
    "公司优化融资结构，置换部分高成本借款，财务费用同比下降。",
    "报告期内公司新增授权专利若干项，产品竞争力进一步增强。",
    "公司在{reg}新建的生产基地进入试运行阶段，产能爬坡进展顺利。",
    "公司针对{prod}产品开展质量专项提升行动，客户投诉率有所下降。",
    "报告期内公司强化资金集中管理，资金使用效率提高。",
    "公司持续关注行业政策变化，及时调整经营策略以应对外部环境不确定性。",
    "本年度公司完成董事会换届选举，聘任了新的高级管理人员。",
    "公司加强知识产权保护，完成多项商标注册与专利续费工作。",
    "报告期内公司与多家{p}保持良好合作，渠道结构进一步多元化。",
    "公司针对{p}建立分级管理机制，对重点客户实行专人对接。",
    "报告期内公司对{f}进行专项梳理，相关会计处理符合企业会计准则要求。",
    "公司{f}规模较年初有所变动，主要系本期经营规模扩张所致。",
    "公司加强{f}管理，建立了月度滚动预测与预警机制。",
    "报告期内公司推进{h}工作，员工满意度与稳定性保持在较好水平。",
    "公司完善{h}制度，将业绩指标层层分解至各业务单元。",
    "公司组织多轮{h}活动，累计参训人次较上年增加。",
    "面对{r}，公司制定了专项应对预案并定期评估执行效果。",
    "公司识别出{r}对经营的可能影响，并在投资决策中加以考虑。",
    "报告期内公司未发生因{r}导致的重大损失事项。",
    "公司{r}管理体系通过外部机构认证，相关流程文件同步更新。",
    "公司{p}回款情况总体良好，账龄结构较上年改善。",
    "报告期内公司对部分{p}的授信额度进行了动态调整。",
    "公司{f}周转天数同比缩短，营运资金占用有所下降。",
    "公司{h}改革方案经职工代表大会审议通过后实施。",
    "公司针对{r}购买了相应保险，风险敞口总体可控。",
    "报告期内公司{p}数量增加，市场覆盖率进一步提高。",
    "公司按季度对{f}进行减值测试，测试结果已履行审议程序。",
    "公司{h}投入持续增加，人均培训时长较上年提高。",
    "公司与{p}签订年度框架协议，约定价格调整与结算方式。",
    "报告期内公司{f}余额变动较大，主要受季节性因素与项目进度影响。",
    "公司{h}体系与职业晋升通道实现衔接，员工流失率有所下降。",
    "公司建立{r}定期排查机制，报告期内完成两轮全覆盖排查。",
    "公司{p}结构优化，单一客户依赖度有所下降。",
    "报告期内公司{f}相关内控制度执行有效，未发现重大缺陷。",
    "公司{h}工作与年度经营目标挂钩，考核结果应用于薪酬分配。",
    "公司针对{r}加强信息披露，及时提示投资者注意相关不确定性。",
    "报告期内公司{p}投诉率下降，售后服务响应时效明显改善。",
    "公司{f}规模与经营规模基本匹配，未出现异常波动。",
    "公司{h}方案兼顾短期激励与长期约束，有利于稳定骨干队伍。",
    "公司{r}应急预案完成演练，相关责任部门职责清晰。",
    "报告期内公司{p}结算方式调整，票据结算占比有所上升。",
    "公司{f}管理纳入预算考核，超预算支出需履行追加审批程序。",
    "公司{h}档案实现电子化管理，查询与统计效率提高。",
    "公司{r}敞口通过衍生工具进行部分对冲，剩余敞口处于可接受范围。",
    "报告期内公司{p}满意度调查结果显示整体评价稳中有升。",
    "公司{f}相关指标处于行业合理水平，未触发债务契约条款。",
    "公司{h}预算执行率较高，培训计划基本按进度完成。",
    "公司{r}相关制度完成修订，明确了报告路径与处置时限。",
    "报告期内公司{p}合作年限普遍较长，合作关系较为稳定。",
    "公司{f}结构明细已在财务报表附注中充分披露。",
    "公司{h}工作获评行业优秀实践案例，经验已在集团内部推广。",
]

PARTNERS = ["经销商", "代理商", "直营门店", "电商平台", "海外客户", "战略客户",
            "核心供应商", "外包服务商", "工程承包商", "终端用户"]
FINTERMS = ["应收账款", "存货", "商誉", "固定资产", "在建工程", "短期借款",
            "应付票据", "经营性现金流", "预付账款", "合同资产"]
HRTERMS = ["校园招聘", "绩效考核", "股权激励", "岗位竞聘", "技能培训", "人才梯队"]
RISKS = ["汇率风险", "利率风险", "信用风险", "合规风险", "安全生产风险",
         "地缘政治因素", "原材料价格波动", "客户集中度风险"]

IND_WORDS = {
    "C1": ["通用设备", "装备制造"], "C2": ["电子元器件", "消费电子"],
    "C3": ["汽车零部件", "整车制造"], "C4": ["医药制造", "生物制品"],
    "C5": ["食品加工", "快消品"], "D": ["电力生产", "热力供应"],
    "E": ["工程建设", "施工总承包"], "F": ["批发零售", "商贸流通"],
    "G": ["交通运输", "物流服务"], "I": ["软件开发", "信息技术服务"],
}
PROD_WORDS = ["核心", "主力", "传统", "新一代", "高附加值", "定制化", "标准化", "出口型"]
REGIONS = ["华东", "华南", "华北", "西南", "华中", "东北", "西北", "海外"]

DIGI_TEMPLATES = [
    "公司持续推进{tech}技术在{scene}中的应用，相关{plat}建设取得阶段性成果，{effect}。",
    "报告期内，公司以{tech}为核心构建{scene}{plat}，推动业务流程重构，{effect}。",
    "公司依托{tech}能力，对{scene}环节实施数字化改造，{effect}。",
    "本年度公司加大{tech}投入，完成{scene}场景下{plat}的迭代升级，{effect}。",
    "公司基于{tech}开展{apply}实践，在{scene}领域形成可复制的解决方案。",
    "公司深化{apply}建设，将{tech}能力嵌入{scene}全流程，{effect}。",
    "围绕{apply}目标，公司搭建了覆盖{scene}的{plat}，实现数据贯通与业务协同。",
    "公司稳步推进{org}，明确以{tech}为支撑、以{apply}为抓手的实施路径。",
    "报告期内公司将{org}列入年度重点工作，成立专项工作组统筹推进{apply}落地。",
    "公司坚持{org}总体方向，推动{scene}环节向数据驱动型模式转变，{effect}。",
    "公司认为{tech}是{org}的关键底座，报告期内相关投入占信息化投入较大比重。",
    "公司通过{apply}与{org}协同推进，带动{scene}环节的组织方式与考核方式调整。",
    "公司在{u}试点部署{tech}应用，{scene}环节的{effect}初步显现。",
    "报告期内公司引入{tech}服务商，联合开展{apply}场景的方案设计与验证。",
    "公司围绕{org}目标完善数据治理机制，{scene}数据的采集口径逐步统一。",
    "公司依托{apply}{plat}，实现{scene}全过程在线留痕，{effect}。",
    "本年度公司围绕{tech}开展专项培训，{u}员工数字技能水平有所提升。",
    "公司将{apply}作为{org}的突破口，在{scene}场景中先行先试后再推广复制。",
]


def fill(tpl: str, industry: str) -> str:
    return tpl.format(
        ind=rng.choice(IND_WORDS[industry]),
        prod=rng.choice(PROD_WORDS), reg=rng.choice(REGIONS),
        gt=rng.choice(GENERIC_TECHY), u=rng.choice(UNITS),
        p=rng.choice(PARTNERS), f=rng.choice(FINTERMS),
        h=rng.choice(HRTERMS), r=rng.choice(RISKS),
        tech=rng.choice(POOL_TECH), apply=rng.choice(POOL_APPLY),
        org=rng.choice(POOL_ORG), scene=rng.choice(SCENES),
        effect=rng.choice(EFFECTS), plat=rng.choice(PLATFORMS),
    )


def make_document(dt_level: float, year: int, industry: str) -> str:
    """拼装一篇 MD&A：数字化句与通用句交错（模拟真实年报中数字化内容散布的特征）"""
    growth = (year - YEAR_START) / (YEAR_END - YEAR_START)
    target = int(rng.integers(MIN_DOC_CHARS * (1 + 0.4 * growth),
                              MAX_DOC_CHARS * (1 + 0.4 * growth)))
    # 数字化句子数量由潜变量决定：先定句数，再按二项抽样决定数字化句占比，
    # 使数字化披露密度与转型强度之间呈稳定的剂量—反应关系
    n_sent = max(8, int(target / 45))
    p_digi = float(np.clip(0.02 + 0.62 * dt_level * (0.55 + 0.6 * growth), 0.0, 0.85))
    n_digi = int(rng.binomial(n_sent, p_digi))
    n_gen = max(3, n_sent - n_digi)
    digi = [fill(rng.choice(DIGI_TEMPLATES), industry) for _ in range(n_digi)]
    gen = [fill(rng.choice(GENERIC), industry) for _ in range(n_gen)]
    sents = digi + gen
    rng.shuffle(sents)          # 打乱顺序，避免数字化内容集中出现在段首
    return "".join(sents)


def main():
    firms = []
    for i in range(1, N_FIRMS + 1):
        firms.append(dict(
            firm_id=f"{100000 + i:06d}",
            industry=rng.choice(list(INDUSTRIES.keys())),
            province=rng.choice(PROVINCES),
            soe=int(rng.random() < 0.35),
            founded=int(rng.integers(1993, 2015)),
            base_size=float(rng.normal(21.5, 1.1)),
            dt_base=float(rng.beta(2.0, 3.0)),      # 企业固有数字化倾向
            dt_speed=float(rng.uniform(0.2, 1.9)),  # 转型推进速度
        ))
    firm_df = pd.DataFrame(firms)
    # 省份×行业层面的数字化集聚效应（政策扩散、示范效应），也是 IV 的相关性来源
    pi_effect = {(pv, ind): float(rng.normal(0, 0.15))
                 for pv in PROVINCES for ind in INDUSTRIES}
    # 同省份-同行业的"转型扩散速度"：随时间推进，是工具变量相关性的来源
    pi_trend = {(pv, ind): float(rng.normal(0, 0.22))
                for pv in PROVINCES for ind in INDUSTRIES}
    ind_trend = {k: float(rng.uniform(-0.08, 0.18)) for k in INDUSTRIES}

    rows, corpus = [], []
    for f in firm_df.to_dict("records"):
        for year in range(YEAR_START, YEAR_END + 1):
            t = (year - YEAR_START) / (YEAR_END - YEAR_START)
            age = max(1, year - f["founded"])
            size = f["base_size"] + 0.25 * t + float(rng.normal(0, 0.10))
            lev = float(np.clip(rng.normal(0.45, 0.16), 0.05, 0.92))
            roa = float(rng.normal(0.045, 0.055))
            cash = float(np.clip(rng.normal(0.16, 0.09), 0.01, 0.60))
            tang = float(np.clip(rng.normal(0.38, 0.15), 0.03, 0.85))
            top1 = float(np.clip(rng.normal(0.35, 0.14), 0.05, 0.80))
            rd_ratio = float(np.clip(rng.normal(0.032, 0.020), 0.0, 0.20))

            # --- 潜变量：数字化转型真实强度（不可观测）---
            dt_latent = float(np.clip(
                f["dt_base"] * 0.45 + f["dt_speed"] * 0.55 * t
                + ind_trend[f["industry"]] * t
                + pi_effect[(f["province"], f["industry"])]
                + pi_trend[(f["province"], f["industry"])] * t
                + float(rng.normal(0, 0.05)), 0, 1))

            # --- 机制变量：先于创新产出生成，使其真正构成中介路径 ---
            opcost = float(np.clip(0.18 - 0.05 * dt_latent + rng.normal(0, 0.03), 0.01, 0.40))
            cost_ratio = float(np.clip(0.72 - 0.10 * dt_latent + rng.normal(0, 0.05), 0.2, 0.95))
            analyst = int(rng.poisson(max(np.exp(1.0 + 0.8 * (size - 21.5)
                                                + 0.9 * dt_latent), 0.1)))
            ln_analyst = float(np.log1p(analyst))

            # --- 创新绩效：专利申请量 ---
            # 总效应 = 直接效应 0.55 + 成本路径 (-2.0)×(-0.05) + 透明度路径 0.25×0.9 ≈ 0.88
            manuf = int(f["industry"].startswith("C"))
            large = int(size > 21.5)
            dt_effect = 0.55 + 0.30 * (1 - f["soe"]) + 0.20 * manuf - 0.18 * large
            lam = np.exp(3.0 + dt_effect * dt_latent - 2.0 * opcost + 0.25 * ln_analyst
                         + 0.28 * (size - 21.5) + 8.0 * rd_ratio + 0.10 * roa
                         + float(rng.normal(0, 0.22)))
            patents = int(rng.poisson(max(lam, 0.05)))

            # --- 效标变量（效标关联效度用）---
            digital_invest = float(np.exp(14.0 + 2.3 * dt_latent
                                          + 0.6 * (size - 21.5) + rng.normal(0, 0.4)))
            software_cr = int(rng.poisson(max(np.exp(0.3 + 2.2 * dt_latent), 0.05)))

            rows.append(dict(
                firm_id=f["firm_id"], year=year, industry=f["industry"],
                province=f["province"], soe=f["soe"], age=age,
                size=round(size, 4), lev=round(lev, 4), roa=round(roa, 4),
                cash=round(cash, 4), tangibility=round(tang, 4), top1=round(top1, 4),
                rd_ratio=round(rd_ratio, 5), patents=patents,
                digital_invest=round(digital_invest, 2),
                software_copyright=software_cr, analyst=analyst,
                cost_ratio=round(cost_ratio, 4), opcost=round(opcost, 4),
                dt_latent=round(dt_latent, 4),
            ))
            corpus.append(dict(firm_id=f["firm_id"], year=year,
                               industry=f["industry"],
                               text=make_document(dt_latent, year, f["industry"])))

    panel = pd.DataFrame(rows)
    panel.to_csv(DATA_RAW / "panel_raw.csv", index=False, encoding="utf-8-sig")
    with open(DATA_RAW / "mda_corpus.jsonl", "w", encoding="utf-8") as fh:
        for c in corpus:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[01] 企业数 {N_FIRMS}，年份 {YEAR_START}-{YEAR_END}，观测 {len(panel)}")
    print(f"[01] 语料篇数 {len(corpus)}，总字数 {sum(len(c['text']) for c in corpus):,}")


if __name__ == "__main__":
    main()
