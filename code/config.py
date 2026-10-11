# -*- coding: utf-8 -*-
"""
全局配置：路径、语料参数、词向量参数、测度参数、实证参数
所有脚本统一从 config 读取，保证实验可复现。
"""
from pathlib import Path
import random

# ---------------- 路径 ----------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
LEXICON = ROOT / "data" / "lexicon"
OUT_TAB = ROOT / "output" / "tables"
OUT_FIG = ROOT / "output" / "figures"
DOCS = ROOT / "docs"

for p in [DATA_RAW, DATA_PROC, LEXICON, OUT_TAB, OUT_FIG, DOCS]:
    p.mkdir(parents=True, exist_ok=True)

SEED = 20240517
random.seed(SEED)

# ---------------- 面板设定 ----------------
N_FIRMS = 300          # 企业数
YEAR_START, YEAR_END = 2012, 2023
INDUSTRIES = {
    "C1": "制造业-通用设备", "C2": "制造业-电子设备", "C3": "制造业-汽车",
    "C4": "制造业-医药", "C5": "制造业-食品", "D": "电力热力燃气",
    "E": "建筑业", "F": "批发零售", "G": "交通运输", "I": "信息技术服务",
}
PROVINCES = ["北京", "上海", "广东", "江苏", "浙江", "山东", "四川", "湖北", "安徽", "河南"]

# ---------------- 语料参数 ----------------
MIN_DOC_CHARS = 700     # 每篇 MD&A 最小字数
MAX_DOC_CHARS = 2600    # 每篇 MD&A 最大字数

# ---------------- 词向量参数 ----------------
W2V = dict(
    vector_size=200,      # 维度
    window=5,             # 窗口
    min_count=5,          # 最小词频
    sg=1,                 # 1=Skip-gram, 0=CBOW
    negative=10,          # 负采样个数
    workers=4,            # 线程
    epochs=8,             # 迭代轮次
    seed=SEED,
)

# 相似词扩展参数
EXPAND = dict(
    topn=25,              # 每个种子词取多少相似词
    sim_threshold=0.55,   # 余弦相似度阈值
    min_len=2,            # 词长过滤（去除单字噪声）
)

# ---------------- 数字化转型种子词（三层） ----------------
# 第一层：底层数字技术（Technology）
SEED_TECH = ["人工智能", "大数据", "云计算", "区块链", "物联网", "机器学习",
             "深度学习", "自然语言处理", "算力", "边缘计算"]
# 第二层：数字化应用场景（Application）
SEED_APPLY = ["智能制造", "工业互联网", "数字孪生", "智能工厂", "智慧物流",
              "精准营销", "线上化", "供应链协同", "远程运维", "数字营销"]
# 第三层：组织与商业模式（Organization / Business model）
SEED_ORG = ["数字化转型", "数字化战略", "数据驱动", "平台化", "数字化运营",
            "智能化升级", "数字生态", "数字化人才"]

SEED_DICT = {
    "tech": SEED_TECH,
    "apply": SEED_APPLY,
    "org": SEED_ORG,
}

# ---------------- 停用词 ----------------
STOPWORDS = set("""
的 了 和 是 在 我 有 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有
看 好 自己 这 那 个 我们 他们 它 其 之 与 及 或 等 为 以 对 于 中 下 上 由 被 把
并 但 而 且 若 如 使 从 向 到 过 还 再 又 更 最 已 将 可 能 会 应 需 应 该 等
公司 本公司 集团 企业 报告 报告期 年度 期内 期内 本期 上期 同比 万元 元 人民币
继续 不断 进一步 积极 持续 加强 推进 提升 提高 实现 完成 开展 进行 通过 基于
截至 其中 主要 相关 各项 方面 工作 情况 发展 增长 下降 增加 减少 公司及 以及
一 二 三 四 五 六 七 八 九 十 百 千 万 亿 第 个 家 项 年 月 日
""".split())

# ---------------- 实证参数 ----------------
CONTROLS = ["size", "lev", "roa", "age", "cash", "tangibility", "top1", "rd_ratio"]
