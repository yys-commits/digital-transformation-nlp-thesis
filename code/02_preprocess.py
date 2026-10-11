# -*- coding: utf-8 -*-
"""
步骤 2：文本清洗、分词与停用词处理
-------------------------------------------------
处理链条（对应论文 4.2 节）：
  全半角统一 -> 剔除图表/页码/数字噪声 -> 正则清洗 -> 领域词典注入
  -> jieba 精确模式分词 -> 停用词与单字过滤 -> 语料落盘
领域词典注入是本步的关键：通用分词器会把"数字孪生""工业互联网"
等术语切碎，导致测度系统性低估，必须先把术语表写入用户词典。
"""
import json
import re
import sys
from collections import Counter

import jieba
import pandas as pd

from config import DATA_RAW, DATA_PROC, STOPWORDS, SEED_DICT
from config import ROOT

# 1) 领域术语注入分词器
#    通用分词器会把"数字孪生""工业互联网""数据湖"等术语切碎，
#    造成测度系统性低估，因此分词前必须把领域术语写入用户词典。
import importlib.util

DOMAIN_TERMS = set()
for layer in SEED_DICT.values():
    DOMAIN_TERMS.update(layer)

# 以文件方式加载 01 的候选词池（保证与语料生成端完全一致，且不触发 main()）
spec = importlib.util.spec_from_file_location(
    "sim_corpus", ROOT / "code" / "01_simulate_corpus.py")
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)
for pool in [sim.POOL_TECH, sim.POOL_APPLY, sim.POOL_ORG, sim.GENERIC_TECHY]:
    DOMAIN_TERMS.update(pool)

EXTRA_TERMS = """
数字化 智能化 信息化 数字技术 数字孪生 管理层讨论与分析 报告期
""".split()
DOMAIN_TERMS.update(EXTRA_TERMS)
for w in DOMAIN_TERMS:
    jieba.add_word(w)

PUNCT = re.compile(r"[^\u4e00-\u9fa5A-Za-z]+")
DIGITS = re.compile(r"\d+")


def clean(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = text.replace("（", "(").replace("）", ")")
    text = DIGITS.sub("", text)               # 去除金额/年份等数字噪声
    text = PUNCT.sub(" ", text)               # 仅保留中英文
    return re.sub(r"\s+", " ", text).strip()


def segment(text: str):
    return [w for w in jieba.lcut(text) if w not in STOPWORDS and len(w) > 1]


def main():
    docs = [json.loads(l) for l in open(DATA_RAW / "mda_corpus.jsonl", encoding="utf-8")]
    meta, lines = [], []
    for d in docs:
        cleaned = clean(d["text"])
        toks = segment(cleaned)
        meta.append(dict(firm_id=d["firm_id"], year=d["year"], industry=d["industry"],
                         n_chars=len(d["text"]), n_tokens=len(toks),
                         n_sent=d["text"].count("。")))
        lines.append(" ".join(toks))

    pd.DataFrame(meta).to_csv(DATA_PROC / "corpus_meta.csv", index=False, encoding="utf-8-sig")
    with open(DATA_PROC / "corpus_seg.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    # 词频统计（用于描述性统计与人工校验抽样）
    cnt = Counter()
    for l in lines:
        cnt.update(l.split())
    pd.DataFrame(cnt.most_common(3000), columns=["word", "freq"]).to_csv(
        DATA_PROC / "word_freq.csv", index=False, encoding="utf-8-sig")

    df = pd.DataFrame(meta)
    print(f"[02] 清洗分词完成：{len(lines)} 篇，平均词数 {df.n_tokens.mean():.0f}，"
          f"最短 {df.n_tokens.min()}，最长 {df.n_tokens.max()}")
    print(f"[02] 词表规模（去重）: 见 word_freq.csv，总 token {df.n_tokens.sum():,}")


if __name__ == "__main__":
    main()
