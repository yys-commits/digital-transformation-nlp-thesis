# -*- coding: utf-8 -*-
"""
步骤 4：种子词 -> 语义扩展 -> 领域特异性过滤 -> 词典定稿
-------------------------------------------------
纯词向量扩展会引入两类噪声：
  (1) 语义相近但不属于数字化范畴的词（如"互联网""技术""系统"）；
  (2) 分词碎屑与低频噪声词。
为此本文在相似度扩展之后加入**领域特异性过滤**（keyness filter）：

    先用种子词密度给每篇文档打分，取密度最高的 30% 作为"高转型组" D1、
    最低的 30% 作为"低转型组" D0（中间 40% 不用，以拉开对比度）；
    对候选词 w 计算领域集中度（加 0.5 伪计数平滑）
        Keyness(w) = (f_{w,D1}+0.5) / N_D1 ÷ (f_{w,D0}+0.5) / N_D0
    Keyness 越大，说明 w 越集中出现在数字化转型语境中。
    "信息化""自动化""互联网"这类泛化词在两组中 equally 常见，
    Keyness≈1，会被自动剔除；而"数字孪生""预测性维护"等专业词
    高度集中于高转型组，Keyness 明显大于 1。
    保留 Keyness >= 1.5 且余弦相似度 >= 0.55 的词进入最终词典。

随后输出三张表：
  lexicon_final.csv     最终词典（含层级、来源、相似度、Keyness、词频）
  expansion_report.csv  扩展过程明细（含被过滤词及原因）
  candidate_recall.csv  对"未在种子表中的专业术语"的召回情况（教学数据下可验证）
"""
import json
import numpy as np
import math
from collections import Counter

import pandas as pd
from gensim.models import Word2Vec

from config import DATA_PROC, LEXICON, SEED_DICT, EXPAND, ROOT

KEYNESS_THRESHOLD = 1.5
MIN_FREQ = 8          # 候选词最低语料频次


def load_corpus():
    return [l.split() for l in open(DATA_PROC / "corpus_seg.txt", encoding="utf-8")]


def main():
    model = Word2Vec.load(str(LEXICON / "w2v.model"))
    corpus = load_corpus()

    seed_all = {w for ws in SEED_DICT.values() for w in ws}
    # ---- 按种子词密度划分高/低转型组，并统计词频 ----
    density = np.array([sum(1 for w in t if w in seed_all) / max(len(t), 1) for t in corpus])
    hi, lo = np.quantile(density, [0.70, 0.30])
    f1, f0, n1, n0 = Counter(), Counter(), 0, 0
    for d, toks in zip(density, corpus):
        if d >= hi:
            f1.update(toks); n1 += len(toks)
        elif d <= lo:
            f0.update(toks); n0 += len(toks)
    global_freq = Counter()
    for toks in corpus:
        global_freq.update(toks)

    def keyness(w):
        a = (f1.get(w, 0) + 0.5) / max(n1, 1)
        b = (f0.get(w, 0) + 0.5) / max(n0, 1)
        return a / b

    # ---- 语义扩展 ----
    records, filtered = [], []
    cand = {}          # word -> (max_sim, 来源层)
    for layer, seeds in SEED_DICT.items():
        for s in seeds:
            if s not in model.wv.key_to_index:
                continue
            for w, sim in model.wv.most_similar(s, topn=EXPAND["topn"]):
                if len(w) < EXPAND["min_len"]:
                    continue
                reason = None
                if sim < EXPAND["sim_threshold"]:
                    reason = "相似度低于阈值"
                elif global_freq.get(w, 0) < MIN_FREQ:
                    reason = "语料频次过低"
                elif keyness(w) < KEYNESS_THRESHOLD:
                    reason = "领域特异性不足（泛化词）"
                rec = dict(layer=layer, seed=s, word=w, cosine=round(float(sim), 4),
                           freq=int(global_freq.get(w, 0)),
                           keyness=round(keyness(w), 3), status="保留" if reason is None else "剔除",
                           reason=reason or "")
                records.append(rec)
                if reason is None:
                    if w not in cand or sim > cand[w][0]:
                        cand[w] = (round(float(sim), 4), layer)
                else:
                    filtered.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(LEXICON / "expansion_report.csv", index=False, encoding="utf-8-sig")

    # ---- 词典定稿：种子词（相似度记为 1.0，来源=种子）+ 扩展词 ----
    rows = []
    for layer, seeds in SEED_DICT.items():
        for s in seeds:
            rows.append(dict(word=s, layer=layer, source="种子词", cosine=1.0,
                             freq=int(global_freq.get(s, 0)), keyness=round(keyness(s), 3)))
    for w, (sim, layer) in cand.items():
        if w in seed_all:
            continue
        rows.append(dict(word=w, layer=layer, source="词向量扩展", cosine=sim,
                         freq=int(global_freq.get(w, 0)), keyness=round(keyness(w), 3)))
    lex = pd.DataFrame(rows).sort_values(["layer", "source", "word"]).reset_index(drop=True)
    lex.to_csv(LEXICON / "lexicon_final.csv", index=False, encoding="utf-8-sig")

    # ---- 召回检验（教学数据下已知的专业术语清单）----
    known = set("""
计算机视觉 智能语音 知识图谱 大模型 算法模型 智能算法 分布式存储 公有云 私有云 混合云
云原生 容器化 智能合约 共识机制 数字人民币 传感器 射频识别 柔性制造 无人工厂 自动化产线
工业机器人 预测性维护 可视化大屏 数据中台 业务中台 用户画像 标签体系 私域流量 全渠道
直播电商 智慧仓储 智能客服 智慧园区 数字工厂 智能排产 质量追溯 数字化组织 敏捷迭代
中台战略 数字治理 数据要素 数据资产化 流程再造 组织扁平化 数字化考核 生态协同
""".split())
    recalled = known & set(lex.word)
    pd.DataFrame(sorted(known), columns=["known_term"]).assign(
        recalled=lambda d: d.known_term.isin(recalled)).to_csv(
        LEXICON / "candidate_recall.csv", index=False, encoding="utf-8-sig")

    n_seed = sum(len(v) for v in SEED_DICT.values())
    print(f"[04] 种子词 {n_seed} 个 -> 候选 {len(records)} 条 -> 保留 {len(cand)} 词")
    print(f"[04] 最终词典规模：{len(lex)} 词（种子 {n_seed} + 扩展 {len(lex)-n_seed}）")
    print(f"[04] 已知专业术语召回率：{len(recalled)}/{len(known)} = {len(recalled)/len(known):.1%}")
    print(f"[04] 过滤掉 {len(filtered)} 条，示例：")
    for r in filtered[:8]:
        print(f"      {r['word']}  (sim={r['cosine']}, keyness={r['keyness']}, {r['reason']})")


if __name__ == "__main__":
    main()
