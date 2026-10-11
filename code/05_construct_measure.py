# -*- coding: utf-8 -*-
"""
步骤 5：基于词典的文档级测度构造
-------------------------------------------------
在最终词典基础上，为每篇 MD&A 计算 6 类指标（论文 4.4 节）：

  (1) DT_count   数字化转型词出现的绝对次数
  (2) DT_ratio   相对词频 = 词次数 / 分词后总词数 × 100      【主指标】
  (3) DT_ln      ln(1 + DT_ratio)，缓解右偏
  (4) DT_tfidf   词典词 TF-IDF 权重之和 / 总词数（削弱高频泛化词的贡献）
  (5) DT_sent    含至少一个词典词的句子数 / 总句子数（句子级渗透度）
  (6) DT_tech / DT_apply / DT_org  三个维度的分指标（技术/应用/组织）

构造原则：
  · 采用**相对词频**而非绝对词频，控制"年报篇幅膨胀"带来的机械性上升；
  · 同时保留句子级指标，用于对篇幅不敏感的稳健性检验；
  · 分维度指标用于后续的机制与异质性分析。
"""
import json
from collections import Counter

import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from config import DATA_RAW, DATA_PROC, LEXICON, OUT_TAB, ROOT

# 词典词保证被整体切分
lex = pd.read_csv(LEXICON / "lexicon_final.csv")
for w in lex.word:
    jieba.add_word(w)
LEX = set(lex.word)
LAYER = dict(zip(lex.word, lex.layer))
LAYER_SET = {k: set(lex[lex.layer == k].word) for k in ["tech", "apply", "org"]}


def main():
    corpus = [json.loads(l) for l in open(DATA_RAW / "mda_corpus.jsonl", encoding="utf-8")]
    seg = [l.split() for l in open(DATA_PROC / "corpus_seg.txt", encoding="utf-8")]
    assert len(corpus) == len(seg)

    # ---- TF-IDF 权重（以词袋为特征）----
    raw_texts = [" ".join(toks) for toks in seg]
    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=2)
    tfidf = vec.fit_transform(raw_texts)
    vocab = {w: i for i, w in enumerate(vec.get_feature_names_out())}

    rows = []
    for i, (doc, toks) in enumerate(zip(corpus, seg)):
        n = max(len(toks), 1)
        cnt = Counter(toks)
        # (1)(2)
        hits = {w: c for w, c in cnt.items() if w in LEX}
        dt_count = sum(hits.values())
        dt_ratio = dt_count / n * 100
        # (4) TF-IDF 加权
        row = tfidf[i]
        tf_sum = 0.0
        for w, c in hits.items():
            if w in vocab:
                tf_sum += row[0, vocab[w]] * c
        # (5) 句子级渗透度
        sents = [s for s in doc["text"].split("。") if s.strip()]
        n_dt_sent = sum(1 for s in sents if any(w in s for w in LEX))
        # (6) 分维度
        per_layer = {k: sum(c for w, c in hits.items() if w in v) / n * 100
                     for k, v in LAYER_SET.items()}
        rows.append(dict(
            firm_id=doc["firm_id"], year=doc["year"], n_tokens=n,
            n_sent=len(sents), DT_count=dt_count, DT_ratio=round(dt_ratio, 5),
            DT_ln=round(float(np.log1p(dt_ratio)), 5),
            DT_tfidf=round(float(tf_sum / n), 6),
            DT_sent=round(n_dt_sent / max(len(sents), 1), 5),
            DT_tech=round(per_layer["tech"], 5),
            DT_apply=round(per_layer["apply"], 5),
            DT_org=round(per_layer["org"], 5),
            DT_dummy=int(dt_count > 0),
            DT_types=len(hits),
        ))

    m = pd.DataFrame(rows)
    panel = pd.read_csv(DATA_RAW / "panel_raw.csv", dtype={"firm_id": str})
    df = panel.merge(m, on=["firm_id", "year"], how="left")

    # 缩尾（1%）以削弱极端值影响
    for c in ["DT_ratio", "DT_ln", "DT_tfidf", "DT_sent", "DT_tech", "DT_apply", "DT_org"]:
        lo, hi = df[c].quantile([0.01, 0.99])
        df[c] = df[c].clip(lo, hi)
        df[c + "_z"] = ((df[c] - df[c].mean()) / df[c].std()).round(5)

    df["DT_z"] = ((df.DT_ratio - df.DT_ratio.mean()) / df.DT_ratio.std()).round(5)
    df.to_csv(DATA_PROC / "panel_measure.csv", index=False, encoding="utf-8-sig")

    desc = df[["DT_z", "DT_count", "DT_ratio", "DT_ln", "DT_tfidf", "DT_sent",
               "DT_tech", "DT_apply", "DT_org", "dt_latent", "patents",
               "size", "lev", "roa", "age", "cash", "tangibility",
               "top1", "soe", "rd_ratio"]].describe().T
    desc["median"] = df[desc.index].median()
    desc.round(4).to_csv(OUT_TAB / "table2_descriptive.csv", encoding="utf-8-sig")

    print(f"[05] 测度构造完成：{len(df)} 观测")
    print(f"[05] DT_ratio 均值 {df.DT_ratio.mean():.3f}，中位数 {df.DT_ratio.median():.3f}，"
          f"零值占比 {(df.DT_count==0).mean():.1%}")
    print(f"[05] DT_ratio 与 dt_latent 的相关系数：{df.DT_ratio.corr(df.dt_latent):.4f}")
    print(f"[05] 已写出 panel_measure.csv 与 table2_descriptive.csv")


if __name__ == "__main__":
    main()
