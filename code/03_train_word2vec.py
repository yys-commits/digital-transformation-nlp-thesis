# -*- coding: utf-8 -*-
"""
步骤 3：领域语料上的词向量训练
-------------------------------------------------
为什么不用通用语料（如 wiki zh）预训练向量？
  年报 MD&A 是高度专业化的子语言，"中台""数字孪生""柔性制造"等词在
  通用语料中低频甚至缺词，且词语之间的"业务邻近关系"与通用语义不同。
  因此本文在**年报语料内部**训练分布式表示（domain embedding），
  使相似度刻画的是"在同一业务语境中可互换"这一测度所需的关系。

参数设定（论文 4.3 节，表 4-3）：
  Skip-gram（sg=1）对低频专业词更友好；window=5 对应句内共现；
  vector_size=200；negative=10；min_count=5；epochs=8。
  稳健性部分对 (100/200/300) × (3/5/8) 做网格，检验测度的参数敏感性。
"""
import sys
import time
from itertools import product

import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from config import DATA_PROC, LEXICON, W2V, SEED_DICT, SEED


def train(vector_size=200, window=5, sg=1, epochs=8, save=True):
    sents = LineSentence(str(DATA_PROC / "corpus_seg.txt"))
    model = Word2Vec(
        sentences=sents, vector_size=vector_size, window=window,
        min_count=W2V["min_count"], sg=sg, negative=W2V["negative"],
        workers=W2V["workers"], epochs=epochs, seed=SEED,
    )
    if save:
        model.save(str(LEXICON / "w2v.model"))
    return model


def main():
    t0 = time.time()
    model = train()
    vocab = len(model.wv.key_to_index)
    print(f"[03] 训练完成，词表 {vocab} 词，用时 {time.time()-t0:.0f}s")

    # 种子词相似词明细（供人工校验与附录使用）
    rows = []
    for layer, seeds in SEED_DICT.items():
        for s in seeds:
            if s not in model.wv.key_to_index:
                continue
            for w, sim in model.wv.most_similar(s, topn=15):
                rows.append(dict(layer=layer, seed=s, neighbor=w, cosine=round(float(sim), 4)))
    df = pd.DataFrame(rows)
    df.to_csv(LEXICON / "seed_neighbors.csv", index=False, encoding="utf-8-sig")
    print(f"[03] 已保存种子词近邻表 {len(df)} 行 -> seed_neighbors.csv")

    if "--skip-grid" in sys.argv:
        print("[03] 已跳过参数网格")
        return
    # 参数敏感性网格（稳健性检验的输入）
    grid, res = [], []
    for vs, wd in product([100, 200, 300], [3, 5, 8]):
        m = train(vector_size=vs, window=wd, sg=1, epochs=8, save=False)
        grid.append((vs, wd))
        for s in SEED_DICT["tech"][:5]:
            if s in m.wv.key_to_index:
                top = m.wv.most_similar(s, topn=10)
                res.append(dict(vector_size=vs, window=wd, seed=s,
                                top10="、".join(f"{w}({v:.2f})" for w, v in top)))
        del m
    pd.DataFrame(res).to_csv(LEXICON / "param_grid_neighbors.csv", index=False, encoding="utf-8-sig")
    print(f"[03] 参数网格 {len(grid)} 组 -> param_grid_neighbors.csv")


if __name__ == "__main__":
    main()
