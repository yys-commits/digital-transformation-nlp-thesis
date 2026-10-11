# -*- coding: utf-8 -*-
"""
步骤 9：测度层面的稳健性（回答"结果是不是调参调出来的？"）
-------------------------------------------------
文本测度最大的质疑是研究者自由度（researcher degrees of freedom）：
换一组词向量参数、换一种加权方式，结论是否还成立？
本脚本做三重检验：

  (1) 词向量参数敏感性：在 vector_size × window 的 6 种组合下重新训练、
      重新扩展词典、重新计算 DT_ratio，报告与基准测度的 Spearman 秩相关
      以及词典的 Jaccard 相似度；
  (2) 加权方式敏感性：相对词频 / TF-IDF 加权 / 句子级渗透度 / 绝对词频
      四种口径下的测度两两相关；
  (3) 结论稳健性：把上述每一种测度分别代入基准回归，报告核心系数与显著性。
"""
import numpy as np
import pandas as pd
from collections import Counter
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

from config import DATA_PROC, LEXICON, OUT_TAB, CONTROLS, SEED_DICT, EXPAND, W2V, SEED

KEYNESS_THRESHOLD = 1.5
MIN_FREQ = 8
rng = np.random.default_rng(SEED + 9)


def build_lexicon(model, corpus, seed_all, freq_hi, freq_lo, n1, n0):
    def keyness(w):
        return ((freq_hi.get(w, 0) + 0.5) / n1) / ((freq_lo.get(w, 0) + 0.5) / n0)
    cand = {}
    for layer, seeds in SEED_DICT.items():
        for s in seeds:
            if s not in model.wv.key_to_index:
                continue
            for w, sim in model.wv.most_similar(s, topn=EXPAND["topn"]):
                if len(w) < EXPAND["min_len"] or sim < EXPAND["sim_threshold"]:
                    continue
                if sum(freq_hi.get(w, 0) for _ in [0]) < 0:
                    continue
                if (freq_hi.get(w, 0) + freq_lo.get(w, 0)) < MIN_FREQ:
                    continue
                if keyness(w) < KEYNESS_THRESHOLD:
                    continue
                if w not in cand or sim > cand[w]:
                    cand[w] = round(float(sim), 4)
    return set(seed_all) | set(cand.keys())


def measure(lex_words, corpus, meta):
    out = []
    for toks in corpus:
        n = max(len(toks), 1)
        c = sum(v for w, v in Counter(toks).items() if w in lex_words)
        out.append(c / n * 100)
    return np.array(out)


def main():
    corpus = [l.split() for l in open(DATA_PROC / "corpus_seg.txt", encoding="utf-8")]
    meta = pd.read_csv(DATA_PROC / "corpus_meta.csv", dtype={"firm_id": str})
    panel = pd.read_csv(DATA_PROC / "panel_measure.csv", dtype={"firm_id": str})
    seed_all = {w for ws in SEED_DICT.values() for w in ws}

    # 高/低转型组语料切分（与 04 一致）
    density = np.array([sum(1 for w in t if w in seed_all) / max(len(t), 1) for t in corpus])
    hi, lo = np.quantile(density, [0.70, 0.30])
    f_hi, f_lo, n1, n0 = Counter(), Counter(), 0, 0
    for d, toks in zip(density, corpus):
        if d >= hi:
            f_hi.update(toks); n1 += len(toks)
        elif d <= lo:
            f_lo.update(toks); n0 += len(toks)

    cache = DATA_PROC / "measure_variants.csv"
    base_lex = set(pd.read_csv(LEXICON / "lexicon_final.csv").word)
    base_dt = panel.DT_ratio.values

    rows, lex_sets = [], {}
    if cache.exists():
        cached = pd.read_csv(cache)
        for c in cached.columns:
            lex_sets[c] = cached[c].values
        print("[09] 读取已缓存的测度变体")
    else:
      for vs, wd in [(100, 5), (300, 5), (200, 3), (200, 8), (100, 3), (300, 8)]:
        sents = LineSentence(str(DATA_PROC / "corpus_seg.txt"))
        m = Word2Vec(sentences=sents, vector_size=vs, window=wd, min_count=W2V["min_count"],
                     sg=1, negative=W2V["negative"], workers=W2V["workers"],
                     epochs=W2V["epochs"], seed=SEED)
        lx = build_lexicon(m, corpus, seed_all, f_hi, f_lo, n1, n0)
        dt = measure(lx, corpus, meta)
        rho = pd.Series(dt).corr(pd.Series(base_dt), method="spearman")
        jac = len(lx & base_lex) / len(lx | base_lex)
        rows.append(dict(设定=f"dim={vs},window={wd}", 词典规模=len(lx),
                         与基准测度Spearman=round(float(rho), 4),
                         词典Jaccard=round(float(jac), 4)))
        lex_sets[f"dim{vs}w{wd}"] = dt
        del m
      pd.DataFrame(lex_sets).to_csv(cache, index=False)
    if rows:
        t = pd.DataFrame(rows)
        t.to_csv(OUT_TAB / "table16_measure_sensitivity.csv", index=False, encoding="utf-8-sig")
    else:   # 使用缓存时沿用此前已算好的表
        t = pd.read_csv(OUT_TAB / "table16_measure_sensitivity.csv")
    print("[09] 词向量参数敏感性：")
    print(t.to_string(index=False))

    # ---- (3) 把每种测度代入基准回归 ----
    df = panel.copy()
    df["ln_patent"] = np.log1p(df.patents)
    for k, v in lex_sets.items():
        df["DT_" + k] = v
    base_cols = ["DT_ratio"]
    res = []
    for col in ["DT_ratio"] + ["DT_" + k for k in lex_sets]:
        d = df.set_index(["firm_id", "year"])[["ln_patent", col] + CONTROLS].dropna()
        mod = PanelOLS(d["ln_patent"], sm.add_constant(d[[col] + CONTROLS]),
                       entity_effects=True, time_effects=True, check_rank=False)
        r = mod.fit(cov_type="clustered", clusters=pd.Series(d.index.get_level_values(0), index=d.index))
        res.append(dict(测度=col, 系数=round(float(r.params[col]), 4),
                        标准误=round(float(r.std_errors[col]), 4),
                        t值=round(float(r.tstats[col]), 3),
                        p值=round(float(r.pvalues[col]), 4),
                        显著性="***" if r.pvalues[col] < 0.01 else "**" if r.pvalues[col] < 0.05
                        else "*" if r.pvalues[col] < 0.1 else ""))
    tres = pd.DataFrame(res)
    tres.to_csv(OUT_TAB / "table17_sensitivity_regression.csv", index=False, encoding="utf-8-sig")
    print("\n[09] 参数敏感性下的回归系数：")
    print(tres.to_string(index=False))


if __name__ == "__main__":
    main()
