# -*- coding: utf-8 -*-
"""
步骤 6：测度的效度检验（论文第 5 章）
-------------------------------------------------
一个文本测度要能被用于因果推断式的实证研究，至少要回答四个问题：

  【内容效度 Content validity】 词典是否覆盖了构念的全部维度，
      且未被语义无关词污染？      -> 三层构念覆盖 + LDA 主题一致性
  【人工编码效度 Coding validity】机器计数与人工阅读判断是否一致？
                                  -> 抽样人工编码、AUC、Cohen's Kappa
  【效标关联效度 Criterion validity】测度是否与"应当相关"的外部指标相关？
                                  -> 数字化投入、软件著作权、分析师关注
  【区分效度 Discriminant validity】测度是否与"不应相关"的指标无关？
                                  -> 泛化词密度、年报篇幅

同时报告【测量误差】：本文数据由模拟器生成，潜变量 dt_latent 已知，
可直接计算测度对真值的还原程度（真实数据中无法做到，故作为教学演示）。
"""
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression

from config import DATA_RAW, DATA_PROC, LEXICON, OUT_TAB, OUT_FIG, SEED

rng = np.random.default_rng(SEED + 6)
res = {}     # 收集所有结果，统一写表


def log(k, v):
    res[k] = v
    print(f"[06] {k}: {v}")


def main():
    df = pd.read_csv(DATA_PROC / "panel_measure.csv", dtype={"firm_id": str})
    lex = pd.read_csv(LEXICON / "lexicon_final.csv")
    corpus = [json.loads(l) for l in open(DATA_RAW / "mda_corpus.jsonl", encoding="utf-8")]
    seg = [" ".join(l.split()) for l in open(DATA_PROC / "corpus_seg.txt", encoding="utf-8")]

    # ---------- 1. 内容效度：构念覆盖 ----------
    cov = lex.groupby("layer").agg(词数=("word", "count"),
                                   种子词=("source", lambda s: (s == "种子词").sum()),
                                   扩展词=("source", lambda s: (s == "词向量扩展").sum()))
    cov.to_csv(OUT_TAB / "table3_lexicon_coverage.csv", encoding="utf-8-sig")
    log("词典规模", f"总计 {len(lex)} 词（tech {int(cov.loc['tech','词数'])} / "
                  f"apply {int(cov.loc['apply','词数'])} / org {int(cov.loc['org','词数'])}）")

    # ---------- 2. 内容效度：词典词是否构成语义内聚的簇 ----------
    # 思路：若词典捕捉到的是一个真实的构念，则词典中各词在 LDA 主题空间上的
    # 分布应当彼此接近（"语义内聚"），而随机抽取的词对则不应当如此。
    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=5, max_df=0.9)
    X = cv.fit_transform(seg)
    lda = LatentDirichletAllocation(n_components=25, random_state=SEED,
                                    max_iter=12, learning_method="batch")
    lda.fit(X)
    vocab = np.array(cv.get_feature_names_out())
    dt_words = [w for w in lex.word if w in set(vocab)]
    # 词-主题分布（行归一化后即为每个词在主题上的概率分布）
    phi = lda.components_.T                       # (V, K)
    phi = phi / (phi.sum(axis=1, keepdims=True) + 1e-12)
    idx = {w: i for i, w in enumerate(vocab)}
    V = np.array([phi[idx[w]] for w in dt_words])

    def cos_sim_matrix(M):
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        return M @ M.T

    S = cos_sim_matrix(V)
    iu = np.triu_indices(len(dt_words), 1)
    dt_sim = float(S[iu].mean())
    # 随机基线（同样规模的随机词，重复 20 次取均值）
    base = []
    for _ in range(20):
        rw = rng.choice(len(vocab), size=len(dt_words), replace=False)
        Sb = cos_sim_matrix(phi[rw])
        base.append(float(Sb[np.triu_indices(len(rw), 1)].mean()))
    base_sim = float(np.mean(base))
    # 词典词最集中的主题
    load = np.array([phi[idx[w]] for w in dt_words]).mean(axis=0)
    k_star = int(np.argmax(load))
    top_k = vocab[np.argsort(lda.components_[k_star])[::-1][:15]]
    pd.DataFrame([
        dict(检验="词典词两两主题分布相似度", 数值=round(dt_sim, 4)),
        dict(检验="随机词基线相似度（20次均值）", 数值=round(base_sim, 4)),
        dict(检验="内聚比（词典词/随机词）", 数值=round(dt_sim / base_sim, 3)),
        dict(检验=f"词典词最集中主题 k*={k_star}", 数值="、".join(top_k[:10])),
    ]).to_csv(OUT_TAB / "table4_lda_cohesion.csv", index=False, encoding="utf-8-sig")
    log("LDA 语义内聚性", f"词典词相似度 {dt_sim:.3f} vs 随机基线 {base_sim:.3f}，"
                       f"内聚比 {dt_sim/base_sim:.2f}；最集中主题 k*={k_star}：{'、'.join(top_k[:8])}")

    # ---------- 3. 人工编码效度 ----------
    idx = rng.choice(len(df), size=400, replace=False)
    sample = df.iloc[idx].copy()
    # 模拟两位编码者：以真值为依据、带一定误判率（模拟人工阅读噪声）
    truth = (sample.dt_latent > sample.dt_latent.median()).astype(int).values
    def coder(err):
        lab = truth.copy()
        flip = rng.random(len(lab)) < err
        return np.where(flip, 1 - lab, lab)
    c1, c2 = coder(0.10), coder(0.12)
    # Cohen's Kappa（编码者间）
    po = (c1 == c2).mean()
    pe = (c1.mean() * c2.mean()) + ((1 - c1.mean()) * (1 - c2.mean()))
    kappa = (po - pe) / (1 - pe)
    # 机器测度 vs 人工编码（取两人一致的样本）
    agree = c1 == c2
    y = c1[agree]
    score = sample.DT_ratio.values[agree]
    auc = roc_auc_score(y, score)
    cut = np.median(score)
    pred = (score > cut).astype(int)
    po2 = (pred == y).mean()
    pe2 = (pred.mean() * y.mean()) + ((1 - pred.mean()) * (1 - y.mean()))
    kappa2 = (po2 - pe2) / (1 - pe2)
    pd.DataFrame([
        dict(指标="编码者间一致性 Kappa", 数值=round(float(kappa), 4)),
        dict(指标="机器测度 vs 人工编码 AUC", 数值=round(float(auc), 4)),
        dict(指标="机器测度 vs 人工编码 一致率", 数值=round(float(po2), 4)),
        dict(指标="机器测度 vs 人工编码 Kappa", 数值=round(float(kappa2), 4)),
        dict(指标="人工编码样本量", 数值=int(agree.sum())),
    ]).to_csv(OUT_TAB / "table5_coding_validity.csv", index=False, encoding="utf-8-sig")
    log("人工编码 Kappa", f"{kappa:.3f}；机器 vs 人工 AUC={auc:.3f}, Kappa={kappa2:.3f}")

    # ---------- 4. 效标关联效度 ----------
    df["ln_digital_invest"] = np.log1p(df.digital_invest)
    df["ln_patents"] = np.log1p(df.patents)
    crit = {}
    for y in ["ln_digital_invest", "software_copyright", "analyst", "ln_patents"]:
        crit[y] = round(float(df[["DT_ratio", y]].corr().iloc[0, 1]), 4)
    # 控制规模后的偏相关
    def partial(x, y, ctrl):
        X = df[ctrl].values
        rx = LinearRegression().fit(X, df[x]).predict(X)
        ry = LinearRegression().fit(X, df[y]).predict(X)
        return round(float(np.corrcoef(df[x] - rx, df[y] - ry)[0, 1]), 4)
    crit_ctrl = {y: partial("DT_ratio", y, ["size"]) for y in crit}
    pd.DataFrame([
        dict(效标=k, 相关系数=v, 控制规模后偏相关=crit_ctrl[k],
             预期方向="正" if k != "ln_patents" else "正") for k, v in crit.items()
    ]).to_csv(OUT_TAB / "table6_criterion_validity.csv", index=False, encoding="utf-8-sig")
    log("效标关联效度", f"数字化投入 r={crit['ln_digital_invest']}，软件著作权 r={crit['software_copyright']}，"
                    f"分析师关注 r={crit['analyst']}")

    # ---------- 5. 区分效度 ----------
    # 泛化词密度：不含任何数字化语义的高频通用词占比
    generic_words = ["管理", "经营", "业务", "工作", "制度", "情况", "产品", "客户"]
    cnt_all = [l.split() for l in open(DATA_PROC / "corpus_seg.txt", encoding="utf-8")]
    df["generic_density"] = [sum(1 for w in t if w in generic_words) / max(len(t), 1) * 100
                             for t in cnt_all]
    df["doc_len"] = [len(t) for t in cnt_all]
    dis = dict(
        与泛化词密度相关=round(float(df.DT_ratio.corr(df.generic_density)), 4),
        与年报篇幅相关=round(float(df.DT_ratio.corr(df.doc_len)), 4),
        与绝对词频DT_count和篇幅=round(float(df.DT_count.corr(df.doc_len)), 4),
    )
    pd.DataFrame([dict(检验=k, 数值=v) for k, v in dis.items()]).to_csv(
        OUT_TAB / "table7_discriminant.csv", index=False, encoding="utf-8-sig")
    log("区分效度", f"与泛化词密度 r={dis['与泛化词密度相关']}，与篇幅 r={dis['与年报篇幅相关']}")

    # ---------- 6. 收敛效度：三种算法的一致性 ----------
    conv = df[["DT_ratio", "DT_tfidf", "DT_sent", "DT_ln"]].corr(method="spearman").round(4)
    conv.to_csv(OUT_TAB / "table8_convergent.csv", encoding="utf-8-sig")
    log("收敛效度(Spearman)", f"ratio~tfidf={conv.loc['DT_ratio','DT_tfidf']}, "
                            f"ratio~sent={conv.loc['DT_ratio','DT_sent']}")

    # ---------- 7. 测量误差（仅教学数据可得）----------
    err = dict(
        与真值相关系数=round(float(df.DT_ratio.corr(df.dt_latent)), 4),
        与真值Spearman=round(float(df.DT_ratio.corr(df.dt_latent, method="spearman")), 4),
        分年度相关均值=round(float(df.groupby("year").apply(
            lambda g: g.DT_ratio.corr(g.dt_latent), include_groups=False).mean()), 4),
    )
    pd.DataFrame([dict(指标=k, 数值=v) for k, v in err.items()]).to_csv(
        OUT_TAB / "table9_measurement_error.csv", index=False, encoding="utf-8-sig")
    log("测度对真值还原", f"Pearson={err['与真值相关系数']}, Spearman={err['与真值Spearman']}")

    df.to_csv(DATA_PROC / "panel_measure.csv", index=False, encoding="utf-8-sig")
    with open(OUT_TAB / "validity_summary.json", "w", encoding="utf-8") as fh:
        json.dump({k: str(v) for k, v in res.items()}, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
