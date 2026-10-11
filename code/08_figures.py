# -*- coding: utf-8 -*-
"""
步骤 8：结果可视化（论文插图）
生成 8 张图，统一输出到 output/figures/
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from wordcloud import WordCloud

from config import DATA_PROC, LEXICON, OUT_TAB, OUT_FIG, SEED

# ---- 中文字体 ----
for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"]:
    try:
        fm.fontManager.addfont(p)
    except Exception:
        pass
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


def main():
    df = pd.read_csv(DATA_PROC / "panel_measure.csv")
    lex = pd.read_csv(LEXICON / "lexicon_final.csv")
    nb = pd.read_csv(LEXICON / "seed_neighbors.csv")

    # ---------- 图1：技术路线图 ----------
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    boxes = [
        (0.02, "① 语料获取\n年报 MD&A 章节抽取", "#dbeafe"),
        (0.21, "② 清洗分词\n领域词典注入+停用词", "#dbeafe"),
        (0.40, "③ 词向量训练\nSkip-gram(200维,窗口5)", "#fef3c7"),
        (0.59, "④ 种子词扩展\n相似度+领域特异性过滤", "#fef3c7"),
        (0.78, "⑤ 测度构造\n相对词频/TF-IDF/句级", "#dcfce7"),
    ]
    for x, t, c in boxes:
        ax.add_patch(plt.Rectangle((x, 0.55), 0.17, 0.34, facecolor=c,
                                   edgecolor="#334155", lw=1.2))
        ax.text(x + 0.085, 0.72, t, ha="center", va="center", fontsize=9.5)
        if x < 0.78:
            ax.annotate("", xy=(x + 0.18, 0.72), xytext=(x + 0.175, 0.72),
                        arrowprops=dict(arrowstyle="->", lw=1.4, color="#334155"))
    ax.text(0.50, 0.35, "⑥ 效度检验：内容效度 / 编码一致性 / 效标关联 / 区分效度",
            ha="center", fontsize=10, color="#0f172a")
    ax.text(0.50, 0.15, "⑦ 实证应用：双向固定效应 + 工具变量 + 机制与异质性分析",
            ha="center", fontsize=10, color="#0f172a")
    ax.set_title("图1 数字化转型文本测度的构建技术路线", fontsize=12)
    plt.savefig(OUT_FIG / "fig1_roadmap.png"); plt.close()

    # ---------- 图2：词典词云 ----------
    freq = dict(zip(lex.word, lex.freq))
    wc = WordCloud(font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                   width=1200, height=700, background_color="white",
                   colormap="viridis", max_words=90).generate_from_frequencies(freq)
    wc.to_file(str(OUT_FIG / "fig2_wordcloud.png"))

    # ---------- 图3：DT 时间趋势（含分维度）----------
    g = df.groupby("year")[["DT_ratio", "DT_tech", "DT_apply", "DT_org"]].mean()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for c, lab, st in [("DT_ratio", "总体", "-o"), ("DT_tech", "底层技术", "--s"),
                       ("DT_apply", "应用场景", "-.^"), ("DT_org", "组织模式", ":d")]:
        ax.plot(g.index, g[c], st, label=lab, lw=1.8, ms=5)
    ax.set_xlabel("年份"); ax.set_ylabel("数字化词占比（‰，每百词）")
    ax.set_title("图3 2012—2023 年上市公司数字化转型披露强度演变")
    ax.legend(); ax.grid(alpha=0.3)
    plt.savefig(OUT_FIG / "fig3_trend.png"); plt.close()

    # ---------- 图4：测度 vs 真值（教学数据特有）----------
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    s = df.sample(900, random_state=SEED)
    ax.scatter(s.dt_latent, s.DT_ratio, s=8, alpha=0.35, color="#2563eb")
    b = np.polyfit(df.dt_latent, df.DT_ratio, 1)
    xs = np.linspace(0, 1, 50); ax.plot(xs, np.polyval(b, xs), color="#dc2626", lw=2)
    ax.set_xlabel("数字化转型潜变量真值 dt_latent（模拟数据）")
    ax.set_ylabel("文本测度 DT_ratio")
    r = df.DT_ratio.corr(df.dt_latent)
    ax.set_title(f"图4 测度对真值的还原能力（Pearson r = {r:.3f}）")
    ax.grid(alpha=0.3)
    plt.savefig(OUT_FIG / "fig4_scatter_truth.png"); plt.close()

    # ---------- 图5：种子词扩展示例（相似度条形图）----------
    sub = nb[nb.seed.isin(["人工智能", "云计算", "智能制造", "数字化转型"])]
    sub = sub.sort_values("cosine").groupby("seed").tail(8)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = plt.cm.Set2(np.linspace(0, 1, sub.seed.nunique()))
    for (seed, grp), c in zip(sub.groupby("seed"), colors):
        ax.barh([f"{seed}→{w}" for w in grp.neighbor], grp.cosine, color=c, label=seed)
    ax.set_xlabel("余弦相似度"); ax.axvline(0.55, color="red", ls="--", lw=1)
    ax.set_title("图5 种子词语义扩展示例（虚线为相似度阈值 0.55）")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="x")
    plt.savefig(OUT_FIG / "fig5_similarity.png"); plt.close()

    # ---------- 图6：不同测度设定下的系数（森林图）----------
    t = pd.read_csv(OUT_TAB / "table10_baseline.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    y = np.arange(len(t))
    ax.errorbar(t["系数"], y, xerr=1.96 * t["标准误"], fmt="o",
                color="#1d4ed8", ecolor="#64748b", capsize=3)
    ax.set_yticks(y); ax.set_yticklabels(t["模型"], fontsize=9)
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("DT 对 ln(1+专利申请量) 的回归系数（95% 置信区间）")
    ax.set_title("图6 基准回归不同设定下的核心系数")
    ax.grid(alpha=0.25, axis="x")
    plt.savefig(OUT_FIG / "fig6_forest_baseline.png"); plt.close()

    # ---------- 图7：异质性 ----------
    h = pd.read_csv(OUT_TAB / "table14_heterogeneity.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    y = np.arange(len(h))
    ax.barh(y, h["系数"], color=["#16a34a" if v > 0 else "#dc2626" for v in h["系数"]])
    ax.set_yticks(y); ax.set_yticklabels(h["分组维度"] + "：" + h["分组"], fontsize=9)
    for i, (v, p) in enumerate(zip(h["系数"], h["显著性"])):
        ax.text(v + 0.002, i, f"{v:.3f}{p}", va="center", fontsize=8.5)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("DT 回归系数"); ax.set_title("图7 异质性分析：分组回归系数")
    plt.savefig(OUT_FIG / "fig7_heterogeneity.png"); plt.close()

    # ---------- 图8：行业 × 年份 热力图 ----------
    piv = df.pivot_table(index="industry", columns="year", values="DT_ratio", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    im = ax.imshow(piv.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=45, fontsize=8)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_title("图8 分行业—分年度数字化转型强度热力图")
    fig.colorbar(im, ax=ax, shrink=0.8, label="DT_ratio")
    plt.savefig(OUT_FIG / "fig8_heatmap.png"); plt.close()

    # ---------- 图9：机制路径示意（系数标注）----------
    m = pd.read_csv(OUT_TAB / "table13_mechanism.csv")
    fig, ax = plt.subplots(figsize=(9, 3.4)); ax.axis("off")
    ax.text(0.08, 0.6, "数字化转型\nDT", ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round", fc="#dbeafe", ec="#334155"))
    ax.text(0.72, 0.6, "创新绩效\nln(1+专利)", ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round", fc="#dcfce7", ec="#334155"))
    for i, r in m.iterrows():
        yy = 0.85 - i * 0.28
        ax.text(0.40, yy, r["机制变量"], ha="center", va="center", fontsize=9.5,
                bbox=dict(boxstyle="round", fc="#fef3c7", ec="#334155"))
        ax.annotate("", xy=(0.34, yy), xytext=(0.14, 0.6),
                    arrowprops=dict(arrowstyle="->", lw=1.1, color="#475569"))
        ax.text(0.22, (yy + 0.6) / 2 + 0.02, f"a={r['a_路径系数']}", fontsize=8, color="#b91c1c")
        ax.annotate("", xy=(0.66, 0.6), xytext=(0.46, yy),
                    arrowprops=dict(arrowstyle="->", lw=1.1, color="#475569"))
        ax.text(0.55, (yy + 0.6) / 2 + 0.02, f"b={r['b_机制变量系数']}", fontsize=8, color="#b91c1c")
    ax.annotate("", xy=(0.68, 0.6), xytext=(0.16, 0.6),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="#0f172a"))
    ax.text(0.42, 0.63, "直接效应 c'", fontsize=8.5)
    ax.set_title("图9 机制检验路径与系数（a：DT→机制变量；b：机制变量→创新绩效）", fontsize=11)
    plt.savefig(OUT_FIG / "fig9_mechanism.png"); plt.close()

    print(f"[08] 已生成 {len(list(OUT_FIG.glob('*.png')))} 张图片 -> {OUT_FIG}")


if __name__ == "__main__":
    main()
