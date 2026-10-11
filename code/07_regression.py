# -*- coding: utf-8 -*-
"""
步骤 7：实证回归（论文第 6 章）
-------------------------------------------------
研究假说：
  H1（主效应）数字化转型程度越高，企业创新绩效越好。
  H2（机制）  数字化转型通过 (a) 降低运营成本、(b) 提升信息透明度
              (c) 提高研发资源配置效率 三条路径作用于创新绩效。
  H3（异质性）上述效应在非国有企业、技术密集型行业、小规模企业中更强。

计量设定：
  ln(1+Patent)_{it} = β·DT_{it} + γ·X_{it} + μ_i + λ_t + ε_{it}
  μ_i 企业固定效应，λ_t 年份固定效应，标准误在企业层面聚类。

内生性处理：
  (a) 解释变量滞后一期；
  (b) 工具变量：同省份—同行业—同年度其他企业的数字化转型均值（Leave-one-out），
      满足相关性（同群效应/政策扩散）与外生性（剔除自身后不受本企业创新冲击影响）。
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS

from config import DATA_PROC, OUT_TAB, CONTROLS

pd.set_option("display.width", 200)


def load():
    df = pd.read_csv(DATA_PROC / "panel_measure.csv", dtype={"firm_id": str})
    df["ln_patent"] = np.log1p(df.patents)
    df["ln_analyst"] = np.log1p(df.analyst)
    df = df.set_index(["firm_id", "year"])
    return df


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def fit_panel(df, y, x, fe_entity=True, fe_time=True, cluster="entity"):
    d = df[[y] + x].dropna()
    mod = PanelOLS(d[y], sm.add_constant(d[x]), entity_effects=fe_entity,
                   time_effects=fe_time, drop_absorbed=True,
                   check_rank=False)
    return mod.fit(cov_type="clustered", clusters=pd.Series(d.index.get_level_values(0), index=d.index))


def row(res, name, key, note=""):
    b, p = res.params[key], res.pvalues[key]
    return dict(模型=name, 变量=key, 系数=round(float(b), 4),
                标准误=round(float(res.std_errors[key]), 4),
                t值=round(float(res.tstats[key]), 3),
                p值=round(float(p), 4), 显著性=stars(p),
                N=int(res.nobs), 说明=note)


def main():
    df = load()
    out = {}

    # ============ 表：基准回归 ============
    rows = []
    specs = [
        ("(1) 混合OLS", ["DT_z"] + CONTROLS + ["soe"], False, False),
        ("(2) 企业FE", ["DT_z"] + CONTROLS, True, False),
        ("(3) 企业+年份FE", ["DT_z"] + CONTROLS, True, True),
        ("(4) 替换测度DT_ratio", ["DT_ratio"] + CONTROLS, True, True),
        ("(5) 标准化句子级测度", ["DT_sent_z"] + CONTROLS, True, True),
        ("(6) 标准化TF-IDF测度", ["DT_tfidf_z"] + CONTROLS, True, True),
    ]
    for name, xs, fe_e, fe_t in specs:
        r = fit_panel(df, "ln_patent", xs, fe_e, fe_t)
        key = xs[0]
        rows.append(row(r, name, key, "双向固定效应" if (fe_e and fe_t) else ""))
    t_base = pd.DataFrame(rows)
    t_base.to_csv(OUT_TAB / "table10_baseline.csv", index=False, encoding="utf-8-sig")
    out["基准回归"] = t_base
    print("[07] 基准回归：")
    print(t_base[["模型", "系数", "标准误", "t值", "显著性", "N"]].to_string(index=False))

    # ============ 分维度回归 ============
    rows = []
    for k, name in [("DT_tech_z", "底层技术"), ("DT_apply_z", "应用场景"), ("DT_org_z", "组织模式")]:
        r = fit_panel(df, "ln_patent", [k] + CONTROLS, True, True)
        rows.append(row(r, f"分维度-{name}", k))
    t_dim = pd.DataFrame(rows)
    t_dim.to_csv(OUT_TAB / "table11_dimensions.csv", index=False, encoding="utf-8-sig")
    print("\n[07] 分维度回归：")
    print(t_dim[["模型", "系数", "t值", "显著性"]].to_string(index=False))

    # ============ 内生性：滞后一期 ============
    d2 = df.reset_index()
    d2["DT_lag"] = d2.groupby("firm_id")["DT_z"].shift(1)
    d2 = d2.dropna(subset=["DT_lag"]).set_index(["firm_id", "year"])
    r = fit_panel(d2, "ln_patent", ["DT_lag"] + CONTROLS, True, True)
    lag_row = row(r, "(1) DT滞后一期", "DT_lag")

    # ============ 内生性：工具变量（Leave-one-out 同群均值）============
    # 面板 IV 需先做双向 within 变换，再对变换后变量执行 2SLS
    d3 = df.reset_index()
    grp = d3.groupby(["province", "industry", "year"])["DT_z"]
    d3["DT_peer"] = (grp.transform("sum") - d3["DT_z"]) / (grp.transform("count") - 1)
    d3 = d3.dropna(subset=["DT_peer"])
    cols = ["ln_patent", "DT_z", "DT_peer"] + CONTROLS

    def demean(d, cols):
        out = pd.DataFrame(index=d.index)
        for c in cols:
            x = d[c].astype(float)
            out[c] = (x - x.groupby(d.firm_id).transform("mean")
                      - x.groupby(d.year).transform("mean") + x.mean())
        return out

    dm = demean(d3, cols)
    idx = pd.MultiIndex.from_frame(d3[["firm_id", "year"]])
    dm.index = idx
    exog = dm[CONTROLS]
    iv = IV2SLS(dm["ln_patent"], exog, dm["DT_z"], dm["DT_peer"]).fit(
        cov_type="clustered", clusters=pd.Series(idx.get_level_values(0), index=idx))
    # 一阶段：内生变量对工具变量回归
    fs = sm.OLS(dm["DT_z"], pd.concat([dm[["DT_peer"]], dm[CONTROLS]], axis=1)).fit(cov_type="cluster", cov_kwds={"groups": idx.get_level_values(0)})
    f_stat = float(fs.tvalues["DT_peer"]) ** 2
    iv_row = dict(模型="(2) IV-2SLS(双向within)", 变量="DT_z",
                  系数=round(float(iv.params["DT_z"]), 4),
                  标准误=round(float(iv.std_errors["DT_z"]), 4),
                  t值=round(float(iv.tstats["DT_z"]), 3),
                  p值=round(float(iv.pvalues["DT_z"]), 4),
                  显著性=stars(iv.pvalues["DT_z"]), N=int(iv.nobs),
                  说明=f"一阶段 t={fs.tvalues['DT_peer']:.2f}(F≈{f_stat:.1f})，"
                       f"工具=同省同行业同年度均值(leave-one-out)")
    t_endo = pd.DataFrame([lag_row, iv_row])
    t_endo.to_csv(OUT_TAB / "table12_endogeneity.csv", index=False, encoding="utf-8-sig")
    print("\n[07] 内生性处理：")
    print(t_endo[["模型", "系数", "t值", "显著性", "说明"]].to_string(index=False))

    # ============ 机制检验（三步中介法）============
    mech_rows = []
    for mvar, mname in [("opcost", "运营成本率"), ("ln_analyst", "信息透明度(分析师关注)"),
                        ("cost_ratio", "主营成本率")]:
        a = fit_panel(df, mvar, ["DT_z"] + CONTROLS, True, True)
        b = fit_panel(df, "ln_patent", ["DT_z", mvar] + CONTROLS, True, True)
        mech_rows.append(dict(机制变量=mname,
                              a_路径系数=round(float(a.params["DT_z"]), 4),
                              a_p值=round(float(a.pvalues["DT_z"]), 4),
                              b_机制变量系数=round(float(b.params[mvar]), 4),
                              b_p值=round(float(b.pvalues[mvar]), 4),
                              直接效应=round(float(b.params["DT_z"]), 4),
                              是否成立="是" if (a.pvalues["DT_z"] < 0.1 and b.pvalues[mvar] < 0.1) else "弱/否"))
    t_mech = pd.DataFrame(mech_rows)
    t_mech.to_csv(OUT_TAB / "table13_mechanism.csv", index=False, encoding="utf-8-sig")
    print("\n[07] 机制检验：")
    print(t_mech.to_string(index=False))

    # ============ 异质性 ============
    d4 = df.reset_index()
    d4["manuf"] = d4.industry.str.startswith("C").astype(int)
    d4["large"] = (d4["size"] > d4["size"].median()).astype(int)
    het = []
    labels = {"soe": {0: "非国企", 1: "国企"},
              "manuf": {0: "非制造业", 1: "制造业"},
              "large": {0: "小规模", 1: "大规模"}}
    for grp_col, gname in [("soe", "产权性质"), ("manuf", "是否制造业"), ("large", "企业规模")]:
        for gval in [0, 1]:
            sub = d4[d4[grp_col] == gval].set_index(["firm_id", "year"])
            r = fit_panel(sub, "ln_patent", ["DT_z"] + CONTROLS, True, True)
            het.append(dict(分组维度=gname, 分组=labels[grp_col][gval],
                            系数=round(float(r.params["DT_z"]), 4),
                            t值=round(float(r.tstats["DT_z"]), 3),
                            p值=round(float(r.pvalues["DT_z"]), 4),
                            显著性=stars(r.pvalues["DT_z"]), N=int(r.nobs)))
    t_het = pd.DataFrame(het)
    t_het.to_csv(OUT_TAB / "table14_heterogeneity.csv", index=False, encoding="utf-8-sig")
    print("\n[07] 异质性：")
    print(t_het.to_string(index=False))

    # ============ 稳健性 ============
    rob = []
    base = fit_panel(df, "ln_patent", ["DT_z"] + CONTROLS, True, True)
    rob.append(row(base, "(0) 基准", "DT_z"))
    # (a) 剔除未提及数字化的样本
    sub = df[df.DT_count > 0]
    rob.append(row(fit_panel(sub, "ln_patent", ["DT_z"] + CONTROLS, True, True),
                   "(1) 剔除零提及样本", "DT_z"))
    # (b) 剔除 2012-2014（早期年报篇幅短，测度噪声大）
    sub = df[df.index.get_level_values("year") >= 2015]
    rob.append(row(fit_panel(sub, "ln_patent", ["DT_z"] + CONTROLS, True, True),
                   "(2) 剔除2012-2014", "DT_z"))
    # (c) 5% 缩尾
    d5 = df.copy()
    lo, hi = d5.DT_z.quantile([0.05, 0.95])
    d5["DT_w"] = d5.DT_z.clip(lo, hi)
    rob.append(row(fit_panel(d5, "ln_patent", ["DT_w"] + CONTROLS, True, True),
                   "(3) 5%缩尾", "DT_w"))
    # (d) 用绝对词频对数（不控制篇幅）
    d6 = df.copy()
    d6["DT_cnt_ln"] = np.log1p(d6.DT_count)
    rob.append(row(fit_panel(d6, "ln_patent", ["DT_cnt_ln"] + CONTROLS, True, True),
                   "(4) 绝对词频对数", "DT_cnt_ln"))
    # (e) 泊松回归（专利为计数变量）
    d7 = df.reset_index()
    X = sm.add_constant(d7[["DT_z"] + CONTROLS])
    po = sm.GLM(d7.patents, X, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": d7.firm_id})
    rob.append(dict(模型="(5) 泊松回归", 变量="DT_z", 系数=round(float(po.params["DT_z"]), 4),
                    标准误=round(float(po.bse["DT_z"]), 4), t值=round(float(po.tvalues["DT_z"]), 3),
                    p值=round(float(po.pvalues["DT_z"]), 4),
                    显著性=stars(po.pvalues["DT_z"]), N=int(po.nobs), 说明="被解释变量=专利计数"))
    # (f) 加入 行业×年份 联合固定效应
    #     注意：企业固定效应已吸收行业主效应，行业×年份哑变量在每个企业内
    #     逐年求和为 1，会与常数项/企业效应共线，因此 (i) 不加常数项、
    #     (ii) 每个行业删去一个参照年份（此处删 2012）
    d8 = df.reset_index()
    d8["ind_year"] = d8.industry.astype(str) + "_" + d8.year.astype(str)
    fe = pd.get_dummies(d8["ind_year"]).astype(float)
    fe = fe.loc[:, ~fe.columns.str.endswith("_2012")]
    fe.index = pd.MultiIndex.from_frame(d8[["firm_id", "year"]])
    Xd = pd.concat([d8[["DT_z"] + CONTROLS].set_index(fe.index), fe], axis=1)
    yv = d8.set_index(fe.index)["ln_patent"]
    r = PanelOLS(yv, Xd, entity_effects=True, time_effects=False, check_rank=False).fit(
        cov_type="clustered", clusters=pd.Series(yv.index.get_level_values(0), index=yv.index))
    rob.append(row(r, "(6) 行业×年份FE", "DT_z"))
    t_rob = pd.DataFrame(rob)
    t_rob.to_csv(OUT_TAB / "table15_robustness.csv", index=False, encoding="utf-8-sig")
    print("\n[07] 稳健性：")
    print(t_rob[["模型", "系数", "标准误", "t值", "显著性", "N"]].to_string(index=False))

    print(f"\n[07] 全部结果已写入 {OUT_TAB}")


if __name__ == "__main__":
    main()
