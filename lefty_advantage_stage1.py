#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import sys
import time
import warnings
import urllib.request

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- 
# CONFIG
# ----------------------------------------------------------------------------- 
RAW = "https://raw.githubusercontent.com/JeffSackmann"
ATP_BASE = f"{RAW}/tennis_atp/master"
WTA_BASE = f"{RAW}/tennis_wta/master"
MCP_BASE = f"{RAW}/tennis_MatchChartingProject/master"

YEARS = list(range(1968, 2026))          # ATP/WTA 单打巡回赛主赛年份范围（404 会自动跳过）
CACHE = "./_tennis_cache"                 # 下载缓存目录
OUTDIR = "./tennis_lefty_outputs"         # 结果输出目录
MIN_RANK_OK = True                        # 是否要求双方排名非空

os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

SUMMARY = [] 

def log(msg=""):
    print(msg)
    SUMMARY.append(str(msg))


def fetch(url, fname):
    """下载到缓存；已存在则跳过；404 等返回 None。"""
    path = os.path.join(CACHE, fname)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
        time.sleep(0.05)
        return path
    except Exception:
        return None


def read_csv_safe(path):
    if path is None:
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"  [warn] 读取失败 {path}: {e}")
        return None


# ============================================================================= 
# 模块 A：宏观左手优势
# ============================================================================= 
def load_tour_matches(base, tag):
    frames = []
    for y in YEARS:
        fname = f"{tag}_matches_{y}.csv"
        p = fetch(f"{base}/{fname}", fname)
        df = read_csv_safe(p)
        if df is None or len(df) == 0:
            continue
        df["tour"] = tag.upper()
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out


def build_mixed_handed(matches):
    """只保留一方为 L、一方为 R 的单打比赛，整理成以左手球员为视角的一行。"""
    need = ["winner_hand", "loser_hand", "winner_rank", "loser_rank",
            "winner_id", "loser_id", "surface", "best_of", "tourney_date", "tour",
            "winner_rank_points", "loser_rank_points"]
    for c in need:
        if c not in matches.columns:
            matches[c] = np.nan
    m = matches.copy()
    m["winner_hand"] = m["winner_hand"].astype(str).str.upper().str.strip()
    m["loser_hand"] = m["loser_hand"].astype(str).str.upper().str.strip()
    mask = (((m.winner_hand == "L") & (m.loser_hand == "R")) |
            ((m.winner_hand == "R") & (m.loser_hand == "L")))
    m = m[mask].copy()

    lefty_won = (m.winner_hand == "L").astype(int)
    rows = pd.DataFrame({
        "tour": m["tour"].values,
        "lefty_won": lefty_won.values,
        "lefty_id": np.where(m.winner_hand == "L", m.winner_id, m.loser_id),
        "lefty_rank": np.where(m.winner_hand == "L", m.winner_rank, m.loser_rank),
        "righty_rank": np.where(m.winner_hand == "L", m.loser_rank, m.winner_rank),
        "lefty_rp": np.where(m.winner_hand == "L", m.winner_rank_points, m.loser_rank_points),
        "righty_rp": np.where(m.winner_hand == "L", m.loser_rank_points, m.winner_rank_points),
        "surface": m["surface"].values,
        "best_of": m["best_of"].values,
        "date": pd.to_numeric(m["tourney_date"], errors="coerce").values,
    })
    rows["year"] = (rows["date"] // 10000).astype("Int64")
    rows["decade"] = (rows["year"] // 10 * 10).astype("Int64")
    # 排名差（从左手视角：正值=右手排名更靠后=左手被看好）
    rows["rank_diff"] = rows["righty_rank"] - rows["lefty_rank"]
    return rows


def binom_summary(wins, n):
    if n == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    p = wins / n
    lo, hi = stats.binomtest(int(wins), int(n)).proportion_ci(confidence_level=0.95)
    pval = stats.binomtest(int(wins), int(n), 0.5).pvalue
    return (p, lo, hi, pval)


def run_module_A(rows):
    log("\n" + "=" * 78)
    log("模块 A：宏观左手优势（一左一右的混合对阵）")
    log("=" * 78)
    log(f"混合对阵总场次（含缺排名）: {len(rows):,}")

    # ---- A1 未调整总体胜率 ----
    overall = []
    for tour in ["ATP", "WTA", "ALL"]:
        sub = rows if tour == "ALL" else rows[rows.tour == tour]
        n = len(sub); w = int(sub.lefty_won.sum())
        p, lo, hi, pv = binom_summary(w, n)
        overall.append({"group": tour, "n_matches": n, "lefty_win_rate": p,
                        "ci95_lo": lo, "ci95_hi": hi, "p_vs_0.5": pv})
        log(f"  [{tour}] n={n:,}  左手胜率={p:.4f}  95%CI[{lo:.4f},{hi:.4f}]  p={pv:.2e}")
    pd.DataFrame(overall).to_csv(f"{OUTDIR}/A_overall.csv", index=False)

    # ---- A2 分层（性别 x 年代 x 场地 x 赛制）----
    strata = []
    for tour in ["ATP", "WTA"]:
        sub0 = rows[rows.tour == tour]
        for dec in sorted([d for d in sub0.decade.dropna().unique()]):
            s = sub0[sub0.decade == dec]
            n = len(s); w = int(s.lefty_won.sum())
            p, lo, hi, pv = binom_summary(w, n)
            strata.append({"strata": "decade", "tour": tour, "key": int(dec),
                           "n": n, "lefty_win_rate": p, "ci_lo": lo, "ci_hi": hi})
        for surf in sub0.surface.dropna().unique():
            s = sub0[sub0.surface == surf]
            n = len(s); w = int(s.lefty_won.sum())
            p, lo, hi, pv = binom_summary(w, n)
            strata.append({"strata": "surface", "tour": tour, "key": surf,
                           "n": n, "lefty_win_rate": p, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(strata).to_csv(f"{OUTDIR}/A_by_strata.csv", index=False)
    log("  分层结果已写入 A_by_strata.csv（年代/场地 x 性别）")

    # ---- A3 控制排名的 logistic + 聚类稳健 SE ----
    log("\n  -- 控制排名的 logistic 回归（聚类于左手球员）--")
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        logit_rows = []
        for tour in ["ATP", "WTA", "ALL"]:
            sub = rows if tour == "ALL" else rows[rows.tour == tour]
            d = sub.dropna(subset=["lefty_won", "rank_diff", "lefty_id"]).copy()
            if len(d) < 200:
                continue
            d["rank_diff100"] = d["rank_diff"] / 100.0 
            model = smf.glm("lefty_won ~ rank_diff100", data=d,
                            family=sm.families.Binomial())
            res = model.fit(cov_type="cluster",
                            cov_kwds={"groups": d["lefty_id"]})
            b0 = res.params["Intercept"]; se0 = res.bse["Intercept"]
            p_eq = 1 / (1 + np.exp(-b0))   # 排名相同时左手获胜概率
            ci = res.conf_int().loc["Intercept"]
            p_lo = 1 / (1 + np.exp(-ci[0])); p_hi = 1 / (1 + np.exp(-ci[1]))
            logit_rows.append({
                "tour": tour, "n": len(d),
                "intercept": b0, "intercept_se": se0,
                "p_lefty_win_equal_rank": p_eq,
                "p_eq_ci_lo": p_lo, "p_eq_ci_hi": p_hi,
                "p_value_intercept": res.pvalues["Intercept"],
                "beta_rank_diff100": res.params["rank_diff100"],
            })
            log(f"  [{tour}] n={len(d):,}  排名相同时左手胜率={p_eq:.4f} "
                f"[{p_lo:.4f},{p_hi:.4f}]  截距p={res.pvalues['Intercept']:.2e}")
        pd.DataFrame(logit_rows).to_csv(f"{OUTDIR}/A_logit.csv", index=False)
    except ImportError:
        log("  [跳过] 未安装 statsmodels，无法做 logistic。pip install statsmodels 后重跑。")

    # ---- A4 年代趋势（等排名左手胜率随年代变化）----
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        trend = []
        for tour in ["ATP", "WTA"]:
            sub = rows[rows.tour == tour]
            for dec in sorted([d for d in sub.decade.dropna().unique()]):
                d = sub[sub.decade == dec].dropna(
                    subset=["lefty_won", "rank_diff", "lefty_id"]).copy()
                if len(d) < 150:
                    continue
                d["rank_diff100"] = d["rank_diff"] / 100.0
                try:
                    res = smf.glm("lefty_won ~ rank_diff100", data=d,
                                  family=sm.families.Binomial()).fit(
                                  cov_type="cluster",
                                  cov_kwds={"groups": d["lefty_id"]})
                    b0 = res.params["Intercept"]
                    trend.append({"tour": tour, "decade": int(dec), "n": len(d),
                                  "p_lefty_win_equal_rank": 1/(1+np.exp(-b0))})
                except Exception:
                    pass
        td = pd.DataFrame(trend)
        td.to_csv(f"{OUTDIR}/A_decade_trend.csv", index=False)
        if len(td):
            log("\n  -- 等排名左手胜率的年代趋势 --")
            for _, r in td.iterrows():
                log(f"  [{r['tour']}] {int(r['decade'])}s  n={int(r['n']):,}  "
                    f"p={r['p_lefty_win_equal_rank']:.4f}")
            # 图
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(7, 4))
                for tour, g in td.groupby("tour"):
                    ax.plot(g.decade, g.p_lefty_win_equal_rank, marker="o", label=tour)
                ax.axhline(0.5, ls="--", c="grey")
                ax.set_xlabel("Decade"); ax.set_ylabel("P(lefty wins) at equal rank")
                ax.set_title("Left-handed advantage over decades (rank-controlled)")
                ax.legend(); fig.tight_layout()
                fig.savefig(f"{OUTDIR}/A_decade_trend.png", dpi=140)
            except Exception:
                pass
    except ImportError:
        pass

    return rows


# ============================================================================= 
# 模块 B：机制——发球落点的镜像对称性
# ============================================================================= 
SERVE_LOCS = ["deuce_wide", "deuce_middle", "deuce_t", "ad_wide", "ad_middle", "ad_t"]
MIRROR_MAP = {"deuce_wide": "ad_wide", "deuce_middle": "ad_middle", "deuce_t": "ad_t",
              "ad_wide": "deuce_wide", "ad_middle": "deuce_middle", "ad_t": "deuce_t"}


def load_mcp_handmap():
    """从 MCP 比赛元数据构造 (match_id, player) -> (hand, opp_hand) 的长表。"""
    frames = []
    for g in ["m", "w"]:
        p = fetch(f"{MCP_BASE}/charting-{g}-matches.csv", f"charting-{g}-matches.csv")
        df = read_csv_safe(p)
        if df is None:
            continue
        df["gender"] = g
        a = df[["match_id", "Player 1", "Pl 1 hand", "Pl 2 hand", "gender", "Surface"]].copy()
        a.columns = ["match_id", "player", "hand", "opp_hand", "gender", "surface"]
        b = df[["match_id", "Player 2", "Pl 2 hand", "Pl 1 hand", "gender", "Surface"]].copy()
        b.columns = ["match_id", "player", "hand", "opp_hand", "gender", "surface"]
        frames.append(pd.concat([a, b], ignore_index=True))
    hm = pd.concat(frames, ignore_index=True)
    for c in ["hand", "opp_hand"]:
        hm[c] = hm[c].astype(str).str.upper().str.strip()
    return hm


def load_mcp_stat(stat):
    frames = []
    for g in ["m", "w"]:
        p = fetch(f"{MCP_BASE}/charting-{g}-stats-{stat}.csv",
                  f"charting-{g}-stats-{stat}.csv")
        df = read_csv_safe(p)
        if df is None:
            continue
        df["gender"] = g
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_module_B(handmap):
    log("\n" + "=" * 78)
    log("模块 B：发球落点的镜像机制（MCP）")
    log("=" * 78)

    sd = load_mcp_stat("ServeDirection")
    if len(sd) == 0:
        log("  [错误] 未能获取 ServeDirection 数据。")
        return
    sd["player"] = sd["player"].astype(str).str.strip()
    handmap["player"] = handmap["player"].astype(str).str.strip()

    # 用合计行；也保留一发以备后续
    sd_tot = sd[sd["row"].astype(str) == "Total"].copy()
    merged = sd_tot.merge(handmap[["match_id", "player", "hand", "opp_hand", "gender"]],
                          on=["match_id", "player"], how="left")
    join_rate = merged["hand"].notna().mean()
    log(f"  ServeDirection 与惯用手表 join 命中率: {join_rate:.3f} "
        f"(行数 {len(merged):,})")

    # 诊断信息
    diag = {
        "serve_total_rows": len(sd_tot),
        "join_rate": round(float(join_rate), 4),
        "hand_values": dict(merged["hand"].value_counts(dropna=False).head(8)),
        "opp_hand_values": dict(merged["opp_hand"].value_counts(dropna=False).head(8)),
    }
    pd.DataFrame([{"k": k, "v": str(v)} for k, v in diag.items()]).to_csv(
        f"{OUTDIR}/B_diagnostics.csv", index=False)

    m = merged[merged["hand"].isin(["L", "R"])].copy()
    for c in SERVE_LOCS:
        m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0)

    # 关键分析：只看"对手为右手"的接发（人群主体），比较 L vs R 发球者的落点分布
    def dist_for(df_):
        tot = df_[SERVE_LOCS].sum()
        s = tot.sum()
        share = (tot / s) if s > 0 else tot * np.nan
        return tot, share, int(s)

    rows_out = []
    mirror_rows = []
    for opp in ["R", "L", "ALL"]:
        sub = m if opp == "ALL" else m[m.opp_hand == opp]
        L = sub[sub.hand == "L"]; R = sub[sub.hand == "R"]
        Lt, Ls, Ln = dist_for(L)
        Rt, Rs, Rn = dist_for(R)
        for loc in SERVE_LOCS:
            rows_out.append({"opp_hand": opp, "server_hand": "L", "loc": loc,
                             "count": int(Lt[loc]), "share": Ls[loc], "n_serves": Ln})
            rows_out.append({"opp_hand": opp, "server_hand": "R", "loc": loc,
                             "count": int(Rt[loc]), "share": Rs[loc], "n_serves": Rn})
        # 镜像检验：R 的镜像分布 = 交换 deuce<->ad
        if Ln > 0 and Rn > 0:
            for loc in SERVE_LOCS:
                mirror_loc = MIRROR_MAP[loc]
                L_share = Ls[loc]
                Rmirror_share = Rs[mirror_loc]   # R 在镜像位置的占比
                # 两比例 z 检验：L 在 loc 的占比 vs R 在镜像位置的占比
                c1, n1 = int(Lt[loc]), Ln
                c2, n2 = int(Rt[mirror_loc]), Rn
                pbar = (c1 + c2) / (n1 + n2)
                se = np.sqrt(pbar * (1 - pbar) * (1/n1 + 1/n2)) if pbar not in (0, 1) else np.nan
                z = (L_share - Rmirror_share) / se if se and se > 0 else np.nan
                pval = 2 * (1 - stats.norm.cdf(abs(z))) if z == z else np.nan
                mirror_rows.append({
                    "opp_hand": opp, "L_loc": loc, "R_mirror_loc": mirror_loc,
                    "L_share": L_share, "R_mirror_share": Rmirror_share,
                    "diff_L_minus_Rmirror": L_share - Rmirror_share,
                    "z": z, "p": pval, "nL": n1, "nR": n2})
    pd.DataFrame(rows_out).to_csv(f"{OUTDIR}/B_serve_dist.csv", index=False)
    mdf = pd.DataFrame(mirror_rows)
    mdf.to_csv(f"{OUTDIR}/B_serve_mirror.csv", index=False)

    # 控制台打印：对手为右手时，最关键的几个落点
    log("\n  -- 对手为【右手】接发时，发球落点占比 (server hand) --")
    sub = m[m.opp_hand == "R"]
    L = sub[sub.hand == "L"]; R = sub[sub.hand == "R"]
    Lt, Ls, Ln = dist_for(L); Rt, Rs, Rn = dist_for(R)
    log(f"     (左手发球样本 n={Ln:,} 次; 右手发球样本 n={Rn:,} 次)")
    log(f"     {'location':<14}{'L_share':>10}{'R_share':>10}")
    for loc in SERVE_LOCS:
        log(f"     {loc:<14}{Ls[loc]:>10.4f}{Rs[loc]:>10.4f}")
    log("\n  -- 镜像偏离（对手=右手）：L 在某落点占比 - R 在镜像落点占比 --")
    sel = mdf[(mdf.opp_hand == "R")]
    for _, r in sel.iterrows():
        flag = "  <== 显著" if (r["p"] == r["p"] and r["p"] < 0.05) else ""
        log(f"     L:{r['L_loc']:<12} vs R:{r['R_mirror_loc']:<12} "
            f"Δ={r['diff_L_minus_Rmirror']:+.4f}  z={r['z']:.2f}  p={r['p']:.2e}{flag}")

    # ---- 击球方向（探索性）----
    shot = load_mcp_stat("ShotDirection")
    if len(shot):
        shot["player"] = shot["player"].astype(str).str.strip()
        sh = shot.merge(handmap[["match_id", "player", "hand", "opp_hand"]],
                        on=["match_id", "player"], how="left")
        sh = sh[sh["hand"].isin(["L", "R"])]
        dirs = ["crosscourt", "down_middle", "down_the_line", "inside_out", "inside_in"]
        for c in dirs:
            sh[c] = pd.to_numeric(sh.get(c), errors="coerce").fillna(0)
        out = []
        for wing in ["F", "B"]:
            for opp in ["R", "ALL"]:
                ss = sh[sh["row"].astype(str) == wing]
                if opp != "ALL":
                    ss = ss[ss.opp_hand == opp]
                for hand in ["L", "R"]:
                    g = ss[ss.hand == hand]
                    tot = g[dirs].sum(); s = tot.sum()
                    for d in dirs:
                        out.append({"wing": wing, "opp_hand": opp, "hand": hand,
                                    "dir": d, "count": int(tot[d]),
                                    "share": (tot[d] / s) if s > 0 else np.nan,
                                    "n": int(s)})
        pd.DataFrame(out).to_csv(f"{OUTDIR}/B_shot_dist.csv", index=False)
        log("\n  击球方向分布(正手/反手 x 惯用手 x 对手手)已写入 B_shot_dist.csv（探索性）")


# ============================================================================= 
# MAIN
# ============================================================================= 
def main():
    t0 = time.time()
    log("下载并加载 ATP / WTA 单打巡回赛比赛文件……（首次运行会联网下载，之后走缓存）")
    atp = load_tour_matches(ATP_BASE, "atp")
    wta = load_tour_matches(WTA_BASE, "wta")
    log(f"  ATP 比赛行数: {len(atp):,}   WTA 比赛行数: {len(wta):,}")
    matches = pd.concat([atp, wta], ignore_index=True)

    rows = build_mixed_handed(matches)
    run_module_A(rows)

    log("\n下载并加载 MCP 元数据 + 发球/击球方向……")
    handmap = load_mcp_handmap()
    log(f"  MCP 球员-惯用手长表行数: {len(handmap):,}")
    run_module_B(handmap)

    log("\n" + "=" * 78)

    with open(f"{OUTDIR}/summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(SUMMARY))


if __name__ == "__main__":
    main()
