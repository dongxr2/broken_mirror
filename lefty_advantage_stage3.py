#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, warnings, urllib.request, re
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

MCP = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master"
CACHE = "./_tennis_cache"; OUTDIR = "./tennis_lefty_outputs"
os.makedirs(CACHE, exist_ok=True); os.makedirs(OUTDIR, exist_ok=True)

S = []
def log(m=""):
    print(m); S.append(str(m))

def fetch(url, fname):
    path = os.path.join(CACHE, fname)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            open(path, "wb").write(r.read())
        return path
    except Exception as e:
        print("  [warn] download failed:", fname, e); return None

DIR_MAP = {"4": "wide", "5": "middle", "6": "t"}
# 对右手接发者：反手角 = deuce_t + ad_wide；正手角 = deuce_wide + ad_t
BH_ZONES = {"deuce_t", "ad_wide"}
FH_ZONES = {"deuce_wide", "ad_t"}
SCORE_MAP = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "ad": 4}


def build_meta():
    """match_id -> dict(p1,p2 名字与手别, gender, year)."""
    meta = {}
    for g in ["m", "w"]:
        p = fetch(f"{MCP}/charting-{g}-matches.csv", f"charting-{g}-matches.csv")
        if p is None: continue
        df = pd.read_csv(p, low_memory=False)
        for _, r in df.iterrows():
            mid = r["match_id"]
            yr = None
            m = re.match(r"^(\d{8})", str(mid))
            if m:
                yr = int(m.group(1)[:4])
            meta[mid] = {
                "p1": str(r.get("Player 1", "")).strip(),
                "p2": str(r.get("Player 2", "")).strip(),
                "h1": str(r.get("Pl 1 hand", "")).strip().upper(),
                "h2": str(r.get("Pl 2 hand", "")).strip().upper(),
                "gender": g, "year": yr,
            }
    return meta


def load_points():
    frames = []
    for g in ["m", "w"]:
        for era in ["to-2009", "2010s", "2020s"]:
            fn = f"charting-{g}-points-{era}.csv"
            p = fetch(f"{MCP}/{fn}", fn)
            if p is None: continue
            frames.append(pd.read_csv(p, low_memory=False))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def court_from_score(pts_str):
    """由比分串(如 '30-40')判定 deuce/ad：已打分数之和的奇偶。失败返回 None。"""
    if pd.isna(pts_str): return None
    s = str(pts_str).strip()
    if "-" not in s: return None
    a, b = s.split("-", 1)
    a, b = a.strip(), b.strip()
    if a not in SCORE_MAP or b not in SCORE_MAP: return None
    total = SCORE_MAP[a] + SCORE_MAP[b]
    return "deuce" if total % 2 == 0 else "ad"


def in_play_serve(first, second):
    f = "" if pd.isna(first) else str(first).strip()
    s = "" if pd.isna(second) else str(second).strip()
    if s not in ("", "nan"): return s, False
    return f, True


def main():
    t0 = time.time()
    log("加载元数据与逐分文件……（复用缓存）")
    meta = build_meta()
    pts = load_points()
    log(f"  逐分行数: {len(pts):,}  有元数据比赛: {len(meta):,}")
    if len(pts) == 0:
        log("  [错误] 无逐分数据"); _w(); return

    for c in ["match_id", "Pt", "Gm#", "Svr", "1st", "2nd", "PtWinner", "Pts"]:
        if c not in pts.columns:
            log(f"  [错误] 缺列 {c}; 实际: {list(pts.columns)}"); _w(); return

    pts = pts.dropna(subset=["match_id", "Gm#", "Svr", "PtWinner"]).copy()
    pts["Svr"] = pd.to_numeric(pts["Svr"], errors="coerce")
    pts["PtWinner"] = pd.to_numeric(pts["PtWinner"], errors="coerce")
    pts = pts.dropna(subset=["Svr", "PtWinner"])
    pts["Svr"] = pts["Svr"].astype(int); pts["PtWinner"] = pts["PtWinner"].astype(int)
    pts["Pt"] = pd.to_numeric(pts["Pt"], errors="coerce")

    # --- 球场判定：双重校验 ---
    pts = pts.sort_values(["match_id", "Gm#", "Pt"])
    pts["k"] = pts.groupby(["match_id", "Gm#"]).cumcount()
    pts["court_seq"] = np.where(pts["k"] % 2 == 0, "deuce", "ad")
    pts["court_sco"] = pts["Pts"].map(court_from_score)
    both = pts["court_sco"].notna()
    agree = (pts.loc[both, "court_sco"] == pts.loc[both, "court_seq"]).mean()
    pts["court"] = pts["court_sco"].where(pts["court_sco"].notna(), pts["court_seq"])
    log(f"  球场判定一致率(比分 vs 序号): {agree:.4f}  (比分可解析占比 {both.mean():.3f})")

    # --- 发球落区 / 一二发 / 直接得分 ---
    sv = pts.apply(lambda r: in_play_serve(r["1st"], r["2nd"]), axis=1)
    pts["serve_str"] = [x[0] for x in sv]
    pts["is_first"] = [x[1] for x in sv]
    pts["dir"] = pts["serve_str"].str[:1].map(DIR_MAP)
    pts["zone"] = pts["court"] + "_" + pts["dir"]
    ss = pts["serve_str"].fillna("")
    pts["ace_like"] = (ss.str.len() <= 2) & (ss.str[-1:].isin(["*", "#"]))

    # --- 发球手/接发手/球员名/年份 ---
    def gm(mid, key): 
        d = meta.get(mid); return d[key] if d else None
    pts["h1"] = pts["match_id"].map(lambda x: gm(x, "h1"))
    pts["h2"] = pts["match_id"].map(lambda x: gm(x, "h2"))
    pts["p1"] = pts["match_id"].map(lambda x: gm(x, "p1"))
    pts["p2"] = pts["match_id"].map(lambda x: gm(x, "p2"))
    pts["year"] = pts["match_id"].map(lambda x: gm(x, "year"))
    pts["server_hand"] = np.where(pts["Svr"] == 1, pts["h1"], pts["h2"])
    pts["returner_hand"] = np.where(pts["Svr"] == 1, pts["h2"], pts["h1"])
    pts["server_name"] = np.where(pts["Svr"] == 1, pts["p1"], pts["p2"])
    pts["server_won"] = (pts["PtWinner"] == pts["Svr"]).astype(int)
    pts["decade"] = (pd.to_numeric(pts["year"], errors="coerce") // 10 * 10)

    # --- 分析样本：一发, 对手右手, 方向已知, 落在四个角(非 middle) ---
    d = pts[(pts.returner_hand == "R") & (pts.server_hand.isin(["L", "R"])) &
            (pts.is_first) & (pts.zone.isin(BH_ZONES | FH_ZONES))].copy()
    d["target"] = np.where(d.zone.isin(BH_ZONES), "BH", "FH")
    log(f"  分析样本(一发/对手右手/角区): {len(d):,} 分")

    diag = {"court_agree": round(float(agree), 4),
            "score_parsable": round(float(both.mean()), 4),
            "analysis_points": int(len(d)),
            "unique_servers": int(d.server_name.nunique())}
    pd.DataFrame([{"k": k, "v": str(v)} for k, v in diag.items()]).to_csv(
        f"{OUTDIR}/S3_diagnostics.csv", index=False)

    # ============================================================
    # (A) 球员级配对检验
    # ============================================================
    log("\n" + "=" * 70)
    log("(A) 球员级配对检验：每位发球者自己的 (BH得分率 - FH得分率)")
    log("=" * 70)
    g = d.groupby(["server_name", "server_hand"])
    recs = []
    for (name, hand), sub in g:
        bh = sub[sub.target == "BH"]; fh = sub[sub.target == "FH"]
        n_bh, n_fh = len(bh), len(fh)
        if n_bh >= 50 and n_fh >= 50:   # 两类各至少 50 个一发
            wr_bh = bh.server_won.mean(); wr_fh = fh.server_won.mean()
            recs.append({"server": name, "hand": hand,
                         "n_bh": n_bh, "n_fh": n_fh,
                         "wr_bh": wr_bh, "wr_fh": wr_fh,
                         "gap_bh_minus_fh": wr_bh - wr_fh})
    pl = pd.DataFrame(recs)
    pl.to_csv(f"{OUTDIR}/S3_player_level.csv", index=False)
    log(f"  达到样本门槛的发球者: 左手 {sum(pl.hand=='L')} 人, 右手 {sum(pl.hand=='R')} 人")

    test_rows = []
    if sum(pl.hand == "L") >= 5 and sum(pl.hand == "R") >= 5:
        gapsL = pl[pl.hand == "L"]["gap_bh_minus_fh"].values
        gapsR = pl[pl.hand == "R"]["gap_bh_minus_fh"].values
        # 组间：左手差值分布 vs 右手差值分布
        u, pu = stats.mannwhitneyu(gapsL, gapsR, alternative="two-sided")
        t, pt = stats.ttest_ind(gapsL, gapsR, equal_var=False)
        # 组内：各自差值是否显著偏离 0
        t1L, p1L = stats.ttest_1samp(gapsL, 0.0)
        t1R, p1R = stats.ttest_1samp(gapsR, 0.0)
        log(f"  左手球员 BH-FH 差: 均值={gapsL.mean():+.4f} 中位={np.median(gapsL):+.4f} "
            f"(n={len(gapsL)})  vs 0: p={p1L:.2e}")
        log(f"  右手球员 BH-FH 差: 均值={gapsR.mean():+.4f} 中位={np.median(gapsR):+.4f} "
            f"(n={len(gapsR)})  vs 0: p={p1R:.2e}")
        log(f"  组间差异 Mann-Whitney p={pu:.2e}；Welch t p={pt:.2e}")
        test_rows = [{
            "L_mean_gap": gapsL.mean(), "L_median_gap": float(np.median(gapsL)),
            "L_n_players": len(gapsL), "L_vs0_p": p1L,
            "R_mean_gap": gapsR.mean(), "R_median_gap": float(np.median(gapsR)),
            "R_n_players": len(gapsR), "R_vs0_p": p1R,
            "between_mannwhitney_p": pu, "between_welch_p": pt}]
    else:
        log("  [注意] 达标球员太少，无法做稳健组间检验；请看 S3_player_level.csv 原始分布。")
    pd.DataFrame(test_rows).to_csv(f"{OUTDIR}/S3_player_test.csv", index=False)

    # ============================================================
    # (B) 时间趋势
    # ============================================================
    log("\n" + "=" * 70)
    log("(B) 解离指标的年代趋势")
    log("=" * 70)
    trows = []
    for dec in sorted([x for x in d.decade.dropna().unique()]):
        dd = d[d.decade == dec]
        L = dd[dd.server_hand == "L"]; R = dd[dd.server_hand == "R"]
        def wr(df_, tgt): 
            s = df_[df_.target == tgt]; return (s.server_won.mean(), len(s))
        Lbh, nLbh = wr(L, "BH"); Lfh, nLfh = wr(L, "FH")
        Rbh, nRbh = wr(R, "BH"); Rfh, nRfh = wr(R, "FH")
        if min(nLbh, nLfh, nRbh, nRfh) < 100:   # 样本太小跳过
            continue
        trows.append({"decade": int(dec),
                      "L_gap_BH_FH": Lbh - Lfh, "L_BH": Lbh, "L_FH": Lfh,
                      "R_gap_BH_FH": Rbh - Rfh, "R_BH": Rbh, "R_FH": Rfh,
                      "L_minus_R_on_BH": Lbh - Rbh,
                      "nL_bh": nLbh, "nL_fh": nLfh, "nR_bh": nRbh, "nR_fh": nRfh})
    td = pd.DataFrame(trows)
    td.to_csv(f"{OUTDIR}/S3_temporal.csv", index=False)
    if len(td):
        log(f"  {'decade':>7}{'L_gap(BH-FH)':>14}{'R_gap(BH-FH)':>14}{'L-R on BH':>12}")
        for _, r in td.iterrows():
            log(f"  {int(r['decade']):>7}{r['L_gap_BH_FH']:>14.4f}"
                f"{r['R_gap_BH_FH']:>14.4f}{r['L_minus_R_on_BH']:>12.4f}")

    # ============================================================
    # (C) 仅看发球杀伤（ace/直接得分）
    # ============================================================
    log("\n" + "=" * 70)
    log("(C) 各落区直接得分(ace_like)率：把发球质量从整分回合剥离")
    log("=" * 70)
    q = pts[(pts.returner_hand == "R") & (pts.server_hand.isin(["L", "R"])) &
            (pts.is_first) & (pts.zone.isin(BH_ZONES | FH_ZONES))]
    qrows = []
    for hand in ["L", "R"]:
        for zone in sorted(BH_ZONES | FH_ZONES):
            z = q[(q.server_hand == hand) & (q.zone == zone)]
            n = len(z)
            qrows.append({"hand": hand, "zone": zone,
                          "target": "BH" if zone in BH_ZONES else "FH",
                          "n": n, "ace_like_rate": z.ace_like.mean() if n else np.nan})
    qd = pd.DataFrame(qrows)
    qd.to_csv(f"{OUTDIR}/S3_servequality.csv", index=False)
    log(f"  {'hand':>5}{'zone':>14}{'target':>8}{'ace_like':>10}{'n':>9}")
    for _, r in qd.iterrows():
        log(f"  {r['hand']:>5}{r['zone']:>14}{r['target']:>8}"
            f"{r['ace_like_rate']:>10.4f}{int(r['n']):>9}")
    _w()


def _w():
    open(f"{OUTDIR}/summary_stage3.txt", "w", encoding="utf-8").write("\n".join(S))


if __name__ == "__main__":
    main()
