#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, warnings, urllib.request
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

MCP = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master"
CACHE = "./_tennis_cache"
OUTDIR = "./tennis_lefty_outputs"
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
            data = r.read()
        open(path, "wb").write(data); return path
    except Exception as e:
        print("  [warn] download failed:", fname, e); return None

DIR_MAP = {"4": "wide", "5": "middle", "6": "t"}     # MCP 发球方向码
# 对【右手接发者】而言：反手角 = deuce_t + ad_wide；正手角 = deuce_wide + ad_t
BACKHAND_ZONES = {"deuce_t", "ad_wide"}
FOREHAND_ZONES = {"deuce_wide", "ad_t"}


def build_match_map():
    """match_id -> (p1hand, p2hand)；同时记录性别。"""
    mm = {}
    for g in ["m", "w"]:
        p = fetch(f"{MCP}/charting-{g}-matches.csv", f"charting-{g}-matches.csv")
        if p is None: continue
        df = pd.read_csv(p, low_memory=False)
        for _, r in df.iterrows():
            h1 = str(r.get("Pl 1 hand", "")).strip().upper()
            h2 = str(r.get("Pl 2 hand", "")).strip().upper()
            mm[r["match_id"]] = (h1, h2, g)
    return mm


def load_points():
    frames = []
    for g in ["m", "w"]:
        for era in ["to-2009", "2010s", "2020s"]:
            fn = f"charting-{g}-points-{era}.csv"
            p = fetch(f"{MCP}/{fn}", fn)
            if p is None: continue
            df = pd.read_csv(p, low_memory=False)
            df["gender"] = g
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def in_play_serve(first, second):
    """返回 (投入比赛的发球串, is_first)。有二发串=一发失误。"""
    f = "" if pd.isna(first) else str(first).strip()
    s = "" if pd.isna(second) else str(second).strip()
    if s not in ("", "nan"):
        return s, False
    return f, True


def main():
    t0 = time.time()
    log("加载 MCP 比赛元数据与逐分文件……（首次联网，之后走缓存）")
    mm = build_match_map()
    pts = load_points()
    log(f"  逐分总行数: {len(pts):,}   有惯用手映射的比赛数: {len(mm):,}")
    if len(pts) == 0:
        log("  [错误] 没有逐分数据。"); _write(); return

    # 必需列
    for c in ["match_id", "Pt", "Gm#", "Svr", "1st", "2nd", "PtWinner"]:
        if c not in pts.columns:
            log(f"  [错误] 缺列 {c}，实际列: {list(pts.columns)}"); _write(); return

    pts = pts.dropna(subset=["match_id", "Gm#", "Svr", "PtWinner"]).copy()
    pts["Svr"] = pd.to_numeric(pts["Svr"], errors="coerce")
    pts["PtWinner"] = pd.to_numeric(pts["PtWinner"], errors="coerce")
    pts = pts.dropna(subset=["Svr", "PtWinner"])
    pts["Svr"] = pts["Svr"].astype(int); pts["PtWinner"] = pts["PtWinner"].astype(int)
    pts["Pt"] = pd.to_numeric(pts["Pt"], errors="coerce")

    # 1) 发球局内的发球次序 -> 推出 deuce/ad 球场（局内第 k 分，k 偶=deuce）
    pts = pts.sort_values(["match_id", "Gm#", "Pt"])
    pts["k_in_game"] = pts.groupby(["match_id", "Gm#"]).cumcount()
    pts["court"] = np.where(pts["k_in_game"] % 2 == 0, "deuce", "ad")

    # 2) 投入比赛的发球 + 方向码 + 落区
    sv = pts.apply(lambda r: in_play_serve(r["1st"], r["2nd"]), axis=1)
    pts["serve_str"] = [x[0] for x in sv]
    pts["is_first"] = [x[1] for x in sv]
    pts["dir_code"] = pts["serve_str"].str[:1]
    pts["dir"] = pts["dir_code"].map(DIR_MAP)             # wide/middle/t 或 NaN(未知)
    pts["zone"] = pts["court"] + "_" + pts["dir"]
    # ACE/直接得分 近似：发球串很短且以 * 或 # 结尾
    ss = pts["serve_str"].fillna("")
    pts["ace_like"] = (ss.str.len() <= 2) & (ss.str[-1:].isin(["*", "#"]))

    # 3) 发球手 / 接发手（来自元数据；Svr=1 => Player1 发球）
    h1 = pts["match_id"].map(lambda x: mm.get(x, ("", "", ""))[0])
    h2 = pts["match_id"].map(lambda x: mm.get(x, ("", "", ""))[1])
    pts["gender_meta"] = pts["match_id"].map(lambda x: mm.get(x, ("", "", ""))[2])
    pts["server_hand"] = np.where(pts["Svr"] == 1, h1, h2)
    pts["returner_hand"] = np.where(pts["Svr"] == 1, h2, h1)
    pts["server_won"] = (pts["PtWinner"] == pts["Svr"]).astype(int)

    # 诊断
    n_total = len(pts)
    n_dir_known = pts["dir"].notna().sum()
    diag = {
        "points_total": int(n_total),
        "dir_known_rate": round(float(n_dir_known / n_total), 4),
        "server_hand_LR_rate": round(float(pts["server_hand"].isin(["L", "R"]).mean()), 4),
        "dir_code_dist": dict(pts["dir_code"].value_counts(dropna=False).head(8)),
    }
    pd.DataFrame([{"k": k, "v": str(v)} for k, v in diag.items()]).to_csv(
        f"{OUTDIR}/S2_diagnostics.csv", index=False)
    log(f"  方向可识别率: {diag['dir_known_rate']:.3f}; "
        f"发球手为L/R比例: {diag['server_hand_LR_rate']:.3f}")

    # 4) 分析样本：对手=右手，发球手=L/R，方向已知
    d = pts[(pts["returner_hand"] == "R") &
            (pts["server_hand"].isin(["L", "R"])) &
            (pts["dir"].notna())].copy()
    log(f"  分析样本(对手为右手, 方向已知): {len(d):,} 分")
    log(f"  唯一发球手数: L={d[d.server_hand=='L']['match_id'].nunique()} 场来源, "
        f"R={d[d.server_hand=='R']['match_id'].nunique()} 场来源")

    # 5) 各落区效果表（按 发球手 x 一二发 x 落区）
    rows = []
    for hand in ["L", "R"]:
        for first in [True, False]:
            g = d[(d.server_hand == hand) & (d.is_first == first)]
            for zone in ["deuce_wide", "deuce_middle", "deuce_t",
                         "ad_wide", "ad_middle", "ad_t"]:
                z = g[g.zone == zone]
                n = len(z)
                rows.append({
                    "server_hand": hand, "serve": "1st" if first else "2nd",
                    "zone": zone, "n": n,
                    "win_rate": z["server_won"].mean() if n else np.nan,
                    "ace_like_rate": z["ace_like"].mean() if n else np.nan,
                })
    eff = pd.DataFrame(rows)
    eff.to_csv(f"{OUTDIR}/S2_zone_effectiveness.csv", index=False)

    # 一发、对手右手，各落区得分率
    log("\n  -- 一发 / 对手为右手：各落区发球得分率 (win_rate) --")
    log(f"     {'zone':<14}{'L_win':>9}{'L_n':>9}{'R_win':>9}{'R_n':>9}")
    for zone in ["deuce_wide", "deuce_middle", "deuce_t", "ad_wide", "ad_middle", "ad_t"]:
        lr = eff[(eff.zone == zone) & (eff.serve == "1st") & (eff.server_hand == "L")].iloc[0]
        rr = eff[(eff.zone == zone) & (eff.serve == "1st") & (eff.server_hand == "R")].iloc[0]
        tag = "  <-反手角" if zone in BACKHAND_ZONES else ""
        log(f"     {zone:<14}{lr['win_rate']:>9.4f}{int(lr['n']):>9}"
            f"{rr['win_rate']:>9.4f}{int(rr['n']):>9}{tag}")

    # 6) 关键两比例检验
    def prop_test(a_win, a_n, b_win, b_n, label):
        if a_n == 0 or b_n == 0:
            return {"contrast": label, "rate_A": np.nan, "rate_B": np.nan,
                    "diff": np.nan, "z": np.nan, "p": np.nan, "nA": a_n, "nB": b_n}
        pa, pb = a_win / a_n, b_win / b_n
        pbar = (a_win + b_win) / (a_n + b_n)
        se = np.sqrt(pbar * (1 - pbar) * (1 / a_n + 1 / b_n))
        z = (pa - pb) / se if se > 0 else np.nan
        p = 2 * (1 - stats.norm.cdf(abs(z))) if z == z else np.nan
        return {"contrast": label, "rate_A": pa, "rate_B": pb, "diff": pa - pb,
                "z": z, "p": p, "nA": int(a_n), "nB": int(b_n)}

    d1 = d[d.is_first]   # 一发
    def wn(hand, zones):
        sub = d1[(d1.server_hand == hand) & (d1.zone.isin(zones))]
        return int(sub["server_won"].sum()), len(sub)

    contrasts = []
    # (i) 左手 ad_wide(攻反手) vs 右手 ad_wide(同格,同样攻反手) —— 左手切削是否更毒
    contrasts.append(prop_test(*wn("L", {"ad_wide"}), *wn("R", {"ad_wide"}),
                               "L_adWide vs R_adWide (same zone)"))
    # (ii) 左手天然武器 ad_wide vs 右手镜像天然武器 deuce_wide
    contrasts.append(prop_test(*wn("L", {"ad_wide"}), *wn("R", {"deuce_wide"}),
                               "L_adWide(natural,->BH) vs R_deuceWide(natural,->FH)"))
    # (iii) 发向右手反手角 vs 正手角：左手
    contrasts.append(prop_test(*wn("L", BACKHAND_ZONES), *wn("L", FOREHAND_ZONES),
                               "L: serve to BACKHAND vs FOREHAND corner"))
    # (iv) 发向右手反手角 vs 正手角：右手
    contrasts.append(prop_test(*wn("R", BACKHAND_ZONES), *wn("R", FOREHAND_ZONES),
                               "R: serve to BACKHAND vs FOREHAND corner"))
    # (v) 攻反手角：左手 vs 右手（总的反手角进攻收益差）
    contrasts.append(prop_test(*wn("L", BACKHAND_ZONES), *wn("R", BACKHAND_ZONES),
                               "to-BACKHAND: L server vs R server"))
    cdf = pd.DataFrame(contrasts)
    cdf.to_csv(f"{OUTDIR}/S2_key_contrasts.csv", index=False)

    log("\n  -- 关键对比（一发, 对手为右手）--")
    for _, r in cdf.iterrows():
        flag = "  <== 显著" if (r["p"] == r["p"] and r["p"] < 0.05) else ""
        log(f"     {r['contrast']}")
        log(f"        A={r['rate_A']:.4f}(n={r['nA']})  B={r['rate_B']:.4f}(n={r['nB']})"
            f"  Δ={r['diff']:+.4f}  z={r['z']:.2f}  p={r['p']:.2e}{flag}")

    _w()


def _w():
    open(f"{OUTDIR}/summary_stage2.txt", "w", encoding="utf-8").write("\n".join(S))

if __name__ == "__main__":
    main()
