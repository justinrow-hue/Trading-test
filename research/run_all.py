"""End-to-end research run: backtests, validation, robustness, charts.

Usage:  python -m research.run_all
Writes CSV/Markdown tables to results/ and PNG charts to results/charts/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import data as D
from . import strategies as S
from .engine import (
    BacktestResult, block_bootstrap_sharpe_diff, deflated_sharpe, drawdown_series,
    metrics, run_backtest, sharpe,
)

OUT = Path(__file__).resolve().parent.parent / "results"
CH = OUT / "charts"
OUT.mkdir(exist_ok=True)
CH.mkdir(exist_ok=True)

BASE_COST = 15.0          # 5 bps commission + 10 bps slippage per side
START = pd.Timestamp("1971-01-04")
S6_END = pd.Timestamp("2023-12-29")  # Shiller earnings end 2023-06 (+6m lag)
OBSERVED_START = pd.Timestamp("2008-07-01")  # all seven series are live ETF data

# ---------------------------------------------------------------------------
px = D.load_prices()
rets = px.pct_change().fillna(0.0)
cash_r = rets["CASH"]
ecy = S.excess_cape_yield(D.load_shiller(), D.load_y10(), D.load_cpi_monthly(), earn_lag=6)

dates = rets.index[rets.index >= START]
SPLIT = dates[int(len(dates) * 0.7)]
END = dates[-1]


def targets(key: str, params: dict, prices: pd.DataFrame = px, eq: str = "US_EQ") -> pd.DataFrame:
    p = prices
    if eq != "US_EQ":  # asset-substitution test: swap the equity series
        p = prices.copy()
        p["US_EQ"] = prices[eq]
    fn = S.REGISTRY[key]["fn"]
    if fn == "s1":
        return S.s1_tsmom(p, **params)
    if fn == "s2":
        return S.s2_dual_momentum(p, **params)
    if fn == "s3":
        return S.s3_vol_managed(p, **{k: v for k, v in params.items() if k != "band"})
    if fn == "s4":
        return S.s4_risk_parity(p, **params)
    if fn == "s5":
        return S.s5_mean_reversion(p, **params)
    if fn == "s6":
        return S.s6_valuation(p, ecy, **params)
    if fn == "s7":
        return S.s7_turn_of_month(p, **params)
    raise KeyError(key)


def run(key, params, cost=BASE_COST, extra_lag=0) -> BacktestResult:
    spec = S.REGISTRY[key]
    band = params.get("band", spec["band"])
    res = run_backtest(rets, targets(key, params), cost_bps=cost,
                       lag=spec["lag"] + extra_lag, band=band, start=START)
    if key.startswith("S6"):
        res = cut(res, None, S6_END)
    return res


def cut(res: BacktestResult, a, b) -> BacktestResult:
    sl = slice(a, b)
    return BacktestResult(res.returns.loc[sl], res.weights.loc[sl], res.turnover.loc[sl], res.costs.loc[sl])


def rp(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else x


# ---------------------------------------------------------------------------
# 1. Benchmarks and default-parameter backtests
# ---------------------------------------------------------------------------
bench = {
    "60/40 (SPY/IEF)": run_backtest(rets, S.bench_60_40(px), BASE_COST, 1, start=START),
    "US Equity B&H (SPY)": run_backtest(rets, S.bench_buy_hold(px.loc[START:]), BASE_COST, 1, start=START),
}
B6040 = bench["60/40 (SPY/IEF)"].returns
BSPY = bench["US Equity B&H (SPY)"].returns

default = {k: run(k, v["default"]) for k, v in S.REGISTRY.items()}


def period_table(results: dict[str, BacktestResult], trade_flags: dict[str, bool]) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        end = res.returns.index[-1]
        for lab, a, b in [("IS", START, SPLIT - pd.Timedelta(days=1)), ("OOS", SPLIT, end), ("Full", START, end)]:
            c = cut(res, a, b)
            m = metrics(c, cash_r, B6040, trade_based=trade_flags.get(name, False))
            m2 = metrics(c, cash_r, BSPY)
            rows.append({"Strategy": name, "Period": lab, "Start": c.returns.index[0].date(),
                         "End": c.returns.index[-1].date(), **m,
                         "Excess CAGR vs SPY": m2["Excess CAGR"]})
    return pd.DataFrame(rows)


flags = {k: v["trade_based"] for k, v in S.REGISTRY.items()}
all_res = {**default, **bench}
perf = period_table(all_res, flags)
perf.to_csv(OUT / "performance_is_oos.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Parameter sensitivity grids (every variant is a "trial" for DSR)
# ---------------------------------------------------------------------------
grid_rows = []
grid_rets: dict[str, dict[str, pd.Series]] = {}
for key, spec in S.REGISTRY.items():
    grid_rets[key] = {}
    variants = spec["grid"]
    if "grid_band" in spec:
        variants = [{**g, "band": b} for g in variants for b in spec["grid_band"]]
    for params in variants:
        res = run(key, params)
        label = json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()})
        grid_rets[key][label] = res.returns
        r = res.returns
        is_r, oos_r = r.loc[:SPLIT - pd.Timedelta(days=1)], r.loc[SPLIT:]
        grid_rows.append({
            "Strategy": key, "Params": label,
            "IS Sharpe": sharpe(is_r, cash_r), "OOS Sharpe": sharpe(oos_r, cash_r),
            "IS CAGR": (1 + is_r).prod() ** (252 / len(is_r)) - 1,
            "OOS CAGR": (1 + oos_r).prod() ** (252 / len(oos_r)) - 1,
            "Full MaxDD": float(drawdown_series(r).min()),
            "Is default": params == spec["default"] or (
                "band" in params and {k: v for k, v in params.items() if k != "band"} == spec["default"]
                and params["band"] == spec["band"]),
        })
grid_df = pd.DataFrame(grid_rows)
grid_df.to_csv(OUT / "parameter_grid.csv", index=False)
N_TRIALS = len(grid_df)

b_is_sh = sharpe(B6040.loc[:SPLIT - pd.Timedelta(days=1)], cash_r)
b_oos_sh = sharpe(B6040.loc[SPLIT:], cash_r)
b_oos_sh_s6 = sharpe(B6040.loc[SPLIT:S6_END], cash_r)
sens = []
for key, g in grid_df.groupby("Strategy", sort=False):
    bo = b_oos_sh_s6 if key.startswith("S6") else b_oos_sh
    rho = spearmanr(g["IS Sharpe"], g["OOS Sharpe"]).statistic if len(g) > 2 else np.nan
    sens.append({
        "Strategy": key, "Variants": len(g),
        "OOS Sharpe min": g["OOS Sharpe"].min(), "OOS Sharpe median": g["OOS Sharpe"].median(),
        "OOS Sharpe max": g["OOS Sharpe"].max(),
        "Share of variants beating 60/40 OOS Sharpe": (g["OOS Sharpe"] > bo).mean(),
        "Share of variants with MaxDD better than -30%": (g["Full MaxDD"] > -0.30).mean(),
        "Spearman(IS rank, OOS rank)": rho,
        "Best-IS variant OOS Sharpe": g.loc[g["IS Sharpe"].idxmax(), "OOS Sharpe"],
    })
sens_df = pd.DataFrame(sens)
sens_df.to_csv(OUT / "sensitivity_summary.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Walk-forward: re-select params every 3y on trailing 15y Sharpe
# ---------------------------------------------------------------------------
WF_START = pd.Timestamp("1986-01-01")
reopt = pd.date_range(WF_START, END, freq="36MS")
wf_rows, wf_series, wf_choices = [], {}, {}
for key, variants in grid_rets.items():
    pieces, chosen = [], []
    for i, t0 in enumerate(reopt):
        t1 = reopt[i + 1] if i + 1 < len(reopt) else END + pd.Timedelta(days=1)
        train = slice(t0 - pd.DateOffset(years=15), t0 - pd.Timedelta(days=1))
        scores = {lab: sharpe(r.loc[train], cash_r) for lab, r in variants.items() if len(r.loc[train]) > 500}
        if not scores:
            continue
        best = max(scores, key=lambda k: -np.inf if np.isnan(scores[k]) else scores[k])
        seg = variants[best].loc[t0:t1 - pd.Timedelta(days=1)]
        if len(seg):
            pieces.append(seg)
            chosen.append((t0.date().isoformat(), best))
    wf = pd.concat(pieces)
    wf_series[key] = wf
    wf_choices[key] = chosen
    d = default[key].returns.loc[wf.index[0]:wf.index[-1]]
    b = B6040.loc[wf.index[0]:wf.index[-1]]
    wf_rows.append({
        "Strategy": key, "Start": wf.index[0].date(), "End": wf.index[-1].date(),
        "WF Sharpe": sharpe(wf, cash_r), "Default Sharpe (same window)": sharpe(d, cash_r),
        "60/40 Sharpe (same window)": sharpe(b, cash_r),
        "WF CAGR": (1 + wf).prod() ** (252 / len(wf)) - 1,
        "60/40 CAGR": (1 + b).prod() ** (252 / len(b)) - 1,
        "WF MaxDD": float(drawdown_series(wf).min()),
        "Distinct params chosen": len({c[1] for c in chosen}), "Re-selections": len(chosen),
    })
wf_df = pd.DataFrame(wf_rows)
wf_df.to_csv(OUT / "walk_forward.csv", index=False)
(OUT / "walk_forward_choices.json").write_text(json.dumps(wf_choices, indent=1))

# ---------------------------------------------------------------------------
# 4. Stress tests: costs x2/x3, one extra bar of delay, observed-ETF era only,
#    substitute global equities for US equities (single-asset strategies)
# ---------------------------------------------------------------------------
stress_rows = []
for key, spec in S.REGISTRY.items():
    base_oos = default[key].returns.loc[SPLIT:]
    bo = B6040.loc[SPLIT:base_oos.index[-1]]
    row = {"Strategy": key, "OOS Sharpe (base)": sharpe(base_oos, cash_r), "60/40 OOS Sharpe": sharpe(bo, cash_r)}
    for mult in (2, 3):
        r = run(key, spec["default"], cost=BASE_COST * mult).returns.loc[SPLIT:]
        row[f"OOS Sharpe @{int(BASE_COST*mult)}bps"] = sharpe(r, cash_r)
        row[f"OOS CAGR @{int(BASE_COST*mult)}bps"] = (1 + r).prod() ** (252 / len(r)) - 1
    r = run(key, spec["default"], extra_lag=1).returns.loc[SPLIT:]
    row["OOS Sharpe +1 bar delay"] = sharpe(r, cash_r)
    obs = default[key].returns.loc[OBSERVED_START:]
    row["Observed-era Sharpe (2008-07+)"] = sharpe(obs, cash_r)
    row["60/40 observed-era Sharpe"] = sharpe(B6040.loc[OBSERVED_START:obs.index[-1]], cash_r)
    if key[:2] in ("S3", "S5", "S7"):
        # global equities (Ken French developed 1990-07+, VT 2008+) in place of US equities
        r_g = rets.copy()
        r_g["US_EQ"] = rets["GL_EQ"]
        res = run_backtest(r_g, targets(key, spec["default"], px, eq="GL_EQ"), BASE_COST, spec["lag"],
                           spec["band"], start=pd.Timestamp("1991-01-02"))
        bh = run_backtest(r_g, S.bench_buy_hold(px.loc["1991-01-02":]), BASE_COST, 1, start=pd.Timestamp("1991-01-02"))
        row["Global-equity version Sharpe (1991+)"] = sharpe(res.returns, cash_r)
        row["Global-equity B&H Sharpe (1991+)"] = sharpe(bh.returns, cash_r)
        row["Global-equity version MaxDD"] = float(drawdown_series(res.returns).min())
        row["Global-equity B&H MaxDD"] = float(drawdown_series(bh.returns).min())
    stress_rows.append(row)
stress_df = pd.DataFrame(stress_rows)
stress_df.to_csv(OUT / "stress_tests.csv", index=False)

# ---------------------------------------------------------------------------
# 5. Regime analysis (ex-post labels; for description, not for trading)
# ---------------------------------------------------------------------------
eq_nav = (1 + rets["US_EQ"]).cumprod()
# bear market = peak-to-trough spans of drawdowns deeper than 20%
dd = eq_nav / eq_nav.cummax() - 1
bear = pd.Series(False, index=dd.index)
peak_date, in_dd = None, False
trough_val, trough_date = 0.0, None
for d, v in dd.items():
    if v == 0:
        if in_dd and trough_val <= -0.20:
            bear.loc[peak_date:trough_date] = True
        peak_date, in_dd, trough_val, trough_date = d, False, 0.0, None
    else:
        in_dd = True
        if v < trough_val:
            trough_val, trough_date = v, d
if in_dd and trough_val <= -0.20:
    bear.loc[peak_date:trough_date] = True
rv21 = rets["US_EQ"].rolling(21).std() * np.sqrt(252)
y10 = D.load_y10()
y10d = y10.reindex(rets.index, method="ffill")
rising = (y10d - y10d.shift(252)) > 0
cpi = D.load_cpi_monthly()
cpi_yoy = (cpi / cpi.shift(12) - 1).reindex(rets.index, method="ffill")
regimes = {
    "Bull (not in >20% drawdown)": ~bear,
    "Bear (peak-to-trough, >20% DD)": bear,
    "High vol (21d RV > 25%)": rv21 > 0.25,
    "Low/normal vol": rv21 <= 0.25,
    "Rising 10y yield (vs 12m ago)": rising,
    "Falling 10y yield": ~rising,
    "High inflation (CPI YoY > 4%)": cpi_yoy > 0.04,
}
reg_rows = []
series_all = {**{k: v.returns for k, v in default.items()}, "60/40": B6040, "SPY B&H": BSPY}
for rname, mask in regimes.items():
    for sname, r in series_all.items():
        m = mask.reindex(r.index).fillna(False).astype(bool)
        x = r[m]
        reg_rows.append({"Regime": rname, "Strategy": sname, "Days": int(m.sum()),
                         "Ann. return": x.mean() * 252, "Sharpe": sharpe(x, cash_r)})
reg_df = pd.DataFrame(reg_rows)
reg_df.to_csv(OUT / "regimes.csv", index=False)

EPISODES = {
    "1973-74 stagflation bear": ("1973-01-11", "1974-10-03"),
    "1987 crash": ("1987-08-25", "1987-12-04"),
    "1994 bond rout": ("1994-01-31", "1994-11-07"),
    "2000-02 dot-com bust": ("2000-03-24", "2002-10-09"),
    "2008 GFC": ("2007-10-09", "2009-03-09"),
    "2020 COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 stock-bond drawdown": ("2022-01-03", "2022-10-12"),
    "2025 tariff shock": ("2025-02-19", "2025-04-08"),
}
ep_rows = []
for ename, (a, b) in EPISODES.items():
    row = {"Episode": ename}
    for sname, r in series_all.items():
        x = r.loc[a:b]
        row[sname] = (1 + x).prod() - 1 if len(x) > 5 else np.nan
    ep_rows.append(row)
ep_df = pd.DataFrame(ep_rows)
ep_df.to_csv(OUT / "stress_episodes.csv", index=False)

# per-decade Sharpe vs 60/40 (consistency across time)
dec_rows = []
for sname, r in series_all.items():
    for dec in range(1970, 2030, 10):
        x = r.loc[f"{dec}":f"{dec + 9}"]
        if len(x) > 500:
            dec_rows.append({"Strategy": sname, "Decade": f"{dec}s", "Sharpe": sharpe(x, cash_r)})
dec_df = pd.DataFrame(dec_rows).pivot(index="Strategy", columns="Decade", values="Sharpe")
dec_df.to_csv(OUT / "decade_sharpe.csv")

# ---------------------------------------------------------------------------
# 6. Correlations (weekly, to avoid asynchronous-close noise)
# ---------------------------------------------------------------------------
wk = pd.DataFrame(series_all).loc[SPLIT:END]
wk = (1 + wk).resample("W-FRI").prod() - 1
corr_oos = wk.corr()
corr_oos.to_csv(OUT / "correlation_oos_weekly.csv")
wk_full = (1 + pd.DataFrame(series_all)).resample("W-FRI").prod() - 1
wk_full.corr().to_csv(OUT / "correlation_full_weekly.csv")

# ---------------------------------------------------------------------------
# 7. Data-snooping diagnostics: deflated Sharpe + bootstrap vs 60/40
# ---------------------------------------------------------------------------
daily_sr = []
for key, variants in grid_rets.items():
    for lab, r in variants.items():
        ex = r - cash_r.reindex(r.index)
        daily_sr.append(ex.mean() / ex.std())
sr_var = float(np.var(daily_sr))
snoop_rows = []
for key in S.REGISTRY:
    r = default[key].returns
    ex_full = r - cash_r.reindex(r.index)
    oos = r.loc[SPLIT:]
    ex_oos = oos - cash_r.reindex(oos.index)
    lo, hi, p = block_bootstrap_sharpe_diff(oos, B6040, cash_r)
    snoop_rows.append({
        "Strategy": key,
        "DSR full (N=%d trials)" % N_TRIALS: deflated_sharpe(ex_full, N_TRIALS, sr_var),
        "DSR OOS": deflated_sharpe(ex_oos, N_TRIALS, sr_var),
        "OOS Sharpe - 60/40 Sharpe, 90% CI low": lo, "90% CI high": hi,
        "Bootstrap P(diff <= 0)": p,
    })
snoop_df = pd.DataFrame(snoop_rows)
snoop_df.to_csv(OUT / "data_snooping.csv", index=False)

# ---------------------------------------------------------------------------
# 8. Qualification against the pre-registered criteria
# ---------------------------------------------------------------------------
def pget(name, per, col):
    return float(perf[(perf.Strategy == name) & (perf.Period == per)][col].iloc[0])


def bench_oos_for(name, col):
    end = default[name].returns.index[-1]
    c = cut(bench["60/40 (SPY/IEF)"], SPLIT, end)
    return metrics(c, cash_r)[col]


qual_rows = []
single_asset = {"S3", "S5", "S6", "S7"}
for key in S.REGISTRY:
    oos_sh, oos_cagr = pget(key, "OOS", "Sharpe"), pget(key, "OOS", "CAGR")
    b_sh, b_cagr = bench_oos_for(key, "Sharpe"), bench_oos_for(key, "CAGR")
    is_sh = pget(key, "IS", "Sharpe")
    full_dd = pget(key, "Full", "Max DD")
    srow = sens_df[sens_df.Strategy == key].iloc[0]
    st = stress_df[stress_df.Strategy == key].iloc[0]
    dec = dec_df.loc[key].dropna()
    decb = dec_df.loc["60/40"].reindex(dec.index)
    crit = {
        "Beats 60/40 OOS (Sharpe AND CAGR)": oos_sh > b_sh and oos_cagr >= b_cagr,
        "Beats 60/40 OOS on Sharpe only": oos_sh > b_sh,
        "OOS Sharpe > 1.0": oos_sh > 1.0,
        "Full-period MaxDD < 30%": full_dd > -0.30,
        "Beats 60/40 Sharpe in IS too": is_sh > b_is_sh,
        "Beats 60/40 Sharpe in >= 4 of 6 decades": int((dec > decb).sum()) >= 4,
        "Median grid variant beats 60/40 OOS Sharpe": srow["OOS Sharpe median"] > (b_oos_sh_s6 if key.startswith("S6") else b_oos_sh),
        "Not single-asset": key[:2] not in single_asset or (
            "Global-equity version Sharpe (1991+)" in st and st["Global-equity version Sharpe (1991+)"] > st["Global-equity B&H Sharpe (1991+)"]),
        "Survives 2x costs (OOS Sharpe > 60/40)": st["OOS Sharpe @30bps"] > b_sh,
    }
    qual_rows.append({"Strategy": key, **crit, "Score (of 9)": sum(crit.values()),
                      "HIGH QUALITY (all hard criteria)": all(crit[c] for c in [
                          "Beats 60/40 OOS (Sharpe AND CAGR)", "Full-period MaxDD < 30%",
                          "Beats 60/40 Sharpe in IS too", "Median grid variant beats 60/40 OOS Sharpe",
                          "Not single-asset", "Survives 2x costs (OOS Sharpe > 60/40)"])})
qual_df = pd.DataFrame(qual_rows)
qual_df.to_csv(OUT / "qualification.csv", index=False)

# ---------------------------------------------------------------------------
# 9. Current market snapshot and live signals
# ---------------------------------------------------------------------------
snap = []
last = px.index[-1]
for a in px.columns:
    p = px[a]
    r = rets[a]
    snap.append({
        "Asset": a, "Proxy": D.ASSETS[a][1],
        "1m": p.iloc[-1] / p.iloc[-22] - 1, "3m": p.iloc[-1] / p.iloc[-64] - 1,
        "12m": p.iloc[-1] / p.iloc[-253] - 1,
        "vs 200d SMA": p.iloc[-1] / p.iloc[-200:].mean() - 1,
        "63d vol": r.iloc[-63:].std() * np.sqrt(252),
        "10y vol": r.iloc[-2520:].std() * np.sqrt(252),
        "Drawdown from ATH": p.iloc[-1] / p.max() - 1,
    })
snap_df = pd.DataFrame(snap)
snap_df.to_csv(OUT / "market_snapshot.csv", index=False)
vix = D.load_vix()
macro = {
    "as_of_prices": str(last.date()),
    "VIX": float(vix.iloc[-1]), "VIX date": str(vix.index[-1].date()),
    "VIX percentile since 1990": float((vix < vix.iloc[-1]).mean()),
    "US 10y yield (monthly avg)": float(y10.iloc[-1]), "10y date": str(y10.index[-1].date()),
    "10y 12m change (pp)": float(y10.iloc[-1] - y10.iloc[-13]),
    "CPI YoY": float(cpi.iloc[-1] / cpi.iloc[-13] - 1), "CPI date": str(cpi.index[-1].date()),
    "ECY latest": float(ecy.iloc[-1]), "ECY date": str(ecy.index[-1]),
    "ECY percentile": float((ecy < ecy.iloc[-1]).mean()),
    "US_EQ 21d realised vol": float(rv21.iloc[-1]),
    "Split date": str(SPLIT.date()), "Start": str(START.date()), "End": str(END.date()),
    "N trials": N_TRIALS,
}
live = {}
for key, spec in S.REGISTRY.items():
    t = targets(key, spec["default"])
    row = t.iloc[-1]
    live[key] = {"decision_date": str(t.index[-1].date()), **{k: round(float(v), 3) for k, v in row.items() if abs(v) > 1e-9}}
macro["live_targets"] = live
(OUT / "snapshot.json").write_text(json.dumps(macro, indent=1, default=str))

# ---------------------------------------------------------------------------
# 10. Charts
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titlecolor": INK, "legend.frameon": False,
})

names = list(S.REGISTRY)


def small_multiples(fn, fname, title):
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True)
    for ax, key in zip(axes.flat, names + ["__bench__"]):
        fn(ax, key)
        ax.axvline(SPLIT, color=MUTED, lw=1, ls="--")
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(CH / fname, dpi=130)
    plt.close(fig)


def eq_panel(ax, key):
    if key == "__bench__":
        ax.plot((1 + BSPY).cumprod(), color=ORANGE, lw=1.5, label="SPY B&H")
        ax.plot((1 + B6040).cumprod(), color=MUTED, lw=1.5, label="60/40")
        ax.set_title("Benchmarks")
    else:
        r = default[key].returns
        ax.plot((1 + B6040.loc[r.index[0]:r.index[-1]]).cumprod(), color=MUTED, lw=1.5, label="60/40")
        ax.plot((1 + r).cumprod(), color=BLUE, lw=1.5, label=key.split(" ", 1)[0])
        ax.set_title(key)
    ax.set_yscale("log")
    ax.legend(loc="upper left")


def dd_panel(ax, key):
    if key == "__bench__":
        ax.plot(drawdown_series(BSPY), color=ORANGE, lw=1.2, label="SPY B&H")
        ax.plot(drawdown_series(B6040), color=MUTED, lw=1.2, label="60/40")
        ax.set_title("Benchmarks")
    else:
        r = default[key].returns
        ax.plot(drawdown_series(B6040.loc[r.index[0]:r.index[-1]]), color=MUTED, lw=1.2, label="60/40")
        ax.plot(drawdown_series(r), color=BLUE, lw=1.2, label=key.split(" ", 1)[0])
        ax.set_title(key)
    ax.axhline(-0.30, color="#d03b3b", lw=0.8, ls=":")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc="lower left")


small_multiples(eq_panel, "equity_curves.png",
                "Growth of $1, net of 15 bps/side costs (log scale). Dashed line = IS/OOS split")
small_multiples(dd_panel, "drawdowns.png", "Drawdowns vs 60/40. Dotted line = -30% limit")

fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
for ax, key in zip(axes.flat, names):
    g = grid_df[grid_df.Strategy == key]
    bo = b_oos_sh_s6 if key.startswith("S6") else b_oos_sh
    ax.scatter(g["IS Sharpe"], g["OOS Sharpe"], s=28, color=BLUE, edgecolor=SURF, linewidth=1.5, zorder=3)
    d = g[g["Is default"]]
    ax.scatter(d["IS Sharpe"], d["OOS Sharpe"], s=60, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=4)
    ax.axhline(bo, color=MUTED, ls="--", lw=1)
    ax.axvline(b_is_sh, color=MUTED, ls=":", lw=1)
    ax.set_title(key, fontsize=9)
    ax.set_xlabel("In-sample Sharpe")
    ax.set_ylabel("Out-of-sample Sharpe")
ax = axes.flat[-1]
ax.axis("off")
ax.text(0, 0.6, "Each dot = one parameter variant\nRing = a-priori default\nDashed = 60/40 OOS Sharpe\nDotted = 60/40 IS Sharpe",
        color=INK2, fontsize=9, va="top")
fig.suptitle(f"Parameter sensitivity: {N_TRIALS} variants, in-sample vs out-of-sample Sharpe", color=INK, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(CH / "sensitivity.png", dpi=130)
plt.close(fig)

cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#f0efec", "#e34948"])
fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(corr_oos.values, cmap=cmap, vmin=-1, vmax=1)
lab = [c.split(" (")[0] for c in corr_oos.columns]
ax.set_xticks(range(len(lab)), lab, rotation=45, ha="right")
ax.set_yticks(range(len(lab)), lab)
ax.grid(False)
for i in range(len(lab)):
    for j in range(len(lab)):
        ax.text(j, i, f"{corr_oos.values[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color=INK)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title(f"Weekly return correlation, out-of-sample ({SPLIT.date()} to {END.date()})", loc="left")
fig.tight_layout()
fig.savefig(CH / "correlation_oos.png", dpi=130)
plt.close(fig)

print("SPLIT", SPLIT.date(), "END", END.date(), "N_TRIALS", N_TRIALS)
print("done")
