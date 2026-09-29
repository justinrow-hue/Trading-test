"""Second-round diagnostics and iteration, run after the main study.

IMPORTANT: everything here was designed *after* seeing the out-of-sample
results of run_all.py, so the "OOS" numbers in this file are not a clean
hold-out. They are reported for transparency, not as validation.

Usage:  python -m research.iterate
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import strategies as S
from .engine import BacktestResult, drawdown_series, metrics, run_backtest, sharpe

OUT = D.CACHE.parent.parent / "results"
START = pd.Timestamp("1971-01-04")

px = D.load_prices()
rets = px.pct_change().fillna(0.0)
cash_r = rets["CASH"]
dates = rets.index[rets.index >= START]
SPLIT = dates[int(len(dates) * 0.7)]

b6040 = run_backtest(rets, S.bench_60_40(px), 15, 1, start=START).returns


def stats(r: pd.Series, label: str) -> dict:
    out = {"Variant": label}
    for lab, a, b in [("IS", START, SPLIT - pd.Timedelta(days=1)), ("OOS", SPLIT, None), ("1991+", "1991-01-01", None)]:
        x = r.loc[a:b]
        bx = b6040.loc[x.index[0]:x.index[-1]]
        out[f"{lab} CAGR"] = (1 + x).prod() ** (252 / len(x)) - 1
        out[f"{lab} Sharpe"] = sharpe(x, cash_r)
        out[f"{lab} MaxDD"] = float(drawdown_series(x).min())
        out[f"{lab} 60/40 Sharpe"] = sharpe(bx, cash_r)
        out[f"{lab} 60/40 CAGR"] = (1 + bx).prod() ** (252 / len(bx)) - 1
    return out


rows = []

# 1. Leave-one-asset-out for the multi-asset strategies
for name, fn, kw in [("S1", S.s1_tsmom, {}), ("S2", S.s2_dual_momentum, {}), ("S4", S.s4_risk_parity, {})]:
    for drop in [None] + D.RISKY:
        uni = [a for a in D.RISKY if a != drop]
        r = run_backtest(rets, fn(px, universe=uni, **kw), 15, 1, start=START).returns
        rows.append({"Test": "leave-one-out", **stats(r, f"{name} without {drop or '(none)'}")})

# 2. Gross vs net diagnostics for the high-turnover equity strategies
for label, t, lag, cost in [
    ("S5 net, next-close execution (as tested)", S.s5_mean_reversion(px), 1, 15),
    ("S5 gross (0 bps), next-close execution", S.s5_mean_reversion(px), 1, 0),
    ("S5 gross (0 bps), same-close MOC execution", S.s5_mean_reversion(px), 0, 0),
    ("S5 net 2 bps, same-close MOC execution", S.s5_mean_reversion(px), 0, 2),
    ("S7 net (as tested)", S.s7_turn_of_month(px), 0, 15),
    ("S7 gross (0 bps)", S.s7_turn_of_month(px), 0, 0),
    ("S7 net 2 bps", S.s7_turn_of_month(px), 0, 2),
]:
    r = run_backtest(rets, t, cost, lag, start=START).returns
    rows.append({"Test": "gross-vs-net", **stats(r, label)})

# 3. Post-hoc iterations (contaminated OOS; see module docstring)
s1 = run_backtest(rets, S.s1_tsmom(px), 15, 1, start=START).returns
s3 = run_backtest(rets, S.s3_vol_managed(px), 15, 1, band=0.10, start=START).returns
i1 = run_backtest(rets, S.s1_tsmom(px, inv_vol=False), 15, 1, start=START).returns
rows.append({"Test": "iteration", **stats(i1, "I1 Trend, equal-weight sleeves (Faber GTAA)")})
rows.append({"Test": "iteration", **stats(0.5 * s1 + 0.5 * s3, "I2 50% S1 Trend + 50% S3 Vol-managed")})
rows.append({"Test": "iteration", **stats(0.5 * s1 + 0.5 * b6040, "I3 50% S1 Trend + 50% 60/40")})
rows.append({"Test": "iteration", **stats(0.5 * i1 + 0.5 * s3, "I4 50% I1 + 50% S3")})

df = pd.DataFrame(rows)
df.to_csv(OUT / "iteration_and_diagnostics.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
cols = ["Variant", "IS CAGR", "IS Sharpe", "IS 60/40 Sharpe", "OOS CAGR", "OOS Sharpe", "OOS MaxDD",
        "OOS 60/40 Sharpe", "OOS 60/40 CAGR", "1991+ Sharpe", "1991+ 60/40 Sharpe", "1991+ MaxDD"]
print(df[cols].round(3).to_string())
