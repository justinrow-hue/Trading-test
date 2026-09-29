"""Seven structurally different strategies.

Every function takes daily total-return price levels `px` (plus any auxiliary
data) and returns target weights indexed by *decision date*. Only data up to
the decision date's close is used; the engine then trades at the next close.

Defaults were fixed from the published literature *before* running any
backtest; the grids exist only for sensitivity / walk-forward analysis.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .data import RISKY

M = 21  # sessions per month


def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).max().values)


def _frame(idx, cols) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=idx, columns=cols)


# ---------------------------------------------------------------------------
# S1  Time-series momentum / trend following across asset classes
# ---------------------------------------------------------------------------
def s1_tsmom(px: pd.DataFrame, lookback_m: int = 12, vol_win: int = 60, universe: list | None = None,
             inv_vol: bool = True) -> pd.DataFrame:
    """Hold each asset (inverse-vol weighted) only if its trailing total return
    beats cash over `lookback_m` months; otherwise that sleeve sits in cash.
    Refs: Moskowitz, Ooi & Pedersen (2012); Faber (2007); Hurst et al. (2017).
    inv_vol=False gives equal sleeve weights (Faber-style GTAA)."""
    RISKY = universe or globals()["RISKY"]
    r = px.pct_change()
    me = month_ends(px.index)
    L = lookback_m * M
    mom = px[RISKY] / px[RISKY].shift(L) - 1
    cmom = px["CASH"] / px["CASH"].shift(L) - 1
    sig = mom.sub(cmom, axis=0) > 0
    iv = 1 / r[RISKY].rolling(vol_win).std()
    if not inv_vol:
        iv = iv.notna().astype(float).replace(0.0, np.nan)
    base = iv.div(iv.sum(axis=1), axis=0)
    w = (base * sig).loc[me].dropna(how="all")
    w = w[mom.loc[w.index].notna().all(axis=1)]
    out = _frame(w.index, px.columns)
    out[RISKY] = w
    out["CASH"] = 1 - w.sum(axis=1)
    return out


# ---------------------------------------------------------------------------
# S2  Cross-sectional (relative) + absolute "dual" momentum rotation
# ---------------------------------------------------------------------------
def s2_dual_momentum(px: pd.DataFrame, top_k: int = 2, lookbacks: tuple = (1, 3, 6, 12),
                     universe: list | None = None) -> pd.DataFrame:
    """Rank assets on average trailing return over `lookbacks` (months); hold
    the top_k equally, but only those beating cash on the same score.
    Refs: Jegadeesh & Titman (1993); Asness, Moskowitz & Pedersen (2013);
    Antonacci (2014)."""
    RISKY = universe or globals()["RISKY"]
    me = month_ends(px.index)
    score = sum(px / px.shift(l * M) - 1 for l in lookbacks) / len(lookbacks)
    score = score.loc[me].dropna()
    out = _frame(score.index, px.columns)
    for d, row in score.iterrows():
        rk = row[RISKY].sort_values(ascending=False)
        picks = [a for a in rk.index[:top_k] if rk[a] > row["CASH"]]
        for a in picks:
            out.loc[d, a] = 1 / top_k
        out.loc[d, "CASH"] = 1 - len(picks) / top_k
    return out


# ---------------------------------------------------------------------------
# S3  Volatility-managed US equity (vol targeting, no leverage)
# ---------------------------------------------------------------------------
def s3_vol_managed(px: pd.DataFrame, vol_win: int = 21, target: float = 0.12) -> pd.DataFrame:
    """Equity weight = min(1, target / realised vol); rest in cash. Daily
    decisions; the engine's no-trade band limits turnover.
    Refs: Moreira & Muir (2017); Harvey et al. (2018)."""
    r = px["US_EQ"].pct_change()
    rv = r.rolling(vol_win).std() * np.sqrt(252)
    w = (target / rv).clip(upper=1.0).dropna()
    out = _frame(w.index, px.columns)
    out["US_EQ"] = w
    out["CASH"] = 1 - w
    return out


# ---------------------------------------------------------------------------
# S4  Naive risk parity (inverse volatility), unlevered
# ---------------------------------------------------------------------------
def s4_risk_parity(px: pd.DataFrame, vol_win: int = 60, universe: list | None = None) -> pd.DataFrame:
    """Monthly inverse-vol weights across all six risky assets, no timing.
    Refs: Asness, Frazzini & Pedersen (2012); Frazzini & Pedersen (2014)."""
    RISKY = universe or globals()["RISKY"]
    r = px[RISKY].pct_change()
    me = month_ends(px.index)
    iv = 1 / r.rolling(vol_win).std()
    w = iv.div(iv.sum(axis=1), axis=0).loc[me].dropna()
    out = _frame(w.index, px.columns)
    out[RISKY] = w
    return out


# ---------------------------------------------------------------------------
# S5  Short-term mean reversion in US equity (liquidity provision)
# ---------------------------------------------------------------------------
def rsi(p: pd.Series, n: int = 2) -> pd.Series:
    d = p.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def s5_mean_reversion(
    px: pd.DataFrame, rsi_th: float = 10, trend_filter: bool = True, exit_rule: str = "sma5", max_hold: int = 10
) -> pd.DataFrame:
    """Buy US equity after a sharp 2-day oversold reading (RSI(2) < rsi_th),
    optionally only above the 200-day SMA; exit on bounce or after max_hold.
    Refs: Lo & MacKinlay (1990); Nagel (2012) on reversal as liquidity provision;
    Connors & Alvarez (2008) for the RSI(2) rule."""
    p = px["US_EQ"]
    x = rsi(p, 2)
    sma200 = p.rolling(200).mean()
    sma5 = p.rolling(5).mean()
    pv, xv, s200, s5v = p.values, x.values, sma200.values, sma5.values
    pos = np.zeros(len(p))
    held = 0
    inpos = False
    for k in range(200, len(p)):
        if inpos:
            held += 1
            ex = (pv[k] > s5v[k]) if exit_rule == "sma5" else (xv[k] > 70)
            if ex or held >= max_hold:
                inpos = False
        else:
            if xv[k] < rsi_th and (not trend_filter or pv[k] > s200[k]):
                inpos, held = True, 0
        pos[k] = 1.0 if inpos else 0.0
    w = pd.Series(pos, index=p.index).iloc[200:]
    out = _frame(w.index, px.columns)
    out["US_EQ"] = w
    out["CASH"] = 1 - w
    return out


# ---------------------------------------------------------------------------
# S6  Valuation / equity-risk-premium tilt (Excess CAPE Yield)
# ---------------------------------------------------------------------------
def excess_cape_yield(shiller: pd.DataFrame, y10: pd.Series, cpi: pd.Series, earn_lag: int = 6) -> pd.Series:
    """ECY_m = 1/CAPE_m - (10y yield_m - trailing 10y CPI inflation).

    CAPE uses real earnings lagged `earn_lag` months (reporting delay and
    Shiller's interpolation), so the value at month m is knowable at its end.
    """
    sh = shiller.copy()
    sh.index = sh.index.to_period("M")
    e10 = sh["Real Earnings"].shift(earn_lag).rolling(120, min_periods=120).mean()
    cape = sh["Real Price"] / e10
    # long history of 10y yield: Shiller before 1953-04, dataset series after
    yl = sh["Long Interest Rate"].copy()
    y = y10.copy()
    y.index = y.index.to_period("M")
    yl = y.combine_first(yl)
    c = cpi.copy()
    c.index = c.index.to_period("M")
    c = c.combine_first(sh["Consumer Price Index"])
    infl10 = (c.shift(1) / c.shift(121)) ** (1 / 10) - 1
    ecy = 1 / cape - (yl / 100 - infl10)
    return ecy.dropna()


def s6_valuation(
    px: pd.DataFrame, ecy: pd.Series, lo: float = 0.3, hi: float = 0.9, bond: str = "UST_7_10"
) -> pd.DataFrame:
    """Equity weight scales with the expanding-window percentile of ECY
    (cheap equities vs bonds -> more equity). Refs: Campbell & Shiller (1998);
    Asness (2003); Shiller et al. (2020) on the Excess CAPE Yield."""
    pct = ecy.expanding(min_periods=240).apply(lambda s: (s[:-1] < s[-1]).mean() if len(s) > 1 else np.nan, raw=True)
    me = month_ends(px.index)
    per = me.to_period("M")
    w = pd.Series(pct.reindex(per).values, index=me).dropna()
    weq = lo + (hi - lo) * w
    out = _frame(weq.index, px.columns)
    out["US_EQ"] = weq
    out[bond] = 1 - weq
    return out


# ---------------------------------------------------------------------------
# S7  Turn-of-the-month calendar effect
# ---------------------------------------------------------------------------
def s7_turn_of_month(px: pd.DataFrame, before: int = 1, after: int = 3) -> pd.DataFrame:
    """Hold US equity for the last `before` and first `after` sessions of each
    month, cash otherwise. The exchange calendar is known in advance, so the
    returned weights are *execution-date* weights (use lag=0 in the engine).
    Refs: Ariel (1987); Lakonishok & Smidt (1988); McConnell & Xu (2008)."""
    idx = px.index
    s = pd.Series(1, index=idx)
    g = s.groupby([idx.year, idx.month])
    fwd = g.cumsum()                 # 1 = first session of month
    n_in_month = g.transform("sum")
    bwd = n_in_month - fwd + 1       # 1 = last session of month
    window = (fwd <= after) | (bwd <= before)   # sessions whose return we want
    hold_at_close = window.shift(-1).fillna(False).astype(float)  # position into next session
    out = _frame(idx, px.columns)
    out["US_EQ"] = hold_at_close
    out["CASH"] = 1 - hold_at_close
    return out


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def bench_60_40(px: pd.DataFrame) -> pd.DataFrame:
    me = month_ends(px.index)
    out = _frame(me, px.columns)
    out["US_EQ"] = 0.6
    out["UST_7_10"] = 0.4
    return out


def bench_buy_hold(px: pd.DataFrame, asset: str = "US_EQ") -> pd.DataFrame:
    out = _frame(px.index[:1], px.columns)
    out[asset] = 1.0
    return out


# ---------------------------------------------------------------------------
# Registry: default params (a priori) and sensitivity grids
# ---------------------------------------------------------------------------
def grid(**kw):
    keys = list(kw)
    return [dict(zip(keys, v)) for v in itertools.product(*kw.values())]


REGISTRY = {
    "S1 Trend (TSMOM)": dict(
        fn="s1", default=dict(lookback_m=12, vol_win=60), lag=1, band=0.0, trade_based=False,
        grid=grid(lookback_m=[3, 6, 9, 12], vol_win=[20, 60, 120])),
    "S2 Dual Momentum": dict(
        fn="s2", default=dict(top_k=2, lookbacks=(1, 3, 6, 12)), lag=1, band=0.0, trade_based=False,
        grid=grid(top_k=[1, 2, 3], lookbacks=[(1, 3, 6, 12), (6,), (12,)])),
    "S3 Vol-Managed Equity": dict(
        fn="s3", default=dict(vol_win=21, target=0.12), lag=1, band=0.10, trade_based=False,
        grid=grid(vol_win=[10, 21, 63], target=[0.10, 0.12, 0.15, 0.18]), grid_band=[0.05, 0.10, 0.20]),
    "S4 Risk Parity": dict(
        fn="s4", default=dict(vol_win=60), lag=1, band=0.0, trade_based=False,
        grid=grid(vol_win=[20, 60, 120, 250])),
    "S5 Mean Reversion": dict(
        fn="s5", default=dict(rsi_th=10, trend_filter=True, exit_rule="sma5"), lag=1, band=0.0, trade_based=True,
        grid=grid(rsi_th=[5, 10, 15, 20], trend_filter=[True, False], exit_rule=["sma5", "rsi70"])),
    "S6 Valuation (ECY)": dict(
        fn="s6", default=dict(lo=0.3, hi=0.9, bond="UST_7_10"), lag=1, band=0.0, trade_based=False,
        grid=[dict(lo=a, hi=b, bond=c) for (a, b) in [(0.3, 0.9), (0.0, 1.0), (0.5, 1.0)] for c in ["UST_7_10", "CASH"]]),
    "S7 Turn-of-Month": dict(
        fn="s7", default=dict(before=1, after=3), lag=0, band=0.0, trade_based=True,
        grid=grid(before=[1, 2, 3], after=[1, 2, 3, 4, 5])),
}
