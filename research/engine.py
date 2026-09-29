"""Event-driven daily portfolio simulator and performance metrics.

Timing convention (no look-ahead):
  * A strategy emits target weights indexed by *decision date* t, computed only
    from data up to and including the close of t.
  * The engine executes the trade at the close of session t + lag (lag=1 by
    default, i.e. the next bar). The new weights earn returns from t+lag+1 on.
  * Between rebalances, weights drift with realised returns.
  * Transaction costs = |trade weight| x (commission + slippage) per side on
    every risky asset. Moves into/out of CASH are free (models a money-market
    sweep); the cost is charged on the risky leg of every trade.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    returns: pd.Series          # daily net portfolio returns
    weights: pd.DataFrame       # end-of-day weights after trading
    turnover: pd.Series         # daily one-way traded weight (risky assets)
    costs: pd.Series            # daily cost drag (fraction of NAV)


def run_backtest(
    rets: pd.DataFrame,
    targets: pd.DataFrame,
    cost_bps: float = 15.0,
    lag: int = 1,
    band: float = 0.0,
    cash: str = "CASH",
    start: pd.Timestamp | None = None,
) -> BacktestResult:
    """Simulate a portfolio.

    rets:     daily simple returns, one column per asset (must include `cash`).
    targets:  decision-date target weights (rows sum to 1). NaN rows are ignored.
    band:     skip a rebalance if no risky weight moves by more than `band`.
    """
    cols = list(rets.columns)
    targets = targets.reindex(columns=cols).fillna(0.0)
    # map each decision date to its execution session
    idx = rets.index
    pos = idx.get_indexer(targets.index)
    targets = targets[pos >= 0]
    pos = pos[pos >= 0] + lag
    keep = pos < len(idx)
    exec_map = pd.DataFrame(targets.values[keep], index=idx[pos[keep]], columns=cols)
    exec_map = exec_map[~exec_map.index.duplicated(keep="last")]

    if start is not None:
        sl = idx >= start
    else:
        sl = np.ones(len(idx), dtype=bool)
    R = rets.values[sl]
    dates = idx[sl]
    exec_rows = exec_map.reindex(dates)
    has_trade = exec_rows.notna().all(axis=1).values
    T = exec_rows.values

    cash_i = cols.index(cash)
    risky_mask = np.ones(len(cols), dtype=bool)
    risky_mask[cash_i] = False
    c = cost_bps / 1e4

    w = np.zeros(len(cols))
    w[cash_i] = 1.0
    n = len(dates)
    out_r = np.empty(n)
    out_w = np.empty((n, len(cols)))
    out_to = np.zeros(n)
    out_c = np.zeros(n)
    for k in range(n):
        g = w * (1.0 + R[k])
        gross = g.sum()
        pr = gross - 1.0
        w = g / gross
        if has_trade[k]:
            tgt = T[k]
            d = np.abs(tgt - w)
            if d[risky_mask].max() > band:
                to = d[risky_mask].sum()
                cost = to * c
                pr = (1.0 + pr) * (1.0 - cost) - 1.0
                w = tgt.copy()
                out_to[k] = to
                out_c[k] = cost
        out_r[k] = pr
        out_w[k] = w
    return BacktestResult(
        returns=pd.Series(out_r, index=dates),
        weights=pd.DataFrame(out_w, index=dates, columns=cols),
        turnover=pd.Series(out_to, index=dates),
        costs=pd.Series(out_c, index=dates),
    )


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def max_drawdown(r: pd.Series) -> float:
    nav = (1 + r).cumprod()
    return float((nav / nav.cummax() - 1).min())


def drawdown_series(r: pd.Series) -> pd.Series:
    nav = (1 + r).cumprod()
    return nav / nav.cummax() - 1


def episodes(res: BacktestResult, cash: str = "CASH") -> pd.Series:
    """Compounded return of each contiguous risk-on episode (for trade stats)."""
    risk = 1 - res.weights[cash]
    on = (risk.shift(1).fillna(0) > 1e-9)  # exposure held during the day's return
    grp = (on != on.shift()).cumsum()
    ep = (1 + res.returns[on]).groupby(grp[on]).prod() - 1
    return ep


def metrics(
    res: BacktestResult,
    cash_ret: pd.Series,
    bench: pd.Series | None = None,
    trade_based: bool = False,
    cash: str = "CASH",
) -> dict:
    r = res.returns
    cr = cash_ret.reindex(r.index).fillna(0)
    ex = r - cr
    n = len(r)
    yrs = n / TRADING_DAYS
    tot = float((1 + r).prod() - 1)
    cagr = (1 + tot) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = ex.mean() / ex.std() * np.sqrt(TRADING_DAYS) if ex.std() > 0 else np.nan
    dd = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(TRADING_DAYS)
    sortino = ex.mean() * TRADING_DAYS / dd if dd > 0 else np.nan
    mdd = max_drawdown(r)
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    if trade_based:
        ep = episodes(res, cash)
        wins = ep[ep > 0]
        losses = ep[ep <= 0]
        n_units = len(ep)
    else:
        m = (1 + r).resample("ME").prod() - 1
        wins = m[m > 0]
        losses = m[m <= 0]
        n_units = len(m)
    win = len(wins) / n_units if n_units else np.nan
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.nan
    out = {
        "Total Return": tot,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max DD": mdd,
        "Calmar": calmar,
        "Win Rate": win,
        "Profit Factor": pf,
        "Win basis": "trades" if trade_based else "months",
        "N (trades/months)": n_units,
        "Turnover/yr": res.turnover.reindex(r.index).sum() / yrs,
        "Cost drag/yr": res.costs.reindex(r.index).sum() / yrs,
        "Exposure": float((1 - res.weights[cash].reindex(r.index)).mean()),
    }
    if bench is not None:
        b = bench.reindex(r.index).fillna(0)
        bx = b - cr
        beta = np.cov(ex, bx)[0, 1] / bx.var()
        alpha = (ex.mean() - beta * bx.mean()) * TRADING_DAYS
        act = r - b
        ir = act.mean() / act.std() * np.sqrt(TRADING_DAYS) if act.std() > 0 else np.nan
        bt = float((1 + b).prod() - 1)
        out.update({
            "Bench CAGR": (1 + bt) ** (1 / yrs) - 1,
            "Excess CAGR": cagr - ((1 + bt) ** (1 / yrs) - 1),
            "Beta": beta,
            "Alpha/yr": alpha,
            "Info Ratio": ir,
        })
    return out


def sharpe(r: pd.Series, cr: pd.Series) -> float:
    ex = r - cr.reindex(r.index).fillna(0)
    s = ex.std()
    return float(ex.mean() / s * np.sqrt(TRADING_DAYS)) if s > 0 else np.nan


def deflated_sharpe(ex: pd.Series, n_trials: int, sr_var_trials: float) -> float:
    """Bailey & Lopez de Prado (2014) deflated Sharpe ratio, per-period units.

    Probability that the true Sharpe exceeds the maximum expected from
    `n_trials` independent trials with zero true Sharpe.
    """
    from scipy.stats import norm, skew, kurtosis

    sr = ex.mean() / ex.std()
    T = len(ex)
    g = 0.5772156649
    sr0 = np.sqrt(sr_var_trials) * ((1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * np.e)))
    sk = skew(ex)
    ku = kurtosis(ex, fisher=False)
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    return float(norm.cdf(z))


def block_bootstrap_sharpe_diff(
    a: pd.Series, b: pd.Series, cr: pd.Series, n: int = 2000, block: int = 21, seed: int = 7
) -> tuple[float, float, float]:
    """Stationary-ish circular block bootstrap of Sharpe(a) - Sharpe(b).

    Returns (5th pct, 95th pct, share of draws where diff <= 0).
    """
    rng = np.random.default_rng(seed)
    cr = cr.reindex(a.index).fillna(0).values
    xa = a.values - cr
    xb = b.reindex(a.index).fillna(0).values - cr
    T = len(xa)
    nb = int(np.ceil(T / block))
    diffs = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, T, nb)
        ix = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        ya, yb = xa[ix], xb[ix]
        diffs[i] = (ya.mean() / ya.std() - yb.mean() / yb.std()) * np.sqrt(TRADING_DAYS)
    return float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95)), float((diffs <= 0).mean())
