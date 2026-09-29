"""Data acquisition for the multi-asset research study.

All inputs are downloaded from public GitHub repositories, pinned to exact
commits so every number in the report is reproducible. Yahoo Finance, FRED,
Stooq and the Ken French library were not reachable from the research
environment, so these GitHub-hosted datasets are the substitute.

Sources (see REPORT.md for caveats on modelled/spliced segments):
  * mwilczynska/financial_datasets  - daily total-return proxies 1970-2026
  * datasets/finance-vix            - CBOE VIX daily close 1990-2026
  * datasets/s-and-p-500            - Shiller monthly S&P data (earnings end 2023-06)
  * datasets/bond-yields-us-10y     - monthly 10y Treasury yield 1953-2026
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"

FD_SHA = "57a44356c79cdbeb0cc76ee2fa74be88c55f82fb"
VIX_SHA = "3dbea23c70cc2afbeeec8ac232c9ec72ac26ac21"
SHILLER_SHA = "78cdcebfd6725c31a914947efd08ddadac08eb1e"
Y10_SHA = "ba336158ac627766ae23ab8bbc46e6e7eab1294a"

RAW = "https://raw.githubusercontent.com"

# Research alias -> (dataset file, tradable ETF the series is meant to proxy)
ASSETS = {
    "US_EQ": ("us_large_cap_sp500", "SPY"),
    "GL_EQ": ("global_stocks", "VT"),
    "UST_7_10": ("intermediate_term_us_treasury", "IEF"),
    "UST_20P": ("long_term_us_treasury", "TLT"),
    "GOLD": ("gold", "GLD"),
    "CMDTY": ("broad_commodities", "DBC"),
    "CASH": ("short_term_us_treasury", "SHY"),  # 1-3y Treasuries used as cash proxy
}
RISKY = ["US_EQ", "GL_EQ", "UST_7_10", "UST_20P", "GOLD", "CMDTY"]


def _fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            dest.write_bytes(r.read())
    return dest


def load_asset_frames() -> dict[str, pd.DataFrame]:
    out = {}
    for alias, (name, _) in ASSETS.items():
        url = f"{RAW}/mwilczynska/financial_datasets/{FD_SHA}/data/processed/{name}.parquet"
        df = pd.read_parquet(_fetch(url, CACHE / f"{name}.parquet"))
        df["Date"] = pd.to_datetime(df["Date"])
        out[alias] = df.set_index("Date").sort_index()
    return out


def load_prices() -> pd.DataFrame:
    """Total-return levels (Adj Close) on the US equity trading calendar.

    Other series are forward-filled at most 5 sessions (e.g. LBMA holidays),
    which produces a zero return on that day rather than a look-ahead.
    """
    frames = load_asset_frames()
    cal = frames["US_EQ"].index
    px = pd.DataFrame({a: f["Adj Close"] for a, f in frames.items()})
    px = px.reindex(cal).ffill(limit=5)
    return px.dropna()


def load_quality_flags() -> pd.DataFrame:
    frames = load_asset_frames()
    rows = []
    for a, f in frames.items():
        if "Quality Flag" in f:
            g = f.groupby("Quality Flag").apply(lambda x: pd.Series({"start": x.index.min().date(), "end": x.index.max().date(), "rows": len(x)}), include_groups=False)
            g.insert(0, "asset", a)
            rows.append(g.reset_index())
    return pd.concat(rows).sort_values(["asset", "start"])


def load_vix() -> pd.Series:
    url = f"{RAW}/datasets/finance-vix/{VIX_SHA}/data/vix-daily.csv"
    df = pd.read_csv(_fetch(url, CACHE / "vix-daily.csv"), parse_dates=["DATE"])
    return df.set_index("DATE")["CLOSE"].rename("VIX")


def load_shiller() -> pd.DataFrame:
    url = f"{RAW}/datasets/s-and-p-500/{SHILLER_SHA}/data/data.csv"
    df = pd.read_csv(_fetch(url, CACHE / "shiller.csv"), parse_dates=["Date"]).set_index("Date")
    return df.replace(0.0, float("nan"))


def load_y10() -> pd.Series:
    url = f"{RAW}/datasets/bond-yields-us-10y/{Y10_SHA}/data/monthly.csv"
    df = pd.read_csv(_fetch(url, CACHE / "y10.csv"), parse_dates=["Date"])
    return df.set_index("Date")["Rate"].rename("Y10")


def load_cpi_monthly() -> pd.Series:
    url = f"{RAW}/mwilczynska/financial_datasets/{FD_SHA}/data/processed/cpi_inflation.parquet"
    df = pd.read_parquet(_fetch(url, CACHE / "cpi_inflation.parquet"))
    df["Date"] = pd.to_datetime(df["Date"])
    obs = df[df["Quality Flag"] == "observed_bls_monthly_cpi_u_level"]
    return obs.set_index("Date")["Close"].rename("CPI")
