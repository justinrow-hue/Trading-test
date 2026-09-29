# Multi-Asset Systematic Strategy Research: Findings

**As of:** prices to 2026-09-25 · **Run date:** 2026-09-29 · **Code:** `research/` (reproduce with `python -m research.run_all && python -m research.iterate`)

---

## 0. Inputs, tools and data (read this first)

**Inputs.** The prompt's `{{MARKET}}`, `{{BENCHMARK}}` and related fields were left blank. I used the defaults below, shaped by the data I could actually reach. Change them and re-run if you meant something else.

| Input | Value used | Why |
|---|---|---|
| Market | US-investable multi-asset ETFs: US equity (SPY), global equity (VT), 7-10y UST (IEF), 20y+ UST (TLT), gold (GLD), broad commodities (DBC); cash = 1-3y UST (SHY) | Only asset-class-level daily data was reachable (see below) |
| Benchmark | **60/40 SPY/IEF, rebalanced monthly, same cost model** (primary); SPY buy-and-hold (secondary) | Natural hurdle for a long-only multi-asset allocator |
| Capital / direction / leverage | $1M, **long-only, max 1.0x gross** | Conservative default |
| Horizon | Daily-to-monthly rebalancing (swing to multi-month) | Matches the data frequency |
| N strategies | 7 (98 parameter variants in total, plus 4 post-hoc iterations) | |

**Tools available.** A Python sandbox (pandas, numpy, scipy, matplotlib) and web search (snippets only; most news and official sites were blocked).

**Blocked by the environment's network policy.** Yahoo Finance, FRED, Stooq, the Ken French library, Nasdaq Data Link, the SEC, federalreserve.gov and CNBC were all denied. Nothing was fabricated. I substituted these public GitHub datasets, pinned to exact commits in `research/data.py`:

| Dataset | Content | Caveat (labelled **fact** from the dataset's own quality flags) |
|---|---|---|
| `mwilczynska/financial_datasets` @57a4435 | Daily total-return levels, 1970-01-02 → 2026-09-25 | Pre-ETF segments are **reconstructed**. Treasuries before 1986-1991 come from a Fed yield-curve model. Global equity before 1990-07 is a model scaled to US daily moves. **Commodities before 1984 are annual GSCI anchors, smoothed into daily data (see §8).** US equity is observed index data throughout (CRSP, then S&P 500 TR). |
| `datasets/finance-vix` @3dbea23 | VIX daily close 1990 → 2026-09-22 | |
| `datasets/s-and-p-500` @78cdceb | Shiller monthly price, earnings and CPI | **Earnings end 2023-06**, so S6 can only be evaluated to 2023-12 |
| `datasets/bond-yields-us-10y` @ba33615 | Monthly 10y yield → 2026-07 | |

**Not possible with this data:** single stocks or sectors (so no survivorship-free equity cross-section), options, intraday data, or ETF volume for ADV. Liquidity figures in §7 are labelled **assumptions**.

**Backtest conventions:**
- **Split:** 1971-01-04 → 2009-12-18 in-sample (70%); 2009-12-21 → 2026-09-25 out-of-sample (30%).
- **Walk-forward:** re-select parameters every 3 years on the trailing 15 years, from 1986.
- **Costs:** 5 bps commission + 10 bps slippage = **15 bps per side** on every risky-asset trade. Moves into and out of cash are free (a money-market sweep).
- **Timing:** signals use closes up to *t* and trade at the close of *t+1*. S7 is the only exception: its calendar is known in advance, so it trades on schedule.
- **Sharpe definition:** Sharpe is computed on excess return **over the SHY-like cash series**, not T-bills. That series compounded at 8.5%/yr in-sample, well above T-bills over 1971-2009, so **in-sample Sharpe ratios here are materially lower than conventional T-bill-based ones**. Out-of-sample the gap is small (SHY returned 1.3%/yr). All comparisons are like-for-like against the benchmark.

---

## 1. Executive summary

- **No strategy meets the "high quality" bar.** That is the headline result. Qualification required all of the following:
  - beating 60/40 out-of-sample on both Sharpe and CAGR after costs;
  - max drawdown under 30%;
  - also beating it in-sample;
  - not depending on one parameter set or one asset;
  - surviving doubled costs.

  The best score was 5 of 9 criteria (S6).
- **The 2010-2026 hold-out was an exceptionally hard benchmark.** 60/40 posted an out-of-sample Sharpe of 0.87, its best stretch in 50 years (decade Sharpe: 1.10 in the 2010s). Every diversifying strategy looks worse against a backdrop where US equities and Treasuries both delivered.
- **S1 Cross-asset Trend (TSMOM) has the most credible edge, but it fails the brief as specified.**
  - Max drawdown was ~11% across 55 years.
  - It beat 60/40's Sharpe in 4 of 6 decades.
  - Walk-forward Sharpe was 0.63 vs 0.50 for 60/40 (1986-2026).
  - It gained in the 1973-74, 2000-02 and 2008 bear markets.
  - Its deflated Sharpe probability is 0.90 across 98 trials.

  But it runs at ~5% volatility, so unlevered it compounds far below 60/40 (4.8% vs 9.8% CAGR out-of-sample). Its strong 1970s numbers are **inflated by smoothed commodity data**: dropping commodities cuts its in-sample Sharpe from 0.84 to 0.46.
- **S6 Valuation (Excess CAPE Yield) was the only strategy to beat 60/40 out-of-sample on both Sharpe and CAGR,** and only barely (+0.4pp/yr; bootstrap p ≈ 0.06). It also has a 36% max drawdown, depends on a single asset, and its data stops in 2023. It is a small, fragile tilt, not a strategy.
- **Discarded:** S5 short-term mean reversion and S7 turn-of-month.
  - **S5:** the gross edge is real (1991+ gross Sharpe 0.81), but at 15 bps/side and next-bar execution costs eat 2.6%/yr.
  - **S7:** the anomaly is essentially gone. Out-of-sample Sharpe is 0.43 gross and −0.06 net.
- **What I would actually do with this evidence:** hold the benchmark, and optionally add **S1 Trend as a diversifying sleeve of 25-50%**. Combining S1 with 60/40 50/50 gave out-of-sample Sharpe 0.91 vs 0.87 and max drawdown −15% vs −21%, at the cost of ~2.5pp/yr of CAGR. That combination was designed *after* seeing the out-of-sample data, so it is **not** validated.

---

## 2. Market analysis: regime and opportunities

**Facts** (computed from the data as of 2026-09-25; macro from search snippets I could not open):

| Asset (proxy) | 1m | 3m | 12m | vs 200d SMA | 63d vol | Drawdown from ATH |
|---|---|---|---|---|---|---|
| US equity (SPY) | +1.0% | +5.6% | +18.0% | +7.9% | 11.2% | −0.5% |
| Global equity (VT) | −0.3% | +4.2% | +18.8% | +6.7% | 12.1% | −1.2% |
| 7-10y UST (IEF) | −3.2% | −4.3% | −3.0% | −3.5% | 5.2% | −14.2% |
| 20y+ UST (TLT) | −4.4% | −8.2% | −6.8% | −5.8% | 10.0% | **−44.0%** |
| Gold (GLD) | −6.6% | +5.3% | +14.6% | −5.5% | **23.5%** | −20.7% |
| Commodities (DBC) | +7.0% | +22.8% | **+48.9%** | +16.9% | 21.4% | −15.6% |
| Cash (SHY) | −0.7% | −0.3% | +1.5% | 0.0% | 1.4% | −0.8% |

- VIX closed at 14.2 on 2026-09-22, the 26th percentile since 1990. Realised 21-day S&P volatility is 10.8%.
- CPI-U rose 3.7% year over year (Aug 2026, BLS). The 10y yield averaged 4.60% in Jul 2026, up 0.21pp over 12 months.
- The FOMC reportedly **raised** the funds rate 25 bp to 3.75-4.00% on 2026-09-16, citing elevated inflation ([CNBC](https://www.cnbc.com/2026/09/16/fed-rate-decision-september-2026.html), [Advisor Perspectives](https://www.advisorperspectives.com/dshort/updates/2026/09/16/feds-interest-rate-decision-september-16-2026), [JPMorgan AM](https://am.jpmorgan.com/us/en/asset-management/adv/insights/portfolio-insights/fixed-income/fixed-income-perspectives/fomc-statement-september-2026/)).
- Commentary describes a broad commodity rally spreading beyond energy ([NAI 500](https://nai500.com/blog/2026/09/broad-commodity-rally-fuels-inflation-risks-us-stocks-face-mounting-pressure/)). I could not open these primary pages; treat the macro points as reported, not verified.
- **Valuation:** the last computable Excess CAPE Yield (2023-09) sat at the **25th percentile** since 1881, meaning equities were expensive relative to real bond yields. S&P has risen ~45% since then, so it is almost certainly lower now, but I cannot compute that (**assumption**).

**Interpretation:**
- The regime looks like **late-cycle reflation**: equities at all-time highs with low implied volatility, a strong commodity uptrend, duration in a persistent bear market, and a Fed that has turned back to hiking.
- That is the configuration where stock-bond correlation turns positive (as in 2022). A 60/40 portfolio's bond leg then stops hedging.
- It is historically the environment where cross-asset trend and commodities have helped: 1973-74 and 2022 in the stress table in §4.

**Live signals (defaults, decision date 2026-09-25):**

| Strategy | Target positioning |
|---|---|
| S1 Trend | 16% SPY, 15% VT, 8% GLD, 8% DBC, **53% cash**. Both Treasury sleeves are off. |
| S2 Dual momentum | 50% SPY / 50% DBC |
| S3 Vol-managed | 100% SPY (volatility is below the 12% target) |
| S4 Risk parity | 35% IEF, 18% TLT, 16% SPY, 15% VT, 8% DBC, 8% GLD |

**Opportunity scan (interpretation):**
- **Commodities** show the strongest relative strength (+49% over 12 months).
- **Long duration** shows the weakest, and structurally so (TLT −44% from its peak).
- **Equities** are strong but expensive and priced for calm.

Mispricing claims cannot be supported with asset-class data alone.

---

## 3. Strategy details

Each strategy's defaults were fixed from the literature **before** any backtest. The grids exist only for sensitivity and walk-forward testing. Code is in `research/strategies.py`.

| # | Strategy | Thesis / who is on the other side | Rules (default) | Grid tested |
|---|---|---|---|---|
| S1 | **Cross-asset trend (TSMOM)** | Slow information diffusion, then herding. Other side: hedgers and rebalancers who fade trends. (Moskowitz-Ooi-Pedersen 2012; Hurst et al. 2017) | Monthly. Hold each asset only if its 12m total return > cash. Sleeves inverse-vol weighted (60d); switched-off sleeves go to cash. | lookback 3/6/9/12m × vol window 20/60/120 (12) |
| S2 | **Dual momentum rotation** | Cross-sectional plus absolute momentum. Other side: value and rebalancing flows. (Asness-Moskowitz-Pedersen 2013; Antonacci 2014) | Monthly. Rank on the average of 1/3/6/12m returns; hold the top 2 equally, only if each beats cash. | top-k 1/2/3 × lookback set (9) |
| S3 | **Volatility-managed equity** | Volatility clusters but expected returns don't rise proportionally, so de-risking in high volatility raises Sharpe. Other side: investors who hold constant weights. (Moreira-Muir 2017) | Daily. SPY weight = min(1, 12% / 21d realised vol); rest cash; 10% no-trade band. | vol window 10/21/63 × target 10/12/15/18% × band 5/10/20% (36) |
| S4 | **Risk parity (inverse vol), unlevered** | Leverage-averse investors overpay for high-risk assets, so low-risk assets carry better risk-adjusted premia. (Asness-Frazzini-Pedersen 2012) | Monthly inverse-vol weights across all 6 assets | vol window 20/60/120/250 (4) |
| S5 | **Short-term mean reversion** | Short-horizon reversal is compensation for providing liquidity to forced sellers. (Lo-MacKinlay 1990; Nagel 2012) | Buy SPY when RSI(2) < 10 and price > 200d SMA. Exit on close > 5d SMA or after 10 days. | RSI threshold 5/10/15/20 × trend filter on/off × exit rule (16) |
| S6 | **Valuation / ERP tilt (Excess CAPE Yield)** | Equity risk premium relative to real bond yields predicts long-horizon returns. Other side: momentum and flow-driven buyers at rich valuations. (Campbell-Shiller 1998; Asness 2003) | Monthly. SPY weight = 30% + 60% × expanding percentile of ECY; rest IEF. Earnings lagged 6 months. | range 30-90/0-100/50-100 × bond leg IEF/cash (6) |
| S7 | **Turn-of-month** | Month-end cash flows (payroll, pension contributions) hit equities around month-end. (Ariel 1987; McConnell-Xu 2008) | Hold SPY for the last trading day plus the first 3 of each month; cash otherwise. | days before 1/2/3 × after 1-5 (15) |

---

## 4. Backtest results (net of 15 bps/side)

### In-sample vs out-of-sample

Win rate is by month (m) or by trade (t). α and β are measured vs 60/40.

| Strategy | Period | CAGR | Vol | Sharpe | Sortino | MaxDD | Calmar | Win% | PF | Turnover/yr | Cost/yr | Exposure | vs 60/40 CAGR | vs SPY CAGR | β | α/yr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 Trend | IS | 11.9% | 4.6% | 0.84 | 1.18 | −10.9% | 1.09 | 76% (m) | 4.66 | 168% | 0.25% | 53% | +2.5pp | +2.1pp | 0.13 | +3.0% |
| S1 Trend | **OOS** | 4.8% | 5.3% | 0.69 | 0.96 | −10.1% | 0.47 | 62% (m) | 2.03 | 184% | 0.28% | 64% | **−5.0pp** | −9.7pp | 0.29 | +1.0% |
| S2 Dual Mom | IS | 13.8% | 13.0% | 0.43 | 0.61 | −30.0% | 0.46 | 64% (m) | 2.33 | 597% | 0.90% | 92% | +4.4pp | +4.0pp | 0.42 | +5.0% |
| S2 Dual Mom | **OOS** | 10.1% | 12.9% | 0.72 | 0.98 | −21.9% | 0.46 | 64% (m) | 2.03 | 607% | 0.91% | 98% | +0.3pp | −4.3pp | 0.47 | +5.1% |
| S3 Vol-Managed | IS | 10.2% | 11.7% | 0.19 | 0.26 | −35.6% | 0.29 | 62% (m) | 1.92 | 155% | 0.23% | 83% | +0.8pp | +0.4pp | 1.03 | +0.8% |
| S3 Vol-Managed | **OOS** | 10.9% | 11.6% | 0.82 | 1.13 | −17.2% | 0.63 | 67% (m) | 2.06 | 184% | 0.28% | 83% | +1.1pp | −3.6pp | 1.08 | +0.5% |
| S4 Risk Parity | IS | 9.3% | 5.4% | 0.17 | 0.25 | −21.2% | 0.44 | 68% (m) | 2.86 | 140% | 0.21% | 100% | −0.1pp | −0.5pp | 0.31 | +0.4% |
| S4 Risk Parity | **OOS** | 5.8% | 7.0% | 0.70 | 0.98 | −18.2% | 0.32 | 62% (m) | 1.86 | 93% | 0.14% | 100% | −4.0pp | −8.6pp | 0.48 | +0.5% |
| S5 Mean Rev | IS | 9.7% | 6.8% | 0.20 | 0.26 | −23.8% | 0.41 | 72% (t) | 1.93 | 1758% | 2.64% | 13% | +0.3pp | −0.1pp | 0.23 | +1.0% |
| S5 Mean Rev | **OOS** | 1.7% | 6.7% | 0.09 | 0.13 | −15.3% | 0.11 | 69% (t) | 1.56 | 1698% | 2.55% | 12% | −8.1pp | −12.7pp | 0.25 | −1.5% |
| S6 Valuation | IS | 9.8% | 10.4% | 0.17 | 0.24 | −35.6% | 0.28 | 64% (m) | 2.15 | 46% | 0.07% | 100% | +0.4pp | +0.0pp | 0.95 | +0.4% |
| S6 Valuation | **OOS†** | 9.6% | 9.8% | 0.89 | 1.26 | −20.9% | 0.46 | 69% (m) | 2.21 | 42% | 0.06% | 100% | +0.4pp | −3.6pp | 0.99 | +0.4% |
| S7 Turn-of-Month | IS | 6.8% | 7.6% | −0.18 | −0.25 | −19.2% | 0.36 | 56% (t) | 1.44 | 2398% | 3.60% | 19% | −2.6pp | −3.0pp | 0.30 | −1.7% |
| S7 Turn-of-Month | **OOS** | 0.6% | 7.4% | −0.06 | −0.08 | −21.8% | 0.03 | 56% (t) | 1.23 | 2415% | 3.62% | 19% | −9.2pp | −13.8pp | 0.32 | −3.1% |
| **60/40** | IS | 9.4% | 10.6% | 0.13 | 0.18 | −32.3% | 0.29 | 63% | 2.03 | 23% | 0.03% | 100% | — | −0.4pp | 1 | — |
| **60/40** | **OOS** | 9.8% | 9.8% | 0.87 | 1.23 | −21.2% | 0.46 | 69% | 2.24 | 23% | 0.03% | 100% | — | −4.6pp | 1 | — |
| SPY B&H | IS | 9.8% | 17.1% | 0.15 | 0.21 | −55.3% | 0.18 | 62% | 1.68 | — | — | 100% | +0.4pp | — | 1.64 | — |
| SPY B&H | **OOS** | 14.5% | 17.2% | 0.79 | 1.11 | −33.8% | 0.43 | 69% | 2.10 | — | — | 100% | +4.6pp | — | 1.73 | — |

† S6 OOS covers 2009-12 → 2023-12 only (earnings data end). The matching 60/40 OOS Sharpe is 0.85.

Charts:
- `results/charts/equity_curves.png`: growth of $1 vs 60/40
- `results/charts/drawdowns.png`
- `results/charts/sensitivity.png`
- `results/charts/correlation_oos.png`

Full tables are the CSVs in `results/`.

### Walk-forward (params re-selected every 3y on trailing-15y Sharpe, 1986 → 2026)

| Strategy | WF Sharpe | Default Sharpe | 60/40 Sharpe | WF CAGR | 60/40 CAGR | WF MaxDD | distinct params chosen / 14 |
|---|---|---|---|---|---|---|---|
| S1 Trend | **0.63** | 0.66 | 0.50 | 7.4% | 9.5% | −10.8% | 5 |
| S2 Dual Mom | 0.45 | 0.42 | 0.50 | 9.6% | 9.5% | −31.0% | 4 |
| S3 Vol-Managed | 0.50 | 0.52 | 0.50 | 10.1% | 9.5% | −29.7% | 9 |
| S4 Risk Parity | 0.53 | 0.54 | 0.50 | 7.6% | 9.5% | −18.2% | 3 |
| S5 Mean Rev | 0.24 | 0.22 | 0.50 | 6.1% | 9.5% | −28.1% | 6 |
| S6 Valuation (to 2023) | 0.51 | 0.52 | 0.47 | 8.8% | 9.2% | −32.3% | 4 |
| S7 Turn-of-Month | −0.09 | −0.18 | 0.50 | 3.0% | 9.5% | −37.9% | 4 |

Re-optimising never beat the a-priori defaults by a meaningful margin, which is the result you expect when the grids carry little real information. S3 flipped parameters 9 times in 14 windows, a sign that its parameter surface is noise.

### Robustness

**Parameter sensitivity (OOS Sharpe across each grid; 60/40 OOS = 0.87):**

| Strategy | min | median | max | % variants beating 60/40 | Spearman(IS rank, OOS rank) |
|---|---|---|---|---|---|
| S1 | 0.46 | 0.62 | 0.69 | 0% | 0.73 |
| S2 | 0.48 | 0.71 | 0.90 | 11% | 0.05 |
| S3 | 0.73 | 0.79 | 0.86 | 0% | 0.29 |
| S4 | 0.66 | 0.71 | 0.72 | 0% | 0.80 |
| S5 | −0.08 | 0.09 | 0.24 | 0% | 0.43 |
| S6 | 0.76 | 0.81 | 0.96 | 33% | 1.00 |
| S7 | −0.51 | 0.08 | 0.30 | 0% | 0.88 |

- **S1, S4, S6 and S7:** in-sample ranks carry over out-of-sample, so their parameter surfaces are structured rather than noisy.
- **S2:** in-sample rank tells you nothing (ρ = 0.05). Its best out-of-sample variant is luck.

**Cost and execution stress (OOS Sharpe):**

| Strategy | base 15bp | 30bp | 45bp | +1 bar delay | Observed-ETF era 2008-07+ (60/40 = 0.68) |
|---|---|---|---|---|---|
| S1 | 0.69 | 0.64 | 0.58 | 0.70 | 0.55 |
| S2 | 0.72 | 0.65 | 0.57 | 0.73 | 0.54 |
| S3 | 0.82 | 0.80 | 0.78 | 0.81 | **0.76** |
| S4 | 0.70 | 0.68 | 0.66 | 0.70 | 0.60 |
| S5 | 0.09 | −0.29 | −0.65 | 0.09 | 0.10 |
| S6 | 0.89 | 0.88 | 0.88 | 0.89 | 0.64 (60/40 0.64) |
| S7 | −0.06 | −0.54 | −1.00 | 0.07 | −0.11 |

**Asset-substitution test (single-asset strategies run on global equities, 1991+):**

| Strategy | Sharpe | Global B&H Sharpe | MaxDD | Global B&H MaxDD |
|---|---|---|---|---|
| S3 | 0.36 | 0.38 | −43% | −57% |
| S5 | 0.01 | 0.38 | | |
| S7 | −0.19 | 0.38 | | |

S3's drawdown reduction transfers to global equities; its Sharpe gain does not.

**Leave-one-asset-out:**

| Strategy | 1991+ Sharpe range | 60/40 (1991+) |
|---|---|---|
| S1 | 0.55-0.72 | 0.56 |
| S2 | 0.50-0.59 | 0.56 |
| S4 | 0.47-0.69 | 0.56 |

- No single asset drives the 1991+ results.
- **Pre-1991, commodities matter a lot:** dropping them cuts S1's in-sample Sharpe from 0.84 to 0.46 and S4's from 0.18 to 0.05.

**Regimes (full period, Sharpe; ex-post labels):**

| Regime (days) | S1 | S2 | S3 | S4 | S5 | S6 | S7 | 60/40 | SPY |
|---|---|---|---|---|---|---|---|---|---|
| Bull (12,009) | 0.93 | 0.55 | 0.93 | 0.83 | 0.39 | 1.06 | 0.12 | 1.01 | 1.00 |
| Bear, >20% DD (2,042) | **0.04** | **0.42** | −2.14 | −1.64 | −0.64 | −1.82 | −1.05 | −1.83 | −1.68 |
| High vol, RV>25% (1,146) | −0.15 | −0.01 | 0.34 | 0.42 | −0.10 | 0.42 | −0.06 | 0.47 | 0.40 |
| Rising 10y (6,736) | **0.83** | 0.68 | 0.31 | 0.22 | 0.02 | 0.07 | −0.11 | 0.13 | 0.25 |
| High inflation, CPI>4% (4,689) | **0.51** | 0.41 | −0.14 | −0.53 | −0.18 | −0.20 | −0.14 | −0.26 | −0.15 |

**Stress episodes (cumulative return):**

| Episode | S1 | S2 | S3 | S4 | S5 | S6 | S7 | 60/40 | SPY |
|---|---|---|---|---|---|---|---|---|---|
| 1973-74 stagflation | +44%* | +146%* | −33% | +22%* | +6% | −32% | −14% | −30% | −46% |
| 1987 crash | −3% | −18% | −20% | −4% | −23% | −17% | −3% | −19% | −32% |
| 2000-02 | +25% | +10% | −34% | +10% | +20% | −5% | −9% | −22% | −47% |
| 2008 GFC | +7% | +15% | −26% | −2% | +10% | −35% | −15% | −32% | −55% |
| 2020 COVID | −2% | +5% | −16% | −10% | −3% | −18% | +6% | −19% | −34% |
| 2022 stock-bond | −9% | −1% | −17% | −16% | −5% | −20% | −5% | −21% | −24% |
| 2025 tariff shock | −4% | −8% | −13% | −5% | −2% | n/a | −3% | −10% | −19% |

\* Inflated by the smoothed pre-1984 commodity series. Do not trust these magnitudes.

**Correlation (weekly, OOS).**
- S3 and S6 are near-clones of the benchmark (ρ 0.89 and 0.92 vs 60/40).
- S1 and S2 are the genuine diversifiers (0.63 and 0.46).
- S1 and S4 overlap heavily (0.85).

---

## 5. Comparison and ranking

Nine pre-registered criteria are defined in `research/run_all.py` §8. A strategy is "high quality" only if it passes all six hard ones:
1. Beats 60/40 OOS on Sharpe **and** CAGR.
2. Full-period MaxDD better than −30%.
3. Beats 60/40 in-sample too.
4. Median grid variant beats 60/40 OOS.
5. Not single-asset.
6. Survives 2× costs.

| Rank | Strategy | Score /9 | High quality? | Pros | Cons |
|---|---|---|---|---|---|
| 1 | **S1 Trend** | 4 | No | Best walk-forward and deflated Sharpe (DSR 0.90). 11% MaxDD over 55 years. Positive in bear, rising-rate and high-inflation regimes. Robust to costs, delay and leave-one-out. | Only ~5% vol, so CAGR far below 60/40 at 1× leverage. OOS Sharpe 0.69 < 0.87. 1970s results inflated by data. |
| 2 | S6 Valuation | 5 | No | Only strategy to beat 60/40 OOS on both Sharpe and CAGR. Cheap to run (46% turnover). IS-to-OOS ranks stable. | Edge is +0.4pp/yr, p ≈ 0.06. MaxDD −36%. Single asset. Data ends 2023. ρ 0.92 with benchmark. |
| 3 | S3 Vol-Managed | 2 | No | Beat 60/40 Sharpe in the observed-ETF era (0.76 vs 0.68). Lower drawdowns than SPY. Nearly cost-insensitive. | Can't dodge gap crashes (1987: −20%). MaxDD −36%. No Sharpe gain on global equities. Parameter churn. |
| 4 | S2 Dual Momentum | 2 | No | Strongest crisis convexity. OOS CAGR matches 60/40. | MaxDD −30%. IS/OOS ranks uncorrelated. Highest turnover among the allocators. |
| 5 | S4 Risk Parity | 3 | No | Smooth and diversified. | Needs leverage to be useful; hit hard in 2022 (−16%). |
| — | S5 Mean Reversion | 2 | **Discard** | Real gross edge (1991+ gross Sharpe 0.81). | Costs 2.6%/yr at 15 bps and next-bar execution. Net OOS Sharpe 0.09. Fails on global equities. |
| — | S7 Turn-of-Month | 1 | **Discard** | — | Gross OOS Sharpe 0.43, net −0.06. The anomaly has decayed. |

**Data-snooping diagnostics:**
- **Trials:** 98 variants plus 4 post-hoc iterations.
- **Deflated Sharpe, full period:** S1 0.90, S2 0.28, S3 0.05, S4 0.04, S5 0.00, S6 0.04, S7 0.00. Only S1 is unlikely to be the best of many zero-edge trials.
- **Block bootstrap of OOS Sharpe difference vs 60/40:** no strategy is significantly better. S6 is the closest (90% CI −0.00 to +0.09).

---

## 6. Final recommendations

**Nothing qualifies against your criteria, and I'm not going to present anything as a validated standalone strategy.** In priority order:

1. **Default: keep the 60/40 benchmark.** No tested strategy reliably beat it after costs in 2010-2026, and none clears the bar over 55 years.
2. **If you want a diversifier: S1 Cross-Asset Trend as a 25-50% sleeve beside 60/40.**
   - It has the strongest evidence of a real, persistent edge: decades of consistency, walk-forward, deflated Sharpe, and positive in bears, rising rates and inflation.
   - It is the natural hedge for today's regime: positive stock-bond correlation, a hiking Fed, and a commodity uptrend.
   - The post-hoc 50/50 blend had OOS Sharpe 0.91 vs 0.87, MaxDD −15% vs −21%, and CAGR 7.3% vs 9.8%.
   - **Expect to give up return in equity bull markets.**
3. **Not recommended now:** S6 (data ends 2023, marginal edge), S3 (benchmark clone), S2 (unstable), S4 (needs leverage), S5 and S7 (cost-dominated).

---

## 7. Deployment considerations (for the S1 sleeve, if adopted)

- **Instruments:** SPY, VT, IEF, TLT, GLD, DBC, with SHY or a T-bill fund as cash.
  - DBC is a K-1-free ETF but has roll yield and an embedded T-bill yield.
  - PDBC is a common alternative.
- **Liquidity (assumption: approximate typical ETF dollar ADV, not measured here).** SPY, IEF, TLT and GLD trade ≫ $500M/day. VT and DBC trade roughly $30-150M/day. At $1M the largest single trade (~$0.2-0.5M) is under 1% of ADV everywhere. Capacity limits bind first in DBC and VT, somewhere around $50-100M of strategy AUM.
- **Execution:** rebalance on the first session after month-end, using market-on-close or VWAP orders. Budget 15 bps per side. Measured turnover is ~180%/yr, a cost drag of ~0.3%/yr.
- **Sizing:** follow the rule weights (inverse 60d vol, capped at the 1× gross limit). No discretionary overrides. Hard cap of 40% per asset if you add assets.
- **Risk limits:**
  - Sleeve drawdown alert at −8%.
  - Portfolio-level drawdown review at −15%.
  - Realised sleeve vol > 10% (2× normal) triggers a review of data and fills.
- **Monitoring (monthly):**
  - realised vs backtested turnover and slippage;
  - rolling 36m Sharpe vs the backtest distribution;
  - live correlation to 60/40 (backtest ≈ 0.6).
- **Kill-switch criteria:**
  - Sleeve drawdown > 15%, which is 1.4× the worst in 55 years.
  - Rolling 36m Sharpe < −0.3.
  - Realised costs > 3× assumed for two quarters.
  - Any data or implementation mismatch between the signal and the reconciled positions.

---

## 8. What could go wrong

- **Data reconstruction.** Pre-ETF series are modelled.
  - Commodities before 1984 are annual anchors smoothed into daily data. This creates **artificial trend persistence and low volatility**, which flatters S1, S2 and S4 in the 1970s (their +44% / +146% / +22% in 1973-74 are not credible magnitudes).
  - Pre-1990 global equity is a scaled copy of US daily moves.
  - The 1991+ and 2008+ subsamples are the honest tests. They were reported, and they are weaker.
- **Cash and Sharpe definition.** SHY is used as cash: it earned ~8.5%/yr in-sample (well above T-bills), and it carries duration risk (2022: −4%).
- **Regime dependence.**
  - Trend loses in sharp V-shaped reversals: 2020 was −2%, and choppy 2023-24-style markets are a risk.
  - Its worst periods are high-vol whipsaws (Sharpe −0.15 in high-vol regimes).
- **Structural breaks.**
  - The two equity anomalies (S5, S7) have already decayed or been arbitraged.
  - Trend-following crowding since the 2010s is a known concern for S1.
- **Out-of-sample contamination.** The iteration step (I1-I4) and this recommendation were formed *after* viewing the 2010-2026 results. Only the original seven defaults are clean out-of-sample tests.

**Invalidation conditions (per thesis):**

| Strategy | What would invalidate it |
|---|---|
| S1 | A decade of negative Sharpe in live trading, or failure to protect in the next >20% equity bear (the thesis says it should be flat or positive). |
| S6 | ECY stops predicting 10y returns once post-2023 earnings are added. A negative OOS alpha extended to 2026 kills it. |
| S3 | A realised-vol spike that no longer precedes lower returns. Watch a live OOS Sharpe below SPY's over 5 years. |
| S2, S4 | Continued sub-benchmark Sharpe on observed ETF data (2008+ already shows it). |
| S5 | Revisit only if you have institutional execution (≤2 bps, same-close MOC fills). Diagnostic net Sharpe was 0.71 (1991+) under those assumptions, which lie **outside this brief**. |

---

## Final check

**What am I missing, and which assumptions are weakest?**
- Real tradable-ETF history exists only from 2002-2008; everything earlier is reconstructed.
- I used SHY rather than T-bills as cash.
- No single stocks, sectors, futures or options were available. True cross-sectional equity factors, carry (futures curves) and options-based strategies were therefore **not testable**. Those are the classes where most published multi-factor edges live.
- ADV figures are assumptions.
- Macro facts come from search snippets I could not open.

**Is the edge real or overfit?**
- **S1:** likely real, with DSR 0.90, 55 years, consistent decades and walk-forward. But it is smaller than the 1970s numbers suggest and does not fit a 1× leverage mandate.
- **S6:** indistinguishable from noise after 98 trials (DSR 0.04).
- **S5 and S7:** anomalies that exist gross, or used to, but are not monetisable at these costs.

**Does it survive different regimes and higher costs?**
- S1, S3, S4 and S6 are cost-robust: at 3× costs OOS Sharpe drops by at most 0.11.
- S5 and S7 are not.
- Only S1 and S2 help in bear, rising-rate and inflationary regimes.

**What would change my mind?**
- Clean T-bill and ETF-level data from Yahoo/FRED, rerun unchanged. The code accepts any daily total-return frame.
- Allowing leverage up to ~2×. S1 at ~10% vol would likely clear the CAGR hurdle; this is untested here and would need its own OOS test.
- Post-2023 earnings data for S6.
- Single-stock data to test cross-sectional factors properly.

---

### Reproduce

```bash
pip install -r requirements.txt
python -m research.run_all    # ~30s; writes results/*.csv and results/charts/*.png
python -m research.iterate    # post-hoc diagnostics (contaminated OOS; see docstring)
```

Search sources:
- [CNBC, Fed decision Sep 16 2026](https://www.cnbc.com/2026/09/16/fed-rate-decision-september-2026.html)
- [Advisor Perspectives](https://www.advisorperspectives.com/dshort/updates/2026/09/16/feds-interest-rate-decision-september-16-2026)
- [JPMorgan AM FOMC note](https://am.jpmorgan.com/us/en/asset-management/adv/insights/portfolio-insights/fixed-income/fixed-income-perspectives/fomc-statement-september-2026/)
- [NAI 500 commodity rally](https://nai500.com/blog/2026/09/broad-commodity-rally-fuels-inflation-risks-us-stocks-face-mounting-pressure/)
- [mwilczynska/financial_datasets](https://github.com/mwilczynska/financial_datasets)
