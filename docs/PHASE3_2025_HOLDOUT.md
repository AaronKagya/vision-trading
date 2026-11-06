# Phase 3 — Locked 2025-Q1 validation (BTCUSDT spot)

This study answers a **specific question** after the 2024 five-seed benchmark: does the numerical OHLC LSTM retain directional usefulness on a later market period, and does a **fixed confidence-based abstention rule** improve the trading proxy?

## Predeclared protocol

- Market: Binance **spot** BTCUSDT, continuous UTC 15-minute OHLCV, downloaded from the monthly public archive and checked for complete coverage.
- Fit/validation period: **2024-01-01 through 2024-12-31**. The final 15% of eligible 2024 training observations is validation; 10 candles are embargoed before that validation section. Labels crossing the year boundary are excluded from fitting.
- Untouched test: **2025-01-01 through 2025-03-31**. No training, scaling, model selection, threshold selection, or early stopping uses 2025 observations.
- Candidate: **numerical LSTM**, selected from the completed 2024 eight-model five-seed study. Simple train-only majority and lagged logistic-regression comparators are included.
- Model settings: 32-candle chart feature window (15m / 1h / 4h), horizon 4 candles, training epochs 10, batch size 128, learning rate 0.001, seed 0.
- Two *predeclared* readings: all directional forecasts, and abstain when `abs(P(up)-0.5) < 0.05`. Position is +1 for P>=0.55, −1 for P<=0.45, otherwise 0.
- Fees: illustrative 0.0005 log-return-equivalent per unit change in exposure, including closure of the final position; **not** a live or executable BTCUSDT spot shorting strategy.
- Report accuracy, AUC, Brier, ECE, coverage, conditional accuracy, and nonoverlapping simulated log-P&L proxy. No claiming success from selection-biased thresholds, Sharpe rankings or one profitable outcome.
- Volatility 75 Index explicitly excluded until a later separately defined project.

## Known limits

- The 2024 trading sample is **already examined**; confidence gating there is exploratory and should not be used as new selection evidence.
- The price panels are causal. Each 2025 forecast sees its historical trailing prices, but labels remain unseen during fitting. Results are reported for **2025 Q1 only**, not full-year 2025.
- Model fitting on all 2024 data is **not** used: part of 2024 is reserved for early stopping/validation, so the test reflects that predeclared training protocol.
- Trading figures are proxies. Costs exclude bid/ask spread dynamics, slippage, latency, exchange short-sale constraints, margin, borrow, liquidation, funding and risk-limited position sizing. They are not realized profits.
- A single seed is not enough for strong replication. Further tests must not change model settings after observing this holdout without declaring the test exploratory.

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python -m scripts.prepare_btcusdt --start 2024-01 --end 2025-03
python -m scripts.run_2025_holdout \
  --data data/BTCUSDT_15m.csv \
  --seed 0 --epochs 10 --bs 128 --out results_holdout_2025q1
```

Run automatically in GitHub Actions from `.github/workflows/btcusdt-2025-holdout.yml`. The workflow emits `metrics.json`, `preds.npz`, `report.md`, and an input-data manifest, or a clear failure status. The 2025 outcome has not been observed at the time this protocol was authored.

## Historical 2024 confidence experiment (not confirmatory)

The independent source ZIPs for seeds 0–4 were inspected before selecting the fixed abstention hypothesis. Applying a 0.05 distance from 50% returned mean prediction coverage of **21.27%**, conditional accuracy **54.68%**, and **negative** average log-P&L proxy **−0.5493**. This was exploratory *post-hoc* analysis, not a new independent holdout or verified trading performance.

Tool for reproduction (after downloading each seed artifact and naming them `btcusdt_2024_seed0.zip` through `seed4.zip`):

```bash
python -m scripts.analyze_abstention --archive-dir . --out results_2024_abstention.json
```
