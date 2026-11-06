# BTCUSDT 2024 full-year, four-fold, five-seed Phase 2 evaluation

**Status:** Predeclared configuration. These experiments must be allowed to finish before numerical performance conclusions are drawn.

## Exact study specification

- Source: Binance public **spot** BTCUSDT 15m monthly OHLCV, January through December 2024, UTC, complete 35,136 candles (leap year).
- Pipeline: corrected Phase 1 + Phase 2 eight models (majority, persistence, logistic lag baseline, numeric OHLC LSTM, chart CNN, CNN+GRU, causal visual world model, numerical-image fusion).
- Per seed: 32-candle windows, 64px chart images, K=6 historical frames, horizon=4 candles, four expanding walk-forward folds with embargo, maximum 10 epochs, batch 128, 1,000 block-bootstrap resamples; seeds 0,1,2,3,4.
- One same historical time period across all seeds. A seed is a random initialization replicate, *not an independent market period*.
- 5 bps per unit change in position is an **illustrative proxy**; short-selling, financing, slippage, market impact, execution, volatility regimes and risk are not faithfully modeled. No live trading.
- The full-year CSV SHA-256 must be identical for all seeds. Archive the individual metrics and predictions; repeat only with a clearly labeled new experiment version, not stealth tuning on the test period.

## Running and monitoring

A push to `validation/btcusdt-full-2024-five-seeds` triggers a parallel GitHub Actions five-seed matrix. Each seed uses a separate CPU runner (maximum 350 minutes); if a runner times out, the run may fail and no completed full benchmark may be claimed. Original `main` remains untouched.

Results are uploaded to GitHub Actions with a 30-day retention time, including each seed's `metrics.json`, `preds.npz`, `report.html`, diagnostics and dataset fingerprint. The aggregator rejects missing seeds, different settings, mismatched hashes or incomplete periods and writes a Markdown/JSON summary only after verifying all five seeds. A success result for CI tests alone does not indicate a completed experiment.

## Interpretation

Compare the eight models as a locked design. Report accuracy and AUC *alongside* net Sharpe and DSR with caveats. Bootstrap CIs on one historical period do not demonstrate market robustness. Keep Volatility 75 Index testing separate for a later stage and avoid retrospective model selection from test data.
