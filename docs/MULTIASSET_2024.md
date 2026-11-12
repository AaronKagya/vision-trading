# Vision Trading AI — Multi-market controlled comparison

## Purpose

Extend the existing eight-configuration research framework to BTCUSDT, ETHUSDT and SOLUSDT using equal 2024-01-01 through 2024-12-31 Binance SPOT 15-minute historical market intervals. The new dashboard is displayed at `/compare/`; gold, EURUSD and Volatility 75 remain **planned** and must not display made-up results.

## Original baseline and evidence

The BTCUSDT seed-0 JSON is preserved **unaltered**, extracted from the successfully completed GitHub Actions artifact `btcusdt-2024-seed-0` (run 41995284584). Its distinct 2024 training run used 35,136 candles and 17,308 out-of-sample predictions; the separately archived five-seed BTC summary has matching source attribution. Both are committed under `evidence/` for reproducible baseline comparison. They are **not trained anew** for this extension.

## Predeclared Phase 6 protocol

- Markets: Binance SPOT **BTCUSDT, ETHUSDT, SOLUSDT**. Do not combine with perpetual contracts or futures.
- Study period: **2024 calendar year** (35,136 candles; 15 minutes in UTC).
- Input: multi-timeframe causally rendered OHLC: trailing 15-minute, one-hour and four-hour panels, 32 bars per view; image 64 pixels.
- Target: sign of four-candle (one-hour) future close-to-close log-return.
- Models: majority, persistence, lagged logistic regression, numerical LSTM, chart CNN, CNN+GRU, latent visual world model, and numerical–visual fusion.
- Same four expanding walk-forward folds, embargo, training-only scaling, 10 maximum epochs, validation-only early stopping, and deterministic seed 0 for the initial pilot. Use five fixed seeds [0,1,2,3,4] for the full comparison.
- Reporting: average out-of-sample accuracy, AUC, Brier and Sharpe **proxy**. Identical cost=0.0005 is a sensitivity assumption, **not verified execution fees** for each market. No experimental trading claims are justified by that number.
- Data integrity: validate complete UTC period, no missing/duplicate candles, OHLC consistency and SHA-256 manifest. Aggregate only when **both** new-market runs finish and their complete metadata agree. Refuse missing seeds, changed configurations, synthetic datasets and missing provenance.
- Statistical inference: multiple assets are correlated, and all random seeds for an asset use the same market history. Cross-market comparisons are descriptive. Do not select a trading strategy from these results without a **fresh, future holdout** and suitable multiple-testing controls.

## Execution

GitHub Actions workflow `.github/workflows/vision-multiasset-2024.yml`:

1. **On branch push**, run one complete 8-model, 4-fold, 10-epoch-budget seed-0 benchmark for each new instrument. This is a *pilot*, compared against original BTC seed 0.
2. **Manual workflow dispatch**, run seeds 0–4 for each new instrument. This is the full cross-instrument benchmark; it is much more CPU-intensive and consumes GitHub-hosted runner minutes.
3. The aggregator verifies both markets and publishes `gh-pages/compare/data.json` if and only if all required seed artifacts match.

Local initial page (no unsupported performance figures):

```bash
python -m scripts.aggregate_multiasset --initial --out site/compare/data.json
python -m http.server 8000 --directory site
# Visit http://localhost:8000/compare/
```

Local replication with real data (internet and accessible Binance archives required):

```bash
python -m scripts.prepare_multiasset --symbol ETHUSDT --start 2024-01 --end 2024-12
python -m src.run_experiments --data data/ETHUSDT_15m.csv \
  --models phase2 --window 32 --img 64 --horizon 4 --K 6 \
  --folds 4 --epochs 10 --bs 128 --bootstrap 1000 --cost 0.0005 \
  --seed 0 --out results_ETHUSDT_0
```

## Next study, intentionally separate

Cross-market transfer requires **train on BTCUSDT / predict on ETHUSDT or SOLUSDT without retraining**, with an explicitly documented common feature/scaler definition, target alignment and new chronological test period. The per-instrument training experiment above does not test transfer learning; it tests the comparison of each asset's separately fitted models.

For gold and forex, acquire an authorized source, resolve different exchange sessions and calendar breaks, and revalidate costs. Deriv V75 is a **synthetic-index** data source with different microstructure; it stays out of the current Binance spot benchmark.
