# BTCUSDT real-data execution: Phase 2

**Data source:** Binance public *spot* monthly 15-minute kline archives. No credentials, orders, synthetic indices, or account access involved.

## What runs automatically

On a push to `validation/btcusdt-2024-phase2`, GitHub Actions runs unit tests, downloads and validates **January 2024** (2,976 15-minute candles), then executes the eight-model *quick engineering smoke test* using reduced window, image, epoch, and fold settings. It uploads `metrics.json`, `preds.npz`, `phase2_diagnostics.json`, `report.html`, and the dataset integrity manifest as a run artifact. This is **not** the definitive scientific comparison with the original report.

The exact dataset, candle boundaries, and SHA-256 are verified. Missing month, missing candles, timestamps, invalid OHLC or other breaks cause the run to fail rather than quietly evaluating a shortened period.

To run the full predeclared 2024 **single-seed** eight-model research comparison, use the workflow's **Run workflow** button with mode `full` or use the matching commands in `docs/PHASE2_EXPERIMENT.md` on a GPU-enabled machine. Full CPU training on a hosted runner can be lengthy. Multiple-seed and second-asset replication remain subsequent milestones, not part of the quick workflow.

### Local quick command

```bash
python -m scripts.prepare_btcusdt --start 2024-01 --end 2024-01
python -m src.run_experiments --data data/BTCUSDT_15m.csv --models phase2 \
  --window 16 --img 32 --horizon 4 --K 2 --folds 1 --epochs 1 \
  --bs 32 --bootstrap 40 --seed 0 --out results_btc_quick
python -m src.report --results results_btc_quick
python -m src.phase2_eval --results results_btc_quick --data data/BTCUSDT_15m.csv --bootstrap 40
```

### Research guardrails

1. The quick run is for verifying real-data ingestion and program behavior, not tuning or profitability inference.
2. Do not compare quick-run performance with the original paper's 17,308 observations; training and period parameters differ.
3. The full-year data version may differ from the original missing CSV; compare hash, dates and sample count.
4. No live trades, automatic exchange execution, or promise of profitability.
5. **Volatility 75 Index is deferred** and will require separate data sourcing and market-specific execution assumptions.
