# Vision Trading AI — Prospective frozen-model audit

**Aim.** Test whether a five-seed *numerical* LSTM ensemble trained using 2024 BTCUSDT candles produces directional skill *prospectively* on timestamps that did not exist when the weights were selected.

This phase is observational research, not a trading system. **No trading orders or real-money positions** can be placed by this code. No market forecast is evidence of profitability. Model probabilities have not been calibrated.

## Frozen model provenance

- Source: Phase 4 GitHub Actions run `42226885906`, five `epoch-seed-*` artifacts, checkpoint `checkpoint_budget_20.pt` for each seed 0–4.
- Model selection: maximum 20 epochs chosen using October–December 2024 validation BCE, with actual per-seed validation-best checkpoint epochs (often less than 20). Model weights fit on January–September 2024; no 2025/2026 parameters or loss thresholds were tuned in this implementation.
- Source data SHA-256: `2cc72f14a30e03baa1aeff87d4d6c63b4359361146fa51b633b96a21e106de67`.
- Predictor: unweighted mean of five fixed LSTM probabilities. 32 historical OHLC windows on 15m/1h/4h trailing views, each using completed candles at or before observation time. Standardization uses the training-only 12-feature scale saved in each original checkpoint.
- Outcome: sign of the *future four-candle* close-to-close return. A zero return is labeled 0, following the project's existing label convention. All forecasts are timestamped in UTC and scored only when the next one-hour target candle has closed.
- Optional simple comparison: four-candle momentum direction at the same forecast origin. Both use the same realized outcomes, with no tuning.
- No trades, recommended actions, shorting, spread/fees, transaction-cost returns, or Sharpe claims are made. More prospective data and independent benchmarking will be necessary for any financial-product claims.

## Workflow activation

The workflow in `.github/workflows/vision-forward-ai-shadow.yml` runs **once automatically when this feature branch is pushed**, to confirm dependency installation and model loading and arm the clean ledger. It also supports manual dispatch.

**GitHub scheduled Actions only run from the default branch.** The 15-minute cron (at UTC minutes 6, 21, 36 and 51) is *not active while this workflow exists solely on the feature branch*. A separately reviewed merge into the `main` default branch would activate that cadence; do not claim continuous AI inference before such a merge.

The static page at `/ai/` displays `gh-pages/ai/state.json` if published. If not activated, it clearly reports that no prospective forecasts are available. The prior `/live/` baseline remains separate and unchanged.

## Fidelity and limitations

- Data from Binance public market-data-only 15m klines. There are no API credentials; verify regional access and terms. Broken gaps, stale candles, invalid OHLCV, or missing model files fail closed.
- No predictions are generated retrospectively on missed hours. A new forecast is permitted only for a **fresh completed UTC hour** observed in the next 12 minutes and strictly after activation.
- The indicator uses the 2024-trained model on future observations; market drift is not adjusted or concealed.
- GitHub Actions can be late or fail, so recorded coverage may be sparse. A delayed poll does not backfill the missed signal; a future outcome can be settled later if a real completed candle is still available.
- Historical 2024–2025 out-of-sample results have already shown limited or no persistent economic value. This phase tests prospective forecasting reliability, not strategy profitability.
- To compare against the visual world model fairly in a future phase, archive a 2024-only frozen visual checkpoint and apply the *same* prospective observation timestamps. Do not claim the LSTM is the whole original vision research question.

## Run locally

```bash
unzip -o models/frozen_2024_lstm_ensemble.zip -d models/phase4_2024
python -m pip install -r requirements.txt
python -m pytest -q tests/test_forward_ai.py
python -m scripts.run_shadow_ai --state ai/state.json
```

The first successful call **arms** the ledger without any fabricated prior model forecasts. Later calls on newly closed candles may append observations. The published site is built from the raw JSON; it cannot create predictions by itself.
