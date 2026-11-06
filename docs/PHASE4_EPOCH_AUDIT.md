# Phase 4: Does Vision Trading AI have enough training epochs?

**Status:** implemented research protocol; real-data evaluation is reported only from GitHub Actions artifacts, never inferred from passing tests.

## Hypothesis

The original January 2024 engineering smoke used **1 epoch** and is not a model-quality result. The completed full 2024 five-seed walk-forward benchmark and the 2025-Q1 external LSTM test set a maximum of **10 epochs**, with best-validation-checkpoint selection and **patience 3** early stopping. They did **not** save per-epoch train and validation losses or the actual best epoch. Consequently we cannot tell if the old training regime converged.

## New frozen protocol

1. **Asset**: exact full 2024 Binance BTCUSDT 15-minute spot candles (35,136 rows); market-data SHA-256 required. No future candle inputs.
2. **Features**: original 32-candle multi-scale **numerical OHLC** representation, same 48-cell LSTM head and other architecture as Phase 3. Forecast 4 candles / 1 hour ahead. Train-only scaler, Adam learning rate 0.001, batch size 128, weight decay 1e-4.
3. **Training**: January 2024 through September 2024, with an embargo around the validation boundary; **validation** October through December 2024. Never use 2025 labels, values, scores or metrics to select a cap.
4. **Seeds**: 0, 1, 2, 3, 4, independent initializations; all identical hyperparameters. Do not select a lucky seed.
5. **Capacity comparison**: train a single **full 40-epoch trajectory per seed**; retain validation-best checkpoint for maximum-epoch caps **5, 10, 20, 40**. Log actual train BCE and validation BCE **every epoch** and the epoch when the old patience=3 criterion would have stopped. Full training beyond that point is **intentional**: otherwise all four caps could become identical even if later improvement occurs.
6. **Choose cap**: lowest mean **2024 validation BCE** across all five seeds; ties choose smaller cap. Save `selected.json`, per-seed curves and all checkpoints. No manual choice after looking at Q2 outcomes.
7. **Locked external test**: after the selection job succeeds, score **April–June 2025 (2025 Q2)** for each of the five frozen checkpoints. Q1 2025 is available only as causal price history before April 1; there is no 2025 fitting, scaling, threshold tuning, or hyperparameter selection. Verify that the first-year byte prefix matches the training data hash.
8. **Analysis**: report BCE curves and early-stop trigger, accuracy, AUC, Brier score, nonoverlap accuracy, training-only majority baseline, and a research-only confidence-abstention transaction proxy. Do not call the proxy realized performance.

## Execution and evidence

Workflow: `.github/workflows/btcusdt-epoch-audit-2024.yml` on branch `validation/btcusdt-epoch-audit-2024`.

It runs five seeds in parallel; an aggregator refuses missing or incompatible artifacts; a final job scores the locked Q2 holdout **only after selection**. Each job uploads artifacts for checking results. Q2 results are one-time external evidence, **not** a tuning dataset. Run all tests with `python -m pytest -q tests`.

## Caveats

- The same raw OHLC data underlies charts and numbers; a more elaborate visual model is not automatically informative.
- BCE minimization is about probabilistic forecasting quality, **not** maximizing trades or profits.
- Financial direction labels are noisy, nonstationary, and overlap. Normalized probabilistic errors can improve while net trading performance gets worse.
- The historical research trading-cost proxy does not implement order fills, borrow, funding, slippage, exchange fees or liquidation. Live trading remains **disabled**.
- This audit addresses the leading numerical LSTM first; it does not establish the training adequacy of all eight model families. The winning cap is not automatically transferable to CNN, GRU, or fusion architectures.
