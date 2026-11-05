# Source audit and corrections — Vision Trading

## Input evidence

- Official working baseline selected by the project owner: `vision-trading (new).zip` in `AaronKagya/vision-trading`.
- The user-supplied historical `metrics.json` is stored at `historical_evidence/metrics_seed0_original.json` unchanged. Its config states BTCUSDT 15m, 32-candle panels, 64-pixel images, 4-candle target, 6 historical frames, 4 folds, 10 epochs, seed 0.
- The PDF paper reports a world-model accuracy of 50.5%, while the separate HTML/PDF dashboard and the JSON agree on roughly 51.45%. Different experimental runs or code versions may have produced these; the original data, saved predictions and precise code commit are missing.

## High-priority defects corrected

### 1. The future frame entered the world model's forward pass

**Original source:** `offs=np.arange(-(K-1),2)` in `src/models.py` adds frame `t+1` to the image input, including validation and prediction; the prediction head consumed `z[:K]`, but the encoder's BatchNorm operated on a combined input of history and the future frame. This could alter the forecast through shared batch statistics, invalidating a strict causal deployment claim.

**Correction:** Prediction input offsets are `1-K,...,0` only; `WorldModel.forward()` checks that the number of frames is exactly `K`. The `t+1` chart is used only as a stop-gradient training **target**, computed in eval mode outside the forward pass. At inference, any positive offset is rejected. Causality regression tests perturb unseen future frames and ensure earlier predictions remain unchanged.

### 2. Per-candle auto-scaling erased within-view price structure

**Original source:** `render()` uses `.max(-1)` / `.min(-1)` on already sliced per-candle high/low arrays. As a result, the high/low scale was computed independently per candle instead of across a 32-candle panel, artificially making each wick occupy the full image height and changing the intended visual semantics.

**Correction:** Compute high/low extrema over the **window axis** for each timeframe, preserving relative movement and wick-body structure.

### 3. Incorrect interpretation of total return

**Original source:** `total_return=float(rets[m].sum())`, where `rets` is computed as `position * forward_log_return - turnover_cost`. This is a sum of a synthetic P&L proxy; not a simple cumulative brokerage return. The JSON also labels `n_trades` as 4,327 although they are evaluation intervals, not necessarily executed orders.

**Correction:** Retain the legacy keys for compatibility but add explicit `total_log_pnl_proxy`, `implied_compounded_return`, `n_trade_intervals`, and `metric_notes` warnings. Compounding the log proxy is for interpretation only and cannot supply a realistic strategy equity curve. Actual account P&L requires an execution engine.

### 4. Pretrained encoders were missing ImageNet normalization

**Original source:** Model inputs are always centered as `image/255-0.5`, even when encoder pretrained weights expect specific ImageNet mean/std normalization.

**Correction:** Apply conventional channel mean/std on pretrained ResNet18 and ViT backbones. From-scratch backbones retain the project's centered image input. This does not guarantee better financial performance; model domain transfer must be validated.

### 5. Silent market data problems

**Original source:** Missing timestamps and inconsistent OHLC values were mostly warnings. Monthly downloader could silently skip a requested month and produce a different date range. Duplicate timestamps were discarded without documentation.

**Correction:** Data parser rejects missing intervals, bad prices, invalid OHLC, duplicates, NaN/Inf and nonpositive closes; monthly importer stops instead of skipping a month. It records a SHA-256 hash and start/end timestamps for real-data runs.

### 6. Reproducibility and report shortcomings

**Original source:** No exact source or dataset fingerprint in summarized metrics; the original report and the separately published paper disagree in headline numbers. Model checkpoint export was not available in the original experimental runner.

**Correction:** Include software and fold metadata; add optional fold-specific checkpoints; keep historical JSON isolated; provide a deterministic smoke-test suite. Generated HTML report now escapes safely, including CSS formatting.

## Verification completed in this package

- Automated unit tests cover malformed candles, panel causality, numerical twin causality, exact input frame-count enforcement, future-image invariance, moving-block bootstrap, interval/turnover cost interpretation, embargo boundaries and report generation.
- A seven-model end-to-end **synthetic-null** experiment generates `metrics.json`, `preds.npz`, HTML report, and descriptive analysis.
- Software smoke tests do **not** validate predictive performance on original BTCUSDT data; they only show code paths execute.

## What was *not* verified

- Exact BTCUSDT historical rows corresponding to the previously reported 17,308 held-out samples.
- Original per-row prediction files or per-fold training checkpoints.
- Mean/variance across five independent seeds on historical market data.
- Out-of-period or second-asset robustness.
- Genuine executable net returns, order fills, bid/ask data, financing, borrow, margin, leverage or live forward performance.

These are dependencies for defensible future research claims, not bugs that can be honestly fixed by changing code alone.
