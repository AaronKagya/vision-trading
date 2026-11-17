# Vision Trading AI — Phase 9: Deriv Synthetic Indices (research-first)

**Why:** Test the original visual-vs-numerical research question on algorithmic 24/7 synthetic indices rather than assume an edge from price charts. BTCUSDT evidence and the frozen visual training workflow remain distinct and unchanged.

## Universe and instrument identities

Initial research labels: Volatility 10 Index, Volatility 25 Index, Volatility 75 Index, Jump 50 Index, Jump 75 Index, Jump 100 Index. Fetch exact API codes from `active_symbols` at run time. **Do not guess** a symbol from a display label. Do not silently mix standard Volatility Indices with `1s` and high-frequency versions. More indices can be considered in future under separate declared experiments.

Deriv says Volatility Indices have predefined **volatility**, not an exploitable deterministic directional trend. Jump Indices have up/down jumps on average every 20 minutes, **not predictable timed signals**. The public feed uses a cryptographically secure algorithm. Treat chance-level forecasting as the primary null hypothesis.

## First experiment and data requirements

1. Unauthenticated `active_symbols: brief` identifies each market. Public `ticks_history` candle style, 900-second granularity, 1000 rows/page, paged backwards, excludes open candle. No usernames or tokens.
2. Fail closed if time gaps, duplicate timestamps, invalid OHLC, stale latest observation, too few rows (<520 for a 32×4h trailing panel), unavailable market history, unsupported symbol or unverified source. Source is `wss://api.derivws.com/trading/v1/options/ws/public`.
3. Deriv candle OHLC **does not include observed traded volume**. The CSV uses a zero placeholder solely to satisfy existing OHLCV file schema; the current numerical and visual models use OHLC-derived features, not volume. Never treat zero as measured volume.
4. Archive exact exported CSV SHA-256 and endpoint timestamps. Determine availability of adequate multi-month history before selecting chronological splits. Do **not** silently treat limited recent Deriv candles as equivalent to the BTCUSDT 2024 full-year benchmark.
5. Predeclare a chronological train/validation/test plan **per index**, then compare the eight configurations using matched data, multiple seeds and a no-trade baseline. Select thresholds only from the training/validation periods. Preserve untouched test periods and risk of correlated random draws. Compare at a matched one-hour forecast horizon (four 15m bars) first. Track volatility/jumps as distinct regimes. If data are inadequate, report *insufficient history* rather than synthetic performance numbers.
6. Use market-appropriate broker contract specifications and costs for trade simulations. The BTC spot 5 bps turnover assumption is **not** an execution model for Deriv CFDs, options or multipliers.
7. A separate 2024 visual CNN+GRU training workflow is still investigating whether chart-image latent learning helps BTC forecasts. The original research question transfers to Deriv **only after** adequate data are archived and split definitions are predeclared.

## Risk-engine and trading modes

- **Research**: historical comparisons and forward-only observational forecasts. No orders.
- **Manual trade plan**: user supplies observed entry, SL, TP, verified currency-denominated value per price point per lot, broker lot sizes/increments and estimated fees/slippage. The calculator caps example risk at 2% and rounds down; if broker minimum volume exceeds risk budget, returns **NO TRADE**. It never claims that a stop fill is guaranteed.
- **Demo execution (future)**: after confirming account/product type and official supported execution route, allow an explicit opt-in authenticated demo connection, position preview, a single order confirmation, idempotent request reconciliation, maximum position/day loss, kill switch, SL/TP read-back, and audit trail.
- **Real-money execution (future, disabled)**: would require separate user approval, broker eligibility and compliance, fully verified contract sizing, test evidence, and explicit per-order confirmation. No auto-execution exists in this phase.
- **External advisor AI (future)**: integrate only a licensed, user-authorized provider, require signed/model-versioned timestamped forecasts, forward-vs-baseline results, calibration, fees, independent verification and disclosure of conflicts. *Claims of guaranteed profits or 90%+ accuracy do not qualify as evidence.* This source is a plugin candidate, not a demonstrated strategy.

## Deriv platform distinction

Deriv's new Options WebSocket supports public data and, when separately authenticated, options/multipliers contracts. A Deriv MT5 account-management API does **not** directly place MT5 CFD trades; an EA installed on an authorized MT5 terminal would be a different separately tested integration. SL/TP availability and semantics depend on contract/product; never assume all modes support the same bracket orders.

## Run locally

```bash
python -m pip install -r requirements.txt -r requirements-deriv.txt
python -m pytest -q tests/test_deriv_synthetics.py tests/test_trade_risk.py
python -m scripts.prepare_deriv_synthetics --discover
python -m scripts.prepare_deriv_synthetics --symbol 'Volatility 75 Index' --pages 5
```

No network is required for unit tests. Public discovery/history require an available connection to Deriv. Sources: https://developers.deriv.com/docs/data/active-symbols/ ; https://developers.deriv.com/docs/data/ticks-history/ ; https://developers.deriv.com/docs/options/ws-public/ ; https://legacy-docs.deriv.com/docs/mt5 ; https://deriv.com/markets/derived-indices/synthetic-indices
