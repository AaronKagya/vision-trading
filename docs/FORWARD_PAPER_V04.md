# Vision Trading AI · v0.4 forward-only paper monitor

**Important:** This is a research-only, near-live **baseline demonstration**. It is NOT an AI-generated trading signal, autonomous bot, real exchange account, broker paper account, or a strategy with demonstrated profitability. The validated LSTM experiment has not proven an out-of-period trading edge; its checkpoints are intentionally **not** used here.

## Engineering plan

- **Source:** Public, unauthenticated Binance spot `BTCUSDT` endpoints `GET /api/v3/klines` (15m) and `GET /api/v3/ticker/price`, on `data-api.binance.vision`. Neither API keys nor permissions to trade are used. A job can fail if Binance data is unavailable, blocked or discontinuous; never substitute fabricated candles.
- **Scheduler:** `7,22,37,52 * * * *` UTC, GitHub Actions cron on repository default branch. GitHub schedule triggers can be delayed or dropped, and are not a real-time streaming service. The workflow retrieves the separate feature branch and reads/updates `gh-pages/live/state.json` using a serialized Git push. The initial push trigger initializes the ledger; first activation never creates historical trade records.
- **Data integrity:** Accept *completed* 15-minute UTC OHLCV bars only; validate price positivity, candle continuity, timestamps, and staleness. Ignore any current incomplete candle. Require the latest closed candle to be recent. Do not create decisions if the verified market data source is unavailable.
- **Baseline rule (fixed before launch):** Once per UTC hour, at most 12 minutes after the hour's 15m candle closes, take `momentum = close[t] / close[t-4] - 1` using **only known historical candles**. If `momentum > 0`, record hypothetical **spot LONG** exposure for one hour; otherwise record **NO TRADE**. No shorts, leverage, or position sizing recommendation. Missed hourly windows are **not backfilled**.
- **Hypothetical entry/exit:** Use the public last-traded BTCUSDT ticker observed on the runner when the decision is recorded (never assign the earlier candle close as a forward fill). If a long was recorded, close it at the first *future* successful polling observation at least one hour later, using the newly fetched public ticker. The simulated holding period can be longer than 60 minutes when runs are missed. The ticker is a market observation, not an executable bid/ask quote or guaranteed fill.
- **Cost:** Assumed `0.0005` log units per side (= 5 bps each way). For each closed paper long, `gross_proxy = ln(exit_ticker / entry_ticker)` and `net_proxy = gross_proxy - 2 * 0.0005`. A flat decision's simulated log-P&L proxy is zero. Summed net log proxies are NOT broker-verified equity or actual dollar returns. Costs omit variable spread, market impact, financing, taxation, liquidity, and outages.
- **Persistence:** A single `state.json` file, committed to `gh-pages` with Actions `contents:write`. It contains activation timestamp, market observation, chart data, open trade if any, decision ledger, and metadata. It is public and carries no secrets. Concurrent runs are serialized; duplicate hourly decisions are suppressed; no rewriting historical rows to optimize results. GitHub Actions artifacts preserve a copy of each successful run.
- **Display:** Static GitHub Pages site `/live/` fetches a cache-busted `state.json`. Shows observed ticker, observation age, current hypothetical position, charts, paper outcomes, and a full ledger. Stale/missing data are marked clearly.

## Limitations and production gates

1. No execution endpoint is ever called; actual P&L, brokerage fill quality, slippage, real-money risk and regulatory compliance have not been established.
2. The momentum rule is deliberately simple. Treat it as a persistence and monitoring *control*, **not as trained AI** and not as a buy/sell recommendation.
3. A public website can report the last completed poll. It cannot guarantee current exchange prices or make scheduled GitHub Actions deterministic.
4. GitHub Pages data may be cached; the UI uses a cache-busting query and age labels, but availability is not assured.
5. A future legitimate AI version will require a versioned, approved checkpoint, same numerical feature/scaler provenance, a stable inference API, loggable predictions, and prospective scoring before any paper position is driven by a model.
6. The current Phase 4 study found no robust external predictive or net-profitability advantage. Do not describe this monitor as having learned the market.

## Local tests and one-shot simulation

```bash
python -m pytest -q tests/test_forward_paper.py
python -m scripts.run_forward_paper --state published/live/state.json
```

The second command contacts Binance's **public market-data API**. Start with a missing `state.json` to activate now, not to reconstruct a fictional historical trade record. The workflow writes a new observation per successful scheduled poll.

## Publishing arrangement

- Feature source: `feature/vision-trading-forward-paper-v04`.
- Static website: `gh-pages/live/index.html`.
- Scheduled monitor: `.github/workflows/vision-forward-paper.yml` on **default `main`** branch, because GitHub Actions only runs cron schedules from the default branch. This workflow checks out the feature branch, not a random internet repository.
- To disable monitoring immediately: disable the *Vision Trading BTCUSDT paper monitor* workflow in GitHub Actions. The published ledger remains viewable and becomes visibly stale.

No Volatility 75 data or live trading has been enabled.
