# Vision Trading AI v0.2 — evidence dashboard

**Status:** research prototype (historical-only; no live prices, brokerage account access, orders, or forecasts for current markets).

`site/index.html` is a self-contained offline dashboard generated from the **completed** GitHub Actions results, not fictional or manually substituted data.

## Open immediately

Open `site/index.html` in a modern browser on desktop or mobile. It works offline; no server or API keys are required. If your device blocks local HTML, run `python -m http.server 8000 --directory site` and open `http://localhost:8000/`.

## Rebuild from evidence

Download these **unaltered** GitHub Actions ZIP artifacts:

- `btcusdt-2025q1-locked-holdout` from [2025-Q1 locked holdout](https://github.com/AaronKagya/vision-trading/actions/runs/42043691500).
- `btcusdt-2024-five-seed-summary` from [2024 benchmark](https://github.com/AaronKagya/vision-trading/actions/runs/41995284584).

From the repository root:

```bash
python -m scripts.build_research_dashboard \
  --holdout-zip path/to/2025_holdout.zip \
  --summary-zip path/to/2024_summary.zip \
  --out site/index.html
python -m pytest -q tests
```

The builder rejects invalid probabilities, missing prediction arrays, unverified training fences, and incomplete five-seed benchmark summaries. It embeds only rounded prediction scores for visualization; use original ZIPs to reproduce exact metrics.

## Findings shown

- **2024**: 5 seeds, 4 walk-forward folds, 8 models, 17,308 unseen fold-test predictions per seed. The LSTM's average directional accuracy was **53.13%**, but its average simulated net Sharpe was **−6.14**. No learned model demonstrated reliable net profitability.
- **2025 Q1**: trained and validated on 2024 data only; 8,636 overlapping one-hour forward forecasts in January–March 2025. LSTM **51.31%** accuracy vs majority **50.37%**; one fixed training-inherited confidence cutoff of 0.05 selected **24.91%** of observations, with selected direction accuracy **51.79%**.
- The fixed abstention strategy's disjoint-interval gross log-P&L proxy was **−0.03526**; turnover **536 units** at per-unit change cost **0.0005** deducted **0.268**, leaving **−0.30326** net log-P&L proxy. This is not a portfolio return, nor a broker-executable backtest.
- Exploratory 96-candle (one-day) paired block bootstrap of the **overlapping** full 8,636 forecasts gives an LSTM-minus-majority accuracy advantage of **+0.94 percentage points**, indicative 95% interval approximately **[−0.79, +2.61] pp**. This includes zero.
- Taking only every 4th (one-hour, nonoverlapping) prediction, LSTM accuracy **49.93%** vs majority **50.86%** (2,159 observations). This is another reason not to claim predictive skill.

## Limitations and safety

The model's 2025 test set has now been inspected. Do not retrospectively optimize against it and call that an external test. A new independently held-out period, registered in advance, is needed before strategy claims. The cost proxy excludes broker execution, bid/ask, spreads, slippage, order-book liquidity, financing, margin, funding, liquidation risk and outage risk. Scores are not calibrated trading probabilities. There is no current live bot or deployed public website.

## Optional publishing

The static `site/index.html` can be hosted on GitHub Pages, Netlify, or Cloudflare Pages. Hosting is *not* automatically enabled and requires account-side deployment configuration. If using GitHub Pages, enable **Settings → Pages → Build and deployment: Deploy from a branch**, and select the appropriate branch and `/docs` or root destination. This repository's self-contained dashboard currently lives under `site/`, so copy `site/index.html` into the chosen Pages publishing root or configure a build/deploy action. Do **not** claim that the project is publicly hosted until an actual URL works.
