"""Self-contained HTML evidence dashboard, generated solely from result files.

python -m src.report --results results
"""
from __future__ import annotations
import argparse
from html import escape
import json
from pathlib import Path
import re

NAMES={'majority':'Majority baseline','persistence':'Trend persistence',
    'logreg_lags':'Logistic regression','lstm_ohlc':'Numerical LSTM',
    'cnn_static':'Static chart CNN','wm_nolatent':'CNN + GRU','wm':'Visual world model',
    'fusion':'Numeric + chart fusion (Phase 2)'}

def build(results='results'):
    root=Path(results)
    r=json.loads((root/'metrics.json').read_text())
    models=r['models']
    ordered=sorted(models,key=lambda m:models[m]['accuracy'],reverse=True)
    trs=[];bars=[]
    for m in ordered:
        x=models[m]
        name=NAMES.get(m,m)
        ci=x['acc_ci'];pct=x['accuracy']*100
        trs.append('<tr><th>'+escape(name)+'</th><td>'+f'{pct:.2f}%</td><td>{ci[0]*100:.2f}–{ci[1]*100:.2f}%</td>'+
                   f'<td>{x["auc"]:.3f}</td><td>{x["sharpe_net"]:.2f}</td><td>{x["dsr"]:.3g}</td></tr>')
        bars.append(f'<div class="barrow"><span>{escape(name)}</span><div class="track"><div class="fill" style="width:{pct:.3f}%"></div></div><b>{pct:.2f}%</b></div>')
    rows='\n'.join(trs);bar_html='\n'.join(bars)
    provenance=r.get('data_provenance') or {}
    source='Synthetic simulation — software smoke test, not trading evidence' if provenance.get('synthetic') else 'Historical out-of-sample run — not forward live testing'
    data_label=escape(str(r['config'].get('data','unknown')))
    s='''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vision Trading | Research Results</title>
<style>body{font:15px/1.6 system-ui,Segoe UI,Arial,sans-serif;background:#0c1425;color:#e9f0ff;margin:0}main{max-width:1030px;margin:0 auto;padding:48px 22px}h1{font-size:2.5rem;line-height:1.12;margin:.25em 0}h2{margin:40px 0 16px}p{color:#b6c4de}small{color:#a5b5d5}.eyebrow{letter-spacing:.13em;text-transform:uppercase;color:#75c5ff;font-weight:700}.banner{border:1px solid #a28142;background:#30281e;color:#ffe0a0;padding:16px;border-radius:12px;margin:22px 0}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}.stat{border:1px solid #283751;border-radius:12px;padding:20px;background:#152139}.stat strong{display:block;font-size:1.75rem}table{width:100%;border-collapse:collapse;overflow:auto;font-size:.9rem}td,th{text-align:left;padding:11px 9px;border-bottom:1px solid #30405a}th{white-space:nowrap}.barrow{display:grid;grid-template-columns:180px 1fr 74px;gap:12px;align-items:center;margin:13px 0}.barrow span{font-size:.84rem}.track{height:14px;background:#1b2b43;border-radius:10px;overflow:hidden}.fill{height:100%;background:#4c9fe1;border-radius:10px}code{color:#bae1ff}footer{margin-top:45px;border-top:1px solid #30405a;padding-top:20px;color:#a5b5d5}@media(max-width:600px){.barrow{grid-template-columns:120px 1fr 65px;font-size:.75rem}main{padding:26px 12px}table{font-size:.73rem}td,th{padding:7px 3px}}</style>
</head><body><main><div class="eyebrow">Research-only · Vision Trading</div><h1>Do chart images help forecast better than numerical candles?</h1>
<p>Reproducible walk-forward evaluation with held-out blocks, embargoes, numerical baselines and visual encoders.</p>
<div class="banner"><strong>Not a verified trading system.</strong> Historical backtests cannot demonstrate future profits. Return estimates are approximate log-P&amp;L proxies and do not include real fills, slippage, leverage, funding or margin.</div>
<div class="stats"><div class="stat"><small>Out-of-sample predictions</small><strong>{n_test:,}</strong></div><div class="stat"><small>Test folds</small><strong>{folds}</strong></div><div class="stat"><small>Configurations evaluated</small><strong>{trials}</strong></div><div class="stat"><small>Test up-rate</small><strong>{up:.1f}%</strong></div></div>
<h2>Direction accuracy</h2>{bars}<h2>All model scores</h2><div style="overflow-x:auto"><table><thead><tr><th>Model</th><th>Accuracy</th><th>95% block CI</th><th>AUC</th><th>Net Sharpe proxy</th><th>DSR probability</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>Experimental provenance</h2><p><b>{source}</b><br>Data: <code>{data}</code><br>Seed: {seed} · Horizon: {horizon} 15-minute candles · Forecast image encoder: {encoder} · 15-minute spot market study.</p>
<p>CSV hash: <code>{hash}</code></p><h2>Interpretation</h2><p>Accuracy differences are not equivalent to a profitable trading strategy. The chart-based hypothesis is evaluated against numerical models receiving the same OHLC history. When additional model configurations are tried, all must be recorded before making adjusted significance claims.</p>
<footer>Produced by src.report from metrics.json. No exchange orders or broker connection.</footer></main></body></html>'''
    page=s.replace('{', '{{').replace('}', '}}')
    page=re.sub(r'\{\{(n_test|folds|trials|up|bars|rows|source|data|seed|horizon|encoder|hash)(:[^{}]+)?\}\}',
                lambda match: '{'+match.group(1)+(match.group(2) or '')+'}', page)
    page=page.format(n_test=r['n_test'],folds=r['config']['folds'],trials=r['n_trials'],up=r['up_rate_test']*100,
        bars=bar_html,rows=rows,source=source,data=data_label,seed=r['config']['seed'],horizon=r['config']['horizon'],
        encoder=escape(r['config']['encoder']),hash=escape(str(provenance.get('sha256') or 'not recorded')))
    path=root/'report.html';path.write_text(page,encoding='utf-8')
    print('Saved',path)
    return path

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='results');a=p.parse_args();build(a.results)
if __name__=='__main__':main()
