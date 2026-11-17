"""Discover Deriv standard synthetic indices and archive real 15m candle history.

PUBLIC data only. No token, no orders. Reject insufficient/gapped history.
"""
import argparse
import json
from src.deriv_synthetics import TRACKED_NAMES, download_recent, get_symbols, save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--discover',action='store_true')
    p.add_argument('--symbol',choices=TRACKED_NAMES)
    p.add_argument('--pages',type=int,default=4)
    p.add_argument('--out-dir',default='data/deriv')
    a=p.parse_args()
    if a.discover:
        print(json.dumps(get_symbols(),indent=2))
        return
    if not a.symbol:
        p.error('Set --symbol or --discover')
    df,meta=download_recent(a.symbol,pages=a.pages)
    csv,manifest=save(df,meta,a.out_dir)
    print(f'Public historical research: {len(df)} candles; {csv}; {manifest}')

if __name__=='__main__':main()
