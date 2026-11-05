"""Causally render 15-minute, trailing one-hour, trailing four-hour chart panels.

Each frame at t uses ONLY 15-minute candles with index <=t. The larger
views represent trailing, rolling aggregation, not exchange-aligned bars.
"""
import numpy as np
import pandas as pd

FACTORS = (1,4,16)

def first_valid_index(window=32):
    if window < 2:
        raise ValueError('window must be >=2')
    return window * max(FACTORS) - 1

def panels(df, window=32):
    first = first_valid_index(window)
    if len(df) <= first:
        raise ValueError(f'Need > {first} candles for window {window}')
    o,h,l,c = (df[col].to_numpy(dtype=np.float64) for col in ('open','high','low','close'))
    allviews=[]
    for f in FACTORS:
        opens = pd.Series(o).shift(f-1).to_numpy()
        highs = pd.Series(h).rolling(f, min_periods=f).max().to_numpy()
        lows = pd.Series(l).rolling(f, min_periods=f).min().to_numpy()
        allviews.append(np.stack([opens,highs,lows,c],axis=-1))
    t_idx = np.arange(first,len(df),dtype=np.int64)
    offs = (window-1-np.arange(window,dtype=np.int64))
    stacked = np.stack([allviews[i][t_idx[:,None]-offs[None,:]*f] for i,f in enumerate(FACTORS)],axis=1)
    assert np.isfinite(stacked).all()
    return t_idx,stacked

def render(pan,size=64,chunk=512):
    pan = np.asarray(pan)
    if pan.ndim!=4 or pan.shape[1]!=3 or pan.shape[-1]!=4:
        raise ValueError('panels must have shape (samples,3,window,4)')
    n,nviews,w,_=pan.shape
    if size%w!=0 or size<16:
        raise ValueError('image size must be >=16 and divisible by window')
    if not np.isfinite(pan).all():
        raise ValueError('Invalid numerical panels')
    px=size//w
    out=np.zeros((n,3,size,size),np.uint8)
    rows=np.arange(size)[None,None,:]
    for start in range(0,n,chunk):
        p=pan[start:start+chunk]
        # Bounds across all bars for EACH timeframe, preserving intra-view structure.
        hi=p[...,1].max(axis=2,keepdims=True)
        lo=p[...,2].min(axis=2,keepdims=True)
        span=np.maximum(hi-lo,1e-12)
        pixel=lambda val: np.rint((hi-val)/span*(size-1)).astype(np.int16)
        O,H,L,C=(pixel(p[...,k]) for k in range(4))
        img=out[start:start+len(p)]
        for j in range(w):
            up=(p[...,j,3]>=p[...,j,0])[...,None]
            wick=(rows>=H[...,j,None])&(rows<=L[...,j,None])
            top=np.minimum(O[...,j],C[...,j])[...,None]
            bottom=np.maximum(O[...,j],C[...,j])[...,None]
            body=(rows>=top)&(rows<=bottom)
            x=j*px
            img[...,x]=np.where(wick,90,img[...,x])
            for dx in range(px):
                img[...,x+dx]=np.where(body,np.where(up,255,128),img[...,x+dx])
    return out

def numeric_features(pan):
    """Numeric representation of identical OHLC, (samples,window,12)."""
    c_t=pan[:,0,-1,3][:,None,None,None]
    return np.log(pan/c_t).transpose(0,2,1,3).reshape(len(pan),pan.shape[2],12).astype(np.float32)
