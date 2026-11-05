"""Historical prediction explorer. Run: streamlit run app.py"""
import json
from pathlib import Path
import numpy as np
import streamlit as st
from src.data import load_candles,synthetic
from src.features import panels,render,first_valid_index

st.set_page_config(page_title='Vision Trading — Research Explorer',layout='wide')
st.title('Vision Trading · Historical Research Explorer')
st.warning('Research / paper simulation only. No live predictions, broker execution or verified investment returns.')
folder=Path(st.sidebar.text_input('Results folder','results'))
if not (folder/'metrics.json').exists() or not (folder/'preds.npz').exists():
    st.error('Missing metrics.json or preds.npz. Run the experiment first.');st.stop()
@st.cache_data(show_spinner='Loading experiment and images...')
def load(folder):
    folder=Path(folder);res=json.loads((folder/'metrics.json').read_text());pred=dict(np.load(folder/'preds.npz'))
    cfg=res['config']
    if cfg['data'].startswith('synthetic'):
        df=synthetic(cfg['n_synth'],'planted' if cfg['data']=='synthetic_planted' else 'null',cfg['seed'])
    else:
        df=load_candles(cfg['data'])
    selected=pred['t_idx']-first_valid_index(cfg['window'])
    if selected.min()<0 or selected.max()>=len(df):raise ValueError('Prediction index mismatch')
    return res,pred,df,selected
try:
    res,p,df,selected=load(str(folder))
except Exception as e:
    st.error(f'Unable to load experiment: {e}');st.stop()
models=list(res['models'])
st.caption(f"{res['n_test']:,} held-out predictions · {res['config']['folds']} folds · {len(models)} models · {res['config']['horizon']} candles ahead")
point=st.slider('Choose a historical test sample',0,len(p['y'])-1,len(p['y'])//2)
timestamp=df.time.iloc[int(p['t_idx'][point])]
st.subheader(f'{timestamp} · Actual: {"UP" if p["y"][point] else "DOWN"}')
_,pan=panels(df.iloc[:int(p['t_idx'][point])+1].copy(),res['config']['window'])
img=render(pan[-1:],res['config']['img'])[0]
c=st.columns(4)
c[0].image(img.transpose(1,2,0),caption='Combined R=15m, G=1h, B=4h',width=220,clamp=True)
for i,label in enumerate(['15-min','Trailing 1-hour','Trailing 4-hour']):
    c[i+1].image(img[i],caption=label,width=200,clamp=True)
rows=[]
for name in models:
    prob=float(p['p_'+name][point])
    rows.append({'model':name,'P(up)':round(prob,4),'forecast':'UP' if prob>=.5 else 'DOWN',
                 'correct':bool((prob>=.5)==p['y'][point])})
st.dataframe(rows,use_container_width=True)
st.subheader('Overall results')
st.dataframe([{'model':m,'accuracy':res['models'][m]['accuracy'],'AUC':res['models'][m]['auc'],
               'net Sharpe proxy':res['models'][m]['sharpe_net'],'DSR probability':res['models'][m]['dsr']} for m in models],
             use_container_width=True)
st.caption('Trading results are based on an approximate log-P&L proxy, not a brokerage backtest.')
