import pytest
import pandas as pd
from src.deriv_synthetics import discover, parse_candles, validate_history, download_recent, GRANULARITY


def make_rows(n=700,now=1794902400):
    now=(now//900)*900
    return [{'epoch':now-(n-j)*GRANULARITY,'open':100+j/10,'high':100.3+j/10,
             'low':99.9+j/10,'close':100.1+j/10} for j in range(n)]


def test_broker_codes_are_discovered_not_assumed():
    x=discover([{'display_name':'Volatility 75 Index','symbol':'R_75'},
                {'display_name':'Volatility 75 (1s) Index','symbol':'1HZ75V'},
                {'display_name':'Jump 50 Index','symbol':'JD50'}])
    assert set(x)=={'Volatility 75 Index','Jump 50 Index'}
    assert x['Volatility 75 Index']['symbol']=='R_75'


def test_missing_symbol_fails_without_guess():
    calls=[]
    def responder(req):
        calls.append(req)
        return {'msg_type':'active_symbols','active_symbols':[{'symbol':'R_10','display_name':'Volatility 10 Index'}]}
    with pytest.raises(ValueError,match='not returned'):
        download_recent('Jump 75 Index',fetch=responder)
    assert len(calls)==1


def test_duplicate_symbol_metadata_rejected():
    with pytest.raises(ValueError,match='Ambiguous'):
        discover([{'display_name':'Jump 75 Index','symbol':'J1'},
                  {'display_name':'Jump 75 Index','symbol':'J2'}])


def test_wrong_message_and_bad_ohlc_rejected():
    with pytest.raises(ValueError,match='Expected'):
        parse_candles({'msg_type':'history','history':{}})
    bad=make_rows(1);bad[0]['high']=0.1
    with pytest.raises(ValueError,match='Inconsistent'):
        parse_candles({'msg_type':'candles','candles':bad})


def test_incomplete_future_bar_excluded():
    now=1794902400
    candles=make_rows(700,now)
    # add a new in-progress bar opening at now
    candles.append({'epoch':now,'open':170,'high':171,'low':169,'close':170.5})
    df=validate_history(candles,observed_epoch=now+60)
    assert len(df)==700
    assert df.close.iloc[-1]==pytest.approx(candles[-2]['close'])
    assert (df.volume == 0).all()


def test_gap_detection_fails_closed():
    candles=make_rows(700)
    candles.pop(13)
    with pytest.raises(ValueError,match='gaps'):
        validate_history(candles,observed_epoch=1794902460)


def test_pagination_and_stale_check():
    now=1794902400
    rows=make_rows(2000,now)
    reqs=[]
    def respond(req):
        reqs.append(req)
        if 'active_symbols' in req:
            return {'msg_type':'active_symbols','active_symbols':[{'symbol':'JD50','display_name':'Jump 50 Index'}]}
        end=now+900 if req['end']=='latest' else int(req['end'])
        candidates=[r for r in rows if r['epoch']<=end]
        return {'msg_type':'candles','candles':candidates[-1000:]}
    df,meta=download_recent('Jump 50 Index',pages=2,fetch=respond,now_epoch=now+60)
    assert len(df)==2000
    assert reqs[1]['ticks_history']=='JD50'
    assert reqs[2]['end']==str(rows[1000]['epoch']-1)
    assert meta['symbol']=='JD50' and 'no traded volume' in meta['volume_note']


def test_reject_trading_request_even_through_public_client():
    from src.deriv_synthetics import fetch_public
    with pytest.raises(ValueError,match='prohibited'):
        fetch_public({'buy':1})


def test_training_panels_can_use_deriv_candles_without_seeing_future():
    from src.features import panels
    df=validate_history(make_rows(700),observed_epoch=1794902460)
    idx,pan=panels(df,32)
    assert len(idx)==len(df)-511
    mutated=df.copy()
    mutated.loc[699,'close']=mutated.loc[699,'close']+0.1
    # Its change cannot affect the panel earlier than final sample.
    _,new=panels(mutated,32)
    assert (pan[:-1]==new[:-1]).all()
