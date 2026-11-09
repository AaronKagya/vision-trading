import copy
import math
import pytest
from src.forward_paper import INTERVAL_MS, HOUR_MS, COST, advance, closed_candles, public_summary


def klines(last_close_hour=10, last_close_min=0, *, n=80, rising=True):
    # Fixed synthetic UTC 2026-11-09; test-only values, not published market results.
    import datetime
    end = int(datetime.datetime(2026,11,9,last_close_hour,last_close_min,tzinfo=datetime.timezone.utc).timestamp()*1000)
    assert end % INTERVAL_MS == 0
    result=[]
    for j in range(n):
        start=end - (n-j)*INTERVAL_MS
        close = (100 + (j * .2)) if rising else (115 - j * .2)
        op=close-.05
        result.append([start,str(op),str(close+.1),str(op-.1),str(close),'10', start+INTERVAL_MS-1])
    # Last kline is open starting exactly end (the still forming candle).
    result.append([end,'101','103','100','102','10',end+INTERVAL_MS-1])
    return result,end


def test_cannot_use_current_open_candle():
    rows,end=klines()
    complete=closed_candles(rows,end+7*60000)
    assert len(complete)==80
    assert complete[-1]['close_ms']==end


def test_rejects_invalid_gap_and_duplicate():
    rows,end=klines()
    rows.pop(6)
    with pytest.raises(ValueError,match='Duplicate, missing'):
        closed_candles(rows,end+7*60000)
    rows,end=klines()
    rows[3]=rows[2]
    with pytest.raises(ValueError,match='Duplicate, missing'):
        closed_candles(rows,end+7*60000)


def test_fail_closed_ohlc():
    rows,end=klines()
    rows[5][2]='-1'
    with pytest.raises(ValueError,match='Invalid price'):
        closed_candles(rows,end+7*60000)
    rows,end=klines()
    rows[5][4]='1000'
    with pytest.raises(ValueError,match='Invalid OHLCV'):
        closed_candles(rows,end+7*60000)


def test_first_poll_arms_only_no_history_trades():
    rows,t=klines()
    state=advance(None,rows,120.,t+7*60000)
    assert state['ledger']==[]
    assert state['activated_at_ms']==t+7*60000
    assert state['market']['last_trade_price_at_observation']==120.
    assert state['open_trade'] is None


def test_trade_is_only_forward_after_activation():
    rows,t=klines(10,0)
    old=advance(None,rows,120.,t+7*60000)
    rows2,t2=klines(11,0)
    newer=advance(old,rows2,121.,t2+7*60000)
    assert len(newer['ledger'])==1
    e=newer['ledger'][0]
    assert e['action']=='long' and e['status']=='open'
    assert e['entry_price']==121.  # NOT earlier candle close
    assert e['signal_candle_close_ms']==t2
    assert e['entry_observed_ms']==t2+7*60000
    assert e['momentum_1h']>0
    assert old['ledger']==[]  # immutable input


def test_flat_on_negative_momentum():
    rows,t=klines(10,0,rising=False)
    old=advance(None,rows,120.,t+7*60000)
    rows2,t2=klines(11,0,rising=False)
    newer=advance(old,rows2,119.,t2+7*60000)
    assert len(newer['ledger'])==1
    assert newer['ledger'][0]['status']=='flat'
    assert newer['open_trade'] is None


def test_settle_next_hour_with_roundtrip_cost():
    rows,t=klines(10,0)
    state=advance(None,rows,120.,t+7*60000)
    rows2,t2=klines(11,0)
    state=advance(state,rows2,121.,t2+7*60000)
    rows3,t3=klines(12,0)
    result=advance(state,rows3,122.21,t3+7*60000)
    first=result['ledger'][0]
    assert first['status']=='closed'
    assert first['exit_price']==122.21
    assert first['gross_log_pnl_proxy']==pytest.approx(math.log(122.21/121.))
    assert first['net_log_pnl_proxy']==pytest.approx(math.log(122.21/121.)-2*COST)
    assert public_summary(result)['metrics']['closed_longs']==1
    assert len(result['ledger'])==2  # new 12:00 hour decision


def test_no_backfill_if_poll_late():
    rows,t=klines(10,0)
    state=advance(None,rows,120.,t+7*60000)
    rows2,t2=klines(11,30)
    later=advance(state,rows2,121.,t2+7*60000)
    assert later['ledger']==[]


def test_no_duplicate_decision_after_repoll():
    rows,t=klines(10,0)
    state=advance(None,rows,120.,t+7*60000)
    rows2,t2=klines(11,0)
    state=advance(state,rows2,121.,t2+7*60000)
    reread=advance(state,rows2,121.1,t2+8*60000)
    assert len(reread['ledger'])==1
    assert reread['ledger'][0]['entry_price']==121.


def test_no_future_data_and_no_time_regression():
    rows,t=klines()
    with pytest.raises(ValueError):
        closed_candles(rows,t+3600*1000)
    s=advance(None,rows,120.,t+7*60000)
    with pytest.raises(ValueError, match='Time regression'):
        advance(s,rows,120.,t+6*60000)


def test_reject_config_changes():
    rows,t=klines()
    s=advance(None,rows,120.,t+7*60000)
    s['transaction_cost_per_side']=0.0
    with pytest.raises(ValueError, match='Cost configuration'):
        advance(s,rows,120.,t+8*60000)


def test_observation_outputs_do_not_fake_ai_forecast():
    rows,t=klines()
    x=public_summary(advance(None,rows,120.,t+7*60000))
    assert x['strategy_type']=='non_ai_baseline'
    assert x['metrics']['decisions']==0
    assert 'lstm_probability' not in str(x)
