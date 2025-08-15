# analysis.py

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime

# ===================== Numeric helpers & Indicators =====================

def np_safe(arr: List[float]) -> np.ndarray:
    return np.array(arr, dtype=float) if arr else np.array([], dtype=float)

def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else 0.0

def safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if x.size else 0.0

def percentile(x: np.ndarray, p: float) -> float:
    return float(np.percentile(x, p)) if x.size else 0.0

def rsi_series(closes: List[float], length: int = 14) -> np.ndarray:
    closes_np = np_safe(closes)
    if closes_np.size < length + 1:
        return np.full(closes_np.size, 50.0)
    delta = np.diff(closes_np)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    roll_up = np.convolve(gains, np.ones(length, dtype=float), 'valid') / length
    roll_down = np.convolve(losses, np.ones(length, dtype=float), 'valid') / length
    rs = np.divide(roll_up, roll_down, out=np.full_like(roll_up, np.nan), where=roll_down != 0)
    rsi = 100 - (100 / (1 + rs))
    pad = np.full(closes_np.size - rsi.size, 50.0)
    return np.concatenate([pad, rsi])

def atr_series(highs: List[float], lows: List[float], closes: List[float], length: int = 14) -> np.ndarray:
    highs_np = np_safe(highs)
    lows_np = np_safe(lows)
    closes_np = np_safe(closes)
    if closes_np.size < length + 1:
        return np.full(closes_np.size, 0.0)
    prev_close = np.concatenate([[closes_np[0]], closes_np[:-1]])
    tr = np.maximum(highs_np - lows_np, np.maximum(np.abs(highs_np - prev_close), np.abs(lows_np - prev_close)))
    atr = np.convolve(tr, np.ones(length, dtype=float), 'valid') / length
    pad = np.full(tr.size - atr.size, atr[0] if atr.size else 0.0)
    return np.concatenate([pad, atr])

def ema_series(values: List[float], length: int) -> np.ndarray:
    values_np = np_safe(values)
    if values_np.size == 0:
        return values_np
    alpha = 2 / (length + 1)
    out = np.empty_like(values_np)
    out[0] = values_np[0]
    for i in range(1, values_np.size):
        out[i] = alpha * values_np[i] + (1 - alpha) * out[i - 1]
    return out

def sma_series(values: List[float], length: int) -> np.ndarray:
    v = np_safe(values)
    if v.size < length:
        return np.full(v.size, safe_mean(v) if v.size else 0.0)
    kernel = np.ones(length) / length
    out = np.convolve(v, kernel, mode='valid')
    pad = np.full(v.size - out.size, out[0] if out.size else 0.0)
    return np.concatenate([pad, out])

def cvd_proxy(closes: List[float], volumes: List[float]) -> np.ndarray:
    closes_np = np_safe(closes)
    volumes_np = np_safe(volumes)
    if closes_np.size < 2 or volumes_np.size != closes_np.size:
        return np.cumsum(np.zeros_like(closes_np))
    delta = np.diff(closes_np)
    sign = np.sign(delta)
    sign = np.concatenate([[0.0], sign])
    delta_vol = sign * volumes_np
    return np.cumsum(delta_vol)

def rel_volume(volumes: List[float], lookback=20) -> float:
    v = np_safe(volumes)
    if v.size < lookback + 1:
        return 1.0
    last = v[-1]
    avg = safe_mean(v[-lookback-1:-1])
    return float(last / avg) if avg > 0 else 1.0

def volatility_metric(closes: List[float], win=30):
    closes_np = np_safe(closes)
    if closes_np.size < win:
        return 0.0
    seg = closes_np[-win:]
    mu = float(np.mean(seg))
    return float(np.std(seg) / mu) if mu else 0.0

# ===================== Pattern & Signal Detection =====================

def sell_side_liquidity_sweep_bullish(highs, lows, opens, closes, lookback=20):
    lows_np = np_safe(lows)
    opens_np = np_safe(opens)
    closes_np = np_safe(closes)
    if lows_np.size < lookback + 2:
        return False
    prior_low = float(np.min(lows_np[-(lookback+1):-1]))
    sweep = (lows_np[-1] < prior_low) and (closes_np[-1] > prior_low) and (closes_np[-1] > opens_np[-1])
    return bool(sweep)

def displacement_bullish(highs, lows, opens, closes, atr, body_ratio=0.6, atr_mult=1.2):
    highs_np = np_safe(highs)
    lows_np = np_safe(lows)
    opens_np = np_safe(opens)
    closes_np = np_safe(closes)
    if highs_np.size < 2 or atr.size != highs_np.size:
        return False
    rng = highs_np[-1] - lows_np[-1]
    body = closes_np[-1] - opens_np[-1]
    if rng <= 0:
        return False
    cond = (closes_np[-1] > opens_np[-1]) and (rng > atr_mult * atr[-1]) and ((body / rng) >= body_ratio)
    return bool(cond)

def bullish_rsi_divergence(closes: List[float], rsi: np.ndarray, lookback=20) -> bool:
    closes_np = np_safe(closes)
    if closes_np.size < lookback + 5 or rsi.size != closes_np.size:
        return False
    segment = closes_np[-(lookback+5):]
    if segment.size < 6:
        return False
    # find two relative lows
    local_idx = np.argsort(segment)[:6]
    chosen = []
    for idx in local_idx:
        abs_idx = idx + (closes_np.size - segment.size)
        if not chosen or abs(abs_idx - chosen[-1]) > 2:
            chosen.append(int(abs_idx))
        if len(chosen) >= 2:
            break
    if len(chosen) < 2:
        return False
    i1, i2 = chosen[0], chosen[1]
    price_ll = closes_np[i1] < closes_np[i2]
    rsi_hl = rsi[i1] > rsi[i2]
    return bool(price_ll and rsi_hl)

def whale_entry(volumes: List[float], closes: List[float], factor=3.0):
    volumes_np = np_safe(volumes)
    closes_np = np_safe(closes)
    if volumes_np.size < 20 or closes_np.size < 2:
        return False
    last = volumes_np[-1]
    mean = float(np.mean(volumes_np[-20:]))
    return bool((last > mean * factor) and (closes_np[-1] > closes_np[-2]))

def cvd_imbalance_up(cvd: np.ndarray, bars=5, mult=1.6):
    if cvd.size < bars + 1:
        return False
    slope = cvd[-1] - cvd[-bars]
    ref_seg = np.diff(cvd[:-bars]) if cvd.size >= bars + 11 else np.diff(cvd)
    ref_std = safe_std(ref_seg)
    if ref_std == 0:
        return bool(slope > 0)
    return bool(slope > mult * ref_std)

# --- Bands & Squeeze ---
def bbands(values: List[float], length=20, mult=2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v = np_safe(values)
    ma = sma_series(values, length)
    std = np.zeros_like(v)
    if v.size >= length:
        for i in range(length-1, v.size):
            std[i] = np.std(v[i-length+1:i+1])
    upper = ma + mult * std
    lower = ma - mult * std
    width = (upper - lower) / ma.clip(min=1e-9)
    return lower, ma, upper, width

def keltner_channels(highs: List[float], lows: List[float], closes: List[float], length=20, mult=1.5):
    ema = ema_series(closes, length)
    atr = atr_series(highs, lows, closes, 20)
    upper = ema + mult * atr
    lower = ema - mult * atr
    return lower, ema, upper

def is_bb_inside_kc(highs, lows, closes, length=20, bb_mult=2.0, kc_mult=1.5) -> bool:
    bb_l, bb_m, bb_u, _ = bbands(closes, length, bb_mult)
    kc_l, kc_m, kc_u = keltner_channels(highs, lows, closes, length, kc_mult)
    if len(bb_l) == 0:
        return False
    return bool(bb_u[-1] < kc_u[-1] and bb_l[-1] > kc_l[-1])

# --- FVG ---
def find_bullish_fvg_indices(highs: List[float], lows: List[float]) -> List[int]:
    out = []
    for n in range(2, len(highs)):
        if lows[n] > highs[n-2]:
            out.append(n)
    return out

def last_fvg_zone(highs: List[float], lows: List[float]):
    idxs = find_bullish_fvg_indices(highs, lows)
    if not idxs:
        return None
    n = idxs[-1]
    gap_top = lows[n]
    gap_bottom = highs[n-2]
    return (float(gap_top), float(gap_bottom), int(n))

def bullish_fvg_alert_logic(opens, highs, lows, closes, volumes, tf_label: str):
    opens_np = np_safe(opens)
    highs_np = np_safe(highs)
    lows_np = np_safe(lows)
    closes_np = np_safe(closes)
    volumes_np = np_safe(volumes)
    if closes_np.size < 60:
        return None
    zone = last_fvg_zone(highs_np.tolist(), lows_np.tolist())
    if not zone:
        return None
    gap_top, gap_bottom, idx_fvg = zone
    start = idx_fvg + 1
    if start + 3 >= closes_np.size:
        return None
    inside_idx = None
    for i in range(start, closes_np.size - 1):
        body_low = float(min(opens_np[i], closes_np[i]))
        body_high = float(max(opens_np[i], closes_np[i]))
        if (gap_bottom <= body_low) and (body_high <= gap_top):
            inside_idx = i
            break
    if inside_idx is None:
        return None
    cvd = cvd_proxy(closes_np.tolist(), volumes_np.tolist())
    ref_slice = volumes_np[max(0, inside_idx-20):inside_idx]
    ref_mean = float(np.mean(ref_slice)) if ref_slice.size else 0.0
    vol_rise = bool(volumes_np[inside_idx] > 1.25 * ref_mean)
    cvd_slice = cvd[max(0, inside_idx-20):inside_idx+1]
    cvd_diff = np.diff(cvd_slice) if cvd_slice.size >= 2 else np.array([])
    cvd_std = safe_std(cvd_diff)
    cvd_rise = bool((cvd[inside_idx] - cvd[max(0, inside_idx-5)]) > 1.5 * cvd_std) if cvd_std > 0 else bool(cvd[inside_idx] - cvd[max(0, inside_idx-5)] > 0)
    if not (vol_rise or cvd_rise):
        return None
    for j in range(inside_idx + 1, closes_np.size):
        if (closes_np[j] > gap_top) and (closes_np[j] > opens_np[j]):
            return f"Bullish FVG Confirmed ({tf_label})"
    return None

# ===================== Structure & Swing Logic =====================

def swing_high_idx(highs: List[float], lookback=5) -> Optional[int]:
    h = np_safe(highs)
    if h.size < 2*lookback+1:
        return None
    i = h.size - 1 - lookback
    window = h[i-lookback:i+lookback+1]
    return i if h[i] == np.max(window) else None

def swing_low_idx(lows: List[float], lookback=5) -> Optional[int]:
    lows_np = np_safe(lows)
    if lows_np.size < 2*lookback+1:
        return None
    i = lows_np.size - 1 - lookback
    window = lows_np[i-lookback:i+lookback+1]
    return i if lows_np[i] == np.min(window) else None

def broke_above_previous_swing_high(closes: List[float], highs: List[float], lookback=5) -> bool:
    h = np_safe(highs)
    c = np_safe(closes)
    if h.size < lookback*2+2 or c.size < lookback*2+2:
        return False
    for offset in range(lookback+1, min(lookback*6, h.size-1)):
        i = h.size - 1 - offset
        if i < lookback or i+lookback >= h.size:
            continue
        if h[i] == np.max(h[i-lookback:i+lookback+1]):
            return bool(c[-1] > h[i])
    return False

# ===================== Profile Functions =====================

def profile_htf_sweep_mss_fvg(o1h,h1h,l1h,c1h,v1h,
                              o4h,h4h,l4h,c4h,v4h,
                              o15,h15,l15,c15,v15) -> Tuple[bool,List[str]]:
    reasons = []
    sweep_1h = sell_side_liquidity_sweep_bullish(h1h,l1h,o1h,c1h,lookback=20)
    sweep_4h = sell_side_liquidity_sweep_bullish(h4h,l4h,o4h,c4h,lookback=20) if len(c4h)>0 else False
    if sweep_1h or sweep_4h:
        reasons.append("HTF Sell-side Sweep")
    atr15 = atr_series(h15,l15,c15,14)
    disp_15 = displacement_bullish(h15,l15,o15,c15,atr15,0.55,1.15)
    choch_1h = broke_above_previous_swing_high(c1h,h1h,lookback=5)
    if disp_15 or choch_1h:
        reasons.append("MSS/ChoCH + Displacement")
    fvg_15 = bullish_fvg_alert_logic(o15,h15,l15,c15,v15,"15M")
    fvg_1h = bullish_fvg_alert_logic(o1h,h1h,l1h,c1h,v1h,"1H")
    if fvg_15 or fvg_1h:
        reasons.append("FVG Reclaim")
    cvd15 = cvd_proxy(c15,v15)
    cvd_ok = cvd_imbalance_up(cvd15,bars=5,mult=1.5)
    relv_ok = rel_volume(v15,20) >= 1.5
    if cvd_ok and relv_ok:
        reasons.append("CVD Ramp + RelVol↑")
    ok = (len(reasons) >= 3)
    return ok, reasons

def profile_bullish_fvg_htf(o1h,h1h,l1h,c1h,v1h, o4h,h4h,l4h,c4h,v4h, o1d,h1d,l1d,c1d,v1d) -> Tuple[bool,List[str]]:
    reasons=[]
    for tf_name, (o,h,lows,c,v) in [("1D",(o1d,h1d,l1d,c1d,v1d)), ("4H",(o4h,h4h,l4h,c4h,v4h))]:
        if len(c)==0:
            continue
        alert = bullish_fvg_alert_logic(o,h,lows,c,v,tf_name)
        if alert:
            reasons.append(f"{alert}")
    ok = len(reasons) >= 1
    return ok, reasons

def profile_squeeze_expansion(o1h,h1h,l1h,c1h,v1h) -> Tuple[bool,List[str]]:
    reasons=[]
    squeeze = is_bb_inside_kc(h1h,l1h,c1h,length=20, bb_mult=2.0, kc_mult=1.5)
    _,_,_, bb_width = bbands(c1h,20,2.0)
    width_p20_ok = False
    if len(bb_width)>30:
        p20 = percentile(bb_width[-30:], 20)
        width_p20_ok = bb_width[-1] <= p20
    if squeeze and width_p20_ok:
        bb_l, bb_m, bb_u, _ = bbands(c1h,20,2.0)
        relv = rel_volume(v1h,20)
        if len(c1h)>0 and len(bb_u)>0 and (c1h[-1] > bb_u[-1]) and relv >= 1.5:
            cvd1h = cvd_proxy(c1h,v1h)
            if cvd_imbalance_up(cvd1h,bars=3,mult=1.2):
                reasons.append("Squeeze→Expansion (1H)")
    ok = len(reasons) >= 1
    return ok, reasons

def profile_rs_breakout_vs_btc(c1h_coin: List[float], c1h_btc: List[float],
                               h1h: List[float], c15: List[float], h15: List[float]) -> Tuple[bool,List[str]]:
    reasons=[]
    c_coin = np_safe(c1h_coin)
    c_btc = np_safe(c1h_btc)
    if c_coin.size<25 or c_btc.size<25:
        return False, reasons
    rs = c_coin / np.clip(c_btc,1e-9,None)
    rs_hh = bool(rs[-1] >= np.max(rs[-20:]))
    if rs_hh:
        bos_1h = broke_above_previous_swing_high(c1h_coin,h1h,5)
        disp_15 = displacement_bullish(h15,h15,h15,c15,atr_series(h15,h15,c15,14),0.5,1.1) if len(h15)==len(c15) and len(h15)>0 else False
        if bos_1h or disp_15:
            reasons.append("RS Breakout vs BTC (1H)")
    ok = len(reasons) >= 1
    return ok, reasons

def atr_vol_gate(highs_15, lows_15, closes_15):
    atr15 = atr_series(highs_15, lows_15, closes_15, 14)
    atr_slice = atr15[-100:] if atr15.size >= 100 else atr15
    if atr_slice.size < 20:
        return False
    cur_atr = float(atr15[-1])
    p40 = percentile(atr_slice, 40)
    return bool(cur_atr >= p40 and cur_atr > 0)

def htf_bullish_bias(closes_1h: List[float], closes_4h: List[float]) -> bool:
    if len(closes_1h) < 60:
        return False
    ema1h = ema_series(closes_1h, 50)
    cond_1h = (closes_1h[-1] > ema1h[-1]) and (ema1h[-1] > ema1h[-6])
    cond_4h = False
    if len(closes_4h) >= 60:
        ema4h = ema_series(closes_4h, 50)
        cond_4h = (closes_4h[-1] > ema4h[-1]) and (ema4h[-1] > ema4h[-6])
    return bool(cond_1h or cond_4h)