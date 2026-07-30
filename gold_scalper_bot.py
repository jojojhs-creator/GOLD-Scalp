#!/usr/bin/env python3
"""
Gold Scalper signal bot
=======================
Posts XAUUSD "Gold Scalper" signals (sweep + CISD) to a Telegram channel.
Mirrors the TradingView strategy. M15 is the VALIDATED keeper; M5/M1 are
EXPERIMENTAL (unvalidated) and get a warning label. Run each timeframe on its
own schedule (GitHub Actions cron), just after each bar close.

Env vars required:
  TELEGRAM_TOKEN     bot token from @BotFather
  TELEGRAM_CHAT_ID   channel id/username the bot posts to (e.g. @mygoldsignals or -1001234567890)
  TWELVEDATA_KEY     free API key from twelvedata.com
Optional:
  SYMBOL     (default "XAU/USD")
  INTERVAL   (default "15min"; use "5min" / "1min" for the experimental feeds)
  STATE_FILE (default "last_signal.txt"; give each feed its own file)

Signal logic is a faithful port of the Gold Scalper Pine strategy.
It only ever evaluates CLOSED bars, and de-dupes so each signal is posted once.
"""
import os
import sys
import json
import requests
import pandas as pd
import numpy as np

# -- Config -------------------------------------------------------------------
TG_TOKEN = os.environ["TELEGRAM_TOKEN"]
TG_CHAT  = os.environ["TELEGRAM_CHAT_ID"]
TD_KEY   = os.environ["TWELVEDATA_KEY"]
SYMBOL   = os.environ.get("SYMBOL", "XAU/USD")
INTERVAL = os.environ.get("INTERVAL", "15min")
STATE_FILE = os.environ.get("STATE_FILE", "last_signal.txt")

# -- Strategy params -- timeframe-aware, mirrors the Pine auto-tune (Medium) --
# M15 is the VALIDATED keeper. M5/M1 use the Pine's low-TF auto-tune values so
# the feed matches those charts, but they are EXPERIMENTAL / unvalidated.
def tf_params(interval):
    if interval == "1min":
        return 10, 1, 18   # basePiv, baseRun, baseArm  (M1)
    if interval == "5min":
        return 9, 1, 16    # (M5)
    return 9, 2, 14        # (M15 keeper -- default)

PIVOT_LEN, RUN_LEN, ARM_BARS = tf_params(INTERVAL)
COOLDOWN     = max(2, RUN_LEN + 1)
SL_BUF_ATR   = 0.10
MIN_RISK_ATR = 0.4
MAX_RISK_ATR = 1.5
FIRST_TP_R   = 0.6    # trim target
RUNNER_R     = 1.5    # TP2
ATR_LEN      = 14


def fetch_ohlc():
    url = "https://api.twelvedata.com/time_series"
    params = dict(symbol=SYMBOL, interval=INTERVAL, outputsize=300,
                  apikey=TD_KEY, order="ASC", timezone="UTC")
    r = requests.get(url, params=params, timeout=30)
    d = r.json()
    if "values" not in d:
        raise RuntimeError(f"Data API error: {d}")
    df = pd.DataFrame(d["values"])
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def wilder_atr(df, n=ATR_LEN):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()  # Pine ta.atr = RMA


def latest_signal(df):
    """Bar-by-bar port of the Gold Scalper entry. Returns a dict for the most
    recent CLOSED bar that produced a signal, or None."""
    df = df.copy()
    df["atr"] = wilder_atr(df)
    n = len(df)
    hi, lo, op, cl, at = (df[c].values for c in ["high", "low", "open", "close", "atr"])
    L = PIVOT_LEN

    # confirmed pivots: pivot at bar i is known at bar i+L
    piv_hi = [None] * n
    piv_lo = [None] * n
    for i in range(L, n - L):
        seg_h = hi[i - L:i + L + 1]
        seg_l = lo[i - L:i + L + 1]
        if hi[i] == seg_h.max():
            piv_hi[i] = hi[i]
        if lo[i] == seg_l.min():
            piv_lo[i] = lo[i]

    pool_hi, pool_lo = [], []          # unswept liquidity
    sweep_hi_px = sweep_lo_px = None
    armed_bull = armed_bear = -10 ** 9
    # CISD run trackers
    cur_dn_open = cur_up_open = None
    cur_dn_len = cur_up_len = 0
    cisd_bull_lvl = cisd_bear_lvl = None
    last_sig_bar = -10 ** 9
    result = None

    for b in range(L, n):
        atrv = at[b]
        if np.isnan(atrv):
            continue

        # add a pool when a pivot confirms and price hasn't broken it since
        pidx = b - L
        if pidx >= 0:
            if piv_hi[pidx] is not None and hi[pidx + 1:b + 1].max(initial=-1e18) <= piv_hi[pidx]:
                pool_hi.append(piv_hi[pidx])
                pool_hi[:] = pool_hi[-6:]
            if piv_lo[pidx] is not None and lo[pidx + 1:b + 1].min(initial=1e18) >= piv_lo[pidx]:
                pool_lo.append(piv_lo[pidx])
                pool_lo[:] = pool_lo[-6:]

        # sweeps (fade): high takes a pool high -> arm SELL; low takes a pool low -> arm BUY
        for lvl in list(pool_hi):
            if hi[b] > lvl:
                sweep_hi_px = hi[b]
                armed_bear = b
                pool_hi.remove(lvl)
        for lvl in list(pool_lo):
            if lo[b] < lvl:
                sweep_lo_px = lo[b]
                armed_bull = b
                pool_lo.remove(lvl)

        # CISD
        is_dn = cl[b] < op[b]
        is_up = cl[b] > op[b]
        prev_dn = cl[b - 1] < op[b - 1]
        prev_up = cl[b - 1] > op[b - 1]
        if is_dn:
            cur_dn_open = cur_dn_open if prev_dn else op[b]
            cur_dn_len = cur_dn_len + 1 if prev_dn else 1
        if is_up:
            cur_up_open = cur_up_open if prev_up else op[b]
            cur_up_len = cur_up_len + 1 if prev_up else 1
        if is_up and prev_dn and cur_dn_len >= RUN_LEN and cur_dn_open is not None:
            cisd_bull_lvl = cur_dn_open
        if is_dn and prev_up and cur_up_len >= RUN_LEN and cur_up_open is not None:
            cisd_bear_lvl = cur_up_open
        cisd_bull = cisd_bull_lvl is not None and cl[b] > cisd_bull_lvl and cl[b - 1] <= cisd_bull_lvl
        cisd_bear = cisd_bear_lvl is not None and cl[b] < cisd_bear_lvl and cl[b - 1] >= cisd_bear_lvl

        bull_armed = b - armed_bull <= ARM_BARS
        bear_armed = b - armed_bear <= ARM_BARS
        cooled = b - last_sig_bar >= COOLDOWN

        def risk_capped(raw):
            return min(max(raw, atrv * MIN_RISK_ATR), atrv * MAX_RISK_ATR)

        sig = None
        if bull_armed and cisd_bull and sweep_lo_px is not None and cooled:
            riskC = risk_capped(cl[b] - (sweep_lo_px - atrv * SL_BUF_ATR))
            if riskC > 0:
                sig = dict(dir="BUY", entry=cl[b], sl=cl[b] - riskC,
                           trim=cl[b] + riskC * FIRST_TP_R, tp2=cl[b] + riskC * RUNNER_R)
        elif bear_armed and cisd_bear and sweep_hi_px is not None and cooled:
            riskC = risk_capped((sweep_hi_px + atrv * SL_BUF_ATR) - cl[b])
            if riskC > 0:
                sig = dict(dir="SELL", entry=cl[b], sl=cl[b] + riskC,
                           trim=cl[b] - riskC * FIRST_TP_R, tp2=cl[b] - riskC * RUNNER_R)

        if sig is not None:
            last_sig_bar = b
            sig["time"] = str(df["datetime"].iloc[b])
            result = sig

    return result


def load_last():
    try:
        with open(STATE_FILE) as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def save_last(t):
    with open(STATE_FILE, "w") as f:
        f.write(t)


TF_LABELS = {"1min": "M1", "5min": "M5", "15min": "M15", "30min": "M30"}


def fmt(sig):
    emoji = "\U0001F7E2" if sig["dir"] == "BUY" else "\U0001F534"
    p = lambda x: f"{x:.2f}"
    label = TF_LABELS.get(INTERVAL, INTERVAL)
    msg = (f"{emoji} {sig['dir']} XAUUSD {label}\n"
           f"Entry {p(sig['entry'])}\n"
           f"SL {p(sig['sl'])}\n"
           f"TP {p(sig['tp2'])}  (trim 50% at {p(sig['trim'])}, then breakeven)")
    # M15 is the only validated frame; everything else is flagged clearly.
    if INTERVAL != "15min":
        msg += "\n\n⚠️ EXPERIMENTAL -- unvalidated, for testing only."
    return msg


def send(text):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    r = requests.post(url, json=dict(chat_id=TG_CHAT, text=text), timeout=30)
    r.raise_for_status()


def main():
    df = fetch_ohlc()
    # Twelve Data's last row is the still-forming bar; drop it -> only closed bars
    df = df.iloc[:-1].reset_index(drop=True)
    sig = latest_signal(df)
    if sig is None:
        print("No signal on the latest closed bar.")
        return
    # only fire on a fresh signal (within the last 3 closed bars, to tolerate
    # scheduler delays) and never post the same one twice
    recent_bars = {str(t) for t in df["datetime"].iloc[-3:]}
    if sig["time"] not in recent_bars:
        print(f"Latest signal is stale ({sig['time']}). Skip.")
        return
    if load_last() == sig["time"]:
        print("Already posted this signal.")
        return
    send(fmt(sig))
    save_last(sig["time"])
    print(f"Posted {sig['dir']} @ {sig['time']}")


if __name__ == "__main__":
    main()
