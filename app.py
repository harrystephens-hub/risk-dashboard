import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
from datetime import datetime, timedelta
import os

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="Global Risk Dashboard", page_icon="🌍", layout="wide")
st.title("🌍 Global Risk & Capital Flow Dashboard")

# ============================================================
# LEGEND DATA (keys aligned with classification function)
# ============================================================
GAUGE_LEGENDS = {
    "VIX": [
        ("< 15", "Very low — complacency risk", "#22C55E"),
        ("15–20", "Calm", "#86EFAC"),
        ("20–25", "Elevated caution", "#FDE047"),
        ("25–30", "High caution", "#FB923C"),
        ("> 30", "Panic / extreme fear", "#EF4444"),
    ],
    "HY Spread": [
        ("< 300 bps", "Low stress", "#22C55E"),
        ("300–500 bps", "Moderate", "#FDE047"),
        ("> 500 bps", "Credit stress", "#EF4444"),
    ],
    "10Y-3M": [
        ("> 1.0%", "Steep — normal", "#22C55E"),
        ("0.0–1.0%", "Flattening — watch", "#FDE047"),
        ("< 0.0%", "Inverted — recession risk", "#EF4444"),
    ],
    "DXY": [
        ("< 95", "Weak USD — risk-on", "#22C55E"),
        ("95–105", "Neutral range", "#FDE047"),
        ("> 105", "Strong USD — risk-off", "#EF4444"),
    ],
    "SPY/TLT": [
        ("Rising", "Risk-on — stocks leading", "#22C55E"),
        ("Flat", "Neutral", "#FDE047"),
        ("Falling", "Risk-off — bonds leading", "#EF4444"),
    ],
    "Copper/Gold": [
        ("Rising", "Growth optimism", "#22C55E"),
        ("Flat", "Neutral", "#FDE047"),
        ("Falling", "Safety demand", "#EF4444"),
    ],
    "USD/JPY": [
        ("Rising", "Risk-on — yen weakening", "#22C55E"),
        ("Flat", "Neutral", "#FDE047"),
        ("Falling", "Risk-off — yen strengthening", "#EF4444"),
    ],
}

EXPLAINERS = {
    "VIX": "The VIX measures expected S&P 500 volatility implied by options prices. It's called the 'fear index' — when investors pay up for protection, VIX rises. Values above 30 usually coincide with market panics.",
    "HY Spread": "The high-yield (junk) bond spread is the extra yield investors demand over Treasuries to hold risky corporate debt. Widening spreads mean bond markets are pricing in higher default risk — often an early warning of economic trouble.",
    "10Y-3M": "The 10-year vs 3-month Treasury spread is the Fed's preferred recession indicator. When short-term rates exceed long-term rates (inversion), it signals markets expect a sharp slowdown and rate cuts. Unlike 10Y-2Y, it avoided false inversion signals in 2022-23. It has preceded every US recession since 1960.",
    "DXY": "The US Dollar Index measures the dollar against a basket of major currencies. A rising dollar means global capital is flowing into USD safe-haven assets, tightening financial conditions worldwide. A falling dollar often signals risk appetite and capital flowing to emerging markets.",
    "SPY/TLT": "This ratio compares US equities (SPY) to long-term Treasury bonds (TLT). When it rises, stocks are outperforming bonds — investors are taking risk. When it falls, bonds are winning — investors are seeking safety. It's a pure risk appetite gauge.",
    "Copper/Gold": "Copper is an industrial metal tied to global growth. Gold is the ultimate safe haven. The ratio rises when growth optimism dominates, and falls when fear and safety demand take over. Often called 'Dr. Copper vs Dr. Gold'.",
    "USD/JPY": "USD/JPY is the most reliable real-time risk barometer in FX markets. The yen is the ultimate safe-haven currency due to Japan's creditor status. When USD/JPY falls, capital is fleeing to yen safety — often faster than VIX reacts. When it rises, risk appetite is strong.",
    "AUD/USD": "The Australian dollar is a commodity currency tied to global growth and Chinese demand. When AUD/USD rises, it signals risk appetite and growth expectations. When it falls, markets are pricing in a global slowdown.",
    "EEM/SPY": "This ratio compares emerging market equities to US equities. When it rises, capital is flowing into riskier developing markets — a sign of genuine risk appetite. When it falls, money is retreating to the perceived safety of US markets.",
    "HYG/LQD": "This ratio compares high-yield (junk) bonds to investment-grade corporate bonds. When it rises, investors are reaching for yield and taking credit risk. When it falls, they're hiding in safer IG debt.",
    "XLY/XLP": "This ratio compares consumer discretionary stocks to consumer staples. When it rises, investors expect growth and are buying cyclical names. When it falls, they're hiding in defensive staples — a classic risk-off rotation.",
}

# ============================================================
# LOAD DATA (cached for 12 hours)
# ============================================================
@st.cache_data(ttl=14400)
def load_all_data():
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        st.error("FRED_API_KEY not set in Streamlit secrets")
        st.stop()
    fred = Fred(api_key=fred_key)

    today = datetime.today()
    hist_start = today - timedelta(days=3*365)

    # Yahoo Finance: 5 years of daily data
    tickers = {"VIX":"^VIX","SPY":"SPY","TLT":"TLT","EEM":"EEM","HYG":"HYG","LQD":"LQD",
               "DXY":"DX-Y.NYB","Copper":"HG=F","Gold":"GC=F",
               "USDJPY":"JPY=X","AUDUSD":"AUDUSD=X",
               "XLY":"XLY","XLP":"XLP"}
    yf_hist = yf.download(list(tickers.values()), start=hist_start, end=today, progress=False)["Close"]
    yf_hist = yf_hist.rename(columns={v:k for k,v in tickers.items()})

    # FRED: 5 years
    fred_series = {"HY_OAS":"BAMLH0A0HYM2","T10Y3M":"T10Y3M"}
    
    fred_hist = {}
    for name, sid in fred_series.items():
        fred_hist[name] = fred.get_series(sid, hist_start, today)
    fred_hist = pd.DataFrame(fred_hist)
    fred_hist.index = fred_hist.index.tz_localize(None)

    # Combine historical
    hist = yf_hist.join(fred_hist, how="outer").ffill()
    hist["SPY_vs_TLT"] = hist["SPY"] / hist["TLT"]
    hist["Copper_vs_Gold"] = hist["Copper"] / hist["Gold"]
    hist["HY_bps"] = hist["HY_OAS"] * 100
    hist["EEM_vs_SPY"] = hist["EEM"] / hist["SPY"]
    hist["HYG_vs_LQD"] = hist["HYG"] / hist["LQD"]
    hist["XLY_vs_XLP"] = hist["XLY"] / hist["XLP"]

    # Dashboard data (6 months)
    dash_start = today - timedelta(days=180)
    df = hist.loc[dash_start:].copy()
    for col in ["VIX","HY_bps","T10Y3M","DXY","SPY_vs_TLT","Copper_vs_Gold",
                 "EEM_vs_SPY","HYG_vs_LQD","XLY_vs_XLP","USDJPY","AUDUSD"]:
        df[col+"_SMA"] = df[col].rolling(20).mean()

    return df, hist

# Manual refresh button
if st.button("🔄 Refresh data now (clear cache)"):
    st.cache_data.clear()
    st.rerun()

with st.spinner("Loading 3 years of historical data for context..."):
    df, hist = load_all_data()

latest = df.iloc[-1]
# Use calendar date lookups, not row counts (avoids weekend padding issues)
one_week_date = latest.name - timedelta(days=7)
one_month_date = latest.name - timedelta(days=30)
# Find the closest available trading day
one_week_ago = df.iloc[df.index.get_indexer([one_week_date], method='ffill')[0]]
one_month_ago = df.iloc[df.index.get_indexer([one_month_date], method='ffill')[0]]

# Timestamp: actual data freshness
data_date = df.index[-1].date()
cache_time = datetime.now()
hours_since_data = (cache_time - pd.Timestamp(data_date)).total_seconds() / 3600
if hours_since_data < 24:
    freshness = f"Data as of: {data_date} (within 24 hours)"
else:
    freshness = f"Data as of: {data_date} ({(hours_since_data/24):.0f} days ago — may be cached)"

# Check if data is stale (last row is today but values match yesterday)
is_stale = False
if len(df) >= 2:
    last_two = df.iloc[-2:]
    if last_two.index[-1].date() == datetime.now().date():
        # Today's row exists — check if it's just yesterday's data copied forward
        key_cols = ["VIX", "SPY_vs_TLT", "DXY"]
        if all(abs(last_two.iloc[-1][c] - last_two.iloc[-2][c]) < 0.001 for c in key_cols if pd.notna(last_two.iloc[-1][c])):
            is_stale = True

if is_stale:
    stale_warning = "⚠️ Today's data may not reflect today's close yet. Cache updates every 12 hours."
else:
    stale_warning = ""

st.caption(f"Dashboard: {df.index[0].date()} – {data_date} | Percentiles based on 3-year history | Page loaded: {cache_time.strftime('%Y-%m-%d %H:%M UTC')}")
if stale_warning:
    st.warning(stale_warning)

# ============================================================
# PERCENTILE CALCULATOR
# ============================================================
def get_percentile(series, value):
    clean = series.dropna()
    pct = (clean < value).sum() / len(clean) * 100
    is_extreme = pct < 5 or pct > 95
    return pct, is_extreme

def normal_flag(pct, is_extreme):
    if is_extreme:
        if pct < 5: return f"🔴 Unusually low (bottom {pct:.0f}%)"
        else: return f"🔴 Unusually high (top {100-pct:.0f}%)"
    elif pct < 15 or pct > 85: return f"🟡 Somewhat unusual ({pct:.0f}th %ile)"
    else: return f"🟢 Normal range ({pct:.0f}th %ile)"

# Calculate percentiles
pct_keys = ["VIX","HY_bps","T10Y3M","DXY","SPY_vs_TLT","Copper_vs_Gold",
            "USDJPY","AUDUSD","EEM_vs_SPY","HYG_vs_LQD","XLY_vs_XLP"]
pct_data = {}
for key in pct_keys:
    if key in hist.columns:
        pct_data[key] = get_percentile(hist[key], latest[key])

# ============================================================
# HELPER: classify gauge
# ============================================================
def classify_gauge(name, value, momentum=""):
    if name == "VIX":
        if value > 30: return ("🔴 Panic", "#EF4444")
        elif value > 25: return ("🟠 High caution", "#FB923C")
        elif value > 20: return ("🟡 Elevated", "#FDE047")
        elif value < 15: return ("🟢 Very low", "#22C55E")
        else: return ("🟢 Calm", "#86EFAC")
    elif name == "HY Spread":
        if value > 500: return ("🔴 Stress", "#EF4444")
        elif value > 300: return ("🟡 Moderate", "#FDE047")
        else: return ("🟢 Low stress", "#22C55E")
    elif name == "10Y-3M":
        if value < 0: return ("🔴 Inverted", "#EF4444")
        elif value < 1.0: return ("🟡 Flattening", "#FDE047")
        else: return ("🟢 Steep", "#22C55E")
    elif name == "DXY":
        if value > 105: return ("🔴 Strong USD", "#EF4444")
        elif value < 95: return ("🟢 Weak USD", "#22C55E")
        else: return ("🟡 Neutral", "#FDE047")
    elif name == "SPY/TLT":
        if momentum == "rising": return ("🟢 Risk-on", "#22C55E")
        elif momentum == "falling": return ("🔴 Risk-off", "#EF4444")
        else: return ("🟡 Neutral", "#FDE047")
    elif name == "Copper/Gold":
        if momentum == "rising": return ("🟢 Growth bid", "#22C55E")
        elif momentum == "falling": return ("🔴 Safety bid", "#EF4444")
        else: return ("🟡 Neutral", "#FDE047")
    elif name == "USD/JPY":
        if momentum == "rising": return ("🟢 Risk-on", "#22C55E")
        elif momentum == "falling": return ("🔴 Risk-off", "#EF4444")
        else: return ("🟡 Neutral", "#FDE047")
    elif name == "AUD/USD":
        if momentum == "rising": return ("🟢 Risk-on", "#22C55E")
        elif momentum == "falling": return ("🔴 Risk-off", "#EF4444")
        else: return ("🟡 Neutral", "#FDE047")
    return ("⚪ Unknown", "#9CA3AF")

# ============================================================
# SIGNAL COUNT (replaces risk score as primary readout)
# ============================================================
vix_val = latest["VIX"]
hy_val = latest["HY_bps"]
yc_val = latest["T10Y3M"]
dxy_val = latest["DXY"]
spy_val = latest["SPY_vs_TLT"]
cg_val = latest["Copper_vs_Gold"]
usdjpy_val = latest["USDJPY"]
audusd_val = latest["AUDUSD"]

spy_mom_1m = (spy_val / one_month_ago["SPY_vs_TLT"] - 1) * 100
spy_dir = "rising" if spy_mom_1m > 1 else ("falling" if spy_mom_1m < -1 else "flat")
cg_mom_1m = (cg_val / one_month_ago["Copper_vs_Gold"] - 1) * 100
cg_dir = "rising" if cg_mom_1m > 0.5 else ("falling" if cg_mom_1m < -0.5 else "flat")
jpy_mom_1m = (usdjpy_val / one_month_ago["USDJPY"] - 1) * 100
jpy_dir = "rising" if jpy_mom_1m > 1 else ("falling" if jpy_mom_1m < -1 else "flat")
aud_mom_1m = (audusd_val / one_month_ago["AUDUSD"] - 1) * 100
aud_dir = "rising" if aud_mom_1m > 0.5 else ("falling" if aud_mom_1m < -0.5 else "flat")

# Count warning signals
warnings = []
if vix_val > 28: warnings.append(("VIX elevated", "🔴"))
if hy_val > 500: warnings.append(("HY spreads wide", "🔴"))
if yc_val < 0: warnings.append(("Yield curve inverted", "🔴"))
if dxy_val > 105: warnings.append(("Dollar surging", "🟠"))
if spy_dir == "falling" and spy_mom_1m < -3: warnings.append(("Defensive rotation", "🟠"))
if cg_dir == "falling" and cg_mom_1m < -3: warnings.append(("Gold outperforming copper", "🟠"))
if jpy_dir == "falling" and jpy_mom_1m < -2: warnings.append(("Yen strengthening sharply", "🔴"))
if aud_dir == "falling" and aud_mom_1m < -2: warnings.append(("AUD weakening notably", "🔴"))

warning_count = len(warnings)
total_signals = 8

if warning_count == 0:
    signal_summary = "🟢 All clear — no warnings active"
elif warning_count <= 2:
    signal_summary = f"🟡 {warning_count}/{total_signals} signals warning — modest caution"
elif warning_count <= 4:
    signal_summary = f"🟠 {warning_count}/{total_signals} signals warning — elevated caution"
else:
    signal_summary = f"🔴 {warning_count}/{total_signals} signals warning — high alert"

# Note on FX thresholds
signal_note = "FX warnings trigger at ±2% monthly move. Summary shows intermediate caution at ±1%."

# Legacy risk score (demoted, SPY momentum bonus removed)
def compute_risk_score(vix, hy_bps, yc, dxy):
    score = 50
    if vix < 15: score -= 10
    elif vix < 20: score -= 5
    elif vix < 25: score += 5
    elif vix < 30: score += 15
    else: score += 25
    if hy_bps < 300: score -= 10
    elif hy_bps > 500: score += 20
    elif hy_bps > 400: score += 10
    if yc < 0: score += 20
    elif yc < 0.5: score += 5
    elif yc > 1.5: score -= 10
    if dxy > 105: score += 15
    elif dxy > 100: score += 5
    elif dxy < 95: score -= 10
    return max(0, min(100, score))

risk_score = compute_risk_score(vix_val, hy_val, yc_val, dxy_val)

# ============================================================
# "WHAT CHANGED THIS WEEK" LINE
# ============================================================
week_changes = []
# Each indicator gets its own natural language verb
for label, col, verb_up, verb_down in [
    ("VIX", "VIX", "rose", "fell"),
    ("credit spreads", "HY_bps", "widened", "tightened"),
    ("DXY", "DXY", "strengthened", "weakened"),
    ("SPY/TLT", "SPY_vs_TLT", "rose", "fell"),
    ("Copper/Gold", "Copper_vs_Gold", "rose", "fell"),
    ("USD/JPY", "USDJPY", "rose", "fell"),
    ("AUD/USD", "AUDUSD", "rose", "fell"),
]:
    direction = verb_up if latest[col] > one_week_ago[col] else verb_down
    week_changes.append(f"{label} {direction}")

st.info(f"**📰 This week:** {', '.join(week_changes)}. "
        f"{signal_summary}.")

# ============================================================
# METRIC CARDS (4x3 grid)
# ============================================================
st.subheader("📊 Current Snapshot")
st.caption("Changes: 1-week | 1-month | Normalcy vs 5-year history")

cols = st.columns(4)

metrics = [
    ("VIX", vix_val, one_week_ago["VIX"], one_month_ago["VIX"], "VIX", "VIX", "VIX"),
    ("HY Spread", hy_val, one_week_ago["HY_bps"], one_month_ago["HY_bps"], "HY_bps", "HY Spread", "HY_bps"),
    ("10Y-3M", yc_val, one_week_ago["T10Y3M"], one_month_ago["T10Y3M"], "T10Y3M", "10Y-3M", "T10Y3M"),
    ("DXY", dxy_val, one_week_ago["DXY"], one_month_ago["DXY"], "DXY", "DXY", "DXY"),
    ("SPY/TLT", spy_val, one_week_ago["SPY_vs_TLT"], one_month_ago["SPY_vs_TLT"], "SPY_vs_TLT", "SPY/TLT", "SPY_vs_TLT"),
    ("Cu/Au", cg_val, one_week_ago["Copper_vs_Gold"], one_month_ago["Copper_vs_Gold"], "Copper_vs_Gold", "Copper/Gold", "Copper_vs_Gold"),
    ("USD/JPY", usdjpy_val, one_week_ago["USDJPY"], one_month_ago["USDJPY"], "USDJPY", "USD/JPY", "USDJPY"),
    ("AUD/USD", audusd_val, one_week_ago["AUDUSD"], one_month_ago["AUDUSD"], "AUDUSD", "AUD/USD", "AUDUSD"),
]

for i, (label, val, w_ago, m_ago, pct_key, gauge_name, col_key) in enumerate(metrics):
    with cols[i % 4]:
        dw = val - w_ago
        dm = val - m_ago
        if label in ("SPY/TLT", "Cu/Au", "AUD/USD", "USD/JPY", "VIX", "DXY"):
            dw_s = f"{(val/w_ago - 1)*100:+.1f}%"
            dm_s = f"{(val/m_ago - 1)*100:+.1f}%"
        elif label == "HY Spread":
            dw_s = f"{dw:+.0f} bps"
            dm_s = f"{dm:+.0f} bps"
        elif label in ("10Y-3M",):
            dw_s = f"{dw:+.2f}%"
            dm_s = f"{dm:+.2f}%"
        else:
            dw_s = f"{dw:+.1f}"
            dm_s = f"{dm:+.1f}"

        display_val = (f"{val:.0f} bps" if label=="HY Spread"
                  else (f"{val:.2f}%" if label=="10Y-3M"
                  else (f"{val:.4f}" if label in ("Cu/Au","AUD/USD")
                  else (f"{val:.2f}" if label=="SPY/TLT"
                  else f"{val:.1f}"))))

        st.metric(label, display_val, delta=f"{dw_s} | {dm_s}", delta_color="normal")

        # Classification
        if gauge_name == "SPY/TLT":
            cls, cls_col = classify_gauge(gauge_name, val, spy_dir)
        elif gauge_name == "Copper/Gold":
            cls, cls_col = classify_gauge(gauge_name, val, cg_dir)
        elif gauge_name == "USD/JPY":
            cls, cls_col = classify_gauge(gauge_name, val, jpy_dir)
        elif gauge_name == "AUD/USD":
            cls, cls_col = classify_gauge(gauge_name, val, aud_dir)
        else:
            cls, cls_col = classify_gauge(gauge_name, val)
        st.markdown(f"<span style='color:{cls_col};font-size:0.85rem;font-weight:600;'>{cls}</span>",
                    unsafe_allow_html=True)

        # Normalcy flag
        if pct_key in pct_data:
            pct, extreme = pct_data[pct_key]
            flag_text = normal_flag(pct, extreme)
            st.caption(flag_text)

# ============================================================
# SIGNAL COUNT + SUMMARY
# ============================================================
st.divider()
sig_col, sum_col = st.columns([1, 2])

with sig_col:
    st.subheader("🚦 Signal Count")
    st.markdown(f"<h1 style='text-align:center;margin:0;'>{warning_count}<span style='font-size:1.5rem;'>/{total_signals}</span></h1>",
                unsafe_allow_html=True)
                unsafe_allow_html=True)
    if warnings:
        for w_label, w_icon in warnings:
            st.markdown(f"{w_icon} {w_label}")
    else:
        st.success("No warning signals active")
    st.caption(signal_note)
    st.divider()
    st.caption(f"Reference score: {risk_score}/100 — composite of VIX, credit spreads, yield curve & DXY only")
    st.progress(risk_score / 100)
    with st.expander("📏 How is this score calculated?"):
        st.markdown("""
        **Reference score (0–100)** is a composite of 4 core macro stress indicators. It's different from the signal count — this measures *magnitude*, not just direction.

        **What goes in:**
        - **VIX:** Below 15 = -10pts, 15–20 = -5, 20–25 = +5, 25–30 = +15, above 30 = +25
        - **HY spreads:** Below 300bps = -10pts, 300–400 = neutral, 400–500 = +10, above 500 = +20
        - **10Y-3M curve:** Above 1.5% = -10pts, 0.5–1.5% = neutral, 0–0.5% = +5, inverted = +20
        - **DXY:** Below 95 = -10pts, 95–100 = -5, 100–105 = +5, above 105 = +15

        **How to read it:**
        - 0–30: Low stress — historically calm conditions
        - 30–50: Below average stress
        - 50–65: Moderate stress — warrants attention
        - 65–80: Elevated — defensive positioning common
        - 80–100: High stress — crisis-level readings

        **Why a separate score?** The signal count tells you how many things are flashing. This tells you how loudly they're flashing. A yield curve at -0.05% and one at -0.80% both count as 1 signal, but the score captures the difference.
        """)

with sum_col:
    st.subheader("🧠 Summary")
    signals = []
    if vix_val > 30: signals.append("🔴 VIX above 30 — extreme fear")
    elif vix_val > 28: signals.append("🔴 VIX elevated — heightened caution")
    elif vix_val > 20: signals.append("🟡 VIX moderately elevated")
    else: signals.append("🟢 VIX in calm range")

    if hy_val > 500: signals.append("🔴 HY spreads >500 bps — significant credit stress")
    elif hy_val > 400: signals.append("🟡 HY spreads widening — monitor closely")
    else: signals.append("🟢 Credit spreads contained")

    if yc_val < 0: signals.append("🔴 10Y-3M inverted — recession warning")
    elif yc_val < 0.5: signals.append("🟡 10Y-3M flattening")
    else: signals.append("🟢 10Y-3M healthy — normal slope")

    if dxy_val > 105: signals.append("🔴 Dollar surging — tightening conditions")
    elif dxy_val > 100: signals.append("🟡 Dollar above parity")
    else: signals.append("🟢 Dollar contained")

    if spy_mom_1m > 3: signals.append("🟢 Stocks leading — risk-on")
    elif spy_mom_1m < -3: signals.append("🔴 Bonds leading — defensive")
    else: signals.append("🟡 Equity/bond ratio balanced")

    if cg_mom_1m > 3: signals.append("🟢 Copper outperforming gold — growth optimism")
    elif cg_mom_1m < -3: signals.append("🔴 Gold outperforming copper — safety demand")
    else: signals.append("🟡 Copper/Gold ratio stable")

    if jpy_mom_1m > 1: signals.append("🟢 Yen weakening — risk-on flows")
    elif jpy_mom_1m < -2: signals.append("🔴 Yen strengthening — flight to safety")
    elif jpy_mom_1m < -1: signals.append("🟡 Yen firming — cautious")
    else: signals.append("🟡 USD/JPY stable")

    if aud_mom_1m > 1: signals.append("🟢 AUD rising — growth appetite")
    elif aud_mom_1m < -2: signals.append("🔴 AUD falling — growth concerns")
    elif aud_mom_1m < -1: signals.append("🟡 AUD soft — watch")
    else: signals.append("🟡 AUD/USD stable")

    st.markdown(
        f"""<div style="background-color:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;padding:16px 20px;">
        {"<br>".join(f'<span style="font-size:0.95rem;">{s}</span>' for s in signals)}
        </div>""",
        unsafe_allow_html=True
    )

# ============================================================
# CHARTS
# ============================================================
st.divider()
chart_col1, chart_col2 = st.columns([3, 1])
with chart_col1:
    st.subheader("📈 Trends")
with chart_col2:
    date_range = st.selectbox("Timeframe", ["1 month", "3 months", "6 months", "1 year"], index=2)

range_map = {"1 month": 22, "3 months": 66, "6 months": 132, "1 year": 260}
cutoff = df.index[-1] - timedelta(days=range_map[date_range])
chart_df = df.loc[cutoff:].copy()
# Remove weekend rows (Monday=0, Sunday=6)
chart_df = chart_df[chart_df.index.dayofweek < 5]

fig = make_subplots(
    rows=11, cols=1, shared_xaxes=True, vertical_spacing=0.04,
    subplot_titles=(
        "① VIX — Fear Index  |  Rising = growing fear",
        "② High-Yield Spread — Credit stress (bps)  |  Wider = more default risk",
        "③ 10Y-3M Spread — Yield curve (%)  |  Below zero = recession warning",
        "④ US Dollar (DXY) — Currency strength  |  Rising = risk-off USD demand",
        "⑤ SPY / TLT — Stocks vs Bonds  |  Rising = risk-on appetite",
        "⑥ Copper / Gold — Growth vs Safety  |  Rising = industrial optimism",
        "⑦ USD/JPY — Safe-haven FX  |  Falling = flight to yen safety",
        "⑧ AUD/USD — Commodity FX  |  Rising = growth / risk appetite",
        "⑨ EEM / SPY — Emerging Markets vs S&P 500  |  Rising = capital flowing to EM",
        "⑩ HYG / LQD — Junk vs IG Bonds  |  Rising = credit risk appetite",
        "⑪ XLY / XLP — Consumer Disc. vs Staples  |  Rising = cyclical rotation",
    ),
    row_heights=[1,1,1,1,1,1,1,1,1,1,1]
)

panels = [
    (1, "VIX", "VIX_SMA", "#DC143C", [20,30], ["orange","red"], 2),
    (2, "HY_bps", "HY_bps_SMA", "#FF8C00", [500], ["red"], 30),
    (3, "T10Y3M", "T10Y3M_SMA", "#4682B4", [0], ["red"], 0.3),
    (4, "DXY", "DXY_SMA", "#008080", [100], ["grey"], 1),
    (5, "SPY_vs_TLT", "SPY_vs_TLT_SMA", "#9370DB", [], [], 0.3),
    (6, "Copper_vs_Gold", "Copper_vs_Gold_SMA", "#DAA520", [], [], 0.0002),
    (7, "USDJPY", "USDJPY_SMA", "#C71585", [], [], 3),
    (8, "AUDUSD", "AUDUSD_SMA", "#228B22", [], [], 0.005),
    (9, "EEM_vs_SPY", "EEM_vs_SPY_SMA", "#2E86AB", [], [], 0.005),
    (10, "HYG_vs_LQD", "HYG_vs_LQD_SMA", "#E07B39", [], [], 0.01),
    (11, "XLY_vs_XLP", "XLY_vs_XLP_SMA", "#6A994E", [], [], 0.02),
]

for row, col, trend, color, hlines, hcolors, pad in panels:
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[col],
        line=dict(color=color, width=2.2), name=col, showlegend=(row==1)), row=row, col=1)
    if trend and trend in chart_df.columns:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[trend],
            line=dict(color="grey", width=1, dash="dash"),
            name="20-day trend", showlegend=(row==1)), row=row, col=1)
    for val, hcol in zip(hlines, hcolors):
        fig.add_hline(y=val, line_dash="dot", line_color=hcol, opacity=0.5, row=row, col=1)
    vals = chart_df[col].dropna()
    if len(vals) > 0:
        fig.update_yaxes(range=[vals.min()-pad, vals.max()+pad], row=row, col=1)

fig.add_hrect(y0=-3, y1=0, line_width=0, fillcolor="red", opacity=0.06, row=3, col=1)

fig.update_layout(
    height=1700, hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5, font=dict(size=11)),
    margin=dict(t=60, b=40, l=40, r=40),
    plot_bgcolor="white", paper_bgcolor="white"
)
for i in range(1,12):
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="#EEE", zeroline=False, row=i, col=1)
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="#EEE", row=i, col=1)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# WEEKEND DEEP-DIVE
# ============================================================
with st.expander("🔬 Weekend Deep-Dive — Cross-Asset Snapshot & Regime Analysis"):
    st.markdown("### 📋 Cross-Asset Snapshot")

    snap_data = []
    for label, val, w_ago, m_ago in [
        ("VIX", vix_val, one_week_ago["VIX"], one_month_ago["VIX"]),
        ("HY Spread (bps)", hy_val, one_week_ago["HY_bps"], one_month_ago["HY_bps"]),
        ("10Y-3M (%)", yc_val, one_week_ago["T10Y3M"], one_month_ago["T10Y3M"]),
        ("DXY", dxy_val, one_week_ago["DXY"], one_month_ago["DXY"]),
        ("SPY/TLT", spy_val, one_week_ago["SPY_vs_TLT"], one_month_ago["SPY_vs_TLT"]),
        ("Cu/Au", cg_val, one_week_ago["Copper_vs_Gold"], one_month_ago["Copper_vs_Gold"]),
        ("USD/JPY", usdjpy_val, one_week_ago["USDJPY"], one_month_ago["USDJPY"]),
        ("AUD/USD", audusd_val, one_week_ago["AUDUSD"], one_month_ago["AUDUSD"]),
        ("EEM/SPY", latest["EEM_vs_SPY"], one_week_ago["EEM_vs_SPY"], one_month_ago["EEM_vs_SPY"]),
        ("HYG/LQD", latest["HYG_vs_LQD"], one_week_ago["HYG_vs_LQD"], one_month_ago["HYG_vs_LQD"]),
        ("XLY/XLP", latest["XLY_vs_XLP"], one_week_ago["XLY_vs_XLP"], one_month_ago["XLY_vs_XLP"]),
    ]:
        ch_w = val - w_ago
        ch_m = val - m_ago
        if label in ("SPY/TLT", "Cu/Au", "EEM/SPY", "HYG/LQD", "XLY/XLP", "AUD/USD"):
            ch_w_s = f"{(val/w_ago - 1)*100:+.1f}%"
            ch_m_s = f"{(val/m_ago - 1)*100:+.1f}%"
        elif "bps" in label:
            ch_w_s = f"{ch_w:+.0f} bps"
            ch_m_s = f"{ch_m:+.0f} bps"
        elif "%" in label:
            ch_w_s = f"{ch_w:+.2f}%"
            ch_m_s = f"{ch_m:+.2f}%"
        else:
            ch_w_s = f"{ch_w:+.1f}"
            ch_m_s = f"{ch_m:+.1f}"

        display = (f"{val:.0f} bps" if "bps" in label
              else (f"{val:.2f}%" if "%" in label
              else (f"{val:.2f}" if label=="SPY/TLT"
              else (f"{val:.4f}" if label in ("Cu/Au","EEM/SPY","HYG/LQD","AUD/USD","XLY/XLP")
              else f"{val:.1f}"))))

        snap_data.append({"Indicator": label, "Current": display, "1-Week Δ": ch_w_s, "1-Month Δ": ch_m_s})

    st.dataframe(pd.DataFrame(snap_data), use_container_width=True, hide_index=True)

    # Regime description
    st.markdown("### 📝 Macro Regime Description")
    regime_parts = []
    if warning_count == 0:
        regime_parts.append("We are in a **low-risk, risk-on regime**. No warning signals are active. Markets are calm with broad risk appetite.")
    elif warning_count <= 2:
        regime_parts.append("We are in a **mostly benign environment** with isolated concerns. The majority of signals are neutral or risk-on.")
    elif warning_count <= 4:
        regime_parts.append("We are in an **elevated caution regime**. Multiple signals warrant attention but no broad panic.")
    else:
        regime_parts.append("We are in a **high-alert, risk-off regime**. Multiple stress signals are active across equities, credit, currencies, and bonds.")

    if yc_val < 0:
        regime_parts.append("The **10Y-3M yield curve is inverted**, the Fed's preferred recession indicator. This has preceded every US recession since 1960 with no false signals.")
    elif yc_val < 0.5:
        regime_parts.append("The **10Y-3M curve is flattening**. Not yet inverted, but worth monitoring closely.")

    if vix_val > 25:
        regime_parts.append("**Equity volatility is elevated**, indicating significant uncertainty.")
    elif vix_val < 15:
        regime_parts.append("**Volatility is very low** — while benign, extremely low VIX can signal complacency and vulnerability to shocks.")

    if spy_dir == "rising":
        regime_parts.append("**Investors are favouring equities over bonds**, a sign of risk appetite and growth expectations.")
    elif spy_dir == "falling":
        regime_parts.append("**Investors are favouring bonds over equities**, a defensive rotation suggesting caution.")

    for part in regime_parts:
        st.markdown(part)
        st.markdown("")

    # Biggest movers — uses absolute change for spreads/yields, % for ratios
    st.markdown("### 📊 Biggest Movers This Week")
    moves_abs = {}
    moves_pct = {}
    for label, val, w_ago, use_abs in [
        ("HY Spread", hy_val, one_week_ago["HY_bps"], True),
        ("10Y-3M", yc_val, one_week_ago["T10Y3M"], True),
        ("VIX", vix_val, one_week_ago["VIX"], False),
        ("DXY", dxy_val, one_week_ago["DXY"], False),
        ("USD/JPY", usdjpy_val, one_week_ago["USDJPY"], False),
        ("SPY/TLT", spy_val, one_week_ago["SPY_vs_TLT"], False),
        ("Copper/Gold", cg_val, one_week_ago["Copper_vs_Gold"], False),
        ("AUD/USD", audusd_val, one_week_ago["AUDUSD"], False),
        ("EEM/SPY", latest["EEM_vs_SPY"], one_week_ago["EEM_vs_SPY"], False),
        ("HYG/LQD", latest["HYG_vs_LQD"], one_week_ago["HYG_vs_LQD"], False),
        ("XLY/XLP", latest["XLY_vs_XLP"], one_week_ago["XLY_vs_XLP"], False),
    ]:
        pct_change = abs((val/w_ago - 1)*100)
        direction = "↑" if val > w_ago else "↓"
        if use_abs:
            moves_abs[label] = (abs(val - w_ago), direction)
        else:
            moves_pct[label] = (pct_change, direction)

    # Show top 2 absolute movers and top 2 % movers
    top_abs = sorted(moves_abs.items(), key=lambda x: x[1][0], reverse=True)[:2]
    top_pct = sorted(moves_pct.items(), key=lambda x: x[1][0], reverse=True)[:2]
    parts = []
    for label, (ch, direction) in top_abs:
        parts.append(f"**{label}** {direction} {ch:.1f}")
    for label, (ch, direction) in top_pct:
        parts.append(f"**{label}** {direction} {ch:.1f}%")
    st.info(f"Top movers this week: {' | '.join(parts)}")

# ============================================================
# EXPLAINERS + LEGEND
# ============================================================
with st.expander("📖 Gauge Explanations & Legend"):
    st.markdown("### What does each gauge mean?")
    for gauge_name, explanation in EXPLAINERS.items():
        st.markdown(f"**{gauge_name}**")
        st.markdown(f"{explanation}")
        st.markdown("")

    st.markdown("---")
    st.markdown("### Threshold Legend")
    cols = st.columns(3)
    legend_items = list(GAUGE_LEGENDS.items())
    for i, (gauge_name, levels) in enumerate(legend_items):
        with cols[i % 3]:
            st.markdown(f"**{gauge_name}**")
            for range_str, meaning, color in levels:
                st.markdown(
                    f"<span style='color:{color};font-size:0.9rem;'>{range_str}</span> — {meaning}",
                    unsafe_allow_html=True
                )
            st.markdown("")

# Raw data
with st.expander("🔍 View raw data (last 30 days)"):
    st.dataframe(
        df[["VIX","HY_bps","T10Y3M","DXY","SPY_vs_TLT","Copper_vs_Gold","USDJPY","AUDUSD"]]
        .tail(30).sort_index(ascending=False).round(2),
        use_container_width=True
    )
