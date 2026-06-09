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
# LEGEND DATA
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
    "10Y-2Y Spread": [
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
}

EXPLAINERS = {
    "VIX": "The VIX measures expected S&P 500 volatility implied by options prices. It's called the 'fear index' — when investors pay up for protection, VIX rises. Values above 30 usually coincide with market panics.",
    "HY Spread": "The high-yield (junk) bond spread is the extra yield investors demand over Treasuries to hold risky corporate debt. Widening spreads mean bond markets are pricing in higher default risk — often an early warning of economic trouble.",
    "10Y-2Y": "This is the difference between 10-year and 2-year Treasury yields. When it goes negative (inverts), short-term rates exceed long-term rates — a signal that markets expect a slowdown and rate cuts ahead. Inverted curves have preceded every US recession since 1950.",
    "DXY": "The US Dollar Index measures the dollar against a basket of major currencies. A rising dollar means global capital is flowing into USD safe-haven assets, tightening financial conditions worldwide. A falling dollar often signals risk appetite and capital flowing to emerging markets.",
    "SPY/TLT": "This ratio compares US equities (SPY) to long-term Treasury bonds (TLT). When it rises, stocks are outperforming bonds — investors are taking risk. When it falls, bonds are winning — investors are seeking safety. It's a pure risk appetite gauge.",
    "Copper/Gold": "Copper is an industrial metal tied to global growth. Gold is the ultimate safe haven. The ratio rises when growth optimism dominates, and falls when fear and safety demand take over. Often called 'Dr. Copper vs Dr. Gold'.",
}

# ============================================================
# LOAD DATA (cached for 12 hours)
# ============================================================
@st.cache_data(ttl=43200)
def load_all_data():
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        st.error("FRED_API_KEY not set in Streamlit secrets")
        st.stop()
    fred = Fred(api_key=fred_key)

    today = datetime.today()

    # --- Historical data (5 years) for percentiles ---
    hist_start = today - timedelta(days=5*365)
    hist_end = today

    # Yahoo Finance: 5 years of daily data
    tickers = {"VIX":"^VIX","SPY":"SPY","TLT":"TLT","EEM":"EEM","HYG":"HYG","LQD":"LQD",
               "DXY":"DX-Y.NYB","Copper":"HG=F","Gold":"GC=F","USDJPY":"JPY=X"}
    yf_hist = yf.download(list(tickers.values()), start=hist_start, end=hist_end, progress=False)["Close"]
    yf_hist = yf_hist.rename(columns={v:k for k,v in tickers.items()})

    # FRED: 5 years
    fred_series = {"HY_OAS":"BAMLH0A0HYM2","IG_OAS":"BAMLC0A0CM",
                   "US10Y":"DGS10","US2Y":"DGS2","T10Y2Y":"T10Y2Y"}
    fred_hist = {}
    for name, sid in fred_series.items():
        fred_hist[name] = fred.get_series(sid, hist_start, hist_end)
    fred_hist = pd.DataFrame(fred_hist)
    fred_hist.index = fred_hist.index.tz_localize(None)

        # Combine historical
    hist = yf_hist.join(fred_hist, how="outer").ffill()
    hist["SPY_vs_TLT"] = hist["SPY"] / hist["TLT"]
    hist["Copper_vs_Gold"] = hist["Copper"] / hist["Gold"]
    hist["HY_bps"] = hist["HY_OAS"] * 100
    hist["EEM_vs_SPY"] = hist["EEM"] / hist["SPY"]
    hist["HYG_vs_LQD"] = hist["HYG"] / hist["LQD"]

    # --- Dashboard data (6 months) ---
    dash_start = today - timedelta(days=180)
    df = hist.loc[dash_start:].copy()
    for col in ["VIX","HY_bps","T10Y2Y","DXY","SPY_vs_TLT","Copper_vs_Gold"]:
        df[col+"_SMA"] = df[col].rolling(20).mean()

    return df, hist

with st.spinner("Loading 5 years of historical data for context..."):
    df, hist = load_all_data()

latest = df.iloc[-1]
one_week_ago = df.iloc[-5] if len(df) > 5 else df.iloc[0]
one_month_ago = df.iloc[-22] if len(df) > 22 else df.iloc[0]
st.caption(f"Last data refresh: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} | Dashboard: {df.index[0].date()} – {df.index[-1].date()} | Percentiles based on 5-year history")

# ============================================================
# PERCENTILE CALCULATOR
# ============================================================
def get_percentile(series, value):
    """Return percentile (0-100) and whether it's unusual (<5 or >95)."""
    clean = series.dropna()
    pct = (clean < value).sum() / len(clean) * 100
    is_extreme = pct < 5 or pct > 95
    return pct, is_extreme

def normal_flag(pct, is_extreme, name=""):
    """Return a flag string showing how normal the reading is."""
    if is_extreme:
        if pct < 5: return f"🔴 Unusually low (bottom {pct:.0f}%)"
        else: return f"🔴 Unusually high (top {100-pct:.0f}%)"
    elif pct < 15 or pct > 85: return f"🟡 Somewhat unusual ({pct:.0f}th %ile)"
    else: return f"🟢 Normal range ({pct:.0f}th %ile)"

# Calculate percentiles for all gauges
pct_data = {
    "VIX": get_percentile(hist["VIX"], latest["VIX"]),
    "HY_bps": get_percentile(hist["HY_bps"], latest["HY_bps"]),
    "T10Y2Y": get_percentile(hist["T10Y2Y"], latest["T10Y2Y"]),
    "DXY": get_percentile(hist["DXY"], latest["DXY"]),
    "SPY_vs_TLT": get_percentile(hist["SPY_vs_TLT"], latest["SPY_vs_TLT"]),
    "Copper_vs_Gold": get_percentile(hist["Copper_vs_Gold"], latest["Copper_vs_Gold"]),
}

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
    elif name == "10Y-2Y":
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
    return ("⚪ Unknown", "#9CA3AF")

# ============================================================
# RISK SCORE
# ============================================================
def compute_risk_score(vix, hy_bps, yc, dxy, spy_mom):
    score = 50
    if vix < 15: score -= 15
    elif vix < 20: score -= 5
    elif vix < 25: score += 10
    elif vix < 30: score += 20
    else: score += 30
    if hy_bps < 300: score -= 10
    elif hy_bps > 500: score += 20
    elif hy_bps > 400: score += 10
    if yc < 0: score += 20
    elif yc < 0.5: score += 5
    elif yc > 1.5: score -= 10
    if dxy > 105: score += 15
    elif dxy > 100: score += 5
    elif dxy < 95: score -= 10
    if spy_mom > 3: score -= 15
    elif spy_mom > 0: score -= 5
    elif spy_mom < -3: score += 15
    elif spy_mom < 0: score += 5
    return max(0, min(100, score))

vix_val = latest["VIX"]
hy_val = latest["HY_bps"]
yc_val = latest["T10Y2Y"]
dxy_val = latest["DXY"]
spy_val = latest["SPY_vs_TLT"]
cg_val = latest["Copper_vs_Gold"]

spy_mom_1m = (spy_val / one_month_ago["SPY_vs_TLT"] - 1) * 100
spy_dir = "rising" if spy_mom_1m > 1 else ("falling" if spy_mom_1m < -1 else "flat")
cg_mom_1m = (cg_val / one_month_ago["Copper_vs_Gold"] - 1) * 100
cg_dir = "rising" if cg_mom_1m > 0.5 else ("falling" if cg_mom_1m < -0.5 else "flat")

risk_score = compute_risk_score(vix_val, hy_val, yc_val, dxy_val, spy_mom_1m)
if risk_score < 30: risk_label = "🟢 LOW RISK"
elif risk_score < 50: risk_label = "🟢 BELOW AVERAGE"
elif risk_score < 65: risk_label = "🟡 MODERATE"
elif risk_score < 80: risk_label = "🟠 ELEVATED"
else: risk_label = "🔴 HIGH RISK"

# ============================================================
# "WHAT CHANGED THIS WEEK" LINE
# ============================================================
week_changes = []
for label, col, direction in [("VIX", "VIX", "fell" if latest["VIX"] < one_week_ago["VIX"] else "rose"),
                                ("credit spreads", "HY_bps", "tightened" if latest["HY_bps"] < one_week_ago["HY_bps"] else "widened"),
                                ("DXY", "DXY", "weakened" if latest["DXY"] < one_week_ago["DXY"] else "strengthened"),
                                ("SPY/TLT", "SPY_vs_TLT", "fell" if latest["SPY_vs_TLT"] < one_week_ago["SPY_vs_TLT"] else "rose"),
                                ("Copper/Gold", "Copper_vs_Gold", "fell" if latest["Copper_vs_Gold"] < one_week_ago["Copper_vs_Gold"] else "rose")]:
    week_changes.append(f"{label} {direction}")

st.info(f"**📰 This week:** {', '.join(week_changes)}. "
        f"Risk score: {risk_score}/100 ({risk_label}). "
        f"{'⚠️ Warning signals active.' if risk_score > 65 else 'No major warning signals.'}")

# ============================================================
# METRIC CARDS
# ============================================================
st.subheader("📊 Current Snapshot")
st.caption("Changes: 1-week | 1-month | Normalcy check vs 5-year history")

col1, col2, col3, col4, col5, col6 = st.columns(6)

metrics = [
    ("VIX", vix_val, one_week_ago["VIX"], one_month_ago["VIX"], "VIX", "VIX"),
    ("HY Spread", hy_val, one_week_ago["HY_bps"], one_month_ago["HY_bps"], "HY_bps", "HY Spread"),
    ("10Y-2Y", yc_val, one_week_ago["T10Y2Y"], one_month_ago["T10Y2Y"], "T10Y2Y", "10Y-2Y"),
    ("DXY", dxy_val, one_week_ago["DXY"], one_month_ago["DXY"], "DXY", "DXY"),
    ("SPY/TLT", spy_val, one_week_ago["SPY_vs_TLT"], one_month_ago["SPY_vs_TLT"], "SPY_vs_TLT", "SPY/TLT"),
    ("Cu/Au", cg_val, one_week_ago["Copper_vs_Gold"], one_month_ago["Copper_vs_Gold"], "Copper_vs_Gold", "Copper/Gold"),
]

for col, (label, val, w_ago, m_ago, pct_key, gauge_name) in zip([col1,col2,col3,col4,col5,col6], metrics):
    with col:
        dw = val - w_ago
        dm = val - m_ago
        if label in ("SPY/TLT", "Cu/Au"):
            dw_s = f"{(val/w_ago - 1)*100:+.1f}%"
            dm_s = f"{(val/m_ago - 1)*100:+.1f}%"
        elif label == "HY Spread":
            dw_s = f"{dw:+.0f} bps"
            dm_s = f"{dm:+.0f} bps"
        elif label == "10Y-2Y":
            dw_s = f"{dw:+.2f}%"
            dm_s = f"{dm:+.2f}%"
        else:
            dw_s = f"{dw:+.1f}"
            dm_s = f"{dm:+.1f}"

        display_val = (f"{val:.0f} bps" if label=="HY Spread"
                  else (f"{val:.2f}%" if label=="10Y-2Y"
                  else (f"{val:.2f}" if label=="SPY/TLT"
                  else (f"{val:.4f}" if label=="Cu/Au" else f"{val:.1f}"))))

        st.metric(label, display_val, delta=f"{dw_s} | {dm_s}", delta_color="normal")

        # Classification
        if gauge_name == "SPY/TLT":
            cls, cls_col = classify_gauge(gauge_name, val, spy_dir)
        elif gauge_name == "Copper/Gold":
            cls, cls_col = classify_gauge(gauge_name, val, cg_dir)
        else:
            cls, cls_col = classify_gauge(gauge_name, val)
        st.markdown(f"<span style='color:{cls_col};font-size:0.85rem;font-weight:600;'>{cls}</span>",
                    unsafe_allow_html=True)

        # Normalcy flag
        pct, extreme = pct_data[pct_key]
        flag_text = normal_flag(pct, extreme)
        st.caption(flag_text)

# ============================================================
# RISK SCORE + SUMMARY
# ============================================================
st.divider()
risk_col, sum_col = st.columns([1, 2])

with risk_col:
    st.subheader("🧭 Risk Score")
    st.markdown(f"<h1 style='text-align:center;margin:0;'>{risk_score}<span style='font-size:1.5rem;'>/100</span></h1>",
                unsafe_allow_html=True)
    st.progress(risk_score / 100)
    st.markdown(f"<p style='text-align:center;font-weight:bold;font-size:1.2rem;'>{risk_label}</p>",
                unsafe_allow_html=True)
    st.caption("0–30: Low | 30–50: Below avg | 50–65: Moderate | 65–80: Elevated | 80–100: High")

with sum_col:
    st.subheader("🧠 Summary")
    signals = []
    if vix_val > 30: signals.append("🔴 VIX above 30 — extreme fear")
    elif vix_val > 25: signals.append("🟠 VIX elevated — heightened caution")
    elif vix_val > 20: signals.append("🟡 VIX moderately elevated")
    else: signals.append("🟢 VIX in calm range")

    if hy_val > 500: signals.append("🔴 HY spreads >500 bps — significant credit stress")
    elif hy_val > 400: signals.append("🟡 HY spreads widening — monitor closely")
    else: signals.append("🟢 Credit spreads contained")

    if yc_val < 0: signals.append("🔴 Yield curve inverted — recession warning")
    elif yc_val < 0.5: signals.append("🟡 Yield curve flattening")
    else: signals.append("🟢 Yield curve healthy")

    if dxy_val > 105: signals.append("🔴 Dollar surging — tightening conditions")
    elif dxy_val > 100: signals.append("🟡 Dollar above parity")
    else: signals.append("🟢 Dollar contained")

    if spy_mom_1m > 3: signals.append("🟢 Stocks leading — risk-on")
    elif spy_mom_1m < -3: signals.append("🔴 Bonds leading — defensive")
    else: signals.append("🟡 Equity/bond ratio balanced")

    if cg_mom_1m > 3: signals.append("🟢 Copper outperforming gold — growth optimism")
    elif cg_mom_1m < -3: signals.append("🔴 Gold outperforming copper — safety demand")
    else: signals.append("🟡 Copper/Gold ratio stable")

    st.markdown(
        f"""<div style="background-color:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;padding:16px 20px;">
        {"<br>".join(f'<span style="font-size:0.95rem;">{s}</span>' for s in signals)}
        </div>""",
        unsafe_allow_html=True
    )

# ============================================================
# DATE RANGE SELECTOR + CHARTS
# ============================================================
st.divider()
chart_col1, chart_col2 = st.columns([3, 1])
with chart_col1:
    st.subheader("📈 Trends")
with chart_col2:
    date_range = st.selectbox("Timeframe", ["1 month", "3 months", "6 months", "1 year"], index=2)

range_map = {"1 month": 22, "3 months": 66, "6 months": 132, "1 year": 260}
cutoff = df.index[-1] - timedelta(days=range_map[date_range])
chart_df = df.loc[cutoff:]

fig = make_subplots(
    rows=7, cols=1, shared_xaxes=True, vertical_spacing=0.05,
    subplot_titles=(
        f"① VIX — Fear Index  |  Rising = growing fear",
        f"② High-Yield Spread — Credit stress (bps)  |  Wider = more default risk",
        f"③ 10Y-2Y Spread — Yield curve (%)  |  Below zero = recession warning",
        f"④ US Dollar (DXY) — Currency strength  |  Rising = risk-off USD demand",
        f"⑤ SPY / TLT — Stocks vs Bonds  |  Rising = risk-on appetite",
        f"⑥ Copper / Gold — Growth vs Safety  |  Rising = industrial optimism",
        f"⑦ EEM / SPY — Emerging Markets vs S&P 500  |  Rising = capital flowing to EM",
    ),
    row_heights=[1,1,1,1,1,1,1]
)

panels = [
    (1, "VIX", "VIX_SMA", "#DC143C", [20,30], ["orange","red"], 2),
    (2, "HY_bps", "HY_bps_SMA", "#FF8C00", [500], ["red"], 30),
    (3, "T10Y2Y", "T10Y2Y_SMA", "#4682B4", [0], ["red"], 0.3),
    (4, "DXY", "DXY_SMA", "#008080", [100], ["grey"], 1),
    (5, "SPY_vs_TLT", "SPY_vs_TLT_SMA", "#9370DB", [], [], 0.3),
    (6, "Copper_vs_Gold", "Copper_vs_Gold_SMA", "#DAA520", [], [], 0.0002),
    (7, "EEM_vs_SPY", None, "#2E86AB", [], [], 0.005),
]

for row, col, trend, color, hlines, hcolors, pad in panels:
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[col],
        line=dict(color=color, width=2.2), name=col, showlegend=(row==1)), row=row, col=1)
    if trend:
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
    height=1150, hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5, font=dict(size=11)),
    margin=dict(t=60, b=40, l=40, r=40),
    plot_bgcolor="white", paper_bgcolor="white"
)
for i in range(1,8):
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="#EEE", zeroline=False, row=i, col=1)
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="#EEE", row=i, col=1)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# WEEKEND DEEP-DIVE (collapsible)
# ============================================================
with st.expander("🔬 Weekend Deep-Dive — Cross-Asset Snapshot & Regime Analysis"):
    st.markdown("### 📋 Cross-Asset Snapshot")

    # Table
    snap_data = []
    for label, val, w_ago, m_ago, gauge_name in [
        ("VIX", vix_val, one_week_ago["VIX"], one_month_ago["VIX"], "VIX"),
        ("HY Spread (bps)", hy_val, one_week_ago["HY_bps"], one_month_ago["HY_bps"], "HY Spread"),
        ("10Y-2Y (%)", yc_val, one_week_ago["T10Y2Y"], one_month_ago["T10Y2Y"], "10Y-2Y"),
        ("DXY", dxy_val, one_week_ago["DXY"], one_month_ago["DXY"], "DXY"),
        ("SPY/TLT", spy_val, one_week_ago["SPY_vs_TLT"], one_month_ago["SPY_vs_TLT"], "SPY/TLT"),
        ("Cu/Au", cg_val, one_week_ago["Copper_vs_Gold"], one_month_ago["Copper_vs_Gold"], "Copper/Gold"),
        ("EEM/SPY", latest["EEM_vs_SPY"], one_week_ago["EEM_vs_SPY"], one_month_ago["EEM_vs_SPY"], ""),
        ("HYG/LQD", latest["HYG_vs_LQD"], one_week_ago["HYG_vs_LQD"], one_month_ago["HYG_vs_LQD"], ""),
    ]:
        ch_w = val - w_ago
        ch_m = val - m_ago
        if label in ("SPY/TLT", "Cu/Au", "EEM/SPY", "HYG/LQD"):
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
              else (f"{val:.4f}" if label in ("Cu/Au","EEM/SPY","HYG/LQD")
              else f"{val:.1f}"))))

        snap_data.append({
            "Indicator": label,
            "Current": display,
            "1-Week Δ": ch_w_s,
            "1-Month Δ": ch_m_s,
        })

    snap_df = pd.DataFrame(snap_data)
    st.dataframe(snap_df, use_container_width=True, hide_index=True)

    # Regime description
    st.markdown("### 📝 Macro Regime Description")
    regime_parts = []

    # Determine overall regime
    if risk_score < 30:
        regime_parts.append("We are in a **low-risk, risk-on regime**. Markets are calm, volatility is subdued, credit spreads are tight, and equities are outperforming bonds.")
    elif risk_score < 50:
        regime_parts.append("We are in a **moderately low-risk environment**. Most indicators are benign with only minor areas of concern.")
    elif risk_score < 65:
        regime_parts.append("We are in a **moderate risk environment**. Some caution signals are present but no full alarm. Monitor closely.")
    elif risk_score < 80:
        regime_parts.append("We are in an **elevated risk regime**. Multiple stress signals are active. Defensive positioning is increasing.")
    else:
        regime_parts.append("We are in a **high-risk, risk-off regime**. Fear is elevated, credit is stressed, and capital is flowing to safety.")

    if yc_val < 0:
        regime_parts.append("The **yield curve is inverted**, a classic recession warning signal. Historically this has preceded downturns by 6-18 months.")
    elif yc_val < 0.5:
        regime_parts.append("The **yield curve is flattening**. While not yet inverted, it bears watching.")
    else:
        regime_parts.append("The **yield curve is healthy and steep**, consistent with normal growth expectations.")

    if vix_val > 25:
        regime_parts.append("**Equity volatility is elevated**, indicating significant uncertainty or fear among market participants.")
    elif vix_val < 15:
        regime_parts.append("**Volatility is very low** — while this feels good, extremely low VIX can signal complacency and vulnerability to shocks.")

    if spy_dir == "rising":
        regime_parts.append("**Investors are favoring equities over bonds**, a sign of risk appetite and growth expectations.")
    elif spy_dir == "falling":
        regime_parts.append("**Investors are favoring bonds over equities**, a defensive rotation suggesting caution or fear.")

    for part in regime_parts:
        st.markdown(part)
        st.markdown("")

    # Biggest mover (all compared by % change for fair comparison)
    st.markdown("### 📊 Biggest Movers This Week")
    moves = {}
    for label, val, w_ago in [
        ("VIX", vix_val, one_week_ago["VIX"]),
        ("HY Spread", hy_val, one_week_ago["HY_bps"]),
        ("10Y-2Y", yc_val, one_week_ago["T10Y2Y"]),
        ("DXY", dxy_val, one_week_ago["DXY"]),
        ("SPY/TLT", spy_val, one_week_ago["SPY_vs_TLT"]),
        ("Copper/Gold", cg_val, one_week_ago["Copper_vs_Gold"]),
        ("EEM/SPY", latest["EEM_vs_SPY"], one_week_ago["EEM_vs_SPY"]),
        ("HYG/LQD", latest["HYG_vs_LQD"], one_week_ago["HYG_vs_LQD"]),
    ]:
        pct_change = abs((val/w_ago - 1)*100)
        direction = "↑" if val > w_ago else "↓"
        moves[label] = (pct_change, direction)

    # Sort by % change, show top 3
    sorted_moves = sorted(moves.items(), key=lambda x: x[1][0], reverse=True)
    top_three = sorted_moves[:3]
    summary_parts = []
    for label, (pct, direction) in top_three:
        summary_parts.append(f"**{label}** {direction} {pct:.1f}%")
    st.info(f"Top movers this week: {', '.join(summary_parts)}")

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
        df[["VIX","HY_bps","T10Y2Y","DXY","SPY_vs_TLT","Copper_vs_Gold"]]
        .tail(30).sort_index(ascending=False).round(2),
        use_container_width=True
    )
