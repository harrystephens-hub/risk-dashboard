import streamlit as st
import yfinance as yf
import pandas as pd
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
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")

# ============================================================
# LEGEND DATA (defines thresholds and labels for each gauge)
# ============================================================
GAUGE_LEGENDS = {
    "VIX": [
        ("< 15", "🟢 Very low — complacency risk", "#22C55E"),
        ("15–20", "🟢 Calm", "#86EFAC"),
        ("20–25", "🟡 Elevated caution", "#FDE047"),
        ("25–30", "🟠 High caution", "#FB923C"),
        ("> 30", "🔴 Panic / extreme fear", "#EF4444"),
    ],
    "HY Spread": [
        ("< 300 bps", "🟢 Low stress", "#22C55E"),
        ("300–500", "🟡 Moderate", "#FDE047"),
        ("> 500", "🔴 Credit stress", "#EF4444"),
    ],
    "10Y-2Y Spread": [
        ("> 1.0%", "🟢 Steep — normal", "#22C55E"),
        ("0.0–1.0%", "🟡 Flattening — watch", "#FDE047"),
        ("< 0.0%", "🔴 Inverted — recession risk", "#EF4444"),
    ],
    "DXY": [
        ("< 95", "🟢 Weak USD — risk-on", "#22C55E"),
        ("95–105", "🟡 Neutral range", "#FDE047"),
        ("> 105", "🔴 Strong USD — risk-off", "#EF4444"),
    ],
    "SPY/TLT": [
        ("Rising", "🟢 Risk-on — stocks leading", "#22C55E"),
        ("Flat", "🟡 Neutral", "#FDE047"),
        ("Falling", "🔴 Risk-off — bonds leading", "#EF4444"),
    ],
    "Copper/Gold": [
        ("Rising", "🟢 Growth optimism", "#22C55E"),
        ("Flat", "🟡 Neutral", "#FDE047"),
        ("Falling", "🔴 Safety demand", "#EF4444"),
    ],
}

# ============================================================
# LOAD DATA (cached for 6 hours)
# ============================================================
@st.cache_data(ttl=21600)
def load_data():
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        st.error("FRED_API_KEY not set in Streamlit secrets")
        st.stop()
    fred = Fred(api_key=fred_key)

    end_date = datetime.today()
    start_date = end_date - timedelta(days=180)

    # Yahoo Finance
    tickers = {"VIX":"^VIX","SPY":"SPY","TLT":"TLT","EEM":"EEM","HYG":"HYG","LQD":"LQD",
               "DXY":"DX-Y.NYB","Copper":"HG=F","Gold":"GC=F","USDJPY":"JPY=X"}
    yf_data = yf.download(list(tickers.values()), start=start_date, end=end_date, progress=False)["Close"]
    yf_data = yf_data.rename(columns={v:k for k,v in tickers.items()})

    # FRED
    fred_series = {"HY_OAS":"BAMLH0A0HYM2","IG_OAS":"BAMLC0A0CM",
                   "US10Y":"DGS10","US2Y":"DGS2","T10Y2Y":"T10Y2Y"}
    fred_dict = {name: fred.get_series(sid, start_date, end_date) for name, sid in fred_series.items()}
    fred_df = pd.DataFrame(fred_dict)
    fred_df.index = fred_df.index.tz_localize(None)

    # Combine
    df = yf_data.join(fred_df, how="outer").ffill()
    df["SPY_vs_TLT"] = df["SPY"] / df["TLT"]
    df["EEM_vs_SPY"] = df["EEM"] / df["SPY"]
    df["HYG_vs_LQD"] = df["HYG"] / df["LQD"]
    df["Copper_vs_Gold"] = df["Copper"] / df["Gold"]
    df["HY_bps"] = df["HY_OAS"] * 100

    # Moving averages
    for col in ["VIX","HY_bps","T10Y2Y","DXY","SPY_vs_TLT","Copper_vs_Gold"]:
        df[col+"_SMA"] = df[col].rolling(20).mean()

    return df

with st.spinner("Loading data..."):
    df = load_data()

latest = df.iloc[-1]
one_week = df.iloc[-5] if len(df) > 5 else df.iloc[0]
one_month = df.iloc[-22] if len(df) > 22 else df.iloc[0]

# ============================================================
# HELPER: classify a gauge value
# ============================================================
def classify_gauge(name, value, momentum=""):
    """Return (label, color_hex) for a given gauge and value."""
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
# COMPUTE RISK SCORE (0-100, higher = more risk)
# ============================================================
def compute_risk_score(vix, hy_bps, yc, dxy, spy_tlt_mom):
    score = 50  # neutral start

    # VIX: 0-30 contribution
    if vix < 15: score -= 15
    elif vix < 20: score -= 5
    elif vix < 25: score += 10
    elif vix < 30: score += 20
    else: score += 30

    # HY spreads: -15 to +20
    if hy_bps < 300: score -= 10
    elif hy_bps > 500: score += 20
    elif hy_bps > 400: score += 10

    # Yield curve: -15 to +20
    if yc < 0: score += 20
    elif yc < 0.5: score += 5
    elif yc > 1.5: score -= 10

    # DXY momentum: -10 to +15
    if dxy > 105: score += 15
    elif dxy > 100: score += 5
    elif dxy < 95: score -= 10

    # SPY/TLT momentum: -15 to +15
    if spy_tlt_mom > 3: score -= 15
    elif spy_tlt_mom > 0: score -= 5
    elif spy_tlt_mom < -3: score += 15
    elif spy_tlt_mom < 0: score += 5

    return max(0, min(100, score))

vix_val = latest["VIX"]
hy_val = latest["HY_bps"]
yc_val = latest["T10Y2Y"]
dxy_val = latest["DXY"]
spy_tlt_val = latest["SPY_vs_TLT"]
cg_val = latest["Copper_vs_Gold"]

spy_mom_1m = (spy_tlt_val / one_month["SPY_vs_TLT"] - 1) * 100
spy_mom_dir = "rising" if spy_mom_1m > 1 else ("falling" if spy_mom_1m < -1 else "flat")
cg_mom_1m = (cg_val / one_month["Copper_vs_Gold"] - 1) * 100
cg_mom_dir = "rising" if cg_mom_1m > 0.5 else ("falling" if cg_mom_1m < -0.5 else "flat")

risk_score = compute_risk_score(vix_val, hy_val, yc_val, dxy_val, spy_mom_1m)

if risk_score < 30: risk_label = "🟢 LOW RISK"
elif risk_score < 50: risk_label = "🟢 BELOW AVERAGE"
elif risk_score < 65: risk_label = "🟡 MODERATE"
elif risk_score < 80: risk_label = "🟠 ELEVATED"
else: risk_label = "🔴 HIGH RISK"

# ============================================================
# METRIC CARDS (with 1-week and 1-month deltas)
# ============================================================
st.subheader("📊 Current Snapshot")
st.caption("Changes shown: 1-week | 1-month vs prior")

col1, col2, col3, col4, col5, col6 = st.columns(6)

metrics = [
    ("VIX", vix_val, one_week["VIX"], one_month["VIX"], "VIX"),
    ("HY Spread", hy_val, one_week["HY_bps"], one_month["HY_bps"], "HY Spread"),
    ("10Y-2Y", yc_val, one_week["T10Y2Y"], one_month["T10Y2Y"], "10Y-2Y"),
    ("DXY", dxy_val, one_week["DXY"], one_month["DXY"], "DXY"),
    ("SPY/TLT", spy_tlt_val, one_week["SPY_vs_TLT"], one_month["SPY_vs_TLT"], "SPY/TLT"),
    ("Cu/Au", cg_val, one_week["Copper_vs_Gold"], one_month["Copper_vs_Gold"], "Copper/Gold"),
]

for col, (label, val, week_ago, month_ago, gauge_name) in zip([col1,col2,col3,col4,col5,col6], metrics):
    with col:
        delta_w = val - week_ago
        delta_m = val - month_ago

        # Format deltas appropriately
        if label in ("SPY/TLT", "Cu/Au"):
            delta_w_str = f"{(val/week_ago - 1)*100:+.1f}%"
            delta_m_str = f"{(val/month_ago - 1)*100:+.1f}%"
        elif label == "HY Spread":
            delta_w_str = f"{delta_w:+.0f} bps"
            delta_m_str = f"{delta_m:+.0f} bps"
        elif label == "10Y-2Y":
            delta_w_str = f"{delta_w:+.2f}%"
            delta_m_str = f"{delta_m:+.2f}%"
        else:
            delta_w_str = f"{delta_w:+.1f}"
            delta_m_str = f"{delta_m:+.1f}"

        st.metric(label, f"{val:.1f}" if label not in ("HY Spread","10Y-2Y","SPY/TLT","Cu/Au")
                  else (f"{val:.0f} bps" if label=="HY Spread"
                  else (f"{val:.2f}%" if label=="10Y-2Y"
                  else (f"{val:.2f}" if label=="SPY/TLT" else f"{val:.4f}"))),
                  delta=f"{delta_w_str} | {delta_m_str}",
                  delta_color="normal")

        # Classification label
        if gauge_name == "SPY/TLT":
            cls_label, cls_color = classify_gauge(gauge_name, val, spy_mom_dir)
        elif gauge_name == "Copper/Gold":
            cls_label, cls_color = classify_gauge(gauge_name, val, cg_mom_dir)
        else:
            cls_label, cls_color = classify_gauge(gauge_name, val)
        st.markdown(f"<span style='color:{cls_color};font-size:0.85rem;font-weight:600;'>{cls_label}</span>",
                    unsafe_allow_html=True)

# ============================================================
# RISK SCORE GAUGE + SUMMARY
# ============================================================
st.divider()
risk_col, summary_col = st.columns([1, 2])

with risk_col:
    st.subheader("🧭 Overall Risk Score")
    # Custom gauge using progress bar
    st.markdown(f"<h1 style='text-align:center;margin:0;'>{risk_score}<span style='font-size:1.5rem;'>/100</span></h1>",
                unsafe_allow_html=True)
    st.progress(risk_score / 100)
    st.markdown(f"<p style='text-align:center;font-weight:bold;font-size:1.2rem;'>{risk_label}</p>",
                unsafe_allow_html=True)
    st.caption("0–30: Low | 30–50: Below avg | 50–65: Moderate | 65–80: Elevated | 80–100: High")

with summary_col:
    st.subheader("🧠 Summary")
    signals = []
    if vix_val > 30: signals.append("🔴 VIX above 30 — extreme fear in equity markets")
    elif vix_val > 25: signals.append("🟠 VIX elevated — heightened caution warranted")
    elif vix_val > 20: signals.append("🟡 VIX moderately elevated")
    else: signals.append("🟢 VIX in calm range")

    if hy_val > 500: signals.append("🔴 HY spreads above 500 bps — significant credit stress")
    elif hy_val > 400: signals.append("🟡 HY spreads widening — monitor closely")
    else: signals.append("🟢 Credit spreads contained")

    if yc_val < 0: signals.append("🔴 Yield curve inverted — classic recession warning")
    elif yc_val < 0.5: signals.append("🟡 Yield curve flattening — watch for inversion")
    else: signals.append("🟢 Yield curve healthy — normal slope")

    if dxy_val > 105: signals.append("🔴 Dollar surging — tightening global financial conditions")
    elif dxy_val > 100: signals.append("🟡 Dollar above parity — modest headwind")
    else: signals.append("🟢 Dollar contained — supportive for risk assets")

    if spy_mom_1m > 3: signals.append("🟢 Stocks strongly outperforming bonds — clear risk-on")
    elif spy_mom_1m < -3: signals.append("🔴 Bonds strongly outperforming stocks — defensive rotation")
    else: signals.append("🟡 Equity/bond ratio balanced — no strong directional signal")

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
st.subheader("📈 6-Month Trends")

fig = make_subplots(
    rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    subplot_titles=(
        "① VIX — Fear Index  |  Rising = growing fear, falling = complacency",
        "② High-Yield Spread — Credit stress (bps)  |  Wider = bond market pricing more default risk",
        "③ 10Y-2Y Spread — Yield curve (%)  |  Below zero (shaded) = recession warning",
        "④ US Dollar (DXY) — Currency strength  |  Rising = risk-off flows into USD",
        "⑤ SPY / TLT — Stocks vs Bonds  |  Rising = equities leading (risk-on appetite)",
        "⑥ Copper / Gold — Growth vs Safety  |  Rising = industrial optimism, falling = safe-haven demand",
    ),
    row_heights=[1,1,1,1,1,1]
)

panels = [
    (1, "VIX", "VIX_SMA", "#DC143C", [20,30], ["orange","red"], 2),
    (2, "HY_bps", "HY_bps_SMA", "#FF8C00", [500], ["red"], 30),
    (3, "T10Y2Y", "T10Y2Y_SMA", "#4682B4", [0], ["red"], 0.3),
    (4, "DXY", "DXY_SMA", "#008080", [100], ["grey"], 1),
    (5, "SPY_vs_TLT", "SPY_vs_TLT_SMA", "#9370DB", [], [], 0.3),
    (6, "Copper_vs_Gold", "Copper_vs_Gold_SMA", "#DAA520", [], [], 0.0002),
]

for row, col, trend, color, hlines, hcolors, pad in panels:
    fig.add_trace(go.Scatter(x=df.index, y=df[col],
        line=dict(color=color, width=2.2), name=col, showlegend=(row==1)), row=row, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df[trend],
        line=dict(color="grey", width=1, dash="dash"),
        name="20-day trend", showlegend=(row==1)), row=row, col=1)
    for val, hcol in zip(hlines, hcolors):
        fig.add_hline(y=val, line_dash="dot", line_color=hcol, opacity=0.5, row=row, col=1)
    vals = df[col].dropna()
    fig.update_yaxes(range=[vals.min()-pad, vals.max()+pad], row=row, col=1)

# Inverted yield curve shading
fig.add_hrect(y0=-3, y1=0, line_width=0, fillcolor="red", opacity=0.06, row=3, col=1)

fig.update_layout(
    height=1050, hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5, font=dict(size=11)),
    margin=dict(t=60, b=40, l=40, r=40),
    plot_bgcolor="white", paper_bgcolor="white"
)

for i in range(1,7):
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="#EEE", zeroline=False, row=i, col=1)
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="#EEE", row=i, col=1)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# LEGEND EXPANDER
# ============================================================
with st.expander("📖 Gauge Legend — What do the levels mean?"):
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
