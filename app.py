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
# LOAD DATA (cached for 6 hours)
# ============================================================
@st.cache_data(ttl=21600)
def load_data():
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        st.error("FRED_API_KEY not set")
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

    # Moving averages
    for col in ["VIX","HY_OAS","T10Y2Y","DXY","SPY_vs_TLT","Copper_vs_Gold"]:
        df[col+"_SMA"] = df[col].rolling(20).mean()
    df["HY_bps"] = df["HY_OAS"] * 100
    df["HY_bps_SMA"] = df["HY_bps"].rolling(20).mean()

    return df

with st.spinner("Loading data..."):
    df = load_data()

latest = df.iloc[-1]
one_month = df.iloc[-22] if len(df) > 22 else df.iloc[0]

# ============================================================
# METRIC CARDS
# ============================================================
st.subheader("📊 Current Snapshot")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    vix = latest["VIX"]
    delta = vix - one_month["VIX"]
    st.metric("VIX", f"{vix:.1f}", delta=f"{delta:+.1f}")
    if vix > 30: st.error("🔴 Panic")
    elif vix > 20: st.warning("🟡 Caution")
    else: st.success("🟢 Calm")

with col2:
    hy = latest["HY_bps"]
    hy_d = hy - one_month["HY_bps"]
    st.metric("HY Spread", f"{hy:.0f} bps", delta=f"{hy_d:+.0f}")
    if hy > 500: st.warning("⚠️ Stress")

with col3:
    yc = latest["T10Y2Y"]
    yc_d = yc - one_month["T10Y2Y"]
    st.metric("10Y-2Y", f"{yc:.2f}%", delta=f"{yc_d:+.2f}")
    if yc < 0: st.error("⚠️ Inverted")

with col4:
    dxy = latest["DXY"]
    dxy_d = dxy - one_month["DXY"]
    st.metric("DXY", f"{dxy:.1f}", delta=f"{dxy_d:+.1f}")

with col5:
    spy_tlt = latest["SPY_vs_TLT"]
    st_d = (spy_tlt/one_month["SPY_vs_TLT"] - 1) * 100
    st.metric("SPY/TLT", f"{spy_tlt:.2f}", delta=f"{st_d:+.1f}%")

with col6:
    cg = latest["Copper_vs_Gold"]
    cg_d = (cg/one_month["Copper_vs_Gold"] - 1) * 100
    st.metric("Cu/Au", f"{cg:.4f}", delta=f"{cg_d:+.1f}%")

# ============================================================
# CHART
# ============================================================
st.subheader("📈 6-Month Trends")

fig = make_subplots(
    rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.04,
    subplot_titles=(
        "① VIX — Fear Index", "② High-Yield Spread — Credit stress (bps)",
        "③ 10Y-2Y Spread — Yield curve (%)", "④ US Dollar (DXY)",
        "⑤ SPY / TLT — Stocks vs Bonds", "⑥ Copper / Gold — Growth vs Safety"
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
        line=dict(color=color, width=2), name=col, showlegend=(row==1)), row=row, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df[trend],
        line=dict(color="grey", width=1, dash="dash"),
        name="20d trend", showlegend=(row==1)), row=row, col=1)
    for val, hcol in zip(hlines, hcolors):
        fig.add_hline(y=val, line_dash="dot", line_color=hcol, opacity=0.5, row=row, col=1)
    vals = df[col].dropna()
    fig.update_yaxes(range=[vals.min()-pad, vals.max()+pad], row=row, col=1)

fig.add_hrect(y0=-3, y1=0, line_width=0, fillcolor="red", opacity=0.06, row=3, col=1)

fig.update_layout(
    height=1000, hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5),
    margin=dict(t=60, b=80, l=40, r=40),
    plot_bgcolor="white", paper_bgcolor="white"
)

for i in range(1,7):
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="#EEE", zeroline=False, row=i, col=1)
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="#EEE", row=i, col=1)

fig.add_annotation(xref="paper", yref="paper", x=0.5, y=-0.05,
    text=("📖 ① VIX >20 caution, >30 panic | ② HY >500 bps = stress | "
          "③ Negative = recession warning | ④ DXY rising = risk-off | "
          "⑤ SPY/TLT rising = stocks leading | ⑥ Cu/Au rising = growth optimism"),
    showarrow=False, align="center", font=dict(size=9, color="#666"),
    bgcolor="#F9F9F9", borderpad=8, bordercolor="#DDD", borderwidth=1)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# SENTIMENT SUMMARY
# ============================================================
st.subheader("🧠 Automated Summary")
signals = []
if vix > 25: signals.append("🔴 VIX elevated — fear in markets")
if pd.notna(yc) and yc < 0: signals.append("⚠️ Yield curve inverted — recession signal")
if hy > 500: signals.append("🔴 High-yield spreads wide — credit stress")
spy_mom = (spy_tlt/one_month["SPY_vs_TLT"] - 1)*100
if spy_mom > 3: signals.append("🟢 Stocks strongly outperforming bonds (risk-on)")
elif spy_mom < -3: signals.append("🔴 Bonds strongly outperforming stocks (risk-off)")
dxy_mom = (dxy/one_month["DXY"] - 1)*100
if dxy_mom > 2: signals.append("🔴 Dollar strengthening sharply")
if not signals: signals.append("⚪ No strong stress signals. Mixed/neutral regime.")
for s in signals: st.write(s)

# Raw data expander
with st.expander("🔍 View raw data"):
    st.dataframe(df[["VIX","HY_bps","T10Y2Y","DXY","SPY_vs_TLT","Copper_vs_Gold"]].tail(30).sort_index(ascending=False),
                 use_container_width=True)
