import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
from datetime import datetime, timedelta, timezone
import os

# ============================================================
# INDICATOR CONFIGURATION
# ============================================================
INDICATOR_CONFIG = {
    "VIX": {"higher_is_riskier": True, "tier": 1, "votes": True, "frequency": "daily", "percentile_window_years": 10, "question": "Is equity fear rising?", "unit": "", "decimals": 1, "color": "#DC143C", "fixed_refs": [20, 30], "ref_labels": ["Elevated", "Panic"], "ref_colors": ["orange", "red"], "summary_verb_up": "rose", "summary_verb_down": "fell", "summary_label": "VIX"},
    "HY_bps": {"higher_is_riskier": True, "tier": 1, "votes": True, "frequency": "daily", "percentile_window_years": 10, "question": "Is credit stress rising?", "unit": "bps", "decimals": 0, "color": "#FF8C00", "fixed_refs": [500], "ref_labels": ["Stress"], "ref_colors": ["red"], "summary_verb_up": "widened", "summary_verb_down": "tightened", "summary_label": "credit spreads"},
    "T10Y3M": {"higher_is_riskier": False, "tier": 1, "votes": True, "frequency": "daily", "percentile_window_years": 10, "question": "Is recession risk increasing?", "unit": "%", "decimals": 2, "color": "#4682B4", "fixed_refs": [0], "ref_labels": ["Inverted"], "ref_colors": ["red"], "summary_verb_up": "steepened", "summary_verb_down": "flattened", "summary_label": "yield curve"},
    "DXY": {"higher_is_riskier": True, "tier": 2, "votes": True, "frequency": "daily", "percentile_window_years": 10, "question": "Are financial conditions tightening?", "unit": "", "decimals": 1, "color": "#008080", "fixed_refs": [100], "ref_labels": ["Parity"], "ref_colors": ["grey"], "summary_verb_up": "strengthened", "summary_verb_down": "weakened", "summary_label": "DXY"},
    "NFCI": {"higher_is_riskier": True, "tier": 2, "votes": True, "frequency": "weekly", "percentile_window_years": 10, "question": "Are financial conditions tightening?", "unit": "", "decimals": 2, "color": "#8B0000", "fixed_refs": [0], "ref_labels": ["Tight"], "ref_colors": ["red"], "summary_verb_up": "tightened", "summary_verb_down": "loosened", "summary_label": "financial conditions"},
    "SPY_vs_TLT": {"higher_is_riskier": False, "tier": 2, "votes": True, "frequency": "daily", "percentile_window_years": 10, "question": "Are investors embracing risk?", "unit": "", "decimals": 2, "color": "#9370DB", "fixed_refs": [], "ref_labels": [], "ref_colors": [], "summary_verb_up": "rose", "summary_verb_down": "fell", "summary_label": "SPY/TLT"},
    "USDJPY": {"higher_is_riskier": False, "tier": 3, "votes": False, "frequency": "daily", "percentile_window_years": 10, "question": "Real-time risk flight?", "unit": "", "decimals": 1, "color": "#C71585", "fixed_refs": [], "ref_labels": [], "ref_colors": [], "summary_verb_up": "rose", "summary_verb_down": "fell", "summary_label": "USD/JPY"},
    "Copper_vs_Gold": {"higher_is_riskier": False, "tier": 3, "votes": False, "frequency": "daily", "percentile_window_years": 10, "question": "Growth optimism or safety demand?", "unit": "", "decimals": 4, "color": "#DAA520", "fixed_refs": [], "ref_labels": [], "ref_colors": [], "summary_verb_up": "rose", "summary_verb_down": "fell", "summary_label": "Copper/Gold"},
    "AUDUSD": {"higher_is_riskier": False, "tier": 3, "votes": False, "frequency": "daily", "percentile_window_years": 10, "question": "Growth proxy confirmation?", "unit": "", "decimals": 4, "color": "#228B22", "fixed_refs": [], "ref_labels": [], "ref_colors": [], "summary_verb_up": "rose", "summary_verb_down": "fell", "summary_label": "AUD/USD"},
    "EEM_vs_SPY": {"higher_is_riskier": False, "tier": 3, "votes": False, "frequency": "daily", "percentile_window_years": "max_available", "question": "Capital flowing to risk?", "unit": "", "decimals": 4, "color": "#2E86AB", "fixed_refs": [], "ref_labels": [], "ref_colors": [], "summary_verb_up": "rose", "summary_verb_down": "fell", "summary_label": "EEM/SPY"},
    "HYG_vs_LQD": {"higher_is_riskier": False, "tier": 3, "votes": False, "frequency": "daily", "percentile_window_years": "max_available", "question": "Credit risk appetite?", "unit": "", "decimals": 4, "color": "#E07B39", "fixed_refs": [], "ref_labels": [], "ref_colors": [], "summary_verb_up": "rose", "summary_verb_down": "fell", "summary_label": "HYG/LQD"},
    "XLY_vs_XLP": {"higher_is_riskier": False, "tier": 3, "votes": False, "frequency": "daily", "percentile_window_years": "max_available", "question": "Cyclical rotation?", "unit": "", "decimals": 4, "color": "#6A994E", "fixed_refs": [], "ref_labels": [], "ref_colors": [], "summary_verb_up": "rose", "summary_verb_down": "fell", "summary_label": "XLY/XLP"},
    "DGS2": {"higher_is_riskier": None, "tier": "deep_dive", "votes": False, "frequency": "daily", "percentile_window_years": None, "question": "What is the market pricing for Fed policy?", "unit": "%", "decimals": 2, "color": "#6B8E23", "fixed_refs": [], "ref_labels": [], "ref_colors": [], "summary_verb_up": "", "summary_verb_down": "", "summary_label": ""},
    "Gold_vs_Oil": {"higher_is_riskier": None, "tier": "deep_dive", "votes": False, "frequency": "daily", "percentile_window_years": None, "question": "Growth scare or inflation scare?", "unit": "", "decimals": 2, "color": "#B8860B", "fixed_refs": [], "ref_labels": [], "ref_colors": [], "summary_verb_up": "", "summary_verb_down": "", "summary_label": ""},
    "WALCL_T": {"higher_is_riskier": None, "tier": "deep_dive", "votes": False, "frequency": "weekly", "percentile_window_years": None, "question": "Is liquidity expanding or contracting?", "unit": "$T", "decimals": 2, "color": "#4B0082", "fixed_refs": [], "ref_labels": [], "ref_colors": [], "summary_verb_up": "", "summary_verb_down": "", "summary_label": ""},
}

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="Global Risk Dashboard", page_icon="🌍", layout="wide")
view_mode = st.radio("🔍 View", ["Quick View", "Full Dashboard"], horizontal=True, index=0)
full_view = view_mode == "Full Dashboard"
st.title("🌍 Global Risk & Capital Flow Dashboard")

with st.expander("ℹ️ About this dashboard"):
    st.markdown("""
    **This dashboard identifies macro regimes, not short-term market moves.** Indicators may remain in one state for extended periods while asset prices move differently.
    
    **Architecture:** 6 core indicators vote on regime (majority wins, ties break conservative). Stress percentile uses 10-year ranks with directional consistency. Weighted signals use a 3/2/1 tier system. Secondary indicators confirm but don't vote. Deep-dive indicators provide additional context.
    """)

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data(ttl=14400)
def load_all_data():
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        st.error("FRED_API_KEY not set")
        st.stop()
    fred = Fred(api_key=fred_key)
    today = datetime.today()
    hist_start = today - timedelta(days=10*365)

    tickers = {"VIX": "^VIX", "SPY": "SPY", "TLT": "TLT", "EEM": "EEM", "HYG": "HYG", "LQD": "LQD", "DXY": "DX-Y.NYB", "Copper": "HG=F", "Gold": "GC=F", "Oil": "CL=F", "USDJPY": "JPY=X", "AUDUSD": "AUDUSD=X", "XLY": "XLY", "XLP": "XLP"}
    yf_hist = yf.download(list(tickers.values()), start=hist_start, end=today, progress=False)["Close"]
    yf_hist = yf_hist.rename(columns={v: k for k, v in tickers.items()})

    fred_series = {"HY_OAS": "BAMLH0A0HYM2", "T10Y3M": "T10Y3M", "NFCI": "NFCI", "DGS2": "DGS2", "WALCL": "WALCL"}
    fred_hist = {}
    for name, sid in fred_series.items():
        fred_hist[name] = fred.get_series(sid, hist_start, today)
    fred_hist = pd.DataFrame(fred_hist)
    fred_hist.index = fred_hist.index.tz_localize(None)

    percentile_source = yf_hist.join(fred_hist, how="outer").ffill()
    percentile_source["HY_bps"] = percentile_source["HY_OAS"] * 100
    percentile_source["SPY_vs_TLT"] = percentile_source["SPY"] / percentile_source["TLT"]
    percentile_source["Copper_vs_Gold"] = percentile_source["Copper"] / percentile_source["Gold"]
    percentile_source["Gold_vs_Oil"] = percentile_source["Gold"] / percentile_source["Oil"]
    percentile_source["EEM_vs_SPY"] = percentile_source["EEM"] / percentile_source["SPY"]
    percentile_source["HYG_vs_LQD"] = percentile_source["HYG"] / percentile_source["LQD"]
    percentile_source["XLY_vs_XLP"] = percentile_source["XLY"] / percentile_source["XLP"]
    percentile_source["WALCL_T"] = percentile_source["WALCL"] / 1_000_000

    dash_start = today - timedelta(days=365)
    df = percentile_source.loc[dash_start:].copy()
    for key in INDICATOR_CONFIG:
        if key in df.columns:
            df[key + "_SMA"] = df[key].rolling(20).mean()

    re_steepening = False
    if "T10Y3M" in df.columns:
        curve_series = df["T10Y3M"].dropna()
        if len(curve_series) >= 252:
            was_inverted = (curve_series.iloc[-252:] < 0).any()
            trough = curve_series.iloc[-252:].min()
            current = curve_series.iloc[-1]
            re_steepening = was_inverted and (current - trough) > 0.5

    return df, percentile_source, re_steepening

if st.button("🔄 Refresh data now (clear cache)"):
    st.cache_data.clear()
    st.rerun()

with st.spinner("Loading data..."):
    df, percentile_source, re_steepening = load_all_data()

latest = df.iloc[-1]
data_date = df.index[-1].date()
cache_time = datetime.now(timezone.utc)
now_date = datetime.now(timezone.utc).date()
days_behind = (now_date - data_date).days
freshness = f"Data as of: {data_date}" + (f" ({days_behind} day{'s' if days_behind > 1 else ''} behind)" if days_behind > 0 else " (current)")
st.caption(f"Dashboard: {df.index[0].date()} – {data_date} | {freshness} | Page loaded: {cache_time.strftime('%Y-%m-%d %H:%M UTC')}")
if days_behind > 0:
    st.warning(f"⚠️ Latest data is from {data_date}. Weekly FRED series update on different schedules. Use refresh to check.")

one_week_date = latest.name - timedelta(days=7)
one_month_date = latest.name - timedelta(days=30)
one_week_ago = df.iloc[df.index.get_indexer([one_week_date], method='ffill')[0]]
one_month_ago = df.iloc[df.index.get_indexer([one_month_date], method='ffill')[0]]

# ============================================================
# PERCENTILES
# ============================================================
def get_percentile(series, value, higher_is_riskier):
    clean = series.dropna()
    if len(clean) < 100:
        return 50, False
    pct = (clean < value).sum() / len(clean) * 100
    return pct, pct < 5 or pct > 95

def normal_flag(pct, extreme):
    if extreme:
        return f"🔴 Unusually {'low' if pct < 5 else 'high'} ({pct:.0f}th %ile)"
    elif pct < 15 or pct > 85:
        return f"🟡 Somewhat unusual ({pct:.0f}th %ile)"
    return f"🟢 Normal range ({pct:.0f}th %ile)"

pct_data = {}
for key, cfg in INDICATOR_CONFIG.items():
    if cfg["percentile_window_years"] is not None and key in percentile_source.columns and key in latest.index:
        val = latest[key]
        if pd.notna(val):
            pct_data[key] = get_percentile(percentile_source[key], val, cfg.get("higher_is_riskier", True))

# ============================================================
# STRESS SCORE
# ============================================================
stress_components = {}
for key in ["VIX", "HY_bps", "DXY", "T10Y3M"]:
    if key in pct_data:
        pct, _ = pct_data[key]
        cfg = INDICATOR_CONFIG[key]
        stress_components[key] = pct if cfg["higher_is_riskier"] else 100 - pct

stress_percentile = sum(stress_components.values()) / len(stress_components) if stress_components else 50

one_month_stress = {}
for key in ["VIX", "HY_bps", "DXY", "T10Y3M"]:
    if key in percentile_source.columns and key in one_month_ago.index:
        series = percentile_source[key].dropna()
        val = one_month_ago[key]
        if pd.notna(val) and len(series) > 0:
            pct_1m, _ = get_percentile(series, val, INDICATOR_CONFIG[key].get("higher_is_riskier", True))
            one_month_stress[key] = pct_1m if INDICATOR_CONFIG[key]["higher_is_riskier"] else 100 - pct_1m

stress_delta = stress_percentile - (sum(one_month_stress.values()) / len(one_month_stress)) if one_month_stress else 0
if stress_delta < -10:
    trend_label, trend_icon = "Improving", "🟢"
elif stress_delta > 10:
    trend_label, trend_icon = "Deteriorating", "🔴"
else:
    trend_label, trend_icon = "Stable", "🟡"

# ============================================================
# REGIME VOTE
# ============================================================
def vote_regime(pct, higher_is_riskier):
    if pct is None:
        return "neutral"
    if higher_is_riskier:
        return "risk_on" if pct < 40 else ("risk_off" if pct > 60 else "neutral")
    return "risk_on" if pct > 60 else ("risk_off" if pct < 40 else "neutral")

votes = {"risk_on": 0, "neutral": 0, "risk_off": 0}
for key, cfg in INDICATOR_CONFIG.items():
    if cfg.get("votes") and key in pct_data:
        votes[vote_regime(pct_data[key][0], cfg["higher_is_riskier"])] += 1

if votes["risk_off"] >= max(votes["neutral"], votes["risk_on"]):
    regime_label, regime_color = "RISK-OFF", "#EF4444"
elif votes["neutral"] >= votes["risk_on"]:
    regime_label, regime_color = "NEUTRAL", "#FDE047"
else:
    regime_label, regime_color = "RISK-ON", "#22C55E"

if votes["risk_off"] >= 4:
    regime_intensity = "STRONG"
elif votes["risk_off"] >= 2 or regime_label == "RISK-OFF":
    regime_intensity = "MODERATE"
elif votes["risk_on"] >= 4:
    regime_intensity = "STRONG"
elif votes["risk_on"] >= 2 or regime_label == "RISK-ON":
    regime_intensity = "MODERATE"
else:
    regime_intensity = ""

regime_display = f"{regime_intensity} {regime_label}".strip()

# ============================================================
# WEIGHTED SIGNALS
# ============================================================
weighted_warnings, weighted_total = 0, 0
unweighted_warnings, unweighted_total = 0, 0
warning_list = []

for key, cfg in INDICATOR_CONFIG.items():
    if isinstance(cfg["tier"], int) and key in pct_data:
        pct, _ = pct_data[key]
        tw = 4 - cfg["tier"]
        weighted_total += tw
        unweighted_total += 1
        is_warning = pct > 85 if cfg["higher_is_riskier"] else pct < 15
        if is_warning:
            weighted_warnings += tw
            unweighted_warnings += 1
            warning_list.append(f"{'🔴' if tw >= 2 else '🟠'} {key}")

signal_pct = (weighted_warnings / weighted_total * 100) if weighted_total > 0 else 0

# ============================================================
# CURRENT VALUES
# ============================================================
def gv(key):
    return latest[key] if key in latest.index and pd.notna(latest[key]) else None

vix_val, hy_val, yc_val, dxy_val = gv("VIX"), gv("HY_bps"), gv("T10Y3M"), gv("DXY")
nfci_val, spy_val = gv("NFCI"), gv("SPY_vs_TLT")
usdjpy_val, cg_val, aud_val = gv("USDJPY"), gv("Copper_vs_Gold"), gv("AUDUSD")
eem_val, hyg_val, xly_val = gv("EEM_vs_SPY"), gv("HYG_vs_LQD"), gv("XLY_vs_XLP")
dgs2_val, go_val, walcl_val = gv("DGS2"), gv("Gold_vs_Oil"), gv("WALCL_T")

def get_dir(val, m_ago, up, down):
    if val is None or m_ago is None or pd.isna(val) or pd.isna(m_ago):
        return "flat"
    chg = (val / m_ago - 1) * 100
    return "rising" if chg > up else ("falling" if chg < -down else "flat")

spy_dir = get_dir(spy_val, one_month_ago.get("SPY_vs_TLT"), 1, 1)
cg_dir = get_dir(cg_val, one_month_ago.get("Copper_vs_Gold"), 0.5, 0.5)
jpy_dir = get_dir(usdjpy_val, one_month_ago.get("USDJPY"), 1, 1)
aud_dir = get_dir(aud_val, one_month_ago.get("AUDUSD"), 0.5, 0.5)

# ============================================================
# REGIME BANNER
# ============================================================
st.markdown(f"""
<div style="background-color:#F8FAFC;border:2px solid {regime_color};border-radius:12px;padding:16px 24px;margin-bottom:16px;">
<table style="width:100%;text-align:center;font-size:1.1rem;">
<tr>
<td><b>Current Regime</b><br><span style="font-size:1.8rem;font-weight:bold;color:{regime_color};">{regime_display}</span></td>
<td><b>Stress Percentile</b><br><span style="font-size:1.8rem;font-weight:bold;">{stress_percentile:.0f}th</span></td>
<td><b>Signals Active</b><br><span style="font-size:1.8rem;font-weight:bold;">{signal_pct:.0f}%</span> <span style="font-size:1rem;">({weighted_warnings}/{weighted_total})</span></td>
<td><b>Trend</b><br><span style="font-size:1.8rem;">{trend_icon}</span> <span style="font-size:1.2rem;">{trend_label}</span></td>
</tr>
</table>
<p style="text-align:center;margin-top:8px;font-size:0.9rem;color:#666;">
Vote: Risk-On {votes['risk_on']} | Neutral {votes['neutral']} | Risk-Off {votes['risk_off']}{' | ⚠️ <b>Curve re-steepening from inversion — historically associated with recession onset</b>' if re_steepening else ''}
</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# WEEKLY SUMMARY
# ============================================================
week_parts = []
for key in ["VIX", "HY_bps", "T10Y3M", "DXY", "NFCI", "SPY_vs_TLT"]:
    cfg = INDICATOR_CONFIG[key]
    if key in latest.index and key in one_week_ago.index:
        cur, prev = latest[key], one_week_ago[key]
        if pd.notna(cur) and pd.notna(prev):
            week_parts.append(f"{cfg['summary_label']} {cfg['summary_verb_up'] if cur > prev else cfg['summary_verb_down']}")
st.info(f"**📰 This week:** {', '.join(week_parts)}. Regime: {regime_display}. Stress: {stress_percentile:.0f}th percentile.")

# ============================================================
# CORE METRIC CARDS
# ============================================================
st.subheader("📊 Core Indicators")
st.caption("Changes: 1-week | 1-month | Normalcy vs 10-year history")

core_specs = [
    ("VIX", "VIX", vix_val), ("HY_bps", "HY Spread", hy_val), ("T10Y3M", "10Y-3M", yc_val),
    ("DXY", "DXY", dxy_val), ("NFCI", "NFCI", nfci_val), ("SPY_vs_TLT", "SPY/TLT", spy_val),
]
cols = st.columns(6)
for col, (key, label, val) in zip(cols, core_specs):
    with col:
        cfg = INDICATOR_CONFIG[key]
        freq = f" [{cfg['frequency'].capitalize()}]" if cfg["frequency"] != "daily" else ""
        w = one_week_ago[key] if key in one_week_ago.index else None
        m = one_month_ago[key] if key in one_month_ago.index else None
        pct_keys = ["SPY_vs_TLT", "Copper_vs_Gold", "AUDUSD", "USDJPY", "EEM_vs_SPY", "HYG_vs_LQD", "XLY_vs_XLP", "VIX", "DXY"]
        dw_s = f"{(val/w - 1)*100:+.1f}%" if (val and w and pd.notna(val) and pd.notna(w) and key in pct_keys) else (f"{val - w:+.{cfg['decimals']}f}" if (val and w and pd.notna(val) and pd.notna(w)) else "N/A")
        dm_s = f"{(val/m - 1)*100:+.1f}%" if (val and m and pd.notna(val) and pd.notna(m) and key in pct_keys) else (f"{val - m:+.{cfg['decimals']}f}" if (val and m and pd.notna(val) and pd.notna(m)) else "N/A")
        disp = f"{val:.{cfg['decimals']}f}{cfg['unit']}" if val is not None and pd.notna(val) else "N/A"
        st.metric(f"{label}{freq}", disp, delta=f"{dw_s} | {dm_s}", delta_color="normal")
        if key in pct_data:
            pct, ext = pct_data[key]
            st.caption(normal_flag(pct, ext))
        else:
            st.caption("No data")

# ============================================================
# SIGNAL COUNT + STRESS BREAKDOWN
# ============================================================
st.divider()
sc1, sc2 = st.columns([1, 1])
with sc1:
    st.subheader("🚦 Weighted Signals")
    st.markdown(f"<h1 style='text-align:center;'>{signal_pct:.0f}%<span style='font-size:1.2rem;'> active</span></h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;'>({weighted_warnings}/{weighted_total} weighted | {unweighted_warnings}/{unweighted_total} unweighted)</p>", unsafe_allow_html=True)
    for w in warning_list:
        st.markdown(w)
    if not warning_list:
        st.success("No warning signals active")
with sc2:
    st.subheader("📏 Stress Percentile")
    st.markdown(f"<h1 style='text-align:center;'>{stress_percentile:.0f}th</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;'>Trend: {trend_icon} {trend_label} ({stress_delta:+.0f} pts this month)</p>", unsafe_allow_html=True)
    st.markdown("**Components:**")
    for k, v in stress_components.items():
        st.markdown(f"- {k}: {v:.0f}th percentile stress")
    st.caption("Higher = more stress. Curve inverted so steep = low stress. VIX, HY, DXY, 10Y-3M equally weighted.")
    
# ============================================================
# SECONDARY METRIC CARDS (Full View only)
# ============================================================
if full_view:
    st.divider()
    st.subheader("📊 Secondary Indicators (Confirmation)")
    st.caption("These confirm or challenge the regime but do not vote on it.")

    sec_specs = [
        ("USDJPY", "USD/JPY", usdjpy_val), ("Copper_vs_Gold", "Cu/Au", cg_val),
        ("AUDUSD", "AUD/USD", aud_val),
    ]
    # Only include ratio columns if they exist in the dataframe
    for key, label, val in [("EEM_vs_SPY", "EEM/SPY", eem_val), ("HYG_vs_LQD", "HYG/LQD", hyg_val), ("XLY_vs_XLP", "XLY/XLP", xly_val)]:
        if key in df.columns and key in one_week_ago.index:
            sec_specs.append((key, label, val))

    cols2 = st.columns(len(sec_specs))
    pct_keys_list = ["SPY_vs_TLT", "Copper_vs_Gold", "AUDUSD", "USDJPY", "EEM_vs_SPY", "HYG_vs_LQD", "XLY_vs_XLP", "VIX", "DXY"]
    for col, (key, label, val) in zip(cols2, sec_specs):
        with col:
            cfg = INDICATOR_CONFIG[key]
            w = one_week_ago[key] if key in one_week_ago.index else None
            m = one_month_ago[key] if key in one_month_ago.index else None
            dw_s = f"{(val/w - 1)*100:+.1f}%" if (val is not None and w is not None and pd.notna(val) and pd.notna(w)) else "N/A"
            dm_s = f"{(val/m - 1)*100:+.1f}%" if (val is not None and m is not None and pd.notna(val) and pd.notna(m)) else "N/A"
            disp = f"{val:.{cfg['decimals']}f}" if val is not None and pd.notna(val) else "N/A"
            st.metric(label, disp, delta=f"{dw_s} | {dm_s}", delta_color="normal")
            if key in pct_data:
                pct, ext = pct_data[key]
                st.caption(normal_flag(pct, ext))
            else:
                st.caption("No data")

# ============================================================
# CHARTS (Full View only)
# ============================================================
if full_view:
    st.divider()
    cc1, cc2 = st.columns([3, 1])
    with cc1:
        st.subheader("📈 Trends")
    with cc2:
        date_range = st.selectbox("Timeframe", ["1 month", "3 months", "6 months", "1 year"], index=2)

    range_map = {"1 month": 22, "3 months": 66, "6 months": 132, "1 year": 260}
    cutoff = df.index[-1] - timedelta(days=range_map[date_range])
    chart_df = df.loc[cutoff:].copy()
    chart_df = chart_df[chart_df.index.dayofweek < 5]

    # Only chart indicators that exist in the dataframe
    chart_keys = [k for k in INDICATOR_CONFIG.keys() if k in chart_df.columns]
    num_charts = len(chart_keys)

    fig = make_subplots(
        rows=num_charts, cols=1, shared_xaxes=True, vertical_spacing=0.035,
        subplot_titles=[f"{INDICATOR_CONFIG[k]['question']}" for k in chart_keys],
        row_heights=[1]*num_charts
    )

    for i, key in enumerate(chart_keys):
        row = i + 1
        cfg = INDICATOR_CONFIG[key]
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[key],
            line=dict(color=cfg["color"], width=2), name=key, showlegend=(row==1)), row=row, col=1)
        sma_col = key + "_SMA"
        if sma_col in chart_df.columns:
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[sma_col],
                line=dict(color="grey", width=1, dash="dash"), name="20d trend", showlegend=(row==1)), row=row, col=1)
        for ref_val, ref_col in zip(cfg["fixed_refs"], cfg["ref_colors"]):
            fig.add_hline(y=ref_val, line_dash="dot", line_color=ref_col, opacity=0.5, row=row, col=1)
        vals = chart_df[key].dropna()
        if len(vals) > 0:
            pad = (vals.max() - vals.min()) * 0.15 or 1
            fig.update_yaxes(range=[vals.min()-pad, vals.max()+pad], row=row, col=1)

    # Yield curve red zone — find which row is T10Y3M
    if "T10Y3M" in chart_keys:
        yc_row = chart_keys.index("T10Y3M") + 1
        fig.add_hrect(y0=-3, y1=0, line_width=0, fillcolor="red", opacity=0.06, row=yc_row, col=1)

    chart_height = max(800, num_charts * 130)
    fig.update_layout(height=chart_height, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(t=60, b=40, l=40, r=40), plot_bgcolor="white", paper_bgcolor="white")
    for i in range(1, num_charts+1):
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="#EEE", zeroline=False, row=i, col=1)
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="#EEE", row=i, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # WEEKEND DEEP-DIVE
    # ============================================================
    with st.expander("🔬 Weekend Deep-Dive — Cross-Asset Snapshot & Regime Analysis"):
        st.markdown("### 📋 Cross-Asset Snapshot")
        snap = []
        deep_specs = [
            ("VIX", "VIX", vix_val), ("HY_bps", "HY Spread", hy_val), ("T10Y3M", "10Y-3M", yc_val),
            ("DXY", "DXY", dxy_val), ("NFCI", "NFCI", nfci_val), ("SPY_vs_TLT", "SPY/TLT", spy_val),
            ("USDJPY", "USD/JPY", usdjpy_val), ("Copper_vs_Gold", "Cu/Au", cg_val),
            ("AUDUSD", "AUD/USD", aud_val),
        ]
        for key, label, val in [("EEM_vs_SPY", "EEM/SPY", eem_val), ("HYG_vs_LQD", "HYG/LQD", hyg_val), ("XLY_vs_XLP", "XLY/XLP", xly_val), ("DGS2", "2Y Yield", dgs2_val), ("Gold_vs_Oil", "Gold/Oil", go_val), ("WALCL_T", "Fed Balance Sheet", walcl_val)]:
            if val is not None:
                deep_specs.append((key, label, val))

        for key, label, val in deep_specs:
            if val is None:
                continue
            cfg = INDICATOR_CONFIG.get(key, {})
            w = one_week_ago[key] if key in one_week_ago.index else None
            m = one_month_ago[key] if key in one_month_ago.index else None
            pct_keys_snap = ["SPY_vs_TLT", "Copper_vs_Gold", "AUDUSD", "USDJPY", "EEM_vs_SPY", "HYG_vs_LQD", "XLY_vs_XLP", "VIX", "DXY", "Gold_vs_Oil"]
            ch_w_s = f"{(val/w - 1)*100:+.1f}%" if (val is not None and w is not None and pd.notna(val) and pd.notna(w) and key in pct_keys_snap) else (f"{val - w:+.{cfg.get('decimals', 1)}f}{cfg.get('unit', '')}" if (val is not None and w is not None and pd.notna(val) and pd.notna(w)) else "N/A")
            ch_m_s = f"{(val/m - 1)*100:+.1f}%" if (val is not None and m is not None and pd.notna(val) and pd.notna(m) and key in pct_keys_snap) else (f"{val - m:+.{cfg.get('decimals', 1)}f}{cfg.get('unit', '')}" if (val is not None and m is not None and pd.notna(val) and pd.notna(m)) else "N/A")
            disp = f"{val:.{cfg.get('decimals', 1)}f}{cfg.get('unit', '')}"
            snap.append({"Indicator": label, "Current": disp, "1-Week": ch_w_s, "1-Month": ch_m_s})
        st.dataframe(pd.DataFrame(snap), use_container_width=True, hide_index=True)

        st.markdown("### 📝 Macro Regime Description")
        if regime_label == "RISK-ON":
            st.markdown("**Risk-on regime.** Markets are pricing in growth, credit is calm, and investors are favouring equities over bonds.")
        elif regime_label == "RISK-OFF":
            st.markdown("**Risk-off regime.** Multiple stress signals are active. Defensive positioning is dominant.")
        else:
            st.markdown("**Neutral/mixed regime.** Indicators are split. No dominant direction — typical of transitional periods.")

        if yc_val is not None and yc_val < 0:
            st.markdown("The **10Y-3M yield curve is inverted**, historically one of the most reliable recession indicators, though timing varies significantly (6-24 months).")
        if re_steepening:
            st.markdown("⚠️ **Curve re-steepening from inversion detected.** Historically, this has often coincided with recession arrival or financial stress events.")
        if stress_percentile > 70:
            st.markdown(f"**Elevated stress ({stress_percentile:.0f}th percentile).** Conditions notably tighter than 10-year average.")
        elif stress_percentile < 30:
            st.markdown(f"**Low stress ({stress_percentile:.0f}th percentile).** Conditions notably calmer than 10-year average.")

        st.markdown("### ⏱ Data Frequency Notes")
        st.markdown("- **Daily:** VIX, HY Spread, 10Y-3M, DXY, SPY/TLT, USD/JPY, AUD/USD, Copper/Gold, EEM/SPY, HYG/LQD, XLY/XLP, 2Y Yield, Gold/Oil")
        st.markdown("- **Weekly (Wed):** NFCI — updated once per week")
        st.markdown("- **Weekly (Thu):** Fed Balance Sheet — updated once per week")
        st.markdown("Weekly indicators may appear stale between updates. This is expected behaviour.")

# ============================================================
# EXPLAINERS + LEGEND
# ============================================================
with st.expander("📖 Gauge Explanations & Threshold Legend"):
    st.markdown("### What does each gauge measure?")
    for key, cfg in INDICATOR_CONFIG.items():
        if cfg.get("question"):
            st.markdown(f"**{key}** — {cfg['question']}")
    st.markdown("---")
    st.markdown("### Fixed Reference Levels")
    for key, cfg in INDICATOR_CONFIG.items():
        if cfg["fixed_refs"]:
            refs = ", ".join([f"{v} ({l})" for v, l in zip(cfg["fixed_refs"], cfg["ref_labels"])])
            st.markdown(f"- **{key}:** {refs}")
    st.markdown("---")
    st.markdown("### Regime Vote Logic")
    st.markdown("6 core indicators vote. Ties break: Risk-Off > Neutral > Risk-On (conservative).")
    st.markdown("---")
    st.markdown("### Philosophy")
    st.markdown("**This dashboard identifies macro regimes, not short-term market moves.** It answers: *what kind of macro environment are we in right now?*")

# ============================================================
# RAW DATA
# ============================================================
with st.expander("🔍 View raw data (last 30 days)"):
    raw_cols = [k for k in INDICATOR_CONFIG.keys() if k in df.columns]
    st.dataframe(df[raw_cols].tail(30).sort_index(ascending=False).round(3), use_container_width=True)
