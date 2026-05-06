"""
Oil Volatility Forecaster — Streamlit Dashboard (V1, 15 features)

Live 4h oil volatility predictions using MAML-adapted MLP.
Pulls GDELT data every 15 minutes, makes a prediction, then after 4 hours
computes actual realized vol and plots actual vs predicted.

Run from the live_deployment/ folder:
  streamlit run app_v1.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from datetime import datetime, timedelta
import joblib
import requests
import urllib3
import yfinance as yf
import sqlite3
import json

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Page config
st.set_page_config(
    page_title="Oil Vol Forecaster",
    page_icon="O",
    layout="wide"
)

DB_PATH = "predictions.db"

# Model definition
class OilVolatilityMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 48), nn.BatchNorm1d(48), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(48, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


FEATURE_COLS = [
    'ovx_close', 'vix_close', 'oil_vol_5d', 'oil_vol_20d',
    'oil_close', 'dxy_close', 'gold_oil_ratio',
    'gs_mean', 'gs_std', 'gs_conflict_pct', 'gs_weighted',
    'tone_mean', 'tone_std', 'n_events', 'mentions_sum',
]
INNER_LR    = 0.01
INNER_STEPS = 5


# Load models
@st.cache_resource
def load_models():
    scaler = joblib.load('../../models/v1/feature_scaler.pkl')
    maml_model = OilVolatilityMLP()
    maml_model.load_state_dict(torch.load('../../models/v1/maml_trained.pth', map_location='cpu'))
    maml_model.eval()
    plain_mlp = OilVolatilityMLP()
    plain_mlp.load_state_dict(torch.load('../../models/v1/mlp_pretrained.pth', map_location='cpu'))
    plain_mlp.eval()
    return scaler, maml_model, plain_mlp

scaler, maml_model, plain_mlp = load_models()


# MAML adaptation
def adapt(model, sup_X, sup_y):
    adapted   = deepcopy(model)
    loss_fn   = nn.HuberLoss(delta=0.5)
    optimizer = torch.optim.SGD(adapted.parameters(), lr=INNER_LR)
    adapted.eval()
    for _ in range(INNER_STEPS):
        optimizer.zero_grad()
        loss = loss_fn(adapted(sup_X), sup_y)
        loss.backward()
        optimizer.step()
    adapted.eval()
    return adapted


def is_ovx_calculating() -> bool:
    """
    OVX calculates in two sessions:
    Session 1: 08:00 - 02:15 UTC (next day)
    Session 2: 02:30 - 21:15 UTC
    15-min break at 02:15-02:30 UTC only.
    Closed weekends.
    """
    now_utc = datetime.utcnow()
    if now_utc.weekday() >= 5:
        return False
    minutes = now_utc.hour * 60 + now_utc.minute
    # Only gap is 02:15-02:30 UTC
    if 135 <= minutes <= 150:
        return False
    return True  # active all other times on weekdays


def get_live_ovx():
    """Get current OVX spot price, not historical close."""
    try:
        ticker = yf.Ticker('^OVX')
        info   = ticker.fast_info
        ovx    = info.last_price
        if ovx and ovx > 0:
            return float(ovx)
    except Exception as e:
        print(f"OVX spot fetch failed: {e}")
    return None


# Fetch market data
@st.cache_data(ttl=300)
def fetch_market_data(days_back=30):
    end   = datetime.utcnow()
    start = end - timedelta(days=days_back)
    tickers = {
        'oil_close'  : 'BZ=F',
        'vix_close'  : '^VIX',
        'ovx_close'  : '^OVX',
        'dxy_close'  : 'DX-Y.NYB',
        'gold_close' : 'GC=F',
    }
    data = {}
    for col, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=start, end=end,
                             progress=False, auto_adjust=True,
                             interval='1h', timeout=10)
            if len(df) > 0:
                s = df['Close'].squeeze()
                s = s[s > 0]
                if len(s) > 0:
                    data[col] = s
        except Exception:
            pass
    if not data:
        return pd.DataFrame()
    market = pd.DataFrame(data).ffill().bfill()
    market.index = pd.to_datetime(market.index)
    if 'oil_close' in market.columns:
        log_ret = np.log(market['oil_close'] / market['oil_close'].shift(1))
        # With hourly data: 5 days = ~120 trading hours, 20 days = ~480
        market['oil_vol_5d']  = log_ret.rolling(120).std() * np.sqrt(252 * 24)
        market['oil_vol_20d'] = log_ret.rolling(480).std() * np.sqrt(252 * 24)
    if 'gold_close' in market.columns and 'oil_close' in market.columns:
        market['gold_oil_ratio'] = market['gold_close'] / market['oil_close']
    market = market.drop(columns=['gold_close'], errors='ignore')

    # Override OVX with live spot price
    live_ovx = get_live_ovx()
    if live_ovx and not market.empty:
        market.loc[market.index[-1], 'ovx_close'] = live_ovx
        print(f"  Live OVX spot: {live_ovx:.2f}")

    return market.dropna()


# Fetch GDELT events
# Constants below are exact copies from the training pipeline so the
# live filtering matches what the model was trained on.

HIGH_IMPACT_COUNTRIES = frozenset({
    "US", "USA", "CH", "RS", "GM", "UK", "JA", "FR", "IR", "KN",
    "IS", "SA", "UP", "TW", "IN", "BR", "TU", "KS", "MX",
})

MIDDLE_EAST_FIPS = frozenset({
    "SA", "IR", "IZ", "KU", "IS", "SY", "LE", "JO",
    "YM", "AE", "QA", "BN", "OM",
})
OIL_PRODUCER_FIPS = frozenset({
    "RS", "VE", "NI", "LY", "AG", "AO", "NO", "EC", "BR", "CA", "KZ",
})
OIL_CONSUMER_FIPS = frozenset({
    "CH", "JA", "IN", "KS", "GM", "FR", "UK", "IT", "SP", "TW",
})
CHOKEPOINT_FIPS = frozenset({"TU", "EG", "SU", "SO", "DJ"})

CAMEO_TASK_MAP = {
    "01": "cooperation_diplomacy", "02": "cooperation_diplomacy",
    "03": "cooperation_diplomacy", "04": "cooperation_diplomacy",
    "05": "cooperation_diplomacy", "06": "cooperation_diplomacy",
    "07": "cooperation_diplomacy", "08": "cooperation_diplomacy",
    "09": "cooperation_diplomacy",
    "061": "sanctions_trade", "163": "sanctions_trade", "164": "sanctions_trade",
    "10": "policy_statement",  "11": "policy_statement",
    "12": "policy_statement",  "13": "policy_statement",
    "14": "political_instability",
    "15": "coercion",
    "16": "diplomatic_tension", "17": "diplomatic_tension",
    "18": "military_conflict",  "19": "military_conflict",
    "20": "military_conflict",
}

COOP_GOLDSTEIN_MIN = 7.0
MIN_MENTIONS       = 10


def assign_oil_region_fips(c: str) -> str:
    if not c or c in ("nan", "UNKNOWN", ""):
        return "other"
    c = str(c).strip().upper()
    if c in ("US", "USA"):
        return "US"
    if c in MIDDLE_EAST_FIPS or c in CHOKEPOINT_FIPS:
        return "middle_east"
    if c in OIL_PRODUCER_FIPS:
        return "oil_producer"
    if c in OIL_CONSUMER_FIPS:
        return "oil_consumer"
    return "other"


def get_task_category(event_code: str) -> str:
    ec = str(event_code).strip()
    for length in (4, 3, 2):
        cat = CAMEO_TASK_MAP.get(ec[:length])
        if cat:
            return cat
    return "other_political"


@st.cache_data(ttl=900)
def fetch_gdelt_events_cached(hours_back=24):
    import zipfile
    import io

    resp = requests.get(
        "http://data.gdeltproject.org/gdeltv2/lastupdate.txt",
        timeout=10, verify=False
    )
    export_url = resp.text.strip().split('\n')[0].split(' ')[2].strip()

    r2 = requests.get(export_url, timeout=30, verify=False)
    with zipfile.ZipFile(io.BytesIO(r2.content)) as z:
        with z.open(z.namelist()[0]) as f:
            raw = pd.read_csv(f, sep='\t', header=None,
                              on_bad_lines='skip', low_memory=False)

    df = pd.DataFrame({
        'actor1_country'    : raw[7].astype(str).str.strip().str.upper(),
        'actor2_country'    : raw[17].astype(str).str.strip().str.upper(),
        'action_country'    : raw[53].astype(str).str.strip().str.upper().replace({'USA': 'US'}),
        'event_code'        : raw[26].astype(str).str.strip(),
        'event_base_code'   : raw[27].astype(str).str.strip(),
        'goldstein'         : pd.to_numeric(raw[30], errors='coerce'),
        'mentions'          : pd.to_numeric(raw[31], errors='coerce').fillna(0),
        'tone'              : pd.to_numeric(raw[34], errors='coerce'),
    }).dropna(subset=['goldstein'])

    # Filter 1: min_mentions >= 10
    df = df[df['mentions'] >= MIN_MENTIONS]

    # Filter 2: HIGH_IMPACT_COUNTRIES
    mask_us = (
        df['actor1_country'].isin(HIGH_IMPACT_COUNTRIES) |
        df['actor2_country'].isin(HIGH_IMPACT_COUNTRIES) |
        df['action_country'].isin(HIGH_IMPACT_COUNTRIES)
    )
    df = df[mask_us]

    # Filter 3: CAMEO codes present in training
    df['task_category'] = df['event_code'].apply(get_task_category)
    fallback_mask = df['task_category'] == 'other_political'
    df.loc[fallback_mask, 'task_category'] = df.loc[
        fallback_mask, 'event_base_code'
    ].apply(get_task_category)
    df = df[df['task_category'] != 'other_political']

    # Filter 4: Goldstein >= 7.0 for cooperation_diplomacy
    coop_mask = df['task_category'] == 'cooperation_diplomacy'
    gold_mask = df['goldstein'].abs() >= COOP_GOLDSTEIN_MIN
    df = df[~coop_mask | gold_mask]

    if df.empty:
        raise ValueError("No relevant events after filtering")

    # Assign oil_region using FIPS-2 primary_country logic
    df['primary_country'] = (
        df['action_country'].replace({'nan': np.nan, 'UNKNOWN': np.nan})
        .fillna(df['actor1_country'].replace({'nan': np.nan}))
        .fillna(df['actor2_country'].replace({'nan': np.nan}))
        .fillna('UNKNOWN')
    )
    df['oil_region'] = df['primary_country'].apply(assign_oil_region_fips)

    gs_vals  = df['goldstein'].values
    men_vals = df['mentions'].values
    tone_vals = df['tone'].dropna().values

    return {
        'gs_mean'         : float(gs_vals.mean()),
        'gs_std'          : float(gs_vals.std()),
        'gs_conflict_pct' : float((gs_vals < 0).mean()),
        'gs_weighted'     : float(np.average(gs_vals, weights=men_vals.clip(min=1))),
        'tone_mean'       : float(tone_vals.mean()) if len(tone_vals) else -1.0,
        'tone_std'        : float(tone_vals.std())  if len(tone_vals) else 3.0,
        'n_events'        : float(len(gs_vals)),
        'mentions_sum'    : float(men_vals.sum()),
        '_source'         : 'gdelt_filtered',
        '_n_articles'     : len(gs_vals),
        '_n_domains'      : 0,
        '_region_counts'  : df['oil_region'].value_counts().to_dict(),
        '_category_counts': df['task_category'].value_counts().to_dict(),
    }


GDELT_FALLBACK = {
    'gs_mean': 0.0, 'gs_std': 2.5, 'gs_conflict_pct': 0.45,
    'gs_weighted': 0.0, 'tone_mean': -1.0, 'tone_std': 3.0,
    'n_events': 24.0, 'mentions_sum': 250.0,
    '_source': 'fallback', '_n_articles': 0, '_n_domains': 0,
}


def fetch_gdelt_events(hours_back=24):
    try:
        result = fetch_gdelt_events_cached(hours_back)
        if 'gdelt_last_failure' in st.session_state:
            st.session_state.gdelt_last_failure = 0
        return result
    except Exception as e:
        print(f"GDELT raw fetch error: {type(e).__name__}: {e}")
        return GDELT_FALLBACK


# Build feature vector
def build_feature_vector(market_row, gdelt):
    features = {
        'ovx_close'      : market_row.get('ovx_close',      np.nan),
        'vix_close'      : market_row.get('vix_close',      np.nan),
        'oil_vol_5d'     : market_row.get('oil_vol_5d',     np.nan),
        'oil_vol_20d'    : market_row.get('oil_vol_20d',    np.nan),
        'oil_close'      : market_row.get('oil_close',      np.nan),
        'dxy_close'      : market_row.get('dxy_close',      np.nan),
        'gold_oil_ratio' : market_row.get('gold_oil_ratio', np.nan),
        'gs_mean'        : gdelt['gs_mean'],
        'gs_std'         : gdelt['gs_std'],
        'gs_conflict_pct': gdelt['gs_conflict_pct'],
        'gs_weighted'    : gdelt['gs_weighted'],
        'tone_mean'      : gdelt['tone_mean'],
        'tone_std'       : gdelt['tone_std'],
        'n_events'       : gdelt['n_events'],
        'mentions_sum'   : gdelt['mentions_sum'],
    }
    return np.array([[features[c] for c in FEATURE_COLS]], dtype=np.float32)


# Compute realized 4h vol.
# IMPORTANT: store all timestamps as UTC. datetime.now() returns LOCAL time;
# yfinance returns UTC. Mixing them creates a timezone offset where no bars match.
def compute_realized_4h_vol(market_df, pred_utc):
    pred_utc = pd.to_datetime(pred_utc)   # naive UTC
    now_utc  = datetime.utcnow()          # naive UTC

    if now_utc < pred_utc + timedelta(hours=4):
        return None

    try:
        # Floor to nearest hour so we don't miss the bar at :00
        pred_hour = pred_utc.replace(minute=0, second=0, microsecond=0)

        start = (pred_hour - timedelta(days=1)).strftime('%Y-%m-%d')
        end   = (pred_hour + timedelta(days=2)).strftime('%Y-%m-%d')

        h = yf.download('CL=F', start=start, end=end,
                        interval='1h', progress=False, auto_adjust=True)
        if h.empty or len(h) < 4:
            return None

        if isinstance(h.columns, pd.MultiIndex):
            h.columns = [c[0] for c in h.columns]

        h.index = pd.to_datetime(h.index)
        if h.index.tz is not None:
            h.index = h.index.tz_localize(None)
        h = h.sort_index()

        mask = (h.index >= pred_hour) & (h.index <= pred_hour + timedelta(hours=4))
        h4   = h[mask]

        if len(h4) < 2:
            return None

        returns = np.log(h4['Close'] / h4['Close'].shift(1)).dropna()
        return float(returns.std(ddof=1) * np.sqrt(252 * 23))

    except Exception as e:
        print(f"compute_realized_4h_vol error: {e}")
        return None


# Database (SQLite)
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL,
            maml_pred     REAL,
            mlp_pred      REAL,
            oil_close     REAL,
            ovx_close     REAL,
            actual_rvol   REAL,
            n_support     INTEGER,
            gdelt_source  TEXT,
            features_json TEXT,
            gdelt_context TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def load_log():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY timestamp DESC", conn,
        parse_dates=['timestamp']
    )
    conn.close()
    # Rename to match what the UI expects
    if 'actual_rvol' in df.columns:
        df = df.rename(columns={'actual_rvol': 'actual_rvol_4h'})
    # Add missing columns from older DB versions
    if 'gdelt_context' not in df.columns:
        df['gdelt_context'] = ''
    return df

def append_prediction(timestamp, maml_pred, mlp_pred, oil_close, ovx_close,
                      n_support=0, gdelt_source='', features=None, gdelt_context=''):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO predictions
        (timestamp, maml_pred, mlp_pred, oil_close, ovx_close,
         n_support, gdelt_source, features_json, gdelt_context)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (str(timestamp), maml_pred, mlp_pred, oil_close,
          ovx_close, n_support, gdelt_source,
          json.dumps(features) if features else '',
          gdelt_context))
    conn.commit()
    conn.close()


# Live support set from resolved predictions
def get_live_support_set():
    """Loads resolved predictions as MAML support set. Seeds from data/seed_v1.csv until enough live examples."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT features_json, actual_rvol FROM predictions
        WHERE actual_rvol IS NOT NULL
        ORDER BY timestamp DESC LIMIT 15
    """).fetchall()
    conn.close()

    n_live = len(rows)

    # Seed with March 17-24 high-OVX data until enough live examples
    if n_live < 15:
        try:
            seed_df   = pd.read_csv('../../data/seed_v1.csv')
            seed_df   = seed_df.dropna(subset=['oil_fwd_rvol_4h'])
            n_seed    = min(15 - n_live, len(seed_df))
            seed_df   = seed_df.sample(n_seed, random_state=42)

            seed_features = seed_df[FEATURE_COLS].values.astype(np.float32)
            seed_actuals  = seed_df['oil_fwd_rvol_4h'].values.astype(np.float32)

            if n_live > 0:
                live_features = np.array([
                    list(json.loads(r[0]).values()) for r in rows
                ], dtype=np.float32)
                live_actuals  = np.array([r[1] for r in rows], dtype=np.float32)
                all_features  = np.vstack([live_features, seed_features])
                all_actuals   = np.concatenate([live_actuals, seed_actuals])
            else:
                all_features = seed_features
                all_actuals  = seed_actuals

            sup_X = torch.tensor(scaler.transform(all_features), dtype=torch.float32)
            sup_y = torch.tensor(np.log1p(all_actuals), dtype=torch.float32).unsqueeze(1)
            return (sup_X, sup_y), n_live, n_seed

        except Exception as e:
            print(f"Seed data load failed: {e}")
            if n_live < 3:
                return None, 0, 0

    # Normal path once enough live examples exist
    features = np.array([
        list(json.loads(r[0]).values()) for r in rows
    ], dtype=np.float32)
    actuals  = np.array([r[1] for r in rows], dtype=np.float32)
    sup_X = torch.tensor(scaler.transform(features), dtype=torch.float32)
    sup_y = torch.tensor(np.log1p(actuals), dtype=torch.float32).unsqueeze(1)
    return (sup_X, sup_y), n_live, 0


def predict_adapted(maml_model, X_tensor, support=None, inner_steps=10):
    """
    If support exists, adapt to it first.
    Otherwise use base model directly.
    """
    if support is None:
        maml_model.eval()
        with torch.no_grad():
            return float(np.expm1(maml_model(X_tensor).item()))

    sup_X, sup_y = support
    adapted   = deepcopy(maml_model)
    loss_fn   = nn.HuberLoss(delta=0.5)
    optimizer = torch.optim.SGD(adapted.parameters(), lr=0.005)
    adapted.train()

    for _ in range(inner_steps):
        optimizer.zero_grad()
        loss = loss_fn(adapted(sup_X), sup_y)
        loss.backward()
        optimizer.step()

    adapted.eval()
    with torch.no_grad():
        return float(np.expm1(adapted(X_tensor).item()))

def update_actuals(market_df):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT id, timestamp FROM predictions
        WHERE actual_rvol IS NULL
    """).fetchall()
    now_utc = datetime.utcnow()
    for row_id, ts in rows:
        pred_utc = pd.to_datetime(str(ts))
        if now_utc >= pred_utc + timedelta(hours=4):
            actual = compute_realized_4h_vol(market_df, pred_utc)
            print(f"Actual for {pred_utc} UTC: {actual}")
            if actual is not None:
                conn.execute(
                    "UPDATE predictions SET actual_rvol=? WHERE id=?",
                    (actual, row_id)
                )
    conn.commit()
    conn.close()
    return load_log()


# Streamlit UI
st.title("Oil Volatility Forecaster")
st.caption("MAML-adapted MLP · GDELT geopolitical events + market data · 4h forward vol")

st.markdown('<meta http-equiv="refresh" content="900">', unsafe_allow_html=True)

with st.expander("About this dashboard", expanded=True):
    st.markdown("""
    This dashboard uses a **Model-Agnostic Meta-Learning (MAML)** model trained on
    GDELT geopolitical event data and Brent oil market features to forecast
    **4-hour forward oil realized volatility**.

    **How predictions are made:**
    The model fetches live oil-related news from GDELT and current market data
    (Brent crude price, OVX, VIX, DXY, Gold/Oil ratio, rolling volatility).
    The MAML model adapts to recent market conditions using the last 5 trading
    days as a support set, then outputs a predicted annualized 4h volatility.

    **How actuals are computed:**
    After 4 hours have elapsed since a prediction was logged, the dashboard
    fetches hourly WTI crude (CL=F) prices and computes the standard deviation of
    returns over that 4-hour window, annualized with √(252 × 23).

    **Label convention used throughout this dashboard:**
    - **(Predicted)** — values output by the model before the fact
    - **(Actual)** — values computed from real market data after the fact

    Click **Make Prediction Now** in the sidebar to log a prediction and
    begin tracking. Actuals will appear automatically after 4 hours.
    """)
st.markdown("---")

with st.sidebar:
    st.header("Settings")
    gdelt_hours   = st.slider("GDELT lookback (hours)", 1, 72, 24)
    make_pred_btn = st.button("Make Prediction Now", type="primary")
    clear_log_btn = st.button("Clear Log")
    if clear_log_btn:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()
        st.success("Log cleared.")
    st.markdown("---")
    st.markdown("**Files loaded:**")
    st.markdown("- `../../models/v1/maml_trained.pth`")
    st.markdown("- `../../models/v1/mlp_pretrained.pth`")
    st.markdown("- `../../models/v1/feature_scaler.pkl`")
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown(
        "1. Fetches GDELT oil events\n"
        "2. Fetches live market data\n"
        "3. MAML adapts on last 5 days\n"
        "4. Predicts 4h forward vol\n"
        "5. After 4h, computes actual vol\n"
        "6. Plots actual vs predicted"
    )

col1, col2 = st.columns(2)

with col1:
    st.subheader("Market Data")
    with st.spinner("Fetching from Yahoo Finance..."):
        market = fetch_market_data(days_back=60)
    if market.empty:
        st.error("Could not fetch market data.")
        st.stop()
    latest = market.iloc[-1].to_dict()
    m1, m2, m3 = st.columns(3)
    m1.metric("Brent Crude (Actual)", f"${latest.get('oil_close', 0):.2f}")
    m2.metric("OVX (Actual)", f"{latest.get('ovx_close', 0):.2f}")
    m3.metric("VIX (Actual)", f"{latest.get('vix_close', 0):.2f}")
    if not is_ovx_calculating():
        st.warning(
            "OVX is not currently updating (outside 08:00-21:15 UTC). "
            "Displayed value is last known. Oil price, DXY and gold remain live."
        )
    st.caption("All market values are actual live data from Yahoo Finance")
    st.dataframe(market.tail(5).round(4), use_container_width=True)

with col2:
    st.subheader("GDELT Events")
    fetch_gdelt_btn = st.button("Fetch GDELT Data")

    if 'gdelt_data' not in st.session_state:
        st.session_state.gdelt_data = None

    if fetch_gdelt_btn or st.session_state.gdelt_data is None:
        with st.spinner("Fetching from GDELT..."):
            st.session_state.gdelt_data = fetch_gdelt_events(hours_back=gdelt_hours)

    gdelt = st.session_state.gdelt_data
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Events (Actual)",    f"{int(gdelt['n_events'])}")
    g2.metric("Sources (Actual)",   f"{int(gdelt['mentions_sum'] / max(gdelt['n_events'],1)):.0f}")
    g3.metric("Avg Tone (Actual)",  f"{gdelt['tone_mean']:.2f}")
    g4.metric("Conflict% (Actual)", f"{gdelt['gs_conflict_pct']*100:.0f}%")
    if gdelt.get('_source') == 'gdelt_filtered':
        st.success(
            f"Live GDELT data -- {gdelt['_n_articles']} events after pipeline filters"
        )
    elif gdelt.get('_source') == 'fallback':
        st.warning(
            "GDELT fetch failed -- using neutral fallback values. "
            "Predictions will rely more heavily on market features."
        )

st.markdown("---")
st.subheader("Volatility Forecast")

raw_X = build_feature_vector(latest, gdelt)

if np.isnan(raw_X).any():
    missing = [FEATURE_COLS[i] for i in range(15) if np.isnan(raw_X[0][i])]
    st.error(f"Missing features: {missing}")
    st.stop()

scaled_X = scaler.transform(raw_X)
X_tensor = torch.tensor(scaled_X, dtype=torch.float32)

plain_mlp.eval()
with torch.no_grad():
    mlp_pred_log = plain_mlp(X_tensor).item()
    mlp_pred     = float(np.expm1(mlp_pred_log))

# Get live support from resolved predictions
support, n_live, n_seed = get_live_support_set()
n_support = n_live + n_seed

# Predict with adaptation if support exists
maml_pred = predict_adapted(
    maml_model, X_tensor,
    support=support,
    inner_steps=10
)

print(f"Predicted using {n_support} resolved examples as support")

st.markdown("Values below are **model predictions** -- not yet verified against actuals")
p1, p2, p3, p4 = st.columns(4)
p1.metric("Plain MLP (Predicted 4h vol)", f"{mlp_pred:.4f}")
p2.metric("MAML (Predicted 4h vol)",
          f"{maml_pred:.4f}",
          delta=f"{maml_pred - mlp_pred:+.4f} vs MLP")
vol_label = "LOW" if maml_pred < 0.15 else "MODERATE" if maml_pred < 0.30 else "HIGH"
p3.metric("Vol Regime (Predicted)", vol_label)
if n_support > 0:
    p4.metric(
        "Live adaptation",
        f"{n_support} examples",
        delta=f"{n_live} live + {n_seed} seeded"
    )
else:
    p4.metric("Live adaptation", "Base model (no data yet)")
st.caption("MAML adapts on resolved predictions as support set. "
           "Actual vol will be computed from hourly WTI prices after 4 hours.")

# Store timestamp as UTC — this is the only correct way
now_utc = datetime.utcnow()
if make_pred_btn:
    features_dict = {
        col: float(raw_X[0][i])
        for i, col in enumerate(FEATURE_COLS)
    }
    top_category = max(
        gdelt.get('_category_counts', {'unknown': 1}),
        key=gdelt.get('_category_counts', {'unknown': 1}).get
    )
    top_region = max(
        gdelt.get('_region_counts', {'unknown': 1}),
        key=gdelt.get('_region_counts', {'unknown': 1}).get
    )
    gdelt_context = (
        f"{top_category} | {top_region} | "
        f"conflict={gdelt['gs_conflict_pct']*100:.0f}% | "
        f"tone={gdelt['tone_mean']:.1f}"
    )
    append_prediction(
        timestamp     = now_utc,
        maml_pred     = maml_pred,
        mlp_pred      = mlp_pred,
        oil_close     = latest.get('oil_close', np.nan),
        ovx_close     = latest.get('ovx_close', np.nan),
        n_support     = n_support,
        gdelt_source  = gdelt.get('_source', ''),
        features      = features_dict,
        gdelt_context = gdelt_context,
    )
    st.success(f"Prediction logged at {now_utc.strftime('%H:%M:%S')} UTC. "
               f"Actual vol will be computed in 4 hours.")

st.markdown("---")
st.subheader("Actual vs Predicted (4h Realized Vol)")

with st.spinner("Checking for resolved predictions..."):
    log = update_actuals(market)

if log.empty:
    st.info("No predictions logged yet. Click 'Make Prediction Now' to start tracking.")
else:
    resolved = log.dropna(subset=['actual_rvol_4h'])
    pending  = log[log['actual_rvol_4h'].isna()]

    if not resolved.empty:
        plot_df = resolved[['timestamp', 'maml_pred', 'mlp_pred', 'actual_rvol_4h']].copy()
        plot_df = plot_df.set_index('timestamp').sort_index()
        plot_df.columns = ['MAML (Predicted)', 'MLP (Predicted)', 'Realized Vol (Actual)']
        st.markdown("**Predicted** = model forecast at time of logging  |  "
                    "**Actual** = realized vol computed 4h later from hourly WTI prices")
        st.line_chart(plot_df, use_container_width=True)

        mae_maml = float(np.mean(np.abs(resolved['actual_rvol_4h'] - resolved['maml_pred'])))
        mae_mlp  = float(np.mean(np.abs(resolved['actual_rvol_4h'] - resolved['mlp_pred'])))
        r1, r2, r3 = st.columns(3)
        r1.metric("Resolved predictions", len(resolved))
        r2.metric("MAML MAE (Predicted vs Actual)", f"{mae_maml:.4f}")
        r3.metric("MLP MAE (Predicted vs Actual)",  f"{mae_mlp:.4f}")

        if len(resolved) >= 5:
            rs = resolved.sort_values('timestamp')
            rs['maml_rolling_mae'] = (
                (rs['actual_rvol_4h'] - rs['maml_pred'])
                .abs().rolling(5, min_periods=1).mean()
            )
            rs['mlp_rolling_mae'] = (
                (rs['actual_rvol_4h'] - rs['mlp_pred'])
                .abs().rolling(5, min_periods=1).mean()
            )
            st.subheader("Adaptation Convergence (Rolling 5-prediction MAE)")
            st.caption("MAML MAE should trend downward as live support set grows")
            roll_df = rs[['timestamp', 'maml_rolling_mae', 'mlp_rolling_mae']].set_index('timestamp')
            roll_df.columns = ['MAML Rolling MAE', 'MLP Rolling MAE']
            st.line_chart(roll_df, use_container_width=True)

        if len(resolved) >= 3:
            threshold = resolved['actual_rvol_4h'].median()
            resolved['actual_regime'] = (resolved['actual_rvol_4h'] > threshold).astype(int)
            resolved['maml_regime']   = (resolved['maml_pred']       > threshold).astype(int)
            resolved['mlp_regime']    = (resolved['mlp_pred']        > threshold).astype(int)
            maml_acc = (resolved['actual_regime'] == resolved['maml_regime']).mean()
            mlp_acc  = (resolved['actual_regime'] == resolved['mlp_regime']).mean()
            d1, d2 = st.columns(2)
            d1.metric("MAML Regime Accuracy", f"{maml_acc*100:.1f}%")
            d2.metric("MLP Regime Accuracy",  f"{mlp_acc*100:.1f}%")

        display_cols = ['timestamp', 'maml_pred', 'mlp_pred',
                        'actual_rvol_4h', 'oil_close', 'ovx_close']
        col_names = ['Timestamp (UTC)', 'MAML (Predicted)', 'MLP (Predicted)',
                      'Realized Vol (Actual)', 'Oil Close', 'OVX']
        if 'gdelt_context' in resolved.columns:
            display_cols.append('gdelt_context')
            col_names.append('GDELT Context')
        display_df = resolved[display_cols].copy()
        display_df.columns = col_names
        st.dataframe(display_df.sort_values('Timestamp (UTC)', ascending=False).round(5),
                     use_container_width=True)
    else:
        st.info("Predictions made but 4 hours haven't passed yet. Check back later.")

    if not pending.empty:
        st.markdown(f"**⏳ {len(pending)} prediction(s) pending actuals** — check back in 4 hours")
        pending_display = pending[['timestamp', 'maml_pred', 'mlp_pred',
                                    'oil_close', 'ovx_close']].copy()
        pending_display.columns = ['Timestamp (UTC)', 'MAML (Predicted)', 'MLP (Predicted)',
                                    'Oil Close (Actual at prediction time)',
                                    'OVX (Actual at prediction time)']
        pending_display['Time until actual'] = pending_display['Timestamp (UTC)'].apply(
            lambda t: f"{max(0, int((pd.to_datetime(t) + timedelta(hours=4) - datetime.utcnow()).total_seconds() / 60))} min remaining"
        )
        st.dataframe(pending_display.sort_values('Timestamp (UTC)', ascending=False).round(5),
                     use_container_width=True)

st.markdown("---")
st.subheader("Market Charts")
tab1, tab2, tab3 = st.tabs(["Oil Price (Actual)", "Realized Vol (Actual)", "VIX / OVX (Actual)"])
with tab1:
    if 'oil_close' in market.columns:
        st.line_chart(market['oil_close'], use_container_width=True)
with tab2:
    vol_cols = [c for c in ['oil_vol_5d', 'oil_vol_20d'] if c in market.columns]
    if vol_cols:
        st.line_chart(market[vol_cols], use_container_width=True)
with tab3:
    ind_cols = [c for c in ['vix_close', 'ovx_close'] if c in market.columns]
    if ind_cols:
        st.line_chart(market[ind_cols], use_container_width=True)

with st.expander("Feature Vector (debug)"):
    st.dataframe(pd.DataFrame({
        'Feature'      : FEATURE_COLS,
        'Raw Value'    : raw_X[0],
        'Scaled Value' : scaled_X[0],
    }), use_container_width=True)

st.markdown("---")
st.caption(
    f"Last refresh: {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC | "
    f"Auto-refresh: every 15 min | "
    f"Data: Yahoo Finance + GDELT 2.0"
)