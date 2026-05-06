"""
Scheduler — V1 (15 features)
Background prediction loop. Triggers a prediction every 1 hour during OVX hours,
checks for resolved actuals every 5 min. Writes to predictions.db.

Run from the live_deployment/ folder:
  python scheduler_v1.py
"""

import time
import sqlite3
import json
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import requests
import urllib3
import yfinance as yf
from copy import deepcopy
from datetime import datetime, timedelta

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Logging setup: rotating file (10 MB × 5) + console mirror.
# `*.log` files are .gitignored. Reset any prior handlers so re-imports
# (e.g. under pytest) don't stack duplicate output.
logger = logging.getLogger("scheduler_v1")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _fh = RotatingFileHandler("scheduler_v1.log", maxBytes=10_000_000, backupCount=5)
    _fh.setFormatter(_fmt)
    _ch = logging.StreamHandler()
    _ch.setFormatter(_fmt)
    logger.addHandler(_fh)
    logger.addHandler(_ch)

# Model definition (matches v1 training architecture)
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
DB_PATH = 'predictions.db'

# Load once
scaler     = joblib.load('../../models/v1/feature_scaler.pkl')
maml_model = OilVolatilityMLP()
maml_model.load_state_dict(
    torch.load('../../models/v1/maml_trained.pth', map_location='cpu'))
maml_model.eval()
plain_mlp  = OilVolatilityMLP()
plain_mlp.load_state_dict(
    torch.load('../../models/v1/mlp_pretrained.pth', map_location='cpu'))
plain_mlp.eval()
logger.info("Models loaded. Scheduler running.")


# Database helpers
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

def log_prediction(ts, maml_pred, mlp_pred,
                   oil_close, ovx_close, n_support, features, gdelt_context=''):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO predictions
        (timestamp, maml_pred, mlp_pred, oil_close,
         ovx_close, n_support, features_json, gdelt_context)
        VALUES (?,?,?,?,?,?,?,?)
    """, (str(ts), maml_pred, mlp_pred, oil_close,
          ovx_close, n_support, json.dumps(features), gdelt_context))
    conn.commit()
    conn.close()

def update_actuals():
    """Check if any pending predictions have resolved."""
    conn   = sqlite3.connect(DB_PATH)
    rows   = conn.execute("""
        SELECT id, timestamp FROM predictions
        WHERE actual_rvol IS NULL
    """).fetchall()
    now    = datetime.utcnow()
    updated = 0
    for row_id, ts in rows:
        pred_time = datetime.fromisoformat(str(ts))
        if now < pred_time + timedelta(hours=4):
            continue
        actual = compute_actual_vol(pred_time)
        if actual is not None:
            conn.execute("""
                UPDATE predictions SET actual_rvol=? WHERE id=?
            """, (actual, row_id))
            updated += 1
    conn.commit()
    conn.close()
    if updated:
        logger.info("Updated %d actuals", updated)

def get_support_set():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT features_json, actual_rvol FROM predictions
        WHERE actual_rvol IS NOT NULL
        ORDER BY timestamp DESC LIMIT 15
    """).fetchall()
    conn.close()
    if len(rows) < 3:
        return None, 0
    features = np.array([
        list(json.loads(r[0]).values())
        for r in rows
    ], dtype=np.float32)
    actuals = np.array([r[1] for r in rows], dtype=np.float32)
    sup_X = torch.tensor(
        scaler.transform(features), dtype=torch.float32)
    sup_y = torch.tensor(
        np.log1p(actuals), dtype=torch.float32).unsqueeze(1)
    return (sup_X, sup_y), len(rows)

def should_predict():
    conn = sqlite3.connect(DB_PATH)
    row  = conn.execute("""
        SELECT timestamp FROM predictions
        ORDER BY timestamp DESC LIMIT 1
    """).fetchone()
    conn.close()
    if row is None:
        return True
    last = datetime.fromisoformat(str(row[0]))
    return (datetime.utcnow() - last).total_seconds() > 3600


def get_live_ovx():
    """Get current OVX spot price, not historical close."""
    try:
        ticker = yf.Ticker('^OVX')
        info   = ticker.fast_info
        ovx    = info.last_price
        if ovx and ovx > 0:
            return float(ovx)
    except Exception as e:
        logger.warning("OVX spot fetch failed: %s", e)
    return None


# Market + GDELT fetchers
def fetch_market():
    end   = datetime.utcnow()
    start = end - timedelta(days=60)
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
                             interval='1h')
            time.sleep(1)
            if len(df) > 0:
                s = df['Close'].squeeze()
                s = s[s > 0]
                if len(s) > 0:
                    data[col] = s
        except Exception as e:
            logger.warning("FAILED %s: %s", ticker, e)

    if not data:
        return None

    market = pd.DataFrame(data).ffill().bfill()
    market.index = pd.to_datetime(market.index)

    if 'oil_close' in market.columns:
        log_ret = np.log(
            market['oil_close'] / market['oil_close'].shift(1))
        # With hourly data: 5 days = ~120 trading hours, 20 days = ~480
        market['oil_vol_5d']  = log_ret.rolling(120).std() * np.sqrt(252 * 24)
        market['oil_vol_20d'] = log_ret.rolling(480).std() * np.sqrt(252 * 24)

    if 'gold_close' in market.columns and 'oil_close' in market.columns:
        market['gold_oil_ratio'] = (market['gold_close'] /
                                    market['oil_close'])

    market = market.drop(columns=['gold_close'], errors='ignore')

    latest = market.iloc[-1]

    if latest.isna().any():
        missing = latest[latest.isna()].index.tolist()
        logger.warning("NaN in latest row: %s -- using ffill", missing)
        market = market.ffill()
        latest = market.iloc[-1]

    if latest.isna().any():
        logger.warning("Still NaN after ffill -- skipping")
        return None

    # Override OVX with live spot price
    live_ovx = get_live_ovx()
    if live_ovx:
        latest['ovx_close'] = live_ovx
        logger.info("Live OVX spot: %.2f", live_ovx)

    logger.info(
        "Latest row date: %s  OVX=%.1f",
        market.index[-1].date(),
        latest.get('ovx_close', float('nan')),
    )
    return latest.to_dict()

# Exact copies from training pipeline (must match what model was trained on)

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


def assign_oil_region_fips(c):
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


def get_task_category(event_code):
    ec = str(event_code).strip()
    for length in (4, 3, 2):
        cat = CAMEO_TASK_MAP.get(ec[:length])
        if cat:
            return cat
    return "other_political"


GDELT_FALLBACK = {
    'gs_mean':0.0, 'gs_std':2.5, 'gs_conflict_pct':0.45,
    'gs_weighted':0.0, 'tone_mean':-1.0, 'tone_std':3.0,
    'n_events':24.0, 'mentions_sum':250.0
}


def fetch_gdelt():
    import zipfile
    import io
    try:
        resp = requests.get(
            "http://data.gdeltproject.org/gdeltv2/lastupdate.txt",
            timeout=10, verify=False
        )
        export_url = resp.text.strip().split('\n')[0].split(' ')[2].strip()

        r2 = requests.get(export_url, timeout=30, verify=False)
        with zipfile.ZipFile(io.BytesIO(r2.content)) as z:
            with z.open(z.namelist()[0]) as f:
                import pandas as pd
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

        df = df[df['mentions'] >= MIN_MENTIONS]

        mask_us = (
            df['actor1_country'].isin(HIGH_IMPACT_COUNTRIES) |
            df['actor2_country'].isin(HIGH_IMPACT_COUNTRIES) |
            df['action_country'].isin(HIGH_IMPACT_COUNTRIES)
        )
        df = df[mask_us]

        df['task_category'] = df['event_code'].apply(get_task_category)
        fallback_mask = df['task_category'] == 'other_political'
        df.loc[fallback_mask, 'task_category'] = df.loc[
            fallback_mask, 'event_base_code'
        ].apply(get_task_category)
        df = df[df['task_category'] != 'other_political']

        coop_mask = df['task_category'] == 'cooperation_diplomacy'
        gold_mask = df['goldstein'].abs() >= COOP_GOLDSTEIN_MIN
        df = df[~coop_mask | gold_mask]

        if df.empty:
            raise ValueError("No relevant events after filtering")

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
        }
    except Exception as e:
        logger.warning("GDELT raw fetch error: %s -- using fallback", e)
        return GDELT_FALLBACK

def compute_actual_vol(pred_utc):
    try:
        pred_hour = pred_utc.replace(minute=0, second=0, microsecond=0)
        start = (pred_hour - timedelta(days=1)).strftime('%Y-%m-%d')
        end   = (pred_hour + timedelta(days=2)).strftime('%Y-%m-%d')
        h = yf.download('CL=F', start=start, end=end,
                        interval='1h', progress=False,
                        auto_adjust=True)
        if h.empty or len(h) < 4:
            return None
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = [c[0] for c in h.columns]
        h.index = pd.to_datetime(h.index)
        if h.index.tz is not None:
            h.index = h.index.tz_localize(None)
        h = h.sort_index()
        mask = ((h.index >= pred_hour) &
                (h.index <= pred_hour + timedelta(hours=4)))
        h4   = h[mask]
        if len(h4) < 2:
            return None
        rets = np.log(h4['Close'] / h4['Close'].shift(1)).dropna()
        return float(rets.std(ddof=1) * np.sqrt(252 * 23))
    except Exception:
        return None


# Prediction
def make_prediction():
    logger.info("Making prediction...")

    market = fetch_market()
    if market is None:
        logger.warning("Market data unavailable - skipping")
        return

    gdelt = fetch_gdelt()

    raw_X = np.array([[
        market.get('ovx_close', 0),
        market.get('vix_close', 0),
        market.get('oil_vol_5d', 0),
        market.get('oil_vol_20d', 0),
        market.get('oil_close', 0),
        market.get('dxy_close', 0),
        market.get('gold_oil_ratio', 0),
        gdelt['gs_mean'],
        gdelt['gs_std'],
        gdelt['gs_conflict_pct'],
        gdelt['gs_weighted'],
        gdelt['tone_mean'],
        gdelt['tone_std'],
        gdelt['n_events'],
        gdelt['mentions_sum'],
    ]], dtype=np.float32)

    if np.isnan(raw_X).any():
        logger.warning("NaN in features - skipping")
        return

    X_tensor = torch.tensor(
        scaler.transform(raw_X), dtype=torch.float32)

    # Plain MLP prediction
    plain_mlp.eval()
    with torch.no_grad():
        mlp_pred = float(np.expm1(plain_mlp(X_tensor).item()))

    # MAML prediction with live adaptation
    support, n_support = get_support_set()
    loss_fn = nn.HuberLoss(delta=0.5)

    if support is not None:
        sup_X, sup_y = support
        adapted   = deepcopy(maml_model)
        optimizer = torch.optim.SGD(
            adapted.parameters(), lr=0.005)
        adapted.train()
        for _ in range(10):
            optimizer.zero_grad()
            loss = loss_fn(adapted(sup_X), sup_y)
            loss.backward()
            optimizer.step()
        adapted.eval()
        with torch.no_grad():
            maml_pred = float(np.expm1(adapted(X_tensor).item()))
    else:
        maml_model.eval()
        with torch.no_grad():
            maml_pred = float(
                np.expm1(maml_model(X_tensor).item()))

    # Log to database
    features_dict = {
        col: float(raw_X[0][i])
        for i, col in enumerate(FEATURE_COLS)
    }
    now_utc = datetime.utcnow()
    gdelt_context = (
        f"conflict={gdelt['gs_conflict_pct']*100:.0f}% | "
        f"tone={gdelt['tone_mean']:.1f} | "
        f"events={int(gdelt['n_events'])}"
    )
    log_prediction(
        ts            = now_utc,
        maml_pred     = maml_pred,
        mlp_pred      = mlp_pred,
        oil_close     = float(market.get('oil_close', 0)),
        ovx_close     = float(market.get('ovx_close', 0)),
        n_support     = n_support,
        features      = features_dict,
        gdelt_context = gdelt_context
    )

    logger.info(
        "MAML: %.4f  MLP: %.4f  OVX: %.1f  Support: %d examples",
        maml_pred, mlp_pred, market.get('ovx_close', 0), n_support,
    )


# Market hours check

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


# Main loop
def main():
    init_db()
    logger.info("Database ready. Checking every 5 minutes.")
    logger.info("Predictions every 1 hour during market hours. Actuals checked each cycle.")
    while True:
        try:
            update_actuals()
            if not is_ovx_calculating():
                logger.debug("US market closed -- OVX not updating, skipping prediction")
            elif not should_predict():
                logger.debug("Next prediction not due yet")
            else:
                make_prediction()
        except Exception:
            logger.exception("Unhandled error in main loop")
        time.sleep(300)   # check every 5 minutes


if __name__ == "__main__":
    main()