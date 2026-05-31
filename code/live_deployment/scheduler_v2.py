"""
Scheduler — V2 (23 features)
Background prediction loop. Every 1 hour during OVX hours, computes a v2 prediction
(15 original features + 8 per-region GDELT aggregates). Writes to predictions_v2.db.

Run from the live_deployment/ folder:
    python scheduler_v2.py
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
logger = logging.getLogger("scheduler_v2")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _fh = RotatingFileHandler("scheduler_v2.log", maxBytes=10_000_000, backupCount=5)
    _fh.setFormatter(_fmt)
    _ch = logging.StreamHandler()
    _ch.setFormatter(_fmt)
    logger.addHandler(_fh)
    logger.addHandler(_ch)

# Model definition
class OilVolatilityMLP(nn.Module):
    def __init__(self, input_dim: int = 23):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48), nn.BatchNorm1d(48), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(48, 32),        nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(32, 16),        nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


# Feature cols (23)
FEATURE_COLS = [
    # Market (7)
    'ovx_close', 'vix_close', 'oil_vol_5d', 'oil_vol_20d',
    'oil_close', 'dxy_close', 'gold_oil_ratio',
    # GDELT global (8)
    'gs_mean', 'gs_std', 'gs_conflict_pct', 'gs_weighted',
    'tone_mean', 'tone_std', 'n_events', 'mentions_sum',
    # Per-region Middle East (4)
    'me_gs_mean', 'me_conflict_pct', 'me_n_events', 'me_tone_mean',
    # Per-region Oil Producers (4)
    'oi_gs_mean', 'oi_conflict_pct', 'oi_n_events', 'oi_tone_mean',
]

INPUT_DIM = len(FEATURE_COLS)  # 23
assert INPUT_DIM == 23, f"Expected 23 features, got {INPUT_DIM}"

DB_PATH   = 'predictions_v2.db'
SEED_PATH = '../../data/seed_v2.csv'

# Load models
scaler = joblib.load('../../models/v2/feature_scaler_v2.pkl')

maml_model = OilVolatilityMLP(input_dim=INPUT_DIM)
maml_model.load_state_dict(torch.load('../../models/v2/maml_trained_v2.pth', map_location='cpu'))
maml_model.eval()

plain_mlp = OilVolatilityMLP(input_dim=INPUT_DIM)
plain_mlp.load_state_dict(torch.load('../../models/v2/mlp_pretrained_v2.pth', map_location='cpu'))
plain_mlp.eval()

logger.info("Models loaded (v2, %d features). Scheduler running.", INPUT_DIM)
logger.info("DB: %s", DB_PATH)


# GDELT constants
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
    "061": "sanctions_trade",      "163": "sanctions_trade",
    "164": "sanctions_trade",
    "10": "policy_statement",      "11": "policy_statement",
    "12": "policy_statement",      "13": "policy_statement",
    "14": "political_instability",
    "15": "coercion",
    "16": "diplomatic_tension",    "17": "diplomatic_tension",
    "18": "military_conflict",     "19": "military_conflict",
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


GDELT_FALLBACK = {
    'gs_mean': 0.0, 'gs_std': 2.5, 'gs_conflict_pct': 0.45,
    'gs_weighted': 0.0, 'tone_mean': -1.0, 'tone_std': 3.0,
    'n_events': 24.0, 'mentions_sum': 250.0,
    'me_gs_mean': 0.0, 'me_conflict_pct': 0.0,
    'me_n_events': 0.0, 'me_tone_mean': -1.0,
    'oi_gs_mean': 0.0, 'oi_conflict_pct': 0.0,
    'oi_n_events': 0.0, 'oi_tone_mean': -1.0,
}


# Database
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


def log_prediction(ts, maml_pred, mlp_pred, oil_close, ovx_close,
                   n_support, features, gdelt_context=''):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO predictions
        (timestamp, maml_pred, mlp_pred, oil_close, ovx_close,
         n_support, features_json, gdelt_context)
        VALUES (?,?,?,?,?,?,?,?)
    """, (str(ts), maml_pred, mlp_pred, oil_close, ovx_close,
          n_support, json.dumps(features), gdelt_context))
    conn.commit()
    conn.close()


def update_actuals():
    conn    = sqlite3.connect(DB_PATH)
    rows    = conn.execute("""
        SELECT id, timestamp FROM predictions WHERE actual_rvol IS NULL
    """).fetchall()
    now     = datetime.utcnow()
    updated = 0
    for row_id, ts in rows:
        pred_time = datetime.fromisoformat(str(ts))
        if now < pred_time + timedelta(hours=4):
            continue
        actual = compute_actual_vol(pred_time)
        if actual is not None:
            conn.execute(
                "UPDATE predictions SET actual_rvol=? WHERE id=?",
                (actual, row_id)
            )
            updated += 1
    conn.commit()
    conn.close()
    if updated:
        logger.info("Updated %d actuals", updated)


def get_support_set():
    """Loads resolved predictions as MAML support set. Seeds from SEED_PATH until enough live examples."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT features_json, actual_rvol FROM predictions
        WHERE actual_rvol IS NOT NULL
        ORDER BY timestamp DESC LIMIT 15
    """).fetchall()
    conn.close()

    n_live = len(rows)

    if n_live < 15:
        try:
            seed_df = pd.read_csv(SEED_PATH)
            seed_df = seed_df.dropna(subset=FEATURE_COLS + ['oil_fwd_rvol_4h'])
            n_seed  = min(15 - n_live, len(seed_df))
            seed_df = seed_df.sample(n_seed, random_state=42)

            seed_feat = seed_df[FEATURE_COLS].values.astype(np.float32)
            seed_act  = seed_df['oil_fwd_rvol_4h'].values.astype(np.float32)

            if n_live > 0:
                live_feat = np.array([
                    list(json.loads(r[0]).values()) for r in rows
                ], dtype=np.float32)
                live_act  = np.array([r[1] for r in rows], dtype=np.float32)
                all_feat  = np.vstack([live_feat, seed_feat])
                all_act   = np.concatenate([live_act, seed_act])
            else:
                all_feat = seed_feat
                all_act  = seed_act

            sup_X = torch.tensor(scaler.transform(all_feat), dtype=torch.float32)
            sup_y = torch.tensor(np.log1p(all_act), dtype=torch.float32).unsqueeze(1)
            logger.info("Support: %d live + %d seeded", n_live, n_seed)
            return (sup_X, sup_y), n_live + n_seed

        except Exception as e:
            logger.warning("Seed load failed: %s", e)
            if n_live < 3:
                return None, 0

    # Normal path — enough live examples
    features = np.array([
        list(json.loads(r[0]).values()) for r in rows
    ], dtype=np.float32)
    actuals = np.array([r[1] for r in rows], dtype=np.float32)
    sup_X = torch.tensor(scaler.transform(features), dtype=torch.float32)
    sup_y = torch.tensor(np.log1p(actuals), dtype=torch.float32).unsqueeze(1)
    return (sup_X, sup_y), n_live


def should_predict() -> bool:
    conn = sqlite3.connect(DB_PATH)
    row  = conn.execute("""
        SELECT timestamp FROM predictions ORDER BY timestamp DESC LIMIT 1
    """).fetchone()
    conn.close()
    if row is None:
        return True
    last = datetime.fromisoformat(str(row[0]))
    return (datetime.utcnow() - last).total_seconds() > 3600


# Market data
def get_live_ovx():
    try:
        ovx = yf.Ticker('^OVX').fast_info.last_price
        if ovx and ovx > 0:
            return float(ovx)
    except Exception as e:
        logger.warning("OVX spot fetch failed: %s", e)
    return None


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
                             progress=False, auto_adjust=True, interval='1h')
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
        log_ret = np.log(market['oil_close'] / market['oil_close'].shift(1))
        market['oil_vol_5d']  = log_ret.rolling(120).std() * np.sqrt(252 * 24)
        market['oil_vol_20d'] = log_ret.rolling(480).std() * np.sqrt(252 * 24)

    if 'gold_close' in market.columns and 'oil_close' in market.columns:
        market['gold_oil_ratio'] = market['gold_close'] / market['oil_close']

    market = market.drop(columns=['gold_close'], errors='ignore')
    market = market.ffill()
    latest = market.iloc[-1]

    if latest.isna().any():
        logger.warning("Still NaN after ffill - skipping")
        return None

    live_ovx = get_live_ovx()
    if live_ovx:
        latest = latest.copy()
        latest['ovx_close'] = live_ovx
        logger.info("Live OVX spot: %.2f", live_ovx)

    logger.info(
        "Market date: %s  OVX=%.1f  Oil=$%.2f",
        market.index[-1].date(),
        latest.get('ovx_close', 0),
        latest.get('oil_close', 0),
    )
    return latest.to_dict()


# GDELT fetcher (with per-region sub-aggregates)
def fetch_gdelt() -> dict:
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
                raw = pd.read_csv(f, sep='\t', header=None,
                                  on_bad_lines='skip', low_memory=False)

        df = pd.DataFrame({
            'actor1_country' : raw[7].astype(str).str.strip().str.upper(),
            'actor2_country' : raw[17].astype(str).str.strip().str.upper(),
            'action_country' : raw[53].astype(str).str.strip().str.upper().replace({'USA': 'US'}),
            'event_code'     : raw[26].astype(str).str.strip(),
            'event_base_code': raw[27].astype(str).str.strip(),
            'goldstein'      : pd.to_numeric(raw[30], errors='coerce'),
            'mentions'       : pd.to_numeric(raw[31], errors='coerce').fillna(0),
            'tone'           : pd.to_numeric(raw[34], errors='coerce'),
        }).dropna(subset=['goldstein'])

        # Filters
        df = df[df['mentions'] >= MIN_MENTIONS]
        mask = (
            df['actor1_country'].isin(HIGH_IMPACT_COUNTRIES) |
            df['actor2_country'].isin(HIGH_IMPACT_COUNTRIES) |
            df['action_country'].isin(HIGH_IMPACT_COUNTRIES)
        )
        df = df[mask]

        df['task_category'] = df['event_code'].apply(get_task_category)
        fb = df['task_category'] == 'other_political'
        df.loc[fb, 'task_category'] = (
            df.loc[fb, 'event_base_code'].apply(get_task_category)
        )
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

        gs_vals   = df['goldstein'].values
        men_vals  = df['mentions'].values
        tone_vals = df['tone'].dropna().values

        # Global aggregates
        result = {
            'gs_mean'         : float(gs_vals.mean()),
            'gs_std'          : float(gs_vals.std()),
            'gs_conflict_pct' : float((gs_vals < 0).mean()),
            'gs_weighted'     : float(np.average(gs_vals, weights=men_vals.clip(min=1))),
            'tone_mean'       : float(tone_vals.mean()) if len(tone_vals) else -1.0,
            'tone_std'        : float(tone_vals.std())  if len(tone_vals) else 3.0,
            'n_events'        : float(len(gs_vals)),
            'mentions_sum'    : float(men_vals.sum()),
        }

        # Per-region sub-aggregates (v2 addition)
        for region, prefix in [('middle_east', 'me'), ('oil_producer', 'oi')]:
            rg = df[df['oil_region'] == region]
            if len(rg) > 0:
                rgs = rg['goldstein'].values
                rgt = rg['tone'].dropna().values
                result[f'{prefix}_gs_mean']      = float(rgs.mean())
                result[f'{prefix}_conflict_pct'] = float((rgs < 0).mean())
                result[f'{prefix}_n_events']     = float(len(rg))
                result[f'{prefix}_tone_mean']    = float(rgt.mean()) if len(rgt) else -1.0
            else:
                result[f'{prefix}_gs_mean']      = 0.0
                result[f'{prefix}_conflict_pct'] = 0.0
                result[f'{prefix}_n_events']     = 0.0
                result[f'{prefix}_tone_mean']    = -1.0

        logger.info(
            "GDELT: %d events | ME: %d | OI: %d | conflict: %.0f%%",
            int(result['n_events']),
            int(result['me_n_events']),
            int(result['oi_n_events']),
            result['gs_conflict_pct'] * 100,
        )
        return result

    except Exception as e:
        logger.warning("GDELT fetch error: %s -- using fallback", e)
        return GDELT_FALLBACK


# Realized vol
def compute_actual_vol(pred_utc):
    try:
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
        rets = np.log(h4['Close'] / h4['Close'].shift(1)).dropna()
        return float(rets.std(ddof=1) * np.sqrt(252 * 23))
    except Exception:
        return None


# Prediction
def make_prediction():
    logger.info("Making prediction (v2)...")

    market = fetch_market()
    if market is None:
        logger.warning("Market data unavailable - skipping")
        return

    gdelt = fetch_gdelt()

    # Build 23-feature vector in exact FEATURE_COLS order
    raw_X = np.array([[
        market.get('ovx_close',      0.0),
        market.get('vix_close',      0.0),
        market.get('oil_vol_5d',     0.0),
        market.get('oil_vol_20d',    0.0),
        market.get('oil_close',      0.0),
        market.get('dxy_close',      0.0),
        market.get('gold_oil_ratio', 0.0),
        gdelt['gs_mean'],
        gdelt['gs_std'],
        gdelt['gs_conflict_pct'],
        gdelt['gs_weighted'],
        gdelt['tone_mean'],
        gdelt['tone_std'],
        gdelt['n_events'],
        gdelt['mentions_sum'],
        gdelt['me_gs_mean'],
        gdelt['me_conflict_pct'],
        gdelt['me_n_events'],
        gdelt['me_tone_mean'],
        gdelt['oi_gs_mean'],
        gdelt['oi_conflict_pct'],
        gdelt['oi_n_events'],
        gdelt['oi_tone_mean'],
    ]], dtype=np.float32)

    assert raw_X.shape[1] == INPUT_DIM, \
        f"Feature dim mismatch: {raw_X.shape[1]} vs {INPUT_DIM}"

    if np.isnan(raw_X).any():
        logger.warning("NaN in features - skipping")
        return

    X_tensor = torch.tensor(scaler.transform(raw_X), dtype=torch.float32)

    # Plain MLP
    plain_mlp.eval()
    with torch.no_grad():
        mlp_pred = float(np.expm1(plain_mlp(X_tensor).item()))

    # MAML with adaptation
    support, n_support = get_support_set()
    loss_fn = nn.HuberLoss(delta=0.5)

    if support is not None:
        sup_X, sup_y = support
        adapted   = deepcopy(maml_model)
        optimizer = torch.optim.SGD(adapted.parameters(), lr=0.01)
        adapted.eval()
        for _ in range(5):
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
            maml_pred = float(np.expm1(maml_model(X_tensor).item()))
        n_support = 0

    # Build features dict in FEATURE_COLS order for DB storage
    features_dict = {col: float(raw_X[0][i]) for i, col in enumerate(FEATURE_COLS)}

    now_utc = datetime.utcnow()
    gdelt_context = (
        f"conflict={gdelt['gs_conflict_pct']*100:.0f}% | "
        f"tone={gdelt['tone_mean']:.1f} | "
        f"events={int(gdelt['n_events'])} | "
        f"ME_events={int(gdelt['me_n_events'])} | "
        f"ME_conflict={gdelt['me_conflict_pct']*100:.0f}%"
    )

    log_prediction(
        ts            = now_utc,
        maml_pred     = maml_pred,
        mlp_pred      = mlp_pred,
        oil_close     = float(market.get('oil_close', 0)),
        ovx_close     = float(market.get('ovx_close', 0)),
        n_support     = n_support,
        features      = features_dict,
        gdelt_context = gdelt_context,
    )

    logger.info(
        "MAML: %.4f  MLP: %.4f  OVX: %.1f  Support: %d  ME_events: %d",
        maml_pred, mlp_pred,
        market.get('ovx_close', 0),
        n_support,
        int(gdelt['me_n_events']),
    )


# OVX hours check
def is_ovx_calculating() -> bool:
    now_utc = datetime.utcnow()
    if now_utc.weekday() >= 5:
        return False
    minutes = now_utc.hour * 60 + now_utc.minute
    if 135 <= minutes <= 150:
        return False
    return True


# Main loop
def main():
    init_db()
    logger.info("Database ready: %s", DB_PATH)
    logger.info("Seed file: %s", SEED_PATH)
    logger.info("Predictions every 1 hour. Actuals checked every 5 min.")
    while True:
        try:
            update_actuals()
            if not is_ovx_calculating():
                logger.debug("Outside OVX hours - skipping prediction")
            elif not should_predict():
                logger.debug("Not due yet")
            else:
                make_prediction()
        except Exception:
            logger.exception("Unhandled error in main loop")
        time.sleep(300)


if __name__ == "__main__":
    main()