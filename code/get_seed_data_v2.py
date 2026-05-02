"""
get_seed_data_v2.py
====================
Builds the seed CSV for MAML v2 adaptation.
v2 adds 8 per-region GDELT features (Middle East + Oil Producers).

Output: dataset_maml_march1724_v2.csv  (23 features)

Run:
    python get_seed_data_v2.py
"""

from gdelt_pipeline import main as run_pipeline, PipelineConfig
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

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

assert len(FEATURE_COLS) == 23, f"Expected 23 features, got {len(FEATURE_COLS)}"

MIDDLE_EAST_FIPS = frozenset({
    "SA", "IR", "IZ", "KU", "IS", "SY", "LE", "JO",
    "YM", "AE", "QA", "BN", "OM",
})
OIL_PRODUCER_FIPS = frozenset({
    "RS", "VE", "NI", "LY", "AG", "AO", "NO", "EC", "BR", "CA", "KZ",
})
CHOKEPOINT_FIPS = frozenset({"TU", "EG", "SU", "SO", "DJ"})


def assign_oil_region(c: str) -> str:
    if not c or str(c) in ("nan", "UNKNOWN", ""):
        return "other"
    c = str(c).strip().upper()
    if c in ("US", "USA"):
        return "US"
    if c in MIDDLE_EAST_FIPS or c in CHOKEPOINT_FIPS:
        return "middle_east"
    if c in OIL_PRODUCER_FIPS:
        return "oil_producer"
    return "other"


# Step 1: Run GDELT pipeline
print("=" * 60)
print("Step 1: Fetching GDELT data March 17-23 2026")
print("=" * 60)

cfg = PipelineConfig(
    start_date     = "2026-03-17",
    end_date       = "2026-03-23",
    output_dir     = "seed_output_v2",
    checkpoint_dir = "seed_output_v2/checkpoints",
    final_output   = "seed_output_v2/political_events_gdelt.csv",
    sp500_output   = "seed_output_v2/sp500_returns.csv",
    log_file       = "seed_output_v2/pipeline.log",
)

run_pipeline(cfg)

# Step 2: Load pipeline output
print("\n" + "=" * 60)
print("Step 2: Loading pipeline output")
print("=" * 60)

intraday_path = "seed_output_v2/political_events_gdelt_with_intraday.csv"
df = pd.read_csv(intraday_path)
df['date'] = pd.to_datetime(df['date'])
print(f"Loaded: {len(df)} events")
print(f"Columns: {df.columns.tolist()}")

# Step 3: Build market features
print("\n" + "=" * 60)
print("Step 3: Downloading market data")
print("=" * 60)

end   = datetime(2026, 3, 25)
start = datetime(2026, 2, 15)  # buffer for rolling vols

tickers = {
    'oil_close'  : 'BZ=F',
    'vix_close'  : '^VIX',
    'ovx_close'  : '^OVX',
    'dxy_close'  : 'DX-Y.NYB',
    'gold_close' : 'GC=F',
}
market_data = {}
for col, ticker in tickers.items():
    try:
        px = yf.download(ticker, start=start, end=end,
                         interval='1h', progress=False, auto_adjust=True)
        if not px.empty:
            s = px['Close'].squeeze()
            s = s[s > 0]
            market_data[col] = s
            print(f"  {col}: {len(s)} bars, last={s.iloc[-1]:.2f}")
    except Exception as e:
        print(f"  {col}: FAILED ({e})")

market = pd.DataFrame(market_data).ffill().bfill()
market.index = pd.to_datetime(market.index)
if market.index.tz is not None:
    market.index = market.index.tz_localize(None)

# Rolling vols
log_ret = np.log(market['oil_close'] / market['oil_close'].shift(1))
market['oil_vol_5d']     = log_ret.rolling(120).std() * np.sqrt(252 * 24)
market['oil_vol_20d']    = log_ret.rolling(480).std() * np.sqrt(252 * 24)
market['gold_oil_ratio'] = market['gold_close'] / market['oil_close']
market = market.drop(columns=['gold_close'])

# Hourly oil for 4h vol target
print("\nDownloading hourly oil for realized vol targets...")
oil_h = yf.download('BZ=F', start=start, end=end,
                    interval='1h', progress=False, auto_adjust=True)
if isinstance(oil_h.columns, pd.MultiIndex):
    oil_h.columns = [c[0] for c in oil_h.columns]
oil_h.index = pd.to_datetime(oil_h.index)
if oil_h.index.tz is not None:
    oil_h.index = oil_h.index.tz_localize(None)
oil_h = oil_h.sort_index()

oil_h['hourly_return'] = oil_h['Close'].pct_change()
shifted = pd.concat(
    [oil_h['hourly_return'].shift(-k) for k in range(1, 5)], axis=1
)
oil_h['oil_fwd_rvol_4h'] = shifted.std(axis=1, ddof=1) * np.sqrt(252 * 23)
print(f"  oil_fwd_rvol_4h mean: {oil_h['oil_fwd_rvol_4h'].dropna().mean():.4f}")

# Step 4: Aggregate GDELT (global + per-region)
print("\n" + "=" * 60)
print("Step 4: Aggregating GDELT features (global + per-region)")
print("=" * 60)

# Assign oil region to each event row
# primary_country comes from the gdelt_pipeline output
if 'primary_country' not in df.columns:
    # Fallback: derive from action_country if pipeline didn't produce it
    df['primary_country'] = df.get('action_country', 'UNKNOWN').fillna('UNKNOWN')

df['oil_region'] = df['primary_country'].apply(assign_oil_region)

def aggregate_gdelt_v2(group):
    """Aggregate global and per-region GDELT features for one date."""
    gs   = group['goldstein_scale'].dropna().values
    men  = group['num_mentions'].fillna(1).values
    tone = group['avg_tone'].dropna().values

    if len(gs) == 0:
        return None

    result = pd.Series({
        # Global aggregates
        'gs_mean'         : float(gs.mean()),
        'gs_std'          : float(gs.std()) if len(gs) > 1 else 0.0,
        'gs_conflict_pct' : float((gs < 0).mean()),
        'gs_weighted'     : float(np.average(gs, weights=men[:len(gs)].clip(min=1))),
        'tone_mean'       : float(tone.mean()) if len(tone) else -1.0,
        'tone_std'        : float(tone.std())  if len(tone) > 1 else 3.0,
        'n_events'        : float(len(gs)),
        'mentions_sum'    : float(men.sum()),
    })

    # Per-region aggregates
    for region, prefix in [('middle_east', 'me'), ('oil_producer', 'oi')]:
        rg = group[group['oil_region'] == region]
        if len(rg) > 0:
            rgs  = rg['goldstein_scale'].dropna().values
            rgt  = rg['avg_tone'].dropna().values
            result[f'{prefix}_gs_mean']      = float(rgs.mean()) if len(rgs) else 0.0
            result[f'{prefix}_conflict_pct'] = float((rgs < 0).mean()) if len(rgs) else 0.0
            result[f'{prefix}_n_events']     = float(len(rg))
            result[f'{prefix}_tone_mean']    = float(rgt.mean()) if len(rgt) else -1.0
        else:
            result[f'{prefix}_gs_mean']      = 0.0
            result[f'{prefix}_conflict_pct'] = 0.0
            result[f'{prefix}_n_events']     = 0.0
            result[f'{prefix}_tone_mean']    = -1.0

    return result

gdelt_agg = df.groupby('date').apply(aggregate_gdelt_v2).reset_index()
gdelt_agg = gdelt_agg.dropna()
print(f"Aggregated into {len(gdelt_agg)} daily rows")
print(f"ME events mean: {gdelt_agg['me_n_events'].mean():.2f}")
print(f"OI events mean: {gdelt_agg['oi_n_events'].mean():.2f}")

# Expand to 4 snapshots per day
SNAPSHOT_HOURS = [4, 9, 14, 20]
expanded = []
for _, row in gdelt_agg.iterrows():
    for hour in SNAPSHOT_HOURS:
        r = row.to_dict()
        r['snapshot_hour'] = hour
        expanded.append(r)

gdelt_expanded = pd.DataFrame(expanded)
gdelt_expanded['snapshot_dt'] = (
    gdelt_expanded['date'] +
    pd.to_timedelta(gdelt_expanded['snapshot_hour'], unit='h')
)
print(f"Expanded to {len(gdelt_expanded)} snapshot rows")

# Step 5: Merge market + GDELT + realized vol
print("\n" + "=" * 60)
print("Step 5: Merging market + GDELT + realized vol")
print("=" * 60)

market_clean = market.dropna()
rows = []

for _, snap in gdelt_expanded.iterrows():
    snap_dt = snap['snapshot_dt']

    # Nearest market bar
    idx  = market_clean.index.get_indexer([snap_dt], method='nearest')[0]
    mrow = market_clean.iloc[idx]

    # 4h realized vol
    mask = (
        (oil_h.index >= snap_dt) &
        (oil_h.index <= snap_dt + timedelta(hours=4))
    )
    h4 = oil_h[mask]
    if len(h4) < 2:
        continue
    rets = np.log(h4['Close'] / h4['Close'].shift(1)).dropna()
    if len(rets) < 2:
        continue
    rvol = float(rets.std(ddof=1) * np.sqrt(252 * 23))

    rows.append({
        'date'             : snap['date'],
        'snapshot_hour'    : snap['snapshot_hour'],
        # Market features
        'ovx_close'        : float(mrow.get('ovx_close',      np.nan)),
        'vix_close'        : float(mrow.get('vix_close',      np.nan)),
        'oil_vol_5d'       : float(mrow.get('oil_vol_5d',     np.nan)),
        'oil_vol_20d'      : float(mrow.get('oil_vol_20d',    np.nan)),
        'oil_close'        : float(mrow.get('oil_close',      np.nan)),
        'dxy_close'        : float(mrow.get('dxy_close',      np.nan)),
        'gold_oil_ratio'   : float(mrow.get('gold_oil_ratio', np.nan)),
        # GDELT global
        'gs_mean'          : snap['gs_mean'],
        'gs_std'           : snap['gs_std'],
        'gs_conflict_pct'  : snap['gs_conflict_pct'],
        'gs_weighted'      : snap['gs_weighted'],
        'tone_mean'        : snap['tone_mean'],
        'tone_std'         : snap['tone_std'],
        'n_events'         : snap['n_events'],
        'mentions_sum'     : snap['mentions_sum'],
        # Per-region Middle East
        'me_gs_mean'       : snap['me_gs_mean'],
        'me_conflict_pct'  : snap['me_conflict_pct'],
        'me_n_events'      : snap['me_n_events'],
        'me_tone_mean'     : snap['me_tone_mean'],
        # Per-region Oil Producers
        'oi_gs_mean'       : snap['oi_gs_mean'],
        'oi_conflict_pct'  : snap['oi_conflict_pct'],
        'oi_n_events'      : snap['oi_n_events'],
        'oi_tone_mean'     : snap['oi_tone_mean'],
        # Target
        'oil_fwd_rvol_4h'  : rvol,
    })

seed = pd.DataFrame(rows)

# Step 6: Clean and save
print("\n" + "=" * 60)
print("Step 6: Cleaning and saving")
print("=" * 60)

seed_clean = seed.dropna(subset=FEATURE_COLS + ['oil_fwd_rvol_4h'])
output_path = "dataset_maml_march1724_v2.csv"
seed_clean.to_csv(output_path, index=False)

print(f"\nSaved: {output_path}")
print(f"Total snapshot rows  : {len(seed_clean)}")
print(f"OVX range            : {seed_clean['ovx_close'].min():.1f} - {seed_clean['ovx_close'].max():.1f}")
print(f"oil_fwd_rvol_4h mean : {seed_clean['oil_fwd_rvol_4h'].mean():.4f}")
print(f"oil_fwd_rvol_4h std  : {seed_clean['oil_fwd_rvol_4h'].std():.4f}")
print(f"me_n_events mean     : {seed_clean['me_n_events'].mean():.2f}")
print(f"oi_n_events mean     : {seed_clean['oi_n_events'].mean():.2f}")
print()
print("Feature check (no NaNs expected):")
for col in FEATURE_COLS:
    nulls = seed_clean[col].isna().sum()
    status = "✓" if nulls == 0 else f"WARNING: {nulls} nulls"
    print(f"  {col:<25} {status}")
print("=" * 60)
print(f"\nCopy {output_path} to your dashboard directory.")
print("This file seeds MAML adaptation until enough live predictions accumulate.")