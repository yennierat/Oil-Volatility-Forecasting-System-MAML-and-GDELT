"""
MLP Pre-training for MAML Foundation — V2 (23 features)
7 market + 8 GDELT global aggregates + 8 per-region aggregates (Middle East + Oil Producers).
"""

import random
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(8)

# Step 1: Load & aggregate to daily
df = pd.read_csv("../data/dataset_maml_daily_v2.csv")

RAW_COLS = [
    'goldstein_scale', 'num_mentions', 'num_sources', 'avg_tone',
    'oil_close', 'oil_vol_5d', 'oil_vol_20d',
    'vix_close', 'ovx_close', 'dxy_close', 'gold_oil_ratio',
    'me_gs_mean', 'me_conflict_pct', 'me_n_events', 'me_tone_mean',
    'oi_gs_mean', 'oi_conflict_pct', 'oi_n_events', 'oi_tone_mean',
]
TARGET_COL = 'oil_fwd_rvol_1d'

before = len(df)
df = df.dropna(subset=RAW_COLS + [TARGET_COL])
print(f"Dropped {before - len(df)} NaN rows. Remaining: {len(df)}")

daily = (
    df.groupby('date').apply(
        lambda g: pd.Series({
            # GDELT global aggregates (raw per-row GDELT is within-date noise)
            'gs_mean'           : g['goldstein_scale'].mean(),
            'gs_std'            : g['goldstein_scale'].std(ddof=0),
            'gs_conflict_pct'   : (g['goldstein_scale'] < 0).mean(),
            # Mentions-weighted Goldstein: high-coverage hostile events carry more weight
            'gs_weighted'       : np.average(
                                      g['goldstein_scale'],
                                      weights=g['num_mentions'].clip(lower=1)
                                  ),
            'tone_mean'         : g['avg_tone'].mean(),
            'tone_std'          : g['avg_tone'].std(ddof=0),
            'n_events'          : len(g),
            'mentions_sum'      : g['num_mentions'].sum(),
            # Market features (constant per day, take first)
            'oil_close'         : g['oil_close'].iloc[0],
            'oil_vol_5d'        : g['oil_vol_5d'].iloc[0],
            'oil_vol_20d'       : g['oil_vol_20d'].iloc[0],
            'vix_close'         : g['vix_close'].iloc[0],
            'ovx_close'         : g['ovx_close'].iloc[0],
            'dxy_close'         : g['dxy_close'].iloc[0],
            'gold_oil_ratio'    : g['gold_oil_ratio'].iloc[0],
            # Per-region features (already aggregated per date)
            'me_gs_mean'        : g['me_gs_mean'].iloc[0],
            'me_conflict_pct'   : g['me_conflict_pct'].iloc[0],
            'me_n_events'       : g['me_n_events'].iloc[0],
            'me_tone_mean'      : g['me_tone_mean'].iloc[0],
            'oi_gs_mean'        : g['oi_gs_mean'].iloc[0],
            'oi_conflict_pct'   : g['oi_conflict_pct'].iloc[0],
            'oi_n_events'       : g['oi_n_events'].iloc[0],
            'oi_tone_mean'      : g['oi_tone_mean'].iloc[0],
            TARGET_COL          : g[TARGET_COL].iloc[0],
        })
    )
    .reset_index()
    .sort_values('date')
    .reset_index(drop=True)
)
print(f"Daily dataset: {len(daily)} unique dates")

# Step 2: Engineered features (computed but not used in FEATURE_COLS — kept for inspection)
# Vol term structure: short vol / long vol — >1 = fear spike, <1 = vol normalising
daily['vol_term_structure'] = daily['oil_vol_5d'] / (daily['oil_vol_20d'] + 1e-9)
# OVX/VIX ratio: oil-specific fear relative to equity fear
daily['ovx_vix_ratio']      = daily['ovx_close'] / (daily['vix_close'] + 1e-9)
# Vol acceleration: positive = vol regime building, negative = vol mean-reverting
daily['vol_acceleration']   = daily['oil_vol_5d'] - daily['oil_vol_20d']
# Oil price z-score (20-day rolling): extreme price dislocations reliably precede vol spikes
daily['oil_price_zscore']   = (
    (daily['oil_close'] - daily['oil_close'].rolling(20).mean()) /
    (daily['oil_close'].rolling(20).std() + 1e-9)
)
# VIX/OVX spread: equity fear minus oil fear
daily['vix_ovx_spread']     = daily['vix_close'] - daily['ovx_close']

daily = daily.dropna().reset_index(drop=True)
print(f"After engineering: {len(daily)} rows")

# Step 3: Feature set (23 features)
FEATURE_COLS = [
    # Market raw (7)
    'ovx_close', 'vix_close', 'oil_vol_5d', 'oil_vol_20d',
    'oil_close', 'dxy_close', 'gold_oil_ratio',
    # GDELT global aggregates (8)
    'gs_mean', 'gs_std', 'gs_conflict_pct', 'gs_weighted',
    'tone_mean', 'tone_std', 'n_events', 'mentions_sum',
    # Per-region: Middle East (4) and Oil Producers (4)
    'me_gs_mean', 'me_conflict_pct', 'me_n_events', 'me_tone_mean',
    'oi_gs_mean', 'oi_conflict_pct', 'oi_n_events', 'oi_tone_mean',
]

INPUT_DIM = len(FEATURE_COLS)
print(f"\nFeature count:            {INPUT_DIM}")
print(f"Training days:            {len(daily[daily['date'] < '2023-10-01'])}")
print(f"Samples per feature:      {len(daily[daily['date'] < '2023-10-01']) / INPUT_DIM:.1f}  (want > 20)")

# Step 4: Temporal split
train_df = daily[daily['date'] < '2023-10-01'].copy()
val_df   = daily[daily['date'] >= '2023-10-01'].copy()

X_train_raw = train_df[FEATURE_COLS].values.astype(np.float32)
y_train     = train_df[TARGET_COL].values.astype(np.float32)
X_val_raw   = val_df[FEATURE_COLS].values.astype(np.float32)
y_val       = val_df[TARGET_COL].values.astype(np.float32)

print(f"Train: {len(X_train_raw)} days | Val: {len(X_val_raw)} days")

# Step 5: Log-transform target
# Target skew = 1.86. Log1p → ~1.06. Prevents MSE from over-penalising rare vol spikes.
y_train_log = np.log1p(y_train)
y_val_log   = np.log1p(y_val)

# Step 6: Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val   = scaler.transform(X_val_raw)
joblib.dump(scaler, '../models/v2/feature_scaler_v2.pkl')

# Step 7: Model — 23 → 48 → 32 → 16 → 1 with BatchNorm (~3.5k params, regularised by BN + Dropout)
class OilVolatilityMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(48, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)

def make_model():
    return OilVolatilityMLP(INPUT_DIM)

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

print(f"Model parameters:         {count_params(make_model()):,}")

# Step 8: Walk-forward CV
print("\n── Walk-Forward CV (3 folds) ──────────────────────────")
n = len(X_train)
fold_size = n // 4
cv_maes = []

for fold in range(3):
    train_end = n - (3 - fold) * fold_size
    val_start = train_end
    val_end   = val_start + fold_size

    Xf_tr    = X_train[:train_end]
    yf_tr    = y_train_log[:train_end]
    Xf_va    = X_train[val_start:val_end]
    yf_va_log = y_train_log[val_start:val_end]
    yf_va_raw = y_train[val_start:val_end]

    Xf_tr_t = torch.tensor(Xf_tr,     dtype=torch.float32)
    yf_tr_t = torch.tensor(yf_tr,     dtype=torch.float32).unsqueeze(1)
    Xf_va_t = torch.tensor(Xf_va,     dtype=torch.float32)
    yf_va_t = torch.tensor(yf_va_log, dtype=torch.float32).unsqueeze(1)

    fm    = make_model()
    fopt  = torch.optim.Adam(fm.parameters(), lr=3e-3, weight_decay=1e-4)
    floss = nn.HuberLoss(delta=0.5)
    fsched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        fopt, mode='min', factor=0.5, patience=20
    )
    fdl = DataLoader(TensorDataset(Xf_tr_t, yf_tr_t), batch_size=16, shuffle=True, drop_last=True)

    best_vl, best_sd = float('inf'), None
    for ep in range(300):
        fm.train()
        for Xb, yb in fdl:
            fopt.zero_grad()
            loss = floss(fm(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            fopt.step()
        fm.eval()
        with torch.no_grad():
            vl = floss(fm(Xf_va_t), yf_va_t).item()
        fsched.step(vl)
        if vl < best_vl:
            best_vl = vl
            best_sd = {k: v.clone() for k, v in fm.state_dict().items()}

    fm.load_state_dict(best_sd)
    fm.eval()
    with torch.no_grad():
        fold_preds = np.expm1(fm(Xf_va_t).numpy().flatten())
    fold_mae = mean_absolute_error(yf_va_raw, fold_preds)
    cv_maes.append(fold_mae)
    print(f"  Fold {fold+1}  |  train={train_end} days  val={fold_size} days  |  MAE={fold_mae:.5f}")

print(f"  CV MAE:  mean={np.mean(cv_maes):.5f}  std={np.std(cv_maes):.5f}")

# Step 9: Full training
X_train_t = torch.tensor(X_train,     dtype=torch.float32)
y_train_t = torch.tensor(y_train_log, dtype=torch.float32).unsqueeze(1)
X_val_t   = torch.tensor(X_val,       dtype=torch.float32)
y_val_t   = torch.tensor(y_val_log,   dtype=torch.float32).unsqueeze(1)

model     = make_model()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
loss_fn   = nn.HuberLoss(delta=0.5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=25
)
train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True, drop_last=True
)

EPOCHS = 600
best_val_loss, best_state = float('inf'), None

print("\n── Full Training ───────────────────────────────────────")
for epoch in tqdm(range(EPOCHS), desc="Training"):
    model.train()
    epoch_loss = 0.0
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(train_loader)

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_val_t), y_val_t).item()
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
torch.save(best_state, '../models/v2/mlp_pretrained_v2.pth')
print("Weights saved → ../models/v2/mlp_pretrained_v2.pth")

# Step 10: Evaluate
model.eval()
with torch.no_grad():
    mlp_preds = np.expm1(model(X_val_t).numpy().flatten()).clip(0)

# Baselines
ovx_idx = FEATURE_COLS.index('ovx_close')
bl_ovx  = Ridge(alpha=1.0)
bl_ovx.fit(X_train[:, ovx_idx].reshape(-1,1), y_train)
bl_ovx_pred = bl_ovx.predict(X_val[:, ovx_idx].reshape(-1,1))

bl_full = Ridge(alpha=1.0)
bl_full.fit(X_train, y_train)
bl_full_pred = bl_full.predict(X_val)

mlp_mae      = mean_absolute_error(y_val, mlp_preds)
bl_ovx_mae   = mean_absolute_error(y_val, bl_ovx_pred)
bl_full_mae  = mean_absolute_error(y_val, bl_full_pred)

print("\n── Results ─────────────────────────────────────────────")
print(f"  Baseline  OVX only (Ridge):     MAE = {bl_ovx_mae:.5f}")
print(f"  Baseline  all features (Ridge): MAE = {bl_full_mae:.5f}")
print(f"  Model B   MLP (this script):    MAE = {mlp_mae:.5f}")
print(f"\n  CV MAE mean: {np.mean(cv_maes):.5f} | Val MAE: {mlp_mae:.5f}")

# Step 11: Ablation comparison template
print("\n── Ablation Comparison Template ────────────────────────")
print("  Run all three scripts, then fill in this table:")
print()
print(f"  {'Model':<35} {'Features':>8}  {'Val MAE':>8}  {'vs OVX baseline':>15}")
print(f"  {'-'*35} {'-'*8}  {'-'*8}  {'-'*15}")
print(f"  {'A  — 11 raw features':<35} {'11':>8}  {'???':>8}  {'???':>15}")
print(f"  {'B  — 11 raw + 5 engineered (this)':<35} {'16':>8}  {mlp_mae:>8.5f}  {(bl_ovx_mae-mlp_mae)/bl_ovx_mae*100:>14.1f}%")
print(f"  {'C  — all 20 features':<35} {'20':>8}  {'???':>8}  {'???':>15}")
print(f"  {'Baseline  OVX only Ridge':<35} {'1':>8}  {bl_ovx_mae:>8.5f}  {'0.0%':>15}")
print()
print("  Decision rule:")
print("    A < B < C  →  feature engineering is hurting (overfitting)")
print("    A > B, B ≈ C  →  5 engineered features optimal, 20 is too many")
print("    A > B > C  →  more features consistently help (unlikely at n=447)")

# Step 12: Save config for MAML
config = {
    'version'        : 'v2',
    'feature_cols'   : FEATURE_COLS,
    'input_dim'      : INPUT_DIM,
    'target_col'     : TARGET_COL,
    'target_transform': 'log1p',
    'scaler'         : '../models/v2/feature_scaler_v2.pkl',
    'backbone'       : '../models/v2/mlp_pretrained_v2.pth',
    'architecture'   : f'{INPUT_DIM} → 48 BN ReLU D0.2 → 32 BN ReLU D0.15 → 16 BN ReLU → 1',
    'samples_per_feature': round(len(X_train_raw) / INPUT_DIM, 1),
    'engineered_features': {
        'vol_term_structure' : 'oil_vol_5d / oil_vol_20d',
        'ovx_vix_ratio'      : 'ovx_close / vix_close',
        'vol_acceleration'   : 'oil_vol_5d - oil_vol_20d',
        'oil_price_zscore'   : '(oil_close - rolling_mean_20) / rolling_std_20',
        'vix_ovx_spread'     : 'vix_close - ovx_close',
    },
    'cv_mae_mean'    : float(np.mean(cv_maes)),
    'cv_mae_std'     : float(np.std(cv_maes)),
    'val_mae'        : float(mlp_mae),
}
with open('../models/v2/mlp_config_version2.json', 'w') as f:
    json.dump(config, f, indent=2)
print("\nConfig saved → ../models/v2/mlp_config_version2.json")