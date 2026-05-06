"""
MLP Pre-training for MAML Foundation
=================================================
- 11 raw features, daily aggregated (no engineered features)
- Matches exactly what MAML intraday tasks will use
- No feature mismatch between pretraining and MAML
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

# Step 1: Load data
df = pd.read_csv("../data/dataset_maml_daily.csv")

RAW_COLS = [
    'goldstein_scale', 'num_mentions', 'num_sources', 'avg_tone',
    'oil_close', 'oil_vol_5d', 'oil_vol_20d',
    'vix_close', 'ovx_close', 'dxy_close', 'gold_oil_ratio'
]
TARGET_COL = 'oil_fwd_rvol_1d'

before = len(df)
df = df.dropna(subset=RAW_COLS + [TARGET_COL])
print(f"Dropped {before - len(df)} NaN rows. Remaining: {len(df)}")

# Step 2: Aggregate to daily
# ~97 event rows share the same target per date — aggregate to 1 row per date before training.
daily = (
    df.groupby('date').apply(
        lambda g: pd.Series({
            'gs_mean'         : g['goldstein_scale'].mean(),
            'gs_std'          : g['goldstein_scale'].std(ddof=0),
            'gs_conflict_pct' : (g['goldstein_scale'] < 0).mean(),
            'gs_weighted'     : np.average(
                                    g['goldstein_scale'],
                                    weights=g['num_mentions'].clip(lower=1)
                                ),
            'tone_mean'       : g['avg_tone'].mean(),
            'tone_std'        : g['avg_tone'].std(ddof=0),
            'n_events'        : len(g),
            'mentions_sum'    : g['num_mentions'].sum(),
            'oil_close'       : g['oil_close'].iloc[0],
            'oil_vol_5d'      : g['oil_vol_5d'].iloc[0],
            'oil_vol_20d'     : g['oil_vol_20d'].iloc[0],
            'vix_close'       : g['vix_close'].iloc[0],
            'ovx_close'       : g['ovx_close'].iloc[0],
            'dxy_close'       : g['dxy_close'].iloc[0],
            'gold_oil_ratio'  : g['gold_oil_ratio'].iloc[0],
            TARGET_COL        : g[TARGET_COL].iloc[0],
        })
    )
    .reset_index()
    .sort_values('date')
    .reset_index(drop=True)
)
print(f"Daily dataset: {len(daily)} unique dates")

# Step 3: Feature set — 15 features, no engineering
FEATURE_COLS = [
    # Market features (highest signal)
    'ovx_close', 'vix_close', 'oil_vol_5d', 'oil_vol_20d',
    'oil_close', 'dxy_close', 'gold_oil_ratio',
    # GDELT daily aggregates
    'gs_mean', 'gs_std', 'gs_conflict_pct', 'gs_weighted',
    'tone_mean', 'tone_std', 'n_events', 'mentions_sum',
]
INPUT_DIM = len(FEATURE_COLS)

train_df = daily[daily['date'] < '2023-10-01'].copy()
val_df   = daily[daily['date'] >= '2023-10-01'].copy()

X_train_raw = train_df[FEATURE_COLS].values.astype(np.float32)
y_train     = train_df[TARGET_COL].values.astype(np.float32)
X_val_raw   = val_df[FEATURE_COLS].values.astype(np.float32)
y_val       = val_df[TARGET_COL].values.astype(np.float32)

print(f"Train: {len(X_train_raw)} days | Val: {len(X_val_raw)} days")
print(f"Features: {INPUT_DIM} | Samples per feature: {len(X_train_raw)/INPUT_DIM:.1f}")

# Step 4: Log transform target
# Skew = 1.95. Log1p makes loss proportional, prevents the model from just predicting the mean.
y_train_log = np.log1p(y_train)
y_val_log   = np.log1p(y_val)

# Step 5: Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val   = scaler.transform(X_val_raw)
joblib.dump(scaler, '../models/v1/feature_scaler.pkl')

# Step 6: Model — 15 → 48 → 32 → 16 → 1 with BatchNorm, sized for ~430 training days
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

print(f"Model parameters: {count_params(make_model()):,}")

# Step 7: Walk-forward CV
print("\n── Walk-Forward CV (3 folds) ──────────────────────────")
n = len(X_train)
fold_size = n // 4
cv_maes = []

for fold in range(3):
    train_end  = n - (3 - fold) * fold_size
    val_start  = train_end
    val_end    = val_start + fold_size

    Xf_tr     = X_train[:train_end]
    yf_tr     = y_train_log[:train_end]
    Xf_va     = X_train[val_start:val_end]
    yf_va_log = y_train_log[val_start:val_end]
    yf_va_raw = y_train[val_start:val_end]

    Xf_tr_t = torch.tensor(Xf_tr,     dtype=torch.float32)
    yf_tr_t = torch.tensor(yf_tr,     dtype=torch.float32).unsqueeze(1)
    Xf_va_t = torch.tensor(Xf_va,     dtype=torch.float32)
    yf_va_t = torch.tensor(yf_va_log, dtype=torch.float32).unsqueeze(1)

    fm     = make_model()
    fopt   = torch.optim.Adam(fm.parameters(), lr=3e-3, weight_decay=1e-4)
    floss  = nn.HuberLoss(delta=0.5)
    fsched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        fopt, mode='min', factor=0.5, patience=20
    )
    fdl = DataLoader(
        TensorDataset(Xf_tr_t, yf_tr_t),
        batch_size=16, shuffle=True, drop_last=True
    )

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
    print(f"  Fold {fold+1}  |  train={train_end} days  val={fold_size} days  "
          f"|  MAE={fold_mae:.5f}")

print(f"  CV MAE:  mean={np.mean(cv_maes):.5f}  std={np.std(cv_maes):.5f}")

# Step 8: Full training
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
    TensorDataset(X_train_t, y_train_t),
    batch_size=16, shuffle=True, drop_last=True
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
torch.save(best_state, '../models/v1/mlp_pretrained.pth')
print("Weights saved → ../models/v1/mlp_pretrained.pth")

# Step 9: Evaluate
model.eval()
with torch.no_grad():
    mlp_preds = np.expm1(model(X_val_t).numpy().flatten()).clip(0)

ovx_idx     = FEATURE_COLS.index('ovx_close')
bl_ovx      = Ridge(alpha=1.0)
bl_ovx.fit(X_train[:, ovx_idx].reshape(-1, 1), y_train)
bl_ovx_pred = bl_ovx.predict(X_val[:, ovx_idx].reshape(-1, 1))

bl_full     = Ridge(alpha=1.0)
bl_full.fit(X_train, y_train)
bl_full_pred = bl_full.predict(X_val)

mlp_mae    = mean_absolute_error(y_val, mlp_preds)
bl_ovx_mae = mean_absolute_error(y_val, bl_ovx_pred)
bl_full_mae = mean_absolute_error(y_val, bl_full_pred)

print("\n── Results ─────────────────────────────────────────────")
print(f"  Baseline  OVX only (Ridge):     MAE = {bl_ovx_mae:.5f}")
print(f"  Baseline  all features (Ridge): MAE = {bl_full_mae:.5f}")
print(f"  MLP Option A:                   MAE = {mlp_mae:.5f}")
print(f"\n  CV MAE mean: {np.mean(cv_maes):.5f} | Val MAE: {mlp_mae:.5f}")

if mlp_mae < bl_ovx_mae:
    print("\n  ✓ MLP beats OVX baseline — safe to use as MAML backbone")
else:
    print("\n  ✗ MLP does not beat OVX baseline")
    print("    → The backbone will still transfer useful representations to MAML")
    print("    → MAML fine-tuning may recover performance per task")

# Step 10: Save config for MAML
config = {
    'version'          : 'v1',
    'feature_cols'     : FEATURE_COLS,
    'input_dim'        : INPUT_DIM,
    'target_col'       : TARGET_COL,
    'target_transform' : 'log1p',
    'scaler'           : '../models/v1/feature_scaler.pkl',
    'backbone'         : '../models/v1/mlp_pretrained.pth',
    'architecture'     : f'{INPUT_DIM} → 48 BN ReLU D0.2 → 32 BN ReLU D0.15 → 16 BN ReLU → 1',
    'cv_mae_mean'      : float(np.mean(cv_maes)),
    'cv_mae_std'       : float(np.std(cv_maes)),
    'val_mae'          : float(mlp_mae),
    'note'             : 'v1: 15 features, no engineered features, MAML-compatible'
}
with open('../models/v1/mlp_config_version1.json', 'w') as f:
    json.dump(config, f, indent=2)
print("Config saved → ../models/v1/mlp_config_version1.json")