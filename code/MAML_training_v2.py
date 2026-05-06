"""
MAML Training — V2 (23 features: 15 v1 features + 8 per-region GDELT aggregates)
Per-region: me_* (Middle East) and oi_* (Oil Producers).

Inputs:  ../data/dataset_maml_intraday_v2.csv, ../models/v2/mlp_pretrained_v2.pth, ../models/v2/feature_scaler_v2.pkl
Outputs: ../models/v2/maml_trained_v2.pth

Run:
    python maml_training_v2.py
"""

import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
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

# 1. Config
FEATURE_COLS = [
    # Market features (7)
    'ovx_close', 'vix_close', 'oil_vol_5d', 'oil_vol_20d',
    'oil_close', 'dxy_close', 'gold_oil_ratio',
    # GDELT global aggregates (8)
    'gs_mean', 'gs_std', 'gs_conflict_pct', 'gs_weighted',
    'tone_mean', 'tone_std', 'n_events', 'mentions_sum',
    # Per-region features — Middle East (4)
    'me_gs_mean', 'me_conflict_pct', 'me_n_events', 'me_tone_mean',
    # Per-region features — Oil Producers (4)
    'oi_gs_mean', 'oi_conflict_pct', 'oi_n_events', 'oi_tone_mean',
]

INPUT_DIM       = len(FEATURE_COLS)   # 23
TARGET_COL      = 'oil_fwd_rvol_4h'
INNER_LR        = 0.01
META_LR         = 0.001
INNER_STEPS     = 5
META_EPOCHS     = 200
TASKS_PER_BATCH = 8

assert INPUT_DIM == 23, f"Expected 23 features, got {INPUT_DIM}"
print(f"Feature count: {INPUT_DIM}")

# 2. Load data
df = pd.read_csv("../data/dataset_maml_intraday_v2.csv", parse_dates=['date'])

# Columns needed for NaN check
RAW_COLS = [
    'goldstein_scale', 'num_mentions', 'num_sources', 'avg_tone',
    'oil_close', 'oil_vol_5d', 'oil_vol_20d',
    'vix_close', 'ovx_close', 'dxy_close', 'gold_oil_ratio',
    'me_gs_mean', 'me_conflict_pct', 'me_n_events', 'me_tone_mean',
    'oi_gs_mean', 'oi_conflict_pct', 'oi_n_events', 'oi_tone_mean',
]

before = len(df)
df = df.dropna(subset=RAW_COLS + [TARGET_COL])
print(f"Dropped {before - len(df)} NaN rows. Remaining: {len(df)}")
print(f"Tasks available: {df['maml_task'].nunique()}")

# Assign splits
df['split'] = 'train'
df.loc[df['date'].between('2025-06-01', '2025-09-30'), 'split'] = 'val'
df.loc[df['date'] >= '2025-10-01', 'split'] = 'test'

print(f"Split distribution:\n{df['split'].value_counts()}")

# Verify splits
for split in ['train', 'val', 'test']:
    s = df[df['split'] == split].drop_duplicates('date')
    print(f"  {split}: OVX mean={s['ovx_close'].mean():.1f}  "
          f"max={s['ovx_close'].max():.1f}  dates={s['date'].nunique()}")

# 3. Aggregate to daily per task
def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates event-level rows to 1 row per (maml_task, split, date).
    GDELT raw features are aggregated; market and per-region features
    take the first value (already aggregated per date in dataset builder).
    """
    agg = (
        df.groupby(['maml_task', 'split', 'date']).apply(
            lambda g: pd.Series({
                # GDELT global aggregates
                'gs_mean'         : g['goldstein_scale'].mean(),
                'gs_std'          : g['goldstein_scale'].std(ddof=0),
                'gs_conflict_pct' : (g['goldstein_scale'] < 0).mean(),
                'gs_weighted'     : np.average(
                                        g['goldstein_scale'],
                                        weights=g['num_mentions'].clip(lower=1)
                                    ),
                'tone_mean'       : g['avg_tone'].mean(),
                'tone_std'        : g['avg_tone'].std(ddof=0),
                'n_events'        : float(len(g)),
                'mentions_sum'    : g['num_mentions'].sum(),
                # Market features — constant per day
                'oil_close'       : g['oil_close'].iloc[0],
                'oil_vol_5d'      : g['oil_vol_5d'].iloc[0],
                'oil_vol_20d'     : g['oil_vol_20d'].iloc[0],
                'vix_close'       : g['vix_close'].iloc[0],
                'ovx_close'       : g['ovx_close'].iloc[0],
                'dxy_close'       : g['dxy_close'].iloc[0],
                'gold_oil_ratio'  : g['gold_oil_ratio'].iloc[0],
                # Per-region features — already aggregated per date
                'me_gs_mean'      : g['me_gs_mean'].iloc[0],
                'me_conflict_pct' : g['me_conflict_pct'].iloc[0],
                'me_n_events'     : g['me_n_events'].iloc[0],
                'me_tone_mean'    : g['me_tone_mean'].iloc[0],
                'oi_gs_mean'      : g['oi_gs_mean'].iloc[0],
                'oi_conflict_pct' : g['oi_conflict_pct'].iloc[0],
                'oi_n_events'     : g['oi_n_events'].iloc[0],
                'oi_tone_mean'    : g['oi_tone_mean'].iloc[0],
                # Target
                TARGET_COL        : g[TARGET_COL].iloc[0],
            }), include_groups=False
        )
        .reset_index()
        .sort_values('date')
        .reset_index(drop=True)
    )
    return agg

print("\nAggregating to daily rows per task...")
df = aggregate_to_daily(df)
print(f"After aggregation: {len(df)} rows ({df['date'].nunique()} unique dates)")

# Verify all feature columns exist after aggregation
missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns after aggregation: {missing}")
print(f"All {INPUT_DIM} feature columns present ✓")

# 4. Refit scaler on train split
train_only = df[df['split'] == 'train']
scaler = StandardScaler()
scaler.fit(train_only[FEATURE_COLS].values.astype(np.float32))
joblib.dump(scaler, '../models/v2/feature_scaler_v2.pkl')
print("\nScaler saved → ../models/v2/feature_scaler_v2.pkl")
print(f"  OVX — mean={scaler.mean_[0]:.1f}  std={scaler.scale_[0]:.1f}")
print(f"  me_gs_mean index={FEATURE_COLS.index('me_gs_mean')} "
      f"mean={scaler.mean_[FEATURE_COLS.index('me_gs_mean')]:.3f}")

# 5. Model
class OilVolatilityMLP(nn.Module):
    """4-layer MLP with BatchNorm + Dropout. Input dim is 23 for v2 (was 15 for v1)."""
    def __init__(self, input_dim: int = 23):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Load pretrained v2 MLP backbone
model = OilVolatilityMLP(input_dim=INPUT_DIM)
state = torch.load('../models/v2/mlp_pretrained_v2.pth', map_location='cpu')
model.load_state_dict(state)
print("\nLoaded ../models/v2/mlp_pretrained_v2.pth")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# 6. Task sampler
class MAMLTaskSampler:
    def __init__(self, df: pd.DataFrame, feature_cols: list,
                 target_col: str, split: str = 'train'):
        self.df           = df[df['split'] == split].copy()
        self.feature_cols = feature_cols
        self.target_col   = target_col

        # Scale features
        self.df[feature_cols] = scaler.transform(
            self.df[feature_cols].values.astype(np.float32)
        )

        # Build date lookup per task
        self.task_dates = {
            task: sorted(group['date'].unique())
            for task, group in self.df.groupby('maml_task')
        }

        # Inverse frequency weights — rare tasks sampled fairly
        task_counts       = self.df['maml_task'].value_counts()
        inv_freq          = 1.0 / task_counts
        self.task_weights = (inv_freq / inv_freq.sum()).to_dict()

        print(f"Sampler ready [{split}]: "
              f"{len(self.task_dates)} tasks | {len(self.df)} rows")

    def sample_task(self) -> str:
        tasks   = list(self.task_weights.keys())
        weights = [self.task_weights[t] for t in tasks]
        return np.random.choice(tasks, p=weights)

    def sample_episode(self, task: str = None,
                       n_support_dates: int = 5,
                       n_query_dates: int = 3):
        if task is None:
            task = self.sample_task()

        dates = list(self.task_dates[task])

        # Fallback for rare tasks
        if len(dates) < (n_support_dates + n_query_dates):
            n_support_dates = max(2, len(dates) // 2)
            n_query_dates   = max(1, len(dates) - n_support_dates)

        # Split by DATE — never by row to prevent leakage
        shuffled      = dates.copy()
        np.random.shuffle(shuffled)
        support_dates = set(shuffled[:n_support_dates])
        query_dates   = set(shuffled[n_support_dates:
                                     n_support_dates + n_query_dates])

        assert len(support_dates & query_dates) == 0, \
            f"DATE LEAK in task {task}!"

        task_df = self.df[self.df['maml_task'] == task]
        support = task_df[task_df['date'].isin(support_dates)]
        query   = task_df[task_df['date'].isin(query_dates)]

        sup_X = torch.tensor(
            support[self.feature_cols].values.astype(np.float32))
        sup_y = torch.tensor(
            support[self.target_col].values.astype(np.float32)
        ).unsqueeze(1)
        qry_X = torch.tensor(
            query[self.feature_cols].values.astype(np.float32))
        qry_y = torch.tensor(
            query[self.target_col].values.astype(np.float32)
        ).unsqueeze(1)

        # Log transform — matches MLP pretraining
        sup_y = torch.log1p(sup_y)
        qry_y = torch.log1p(qry_y)

        return sup_X, sup_y, qry_X, qry_y, task

# 7. Instantiate samplers
print()
train_sampler = MAMLTaskSampler(df, FEATURE_COLS, TARGET_COL, 'train')
val_sampler   = MAMLTaskSampler(df, FEATURE_COLS, TARGET_COL, 'val')
test_sampler  = MAMLTaskSampler(df, FEATURE_COLS, TARGET_COL, 'test')

# Sanity check
sup_X, sup_y, qry_X, qry_y, task = train_sampler.sample_episode()
print("\nSample episode:")
print(f"  Task:           {task}")
print(f"  Support X:      {sup_X.shape}  (expected: [5, 23])")
print(f"  Query X:        {qry_X.shape}  (expected: [3, 23])")
print(f"  NaN in support: {torch.isnan(sup_X).any()}")
print(f"  NaN in query:   {torch.isnan(qry_X).any()}")

assert sup_X.shape[1] == INPUT_DIM, \
    f"Feature dim mismatch: got {sup_X.shape[1]}, expected {INPUT_DIM}"
print(f"  Feature dim check: ✓ ({INPUT_DIM})")

# 8. MAML training loop
loss_fn        = nn.HuberLoss(delta=0.5)
meta_optimizer = torch.optim.Adam(model.parameters(), lr=META_LR)
best_meta_loss = float('inf')

print(f"\n{'='*60}")
print(f"  MAML Training — {META_EPOCHS} epochs, {TASKS_PER_BATCH} tasks/batch")
print(f"  Inner LR: {INNER_LR}  |  Meta LR: {META_LR}  |  Inner steps: {INNER_STEPS}")
print(f"{'='*60}")

for epoch in tqdm(range(META_EPOCHS), desc="MAML Training"):
    model.train()
    meta_optimizer.zero_grad()
    total_query_loss = 0.0

    for _ in range(TASKS_PER_BATCH):
        sup_X, sup_y, qry_X, qry_y, _ = train_sampler.sample_episode()

        # Inner loop — adapt copy to this task
        # eval() during adaptation: dropout off on tiny support set
        adapted   = deepcopy(model)
        adapted.eval()
        adapt_opt = torch.optim.SGD(adapted.parameters(), lr=INNER_LR)

        for _ in range(INNER_STEPS):
            adapt_opt.zero_grad()
            loss = loss_fn(adapted(sup_X), sup_y)
            loss.backward()
            adapt_opt.step()

        # Outer loop — train() for dropout regularization on query
        adapted.train()
        qry_loss = loss_fn(adapted(qry_X), qry_y)
        qry_loss.backward()

        # Accumulate adapted gradients into meta-model
        for p_meta, p_adapt in zip(model.parameters(),
                                    adapted.parameters()):
            if p_adapt.grad is not None:
                if p_meta.grad is None:
                    p_meta.grad = p_adapt.grad.clone() / TASKS_PER_BATCH
                else:
                    p_meta.grad += p_adapt.grad.clone() / TASKS_PER_BATCH

        total_query_loss += qry_loss.item()

    meta_optimizer.step()
    avg_loss = total_query_loss / TASKS_PER_BATCH

    # Meta-validation every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        val_losses = []

        for val_task in list(val_sampler.task_dates.keys())[:5]:
            try:
                v_sup_X, v_sup_y, v_qry_X, v_qry_y, _ = \
                    val_sampler.sample_episode(task=val_task)

                v_adapted = deepcopy(model)
                v_adapted.eval()
                v_opt = torch.optim.SGD(
                    v_adapted.parameters(), lr=INNER_LR)

                for _ in range(INNER_STEPS):
                    v_opt.zero_grad()
                    v_loss = loss_fn(v_adapted(v_sup_X), v_sup_y)
                    v_loss.backward()
                    v_opt.step()

                with torch.no_grad():
                    val_losses.append(
                        loss_fn(v_adapted(v_qry_X), v_qry_y).item()
                    )
            except Exception:
                pass

        if val_losses:
            meta_val_loss = np.mean(val_losses)
            if meta_val_loss < best_meta_loss:
                best_meta_loss = meta_val_loss
                torch.save(model.state_dict(), '../models/v2/maml_trained_v2.pth')
                tqdm.write(f"  ✓ Best saved at epoch {epoch+1} "
                           f"(val_loss={meta_val_loss:.5f})")

            tqdm.write(f"  Epoch {epoch+1:>3} | "
                       f"Train: {avg_loss:.5f} | "
                       f"Val: {meta_val_loss:.5f} | "
                       f"Best: {best_meta_loss:.5f}")

print(f"\nTraining complete. Best val loss: {best_meta_loss:.5f}")
print("Weights saved → ../models/v2/maml_trained_v2.pth")

# 9. Evaluate per task
print(f"\n{'='*60}")
print("  Evaluating on test set...")
print(f"{'='*60}")

model.load_state_dict(torch.load('../models/v2/maml_trained_v2.pth', map_location='cpu'))
results = {}

for task in test_sampler.task_dates.keys():
    try:
        sup_X, sup_y, qry_X, qry_y, _ = \
            test_sampler.sample_episode(task=task)

        adapted   = deepcopy(model)
        adapted.eval()
        adapt_opt = torch.optim.SGD(adapted.parameters(), lr=INNER_LR)

        for _ in range(INNER_STEPS):
            adapt_opt.zero_grad()
            loss = loss_fn(adapted(sup_X), sup_y)
            loss.backward()
            adapt_opt.step()

        with torch.no_grad():
            preds  = np.expm1(adapted(qry_X).numpy().flatten()).clip(0)
            actual = np.expm1(qry_y.numpy().flatten())

        results[task] = mean_absolute_error(actual, preds)

    except Exception as e:
        print(f"  Skipped {task}: {e}")

print(f"\n{'='*60}")
print(f"{'Task':<45} {'MAE':>8}")
print(f"{'='*60}")
for task, mae in sorted(results.items(), key=lambda x: x[1]):
    print(f"  {task:<43} {mae:>8.5f}")
print(f"{'='*60}")
mean_mae = np.mean(list(results.values()))
print(f"  {'Mean MAML MAE':<43} {mean_mae:>8.5f}")
print(f"{'='*60}")

print(f"""
Summary:
  Tasks evaluated : {len(results)}
  Mean MAE        : {mean_mae:.5f}
  Model saved     : ../models/v2/maml_trained_v2.pth
  Scaler saved    : ../models/v2/feature_scaler_v2.pkl
  Features        : {INPUT_DIM}
""")