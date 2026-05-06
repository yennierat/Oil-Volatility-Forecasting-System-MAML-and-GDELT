"""
MAML Training — v1 (15 features, no per-region GDELT aggregates)
Loads pretrained MLP backbone (mlp_pretrained.pth) and meta-trains it on per-task episodes.
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

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 1. Config
FEATURE_COLS = [
    'ovx_close', 'vix_close', 'oil_vol_5d', 'oil_vol_20d',
    'oil_close', 'dxy_close', 'gold_oil_ratio',
    'gs_mean', 'gs_std', 'gs_conflict_pct', 'gs_weighted',
    'tone_mean', 'tone_std', 'n_events', 'mentions_sum',
]
TARGET_COL      = 'oil_fwd_rvol_4h'
INNER_LR        = 0.01
META_LR         = 0.001
INNER_STEPS     = 5
META_EPOCHS     = 200
TASKS_PER_BATCH = 8

# 2. Load data
df = pd.read_csv("../data/dataset_maml_intraday.csv", parse_dates=['date'])

# Step 1 — use the split that gives 150+ test dates
df['split'] = 'train'
df.loc[df['date'].between('2025-06-01', '2025-09-30'), 'split'] = 'val'
df.loc[df['date'] >= '2025-10-01', 'split'] = 'test'

RAW_COLS = [
    'goldstein_scale', 'num_mentions', 'num_sources', 'avg_tone',
    'oil_close', 'oil_vol_5d', 'oil_vol_20d',
    'vix_close', 'ovx_close', 'dxy_close', 'gold_oil_ratio'
]
df = df.dropna(subset=RAW_COLS + [TARGET_COL])
print(f"Loaded {len(df)} rows after dropping NaN")
print(f"Tasks available: {df['maml_task'].nunique()}")
print(f"Split distribution:\n{df['split'].value_counts()}")

# Verify splits
for split in ['train', 'val', 'test']:
    s = df[df['split'] == split].drop_duplicates('date')
    print(f"{split}: OVX mean={s['ovx_close'].mean():.1f} "
          f"max={s['ovx_close'].max():.1f} "
          f"dates={s['date'].nunique()}")

# 3. Aggregate to daily per task
# Same aggregation as MLP pretraining — 1 row per date per task.
# GDELT features are aggregated; market features take first value.
def aggregate_to_daily(df):
    agg = (
        df.groupby(['maml_task', 'split', 'date']).apply(
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
    return agg

print("\nAggregating to daily rows per task...")
df = aggregate_to_daily(df)
print(f"After aggregation: {len(df)} rows ({df['date'].nunique()} unique dates)")

# 4. Refit scaler on train split
train_only = df[df['split'] == 'train']
scaler = StandardScaler()
scaler.fit(train_only[FEATURE_COLS].values.astype(np.float32))
joblib.dump(scaler, '../models/v1/feature_scaler_live.pkl')
print(f"OVX — mean={scaler.mean_[0]:.1f}  std={scaler.scale_[0]:.1f}")

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

model = OilVolatilityMLP()
model.load_state_dict(torch.load('../models/v1/mlp_pretrained.pth'))
print("Pre-trained weights loaded")

# 5. Task sampler
class MAMLTaskSampler:
    def __init__(self, df, feature_cols, target_col, split='train'):
        self.df           = df[df['split'] == split].copy()
        self.feature_cols = feature_cols
        self.target_col   = target_col

        self.df[feature_cols] = scaler.transform(
            self.df[feature_cols].values.astype(np.float32)
        )

        # Build date lookup per task
        self.task_dates = {
            task: sorted(group['date'].unique())
            for task, group in self.df.groupby('maml_task')
        }

        # Inverse frequency weights so rare tasks are fairly sampled
        task_counts       = self.df['maml_task'].value_counts()
        inv_freq          = 1.0 / task_counts
        self.task_weights = (inv_freq / inv_freq.sum()).to_dict()

        print(f"\nSampler ready [{split}]: "
              f"{len(self.task_dates)} tasks | {len(self.df)} rows")

    def sample_task(self):
        tasks   = list(self.task_weights.keys())
        weights = [self.task_weights[t] for t in tasks]
        return np.random.choice(tasks, p=weights)

    def sample_episode(self, task=None,
                       n_support_dates=5, n_query_dates=3):
        if task is None:
            task = self.sample_task()

        dates = list(self.task_dates[task])

        # Fallback for rare tasks
        if len(dates) < (n_support_dates + n_query_dates):
            n_support_dates = max(2, len(dates) // 2)
            n_query_dates   = max(1, len(dates) - n_support_dates)

        # Split by DATE — never by row
        shuffled      = dates.copy()
        np.random.shuffle(shuffled)
        support_dates = set(shuffled[:n_support_dates])
        query_dates   = set(shuffled[n_support_dates:
                                     n_support_dates + n_query_dates])

        # Safety guarantee — no date leak
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

        # Log transform — oil_fwd_rvol_4h skew = 5.94
        # Must match what MLP was trained on (log1p target)
        sup_y = torch.log1p(sup_y)
        qry_y = torch.log1p(qry_y)

        return sup_X, sup_y, qry_X, qry_y, task

# Instantiate samplers
train_sampler = MAMLTaskSampler(df, FEATURE_COLS, TARGET_COL, 'train')
val_sampler   = MAMLTaskSampler(df, FEATURE_COLS, TARGET_COL, 'val')
test_sampler  = MAMLTaskSampler(df, FEATURE_COLS, TARGET_COL, 'test')

# 6. Quick sanity check
sup_X, sup_y, qry_X, qry_y, task = train_sampler.sample_episode()
print("\nSample episode:")
print(f"  Task:      {task}")
print(f"  Support X: {sup_X.shape}")
print(f"  Query X:   {qry_X.shape}")
print(f"  NaN in support: {torch.isnan(sup_X).any()}")
print(f"  NaN in query:   {torch.isnan(qry_X).any()}")

# 7. MAML training loop
loss_fn        = nn.HuberLoss(delta=0.5)
meta_optimizer = torch.optim.Adam(model.parameters(), lr=META_LR)
best_meta_loss = float('inf')

for epoch in tqdm(range(META_EPOCHS), desc="MAML Training"):
    model.train()
    meta_optimizer.zero_grad()
    total_query_loss = 0.0

    for _ in range(TASKS_PER_BATCH):
        sup_X, sup_y, qry_X, qry_y, task = \
            train_sampler.sample_episode()

        # Inner loop — adapt copy to this task
        # Dropout OFF during adaptation (tiny support set, noise hurts)
        adapted   = deepcopy(model)
        adapted.eval()
        adapt_opt = torch.optim.SGD(
            adapted.parameters(), lr=INNER_LR)

        for _ in range(INNER_STEPS):
            adapt_opt.zero_grad()
            loss = loss_fn(adapted(sup_X), sup_y)
            loss.backward()
            adapt_opt.step()

        # Outer loop — dropout ON for regularization on query set
        adapted.train()
        qry_loss = loss_fn(adapted(qry_X), qry_y)
        qry_loss.backward()

        # Copy adapted gradients back to meta-model
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

    # Meta-validation every 10 epochs on held-out test tasks
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
                torch.save(model.state_dict(), '../models/v1/maml_trained.pth')
                tqdm.write(f"  ✓ Best saved at epoch {epoch+1} "
                           f"(val_loss={meta_val_loss:.5f})")

            tqdm.write(f"Epoch {epoch+1:>3} | "
                       f"Train loss: {avg_loss:.5f} | "
                       f"Val loss: {meta_val_loss:.5f} | "
                       f"Best: {best_meta_loss:.5f}")

# 8. Evaluate per task
model.load_state_dict(torch.load('../models/v1/maml_trained.pth'))
results = {}

for task in test_sampler.task_dates.keys():
    try:
        sup_X, sup_y, qry_X, qry_y, _ = \
            test_sampler.sample_episode(task=task)

        adapted   = deepcopy(model)
        adapted.eval()
        adapt_opt = torch.optim.SGD(
            adapted.parameters(), lr=INNER_LR)

        for _ in range(INNER_STEPS):
            adapt_opt.zero_grad()
            loss = loss_fn(adapted(sup_X), sup_y)
            loss.backward()
            adapt_opt.step()

        with torch.no_grad():
            # Inverse log transform to get original scale MAE
            preds  = np.expm1(adapted(qry_X).numpy().flatten())
            actual = np.expm1(qry_y.numpy().flatten())
        results[task] = mean_absolute_error(actual, preds)

    except Exception as e:
        print(f"Skipped {task}: {e}")

print(f"\n{'='*55}")
print(f"{'Task':<45} {'MAE':>8}")
print(f"{'='*55}")
for task, mae in sorted(results.items(), key=lambda x: x[1]):
    print(f"{task:<45} {mae:>8.5f}")
print(f"{'='*55}")
print(f"{'Mean MAML MAE':<45} "
      f"{np.mean(list(results.values())):>8.5f}")