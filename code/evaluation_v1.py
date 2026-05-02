"""
V1 Evaluation — comprehensive offline evaluation for the V1 MAML pipeline (15 features:
7 market + 8 GDELT global aggregates; no per-region or engineered features).

Reports random- AND chronological-sampling MAE, pooled Diebold-Mariano tests,
bootstrap CIs, permutation feature importance, and a temporally-local k-shot ablation.
GARCH uses sqrt(252) to match the daily-annualised scale of the intraday target
(sqrt(252*23) would overstate GARCH by ~sqrt(23)).

Inputs:
  dataset_maml_intraday (4).csv, mlp_pretrained.pth, maml_trained.pth,
  feature_scaler.pkl  OR  feature_scaler_live.pkl

Outputs (in eval_outputs_v1/):
  results_v1_per_task_random.csv, results_v1_per_task_chronological.csv,
  results_v1_dm_per_task.csv, results_v1_kshot.csv,
  results_v1_feat_importance.csv, results_v1_summary.json
"""

import json
import random
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from arch import arch_model
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Config (V1-specific paths — adjust as needed)
DATA_PATH   = "../data/dataset_maml_intraday.csv"
MLP_PATH    = "../models/v1/mlp_pretrained.pth"
MAML_PATH   = "../models/v1/maml_trained.pth"
# Match whichever scaler was used when V1 MAML was trained.
# Training saves 'feature_scaler_live.pkl', so try that first; fall back to feature_scaler.pkl.
SCALER_PATH = "../models/v1/feature_scaler_live.pkl"
OUT_DIR     = Path("eval_outputs_v1")
OUT_DIR.mkdir(exist_ok=True)

RAW_COLS = [
    'goldstein_scale', 'num_mentions', 'num_sources', 'avg_tone',
    'oil_close', 'oil_vol_5d', 'oil_vol_20d',
    'vix_close', 'ovx_close', 'dxy_close', 'gold_oil_ratio'
]
# V1 = 15 features: 7 market + 8 GDELT global. No per-region features.
FEATURE_COLS = [
    'ovx_close', 'vix_close', 'oil_vol_5d', 'oil_vol_20d',
    'oil_close', 'dxy_close', 'gold_oil_ratio',
    'gs_mean', 'gs_std', 'gs_conflict_pct', 'gs_weighted',
    'tone_mean', 'tone_std', 'n_events', 'mentions_sum',
]
TARGET_COL  = 'oil_fwd_rvol_4h'
INPUT_DIM   = len(FEATURE_COLS)   # 15 for V1
INNER_LR    = 0.01
INNER_STEPS = 5
N_RUNS      = 5
N_SUPPORT   = 5
N_QUERY     = 3
N_BOOTSTRAP = 2000

# Model definition — must match V1 training architecture exactly
class OilVolatilityMLP(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48), nn.BatchNorm1d(48), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(48, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


# Load and aggregate to one row per (task, date)
print("=" * 72)
print("Loading V1 data...")
print("=" * 72)
df = pd.read_csv(DATA_PATH, parse_dates=['date'])
before = len(df)
df = df.dropna(subset=RAW_COLS + [TARGET_COL])
print(f"Dropped {before - len(df)} NaN rows. Remaining: {len(df):,}")
print(f"Tasks: {df['maml_task'].nunique()}")

df = (
    df.groupby(['maml_task', 'split', 'date']).apply(
        lambda g: pd.Series({
            'gs_mean'         : g['goldstein_scale'].mean(),
            'gs_std'          : g['goldstein_scale'].std(ddof=0),
            'gs_conflict_pct' : (g['goldstein_scale'] < 0).mean(),
            'gs_weighted'     : np.average(
                                    g['goldstein_scale'],
                                    weights=g['num_mentions'].clip(lower=1)),
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
        }), include_groups=False
    )
    .reset_index()
    .sort_values('date')
    .reset_index(drop=True)
)
print(f"After aggregation: {len(df):,} rows ({df['date'].nunique()} unique dates)\n")


# Load scaler — try both common filenames for robustness
try:
    scaler = joblib.load(SCALER_PATH)
    print(f"Scaler loaded: {SCALER_PATH}")
except FileNotFoundError:
    fallback = "../models/v1/feature_scaler.pkl"
    print(f"{SCALER_PATH} not found, trying {fallback}...")
    scaler = joblib.load(fallback)
    print(f"Scaler loaded: {fallback}")


# Episode sampler — supports both random and chronological sampling
class MAMLTaskSampler:
    def __init__(self, df, feature_cols, target_col, split='test'):
        self.df           = df[df['split'] == split].copy()
        self.feature_cols = feature_cols
        self.target_col   = target_col
        self.df[feature_cols] = scaler.transform(
            self.df[feature_cols].values.astype(np.float32)
        )
        self.task_dates = {
            task: sorted(group['date'].unique())
            for task, group in self.df.groupby('maml_task')
        }
        print(f"Sampler ready [{split}]: "
              f"{len(self.task_dates)} tasks | {len(self.df):,} rows")

    def _slice(self, task, support_dates, query_dates):
        task_df = self.df[self.df['maml_task'] == task]
        support = task_df[task_df['date'].isin(support_dates)]
        query   = task_df[task_df['date'].isin(query_dates)]
        sup_X = torch.tensor(support[self.feature_cols].values.astype(np.float32))
        sup_y = torch.tensor(support[self.target_col].values.astype(np.float32)
                             ).unsqueeze(1)
        qry_X = torch.tensor(query[self.feature_cols].values.astype(np.float32))
        qry_y = torch.tensor(query[self.target_col].values.astype(np.float32)
                             ).unsqueeze(1)
        sup_y = torch.log1p(sup_y)
        qry_y = torch.log1p(qry_y)
        return sup_X, sup_y, qry_X, qry_y, query['date'].values

    def sample_episode(self, task, n_support=N_SUPPORT, n_query=N_QUERY,
                       run_seed=None, mode='random'):
        dates = list(self.task_dates[task])
        if len(dates) < (n_support + n_query):
            n_support = max(2, len(dates) // 2)
            n_query   = max(1, len(dates) - n_support)

        if mode == 'random':
            rng = np.random.RandomState(run_seed)
            shuffled = dates.copy()
            rng.shuffle(shuffled)
            support_dates = set(shuffled[:n_support])
            query_dates   = set(shuffled[n_support:n_support + n_query])

        elif mode == 'chronological':
            # Support = N most recent dates BEFORE cutoff
            # Query   = N dates strictly AFTER cutoff
            rng = np.random.RandomState(run_seed)
            min_idx = n_support
            max_idx = len(dates) - n_query
            if max_idx <= min_idx:
                cutoff = min_idx
            else:
                cutoff = rng.randint(min_idx, max_idx + 1)
            support_dates = set(dates[cutoff - n_support : cutoff])
            query_dates   = set(dates[cutoff : cutoff + n_query])

        else:
            raise ValueError(f"Unknown mode: {mode}")

        assert len(support_dates & query_dates) == 0, \
            f"DATE LEAK in task {task}!"
        return self._slice(task, support_dates, query_dates)


# Inner-loop adaptation
def adapt(model, sup_X, sup_y, inner_lr=INNER_LR, inner_steps=INNER_STEPS):
    adapted   = deepcopy(model)
    loss_fn   = nn.HuberLoss(delta=0.5)
    optimizer = torch.optim.SGD(adapted.parameters(), lr=inner_lr)
    adapted.eval()
    for _ in range(inner_steps):
        optimizer.zero_grad()
        loss = loss_fn(adapted(sup_X), sup_y)
        loss.backward()
        optimizer.step()
    return adapted


# Rolling GARCH(1,1) baseline.
# GARCH(1,1) on daily log returns gives a daily conditional vol forecast.
# Multiplying by sqrt(252) annualises it onto the same scale as oil_fwd_rvol_4h
# (already annualised via sqrt(252*23) at construction time). Both are annualised
# volatilities — they differ in the horizon measured over, not in scale.
# Using sqrt(252*23) here would overstate GARCH by sqrt(23) ≈ 4.8x.
print("Fitting rolling GARCH(1,1) per test date...")
all_prices = (
    df.drop_duplicates('date')
    .sort_values('date')[['date', 'oil_close']]
    .reset_index(drop=True)
)
test_dates = sorted(df[df['split'] == 'test']['date'].unique())

garch_forecasts = {}
for td in test_dates:
    hist = all_prices[all_prices['date'] < td]['oil_close']
    if len(hist) < 30:
        continue
    log_ret = np.log(hist / hist.shift(1)).dropna() * 100
    try:
        gfit = arch_model(log_ret, vol='Garch', p=1, q=1,
                          dist='normal', rescale=False).fit(disp='off')
        sigma_daily = float(np.sqrt(
            gfit.forecast(horizon=1).variance.values[-1, 0])) / 100
        garch_forecasts[td] = sigma_daily * np.sqrt(252)
    except Exception:
        garch_forecasts[td] = np.nan

median_garch = np.nanmedian(list(garch_forecasts.values()))
for td in test_dates:
    if td not in garch_forecasts or np.isnan(garch_forecasts.get(td, np.nan)):
        garch_forecasts[td] = median_garch
print(f"GARCH forecasts ready for {len(garch_forecasts)} test dates "
      f"(median = {median_garch:.5f})\n")


# Load trained V1 models
print("Loading V1 trained models...")
plain_mlp = OilVolatilityMLP()
plain_mlp.load_state_dict(torch.load(MLP_PATH, map_location='cpu'))
plain_mlp.eval()
print("  Plain MLP (V1) loaded")

maml_model = OilVolatilityMLP()
maml_model.load_state_dict(torch.load(MAML_PATH, map_location='cpu'))
maml_model.eval()
print("  MAML (V1) loaded\n")


# Main evaluation loop — runs both random and chronological samplers
def run_evaluation(sampler, mode):
    ovx_idx = FEATURE_COLS.index('ovx_close')
    all_runs = {task: {
        'ovx': [], 'garch': [], 'mlp': [], 'maml': [],
        'actual_vals': [], 'maml_preds': [], 'mlp_preds': [],
        'garch_preds': [], 'ovx_preds': [],
    } for task in sampler.task_dates.keys()}

    for run in range(N_RUNS):
        for task in sampler.task_dates.keys():
            try:
                sup_X, sup_y, qry_X, qry_y, qry_dates = sampler.sample_episode(
                    task=task, run_seed=run * 1000, mode=mode
                )
                actual = np.expm1(qry_y.numpy().flatten())

                # OVX linear regression
                ovx_sup = sup_X[:, ovx_idx].numpy().reshape(-1, 1)
                ovx_qry = qry_X[:, ovx_idx].numpy().reshape(-1, 1)
                lr = LinearRegression()
                lr.fit(ovx_sup, np.expm1(sup_y.numpy().flatten()))
                ovx_preds = lr.predict(ovx_qry)
                all_runs[task]['ovx'].append(
                    mean_absolute_error(actual, ovx_preds))

                # GARCH
                garch_preds = np.array([
                    garch_forecasts.get(pd.Timestamp(d), median_garch)
                    for d in qry_dates
                ])
                all_runs[task]['garch'].append(
                    mean_absolute_error(actual, garch_preds))

                # Plain MLP
                with torch.no_grad():
                    mlp_preds = np.expm1(
                        plain_mlp(qry_X).numpy().flatten()).clip(0)
                all_runs[task]['mlp'].append(
                    mean_absolute_error(actual, mlp_preds))

                # MAML
                adapted = adapt(maml_model, sup_X, sup_y)
                with torch.no_grad():
                    maml_preds = np.expm1(
                        adapted(qry_X).numpy().flatten()).clip(0)
                all_runs[task]['maml'].append(
                    mean_absolute_error(actual, maml_preds))

                all_runs[task]['actual_vals'].append(actual)
                all_runs[task]['maml_preds'].append(maml_preds)
                all_runs[task]['mlp_preds'].append(mlp_preds)
                all_runs[task]['garch_preds'].append(garch_preds)
                all_runs[task]['ovx_preds'].append(ovx_preds)
            except Exception as e:
                print(f"  [{mode}] Skipped {task} run {run+1}: {e}")
        print(f"  [{mode}] Run {run+1}/{N_RUNS} complete")
    return all_runs


print("=" * 72)
print("V1 EVALUATION — RANDOM EPISODE SAMPLING (5 runs)")
print("=" * 72)
sampler = MAMLTaskSampler(df, FEATURE_COLS, TARGET_COL, split='test')
random_runs = run_evaluation(sampler, mode='random')

print()
print("=" * 72)
print("V1 EVALUATION — CHRONOLOGICAL EPISODE SAMPLING (5 runs)")
print("=" * 72)
chrono_runs = run_evaluation(sampler, mode='chronological')


# Per-task summary
def summarise(all_runs):
    rows = []
    for task, r in all_runs.items():
        if not r['maml']:
            continue
        rows.append({
            'task': task,
            'ovx_mean'  : np.mean(r['ovx']),
            'garch_mean': np.mean(r['garch']),
            'mlp_mean'  : np.mean(r['mlp']),
            'maml_mean' : np.mean(r['maml']),
            'maml_std'  : np.std(r['maml']),
            'maml_beats_mlp'   : np.mean(r['maml']) < np.mean(r['mlp']),
            'maml_beats_ovx'   : np.mean(r['maml']) < np.mean(r['ovx']),
            'maml_beats_garch' : np.mean(r['maml']) < np.mean(r['garch']),
        })
    return pd.DataFrame(rows).sort_values('maml_mean').reset_index(drop=True)


per_task_random = summarise(random_runs)
per_task_chrono = summarise(chrono_runs)
per_task_random.to_csv(OUT_DIR / "results_v1_per_task_random.csv", index=False)
per_task_chrono.to_csv(OUT_DIR / "results_v1_per_task_chronological.csv", index=False)


# Bootstrap CIs and overall summary
def bootstrap_ci(values, n_boot=N_BOOTSTRAP, alpha=0.05, seed=SEED):
    rng = np.random.RandomState(seed)
    boot_means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    return float(np.percentile(boot_means, 100 * alpha / 2)), \
           float(np.percentile(boot_means, 100 * (1 - alpha / 2)))


def overall_summary(per_task, label):
    s = {
        'mode': label,
        'n_tasks': int(len(per_task)),
        'ovx_mean'  : float(per_task['ovx_mean'].mean()),
        'garch_mean': float(per_task['garch_mean'].mean()),
        'mlp_mean'  : float(per_task['mlp_mean'].mean()),
        'maml_mean' : float(per_task['maml_mean'].mean()),
        'maml_wins_vs_mlp'  : int(per_task['maml_beats_mlp'].sum()),
        'maml_wins_vs_garch': int(per_task['maml_beats_garch'].sum()),
        'maml_wins_vs_ovx'  : int(per_task['maml_beats_ovx'].sum()),
    }
    for model in ['ovx', 'garch', 'mlp', 'maml']:
        lo, hi = bootstrap_ci(per_task[f'{model}_mean'].values)
        s[f'{model}_ci95_lo'] = lo
        s[f'{model}_ci95_hi'] = hi
    return s


summary_random = overall_summary(per_task_random, 'random')
summary_chrono = overall_summary(per_task_chrono, 'chronological')

print()
print("=" * 72)
print("V1 OVERALL RESULTS")
print("=" * 72)
for s in [summary_random, summary_chrono]:
    print(f"\n  Sampling mode: {s['mode'].upper()}")
    print(f"  {'Model':<10} {'Mean MAE':>10} {'95% CI':>22} {'Win rate':>15}")
    print(f"  {'-'*60}")
    for m, name in [('ovx', 'OVX'), ('garch', 'GARCH'),
                    ('mlp', 'MLP'), ('maml', 'MAML')]:
        ci = f"[{s[f'{m}_ci95_lo']:.4f}, {s[f'{m}_ci95_hi']:.4f}]"
        wr = f"{s['maml_wins_vs_mlp']}/{s['n_tasks']} vs MLP" if m == 'maml' else "—"
        print(f"  {name:<10} {s[f'{m}_mean']:>10.5f} {ci:>22} {wr:>15}")
    print(f"  MAML wins: vs OVX {s['maml_wins_vs_ovx']}/{s['n_tasks']} | "
          f"vs GARCH {s['maml_wins_vs_garch']}/{s['n_tasks']} | "
          f"vs MLP {s['maml_wins_vs_mlp']}/{s['n_tasks']}")


# Diebold-Mariano — per-task and pooled
def diebold_mariano(actual, pred1, pred2):
    e1 = np.abs(actual - pred1)
    e2 = np.abs(actual - pred2)
    d  = e1 - e2
    n  = len(d)
    if n < 2:
        return 0.0, 1.0
    d_bar = np.mean(d)
    gamma = np.var(d, ddof=1)
    if gamma / n <= 0:
        return 0.0, 1.0
    dm_stat = d_bar / np.sqrt(gamma / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value


def per_task_dm(all_runs, label):
    print(f"\n  Per-task DM tests [{label}]")
    print(f"  {'-'*70}")
    print(f"  {'Task':<45} {'p(MAML<MLP)':>13} {'p(MAML<GARCH)':>14}")
    rows = []
    sig_mlp = sig_garch = 0
    for task, r in all_runs.items():
        if not r['maml_preds']:
            continue
        actual = np.concatenate(r['actual_vals'])
        maml_p = np.concatenate(r['maml_preds'])
        mlp_p  = np.concatenate(r['mlp_preds'])
        gar_p  = np.concatenate(r['garch_preds'])
        _, p_mlp   = diebold_mariano(actual, maml_p, mlp_p)
        _, p_garch = diebold_mariano(actual, maml_p, gar_p)
        if p_mlp   < 0.05: sig_mlp   += 1
        if p_garch < 0.05: sig_garch += 1
        marker_m = "**" if p_mlp   < 0.05 else "  "
        marker_g = "**" if p_garch < 0.05 else "  "
        print(f"  {task:<45} {p_mlp:>11.3f}{marker_m} {p_garch:>12.3f}{marker_g}")
        rows.append({'task': task, 'p_vs_mlp': p_mlp, 'p_vs_garch': p_garch,
                     'sig_vs_mlp': p_mlp < 0.05, 'sig_vs_garch': p_garch < 0.05})
    print(f"\n  Significant at p<0.05: vs MLP {sig_mlp}/{len(rows)} | "
          f"vs GARCH {sig_garch}/{len(rows)}")
    return pd.DataFrame(rows), sig_mlp, sig_garch


def pooled_dm(all_runs, label):
    actual_all, maml_all, mlp_all, garch_all = [], [], [], []
    for r in all_runs.values():
        if not r['maml_preds']:
            continue
        actual_all.append(np.concatenate(r['actual_vals']))
        maml_all.append(np.concatenate(r['maml_preds']))
        mlp_all.append(np.concatenate(r['mlp_preds']))
        garch_all.append(np.concatenate(r['garch_preds']))
    actual_all = np.concatenate(actual_all)
    maml_all   = np.concatenate(maml_all)
    mlp_all    = np.concatenate(mlp_all)
    garch_all  = np.concatenate(garch_all)
    n = len(actual_all)
    dm_mlp,   p_mlp   = diebold_mariano(actual_all, maml_all, mlp_all)
    dm_garch, p_garch = diebold_mariano(actual_all, maml_all, garch_all)
    print(f"\n  Pooled DM test [{label}] — n = {n} pooled observations")
    print(f"  {'-'*60}")
    print(f"  MAML vs MLP   : DM = {dm_mlp:>+8.3f}   p = {p_mlp:.4f}")
    print(f"  MAML vs GARCH : DM = {dm_garch:>+8.3f}   p = {p_garch:.4f}")
    return {'mode': label, 'n_pooled': int(n),
            'dm_vs_mlp': float(dm_mlp), 'p_vs_mlp': float(p_mlp),
            'dm_vs_garch': float(dm_garch), 'p_vs_garch': float(p_garch)}


print()
print("=" * 72)
print("V1 DIEBOLD-MARIANO TESTS")
print("=" * 72)

dm_per_task_random, n_sig_mlp_r, n_sig_g_r = per_task_dm(random_runs, 'random')
dm_per_task_chrono, n_sig_mlp_c, n_sig_g_c = per_task_dm(chrono_runs, 'chronological')
pooled_random = pooled_dm(random_runs, 'random')
pooled_chrono = pooled_dm(chrono_runs, 'chronological')

dm_per_task_random['mode'] = 'random'
dm_per_task_chrono['mode'] = 'chronological'
pd.concat([dm_per_task_random, dm_per_task_chrono]).to_csv(
    OUT_DIR / "results_v1_dm_per_task.csv", index=False)


# K-shot ablation — random vs chronological
print()
print("=" * 72)
print("V1 K-SHOT ABLATION")
print("=" * 72)

kshot_rows = []
for k in [1, 3, 5, 10, 15]:
    for mode in ['random', 'chronological']:
        k_maes = []
        for task in sampler.task_dates.keys():
            try:
                sup_X, sup_y, qry_X, qry_y, _ = sampler.sample_episode(
                    task=task, n_support=k, n_query=3,
                    run_seed=42, mode=mode
                )
                adapted = adapt(maml_model, sup_X, sup_y)
                with torch.no_grad():
                    preds = np.expm1(adapted(qry_X).numpy().flatten())
                actual = np.expm1(qry_y.numpy().flatten())
                k_maes.append(mean_absolute_error(actual, preds))
            except Exception:
                pass
        if k_maes:
            mean_mae = float(np.mean(k_maes))
            kshot_rows.append({'k': k, 'mode': mode, 'mean_mae': mean_mae,
                               'n_tasks': len(k_maes)})
            print(f"  K={k:<3} [{mode:<13}]  mean MAML MAE = {mean_mae:.5f} "
                  f"(over {len(k_maes)} tasks)")
    print()

pd.DataFrame(kshot_rows).to_csv(OUT_DIR / "results_v1_kshot.csv", index=False)


# Permutation feature importance on the pretrained MLP
print("=" * 72)
print("V1 PERMUTATION FEATURE IMPORTANCE")
print("=" * 72)

test_df = df[df['split'] == 'test'].copy()
test_df[FEATURE_COLS] = scaler.transform(
    test_df[FEATURE_COLS].values.astype(np.float32))
X_test = torch.tensor(test_df[FEATURE_COLS].values.astype(np.float32))
y_test = test_df[TARGET_COL].values.astype(np.float32)

with torch.no_grad():
    base_preds = np.expm1(plain_mlp(X_test).numpy().flatten()).clip(0)
base_mae = mean_absolute_error(y_test, base_preds)
print(f"  Baseline MLP MAE on test set: {base_mae:.5f}\n")

importance_rows = []
rng = np.random.RandomState(SEED)
for j, feat in enumerate(FEATURE_COLS):
    deltas = []
    for _ in range(10):
        X_perm = X_test.clone().numpy()
        rng.shuffle(X_perm[:, j])
        with torch.no_grad():
            p = np.expm1(plain_mlp(torch.tensor(X_perm)
                                   ).numpy().flatten()).clip(0)
        deltas.append(mean_absolute_error(y_test, p) - base_mae)
    importance_rows.append({
        'feature': feat,
        'mean_mae_increase': float(np.mean(deltas)),
        'std_mae_increase': float(np.std(deltas)),
    })

imp_df = pd.DataFrame(importance_rows).sort_values(
    'mean_mae_increase', ascending=False).reset_index(drop=True)
imp_df.to_csv(OUT_DIR / "results_v1_feat_importance.csv", index=False)

print(f"  {'Feature':<22} {'deltaMAE when shuffled':>22} {'+/- std':>10}")
print(f"  {'-'*60}")
for _, row in imp_df.iterrows():
    print(f"  {row['feature']:<22} {row['mean_mae_increase']:>22.5f} "
          f"{row['std_mae_increase']:>10.5f}")


# Save final summary
final_summary = {
    'pipeline_version': 'V1',
    'n_features': INPUT_DIM,
    'random_sampling':         summary_random,
    'chronological_sampling':  summary_chrono,
    'pooled_dm_random':        pooled_random,
    'pooled_dm_chronological': pooled_chrono,
    'per_task_dm_significant_random': {
        'vs_mlp': int(n_sig_mlp_r), 'vs_garch': int(n_sig_g_r),
        'total_tasks': int(len(per_task_random)),
    },
    'per_task_dm_significant_chronological': {
        'vs_mlp': int(n_sig_mlp_c), 'vs_garch': int(n_sig_g_c),
        'total_tasks': int(len(per_task_chrono)),
    },
    'top_5_features_by_importance': imp_df.head(5).to_dict(orient='records'),
    'config': {
        'n_runs': N_RUNS, 'n_support': N_SUPPORT, 'n_query': N_QUERY,
        'inner_lr': INNER_LR, 'inner_steps': INNER_STEPS,
        'garch_annualisation': 'sqrt(252)',
        'n_bootstrap': N_BOOTSTRAP,
        'seed': SEED,
    }
}
with open(OUT_DIR / "results_v1_summary.json", 'w') as f:
    json.dump(final_summary, f, indent=2, default=str)


print()
print("=" * 72)
print("V1 EVALUATION COMPLETE")
print("=" * 72)
print(f"\nOutputs written to: {OUT_DIR.resolve()}")
print("  - results_v1_per_task_random.csv")
print("  - results_v1_per_task_chronological.csv")
print("  - results_v1_dm_per_task.csv")
print("  - results_v1_kshot.csv")
print("  - results_v1_feat_importance.csv")
print("  - results_v1_summary.json")
print()
print("Once both V1 and V2 have run, your report can cite:")
print("  - V1 vs V2 MAE comparison (Table 1 / Figure 1)")
print("  - V1 vs V2 win rates (Table 2 / Figure 2)")
print("  - V1 vs V2 K-shot curves (Figure 4)")
print("  - Whether V2's advantage over V1 holds under chronological sampling")
print()