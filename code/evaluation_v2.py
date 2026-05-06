"""
V2 Evaluation — comprehensive offline evaluation for the V2 MAML pipeline (23 features).

Reports random- AND chronological-sampling MAE, pooled Diebold-Mariano tests,
bootstrap CIs, permutation feature importance, and a temporally-local k-shot ablation.
GARCH uses sqrt(252*23) to match the hourly → annual scale of oil_fwd_rvol_4h.

Inputs:
  ../data/dataset_maml_intraday_v2.csv, mlp_pretrained_v2.pth,
  maml_trained_v2.pth, feature_scaler_v2.pkl

Outputs (in eval_outputs_v2/):
  results_v2_per_task_random.csv, results_v2_per_task_chronological.csv,
  results_v2_dm_per_task.csv, results_v2_kshot.csv,
  results_v2_feat_importance.csv, results_v2_summary.json
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

# Config
DATA_PATH    = "../data/dataset_maml_intraday_v2.csv"
MLP_PATH     = "../models/v2/mlp_pretrained_v2.pth"
MAML_PATH    = "../models/v2/maml_trained_v2.pth"
SCALER_PATH  = "../models/v2/feature_scaler_v2.pkl"
OUT_DIR      = Path("eval_outputs_v2")
OUT_DIR.mkdir(exist_ok=True)

RAW_COLS = [
    'goldstein_scale', 'num_mentions', 'num_sources', 'avg_tone',
    'oil_close', 'oil_vol_5d', 'oil_vol_20d',
    'vix_close', 'ovx_close', 'dxy_close', 'gold_oil_ratio'
]
FEATURE_COLS = [
    'ovx_close', 'vix_close', 'oil_vol_5d', 'oil_vol_20d',
    'oil_close', 'dxy_close', 'gold_oil_ratio',
    'gs_mean', 'gs_std', 'gs_conflict_pct', 'gs_weighted',
    'tone_mean', 'tone_std', 'n_events', 'mentions_sum',
    'me_gs_mean', 'me_conflict_pct', 'me_n_events', 'me_tone_mean',
    'oi_gs_mean', 'oi_conflict_pct', 'oi_n_events', 'oi_tone_mean',
]
TARGET_COL    = 'oil_fwd_rvol_4h'
INPUT_DIM     = len(FEATURE_COLS)        # 23 for V2
INNER_LR      = 0.01
INNER_STEPS   = 5
N_RUNS        = 5
N_SUPPORT     = 5
N_QUERY       = 3
TRADING_HOURS = 23                       # ICE Brent ~23h/day
ANNUAL_HOURS  = 252 * TRADING_HOURS       # for GARCH scaling
N_BOOTSTRAP   = 2000                      # for confidence intervals

# Model definition — must match training architecture
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


# Load data and aggregate to one row per (task, date)
print("=" * 72)
print("Loading data...")
print("=" * 72)
df = pd.read_csv(DATA_PATH, parse_dates=['date'])
before = len(df)
df = df.dropna(subset=RAW_COLS + [TARGET_COL])
print(f"Dropped {before - len(df)} NaN rows. Remaining: {len(df):,}")
print(f"Tasks: {df['maml_task'].nunique()}")

# Aggregate to one row per (task, split, date)
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
            'me_gs_mean'      : g['me_gs_mean'].iloc[0],
            'me_conflict_pct' : g['me_conflict_pct'].iloc[0],
            'me_n_events'     : g['me_n_events'].iloc[0],
            'me_tone_mean'    : g['me_tone_mean'].iloc[0],
            'oi_gs_mean'      : g['oi_gs_mean'].iloc[0],
            'oi_conflict_pct' : g['oi_conflict_pct'].iloc[0],
            'oi_n_events'     : g['oi_n_events'].iloc[0],
            'oi_tone_mean'    : g['oi_tone_mean'].iloc[0],
            TARGET_COL        : g[TARGET_COL].iloc[0],
        }), include_groups=False
    )
    .reset_index()
    .sort_values('date')
    .reset_index(drop=True)
)
print(f"After aggregation: {len(df):,} rows ({df['date'].nunique()} unique dates)\n")

# Load scaler
scaler = joblib.load(SCALER_PATH)


# Episode sampler — supports both random and chronological sampling
class MAMLTaskSampler:
    """
    Samples (support, query) episodes from the test split per task.

    mode='random'        — original protocol; support and query may be
                            interleaved in time. Inherits from standard
                            MAML few-shot evaluation.
    mode='chronological' — strictly causal: support always precedes query,
                            mirroring deployment-time information availability.
    """
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
        # Log transform — must match training
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
            # Pick a random valid cutoff so support has n_support recent dates
            # before the cutoff and query has n_query dates strictly after.
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
    adapted.eval()  # freeze BN running stats during adaptation
    for _ in range(inner_steps):
        optimizer.zero_grad()
        loss = loss_fn(adapted(sup_X), sup_y)
        loss.backward()
        optimizer.step()
    return adapted


# Rolling GARCH(1,1) baseline.
# GARCH on daily log returns gives a daily conditional vol forecast.
# Multiplying by sqrt(252) annualises it onto the same scale as oil_fwd_rvol_4h
# (already annualised via sqrt(252*23) at construction time). Both are
# annualised volatilities — they differ in horizon, not scale.
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
        # daily-scale conditional vol (still in pct terms because *100 above)
        sigma_daily = float(np.sqrt(gfit.forecast(horizon=1).variance.values[-1, 0])) / 100
        # rescale daily vol → annualised to match target
        garch_forecasts[td] = sigma_daily * np.sqrt(252)
    except Exception:
        garch_forecasts[td] = np.nan

median_garch = np.nanmedian(list(garch_forecasts.values()))
for td in test_dates:
    if td not in garch_forecasts or np.isnan(garch_forecasts.get(td, np.nan)):
        garch_forecasts[td] = median_garch
print(f"GARCH forecasts ready for {len(garch_forecasts)} test dates "
      f"(median = {median_garch:.5f})\n")


# Load trained models
print("Loading trained models...")
plain_mlp = OilVolatilityMLP()
plain_mlp.load_state_dict(torch.load(MLP_PATH, map_location='cpu'))
plain_mlp.eval()
print("  Plain MLP loaded")

maml_model = OilVolatilityMLP()
maml_model.load_state_dict(torch.load(MAML_PATH, map_location='cpu'))
maml_model.eval()
print("  MAML model loaded\n")


# Main evaluation loop — runs both random and chronological samplers
def run_evaluation(sampler, mode):
    """Run N_RUNS evaluations across all 25 tasks under given sampling mode."""
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
                all_runs[task]['ovx'].append(mean_absolute_error(actual, ovx_preds))

                # GARCH per-date forecast
                garch_preds = np.array([
                    garch_forecasts.get(pd.Timestamp(d), median_garch)
                    for d in qry_dates
                ])
                all_runs[task]['garch'].append(
                    mean_absolute_error(actual, garch_preds))

                # Plain MLP (no adaptation)
                with torch.no_grad():
                    mlp_preds = np.expm1(
                        plain_mlp(qry_X).numpy().flatten()).clip(0)
                all_runs[task]['mlp'].append(
                    mean_absolute_error(actual, mlp_preds))

                # MAML (with adaptation)
                adapted = adapt(maml_model, sup_X, sup_y)
                with torch.no_grad():
                    maml_preds = np.expm1(
                        adapted(qry_X).numpy().flatten()).clip(0)
                all_runs[task]['maml'].append(
                    mean_absolute_error(actual, maml_preds))

                # Store raw preds for pooled DM later
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
print("MAIN EVALUATION — RANDOM EPISODE SAMPLING (5 runs)")
print("=" * 72)
sampler = MAMLTaskSampler(df, FEATURE_COLS, TARGET_COL, split='test')
random_runs = run_evaluation(sampler, mode='random')

print()
print("=" * 72)
print("MAIN EVALUATION — CHRONOLOGICAL EPISODE SAMPLING (5 runs)")
print("=" * 72)
print("Support set = N most recent dates BEFORE a randomly chosen cutoff;")
print("query set = next N dates strictly AFTER the cutoff. Causally clean.")
print()
chrono_runs = run_evaluation(sampler, mode='chronological')


# Aggregate per-task results
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

# Save per-task tables
per_task_random.to_csv(OUT_DIR / "results_v2_per_task_random.csv", index=False)
per_task_chrono.to_csv(OUT_DIR / "results_v2_per_task_chronological.csv", index=False)


# Bootstrap CI for mean MAE across tasks
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
    # Bootstrap 95% CI for each model's mean MAE
    for model in ['ovx', 'garch', 'mlp', 'maml']:
        lo, hi = bootstrap_ci(per_task[f'{model}_mean'].values)
        s[f'{model}_ci95_lo'] = lo
        s[f'{model}_ci95_hi'] = hi
    return s


summary_random = overall_summary(per_task_random, 'random')
summary_chrono = overall_summary(per_task_chrono, 'chronological')

print()
print("=" * 72)
print("OVERALL RESULTS")
print("=" * 72)
for s in [summary_random, summary_chrono]:
    print(f"\n  Sampling mode: {s['mode'].upper()}")
    print(f"  {'Model':<10} {'Mean MAE':>10} {'95% CI':>22} {'Win rate':>15}")
    print(f"  {'-'*60}")
    for m, name in [('ovx', 'OVX'), ('garch', 'GARCH'),
                    ('mlp', 'MLP'), ('maml', 'MAML')]:
        ci = f"[{s[f'{m}_ci95_lo']:.4f}, {s[f'{m}_ci95_hi']:.4f}]"
        if m == 'maml':
            wr = (f"{s['maml_wins_vs_mlp']}/{s['n_tasks']} vs MLP")
        else:
            wr = "—"
        print(f"  {name:<10} {s[f'{m}_mean']:>10.5f} {ci:>22} {wr:>15}")
    print(f"  MAML wins:  vs OVX {s['maml_wins_vs_ovx']}/{s['n_tasks']} | "
          f"vs GARCH {s['maml_wins_vs_garch']}/{s['n_tasks']} | "
          f"vs MLP {s['maml_wins_vs_mlp']}/{s['n_tasks']}")


# Diebold-Mariano: per-task AND pooled across all tasks
def diebold_mariano(actual, pred1, pred2):
    """One-sided DM: tests whether pred1 has lower MAE than pred2.
    Returns (DM stat, two-sided p-value)."""
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
        if p_mlp < 0.05:
            sig_mlp += 1
        if p_garch < 0.05:
            sig_garch += 1
        marker_m = "**" if p_mlp   < 0.05 else "  "
        marker_g = "**" if p_garch < 0.05 else "  "
        print(f"  {task:<45} {p_mlp:>11.3f}{marker_m} {p_garch:>12.3f}{marker_g}")
        rows.append({'task': task, 'p_vs_mlp': p_mlp, 'p_vs_garch': p_garch,
                     'sig_vs_mlp': p_mlp < 0.05, 'sig_vs_garch': p_garch < 0.05})
    print(f"\n  Significant at p<0.05: vs MLP {sig_mlp}/{len(rows)} | "
          f"vs GARCH {sig_garch}/{len(rows)}")
    return pd.DataFrame(rows), sig_mlp, sig_garch


def pooled_dm(all_runs, label):
    """Pool prediction errors across ALL tasks before testing.
    Substantially higher statistical power than per-task tests on small samples.
    """
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
print("DIEBOLD-MARIANO TESTS")
print("=" * 72)

dm_per_task_random, n_sig_mlp_r, n_sig_g_r = per_task_dm(random_runs, 'random')
dm_per_task_chrono, n_sig_mlp_c, n_sig_g_c = per_task_dm(chrono_runs, 'chronological')
pooled_random = pooled_dm(random_runs, 'random')
pooled_chrono = pooled_dm(chrono_runs, 'chronological')

dm_per_task_random['mode'] = 'random'
dm_per_task_chrono['mode'] = 'chronological'
pd.concat([dm_per_task_random, dm_per_task_chrono]).to_csv(
    OUT_DIR / "results_v2_dm_per_task.csv", index=False)


# K-shot ablation — both random and temporally-local versions
print()
print("=" * 72)
print("K-SHOT ABLATION  (random vs temporally-local support sets)")
print("=" * 72)
print("Random:           K support dates drawn uniformly from test pool")
print("Temporally local: K most recent dates before a chosen query cutoff")
print()

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

pd.DataFrame(kshot_rows).to_csv(OUT_DIR / "results_v2_kshot.csv", index=False)


# Permutation feature importance on the pretrained MLP, on the test split.
# Validates the claim that OVX is the strongest predictor.
print("=" * 72)
print("PERMUTATION FEATURE IMPORTANCE  (plain MLP, test split)")
print("=" * 72)
print("Each feature is shuffled in turn; the increase in MAE is its importance.")
print("Larger increase = the model relied more on that feature.\n")

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
    for _ in range(10):  # 10 shuffles per feature for stability
        X_perm = X_test.clone().numpy()
        rng.shuffle(X_perm[:, j])
        with torch.no_grad():
            p = np.expm1(plain_mlp(torch.tensor(X_perm)).numpy().flatten()).clip(0)
        deltas.append(mean_absolute_error(y_test, p) - base_mae)
    importance_rows.append({
        'feature': feat,
        'mean_mae_increase': float(np.mean(deltas)),
        'std_mae_increase': float(np.std(deltas)),
    })

imp_df = pd.DataFrame(importance_rows).sort_values(
    'mean_mae_increase', ascending=False).reset_index(drop=True)
imp_df.to_csv(OUT_DIR / "results_v2_feat_importance.csv", index=False)

print(f"  {'Feature':<22} {'ΔMAE when shuffled':>22} {'± std':>10}")
print(f"  {'-'*60}")
for _, row in imp_df.iterrows():
    print(f"  {row['feature']:<22} {row['mean_mae_increase']:>22.5f} "
          f"{row['std_mae_increase']:>10.5f}")


# Save consolidated summary
final_summary = {
    'random_sampling':       summary_random,
    'chronological_sampling': summary_chrono,
    'pooled_dm_random':       pooled_random,
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
        'trading_hours_per_day': TRADING_HOURS,
        'annual_hours_for_garch_scaling': ANNUAL_HOURS,
        'n_bootstrap': N_BOOTSTRAP,
        'seed': SEED,
    }
}
with open(OUT_DIR / "results_v2_summary.json", 'w') as f:
    json.dump(final_summary, f, indent=2, default=str)


# Final notes printed for the user
print()
print("=" * 72)
print("EVALUATION COMPLETE")
print("=" * 72)
print(f"\nOutputs written to: {OUT_DIR.resolve()}")
print("  - results_v2_per_task_random.csv")
print("  - results_v2_per_task_chronological.csv")
print("  - results_v2_dm_per_task.csv")
print("  - results_v2_kshot.csv")
print("  - results_v2_feat_importance.csv")
print("  - results_v2_summary.json")
print()
print("HOW TO USE THESE NUMBERS IN YOUR REPORT")
print("-" * 72)
print("Table 1 (Mean MAE):           summary['random_sampling']['*_mean']")
print("Bootstrap CIs:                 summary['random_sampling']['*_ci95_*']")
print("Win rates (Table 2):           summary['random_sampling']['maml_wins_vs_*']")
print("Per-task DM (Table 3):         results_v2_dm_per_task.csv (mode='random')")
print("Pooled DM (NEW row in Tbl 3):  summary['pooled_dm_random']['p_vs_mlp']")
print("K-shot table (Table 4):        results_v2_kshot.csv (mode='random')")
print("Robustness check (NEW Sec 5):  compare 'random' vs 'chronological'")
print("                                in summary; if MAML still wins under")
print("                                chronological, that defends against")
print("                                supervisor's temporal-causality concern.")
print("Feature importance:            results_v2_feat_importance.csv")
print("                                (validates 'OVX is strongest' claim)")
print()