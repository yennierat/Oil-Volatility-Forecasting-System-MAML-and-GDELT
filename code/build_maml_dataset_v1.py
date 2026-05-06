"""
MAML Dataset Builder v1 — GDELT + Brent Oil Volatility (15-feature dataset)

Targets:
  oil_fwd_rvol_1d  — MLP pretraining target (daily dataset)
                     abs(next_day_return) * sqrt(252)
  oil_fwd_rvol_4h  — MAML target (intraday dataset)
                     std of next 4 hourly returns * sqrt(252*23)

Usage:
    python3 build_maml_dataset_v1.py

Inputs:
    gdelt_output/dataset_daily.csv
    gdelt_output/dataset_intraday.csv

Outputs:
    gdelt_output/dataset_maml_daily.csv
    gdelt_output/dataset_maml_intraday.csv
    gdelt_output/maml_task_summary.txt
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

INPUT_DAILY    = "../data/dataset_daily.csv"
INPUT_INTRADAY = "../data/dataset_intraday.csv"
OUTPUT_DIR     = "../data"

BRENT_TICKER = "BZ=F"
WTI_TICKER   = "CL=F"
VIX_TICKER   = "^VIX"
GOLD_TICKER  = "GC=F"
OVX_TICKER   = "^OVX"
DXY_TICKER   = "DX=F"

MIN_EVENTS_PER_TASK = 50
MIN_DATES_PER_TASK  = 20

VAL_START  = "2025-06-01"
TEST_START = "2025-10-01"

OIL_ANN_DAILY  = np.sqrt(252)
OIL_ANN_HOURLY = np.sqrt(252 * 23)   # oil trades ~23h/day


MIDDLE_EAST = frozenset({
    "SA", "IR", "IZ", "KU", "IS", "SY", "LE", "JO",
    "YM", "AE", "QA", "BN", "OM",
})
OIL_PRODUCER_NON_ME = frozenset({
    "RS", "VE", "NI", "LY", "AG", "AO", "NO", "EC", "BR", "CA", "KZ",
})
OIL_CONSUMER = frozenset({
    "CH", "JA", "IN", "KS", "GM", "FR", "UK", "IT", "SP", "TW",
})
CHOKEPOINT = frozenset({"TU", "EG", "SU", "SO", "DJ"})


def assign_oil_region(c):
    if pd.isna(c) or c in ("", "UNKNOWN"):
        return "other"
    c = str(c).strip().upper()
    if c in ("US", "USA"):
        return "US"
    if c in MIDDLE_EAST:
        return "middle_east"
    if c in OIL_PRODUCER_NON_ME:
        return "oil_producer"
    if c in OIL_CONSUMER:
        return "oil_consumer"
    if c in CHOKEPOINT:
        return "middle_east"
    return "other"


def build_maml_task_label(task_category, region):
    tc = "diplomatic_tension" if task_category == "coercion" else task_category
    return f"{tc}__{region}"


def _download(ticker, start, end, interval="1d", name=""):
    print(f"  {name} ({ticker})...", end=" ", flush=True)
    try:
        px = yf.download(
            ticker, start=start, end=end,
            interval=interval, progress=False, auto_adjust=True
        )
        if isinstance(px.columns, pd.MultiIndex):
            px.columns = [c[0] for c in px.columns]
        if px.empty:
            print("EMPTY")
            return None
        px = px.reset_index()
        px.rename(columns={px.columns[0]: "dt"}, inplace=True)
        px["dt"] = pd.to_datetime(px["dt"]).dt.tz_localize(None)
        print(f"{len(px):,} bars")
        return px
    except Exception as e:
        print(f"FAILED ({e})")
        return None


def build_oil_daily(start: str, end: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"  Downloading daily market data: {start} -> {end}")
    print(f"{'='*60}")

    raw = _download(BRENT_TICKER, start, end, name="Brent crude")
    if raw is None:
        raw = _download(WTI_TICKER, start, end, name="WTI fallback")
    if raw is None:
        print("FATAL: No oil data.")
        sys.exit(1)

    oil = raw[["dt", "Close"]].rename(
        columns={"dt": "trade_date", "Close": "oil_close"}
    ).copy()

    oil["oil_return"] = oil["oil_close"].pct_change()

    # MLP pretraining target: abs(next-day return) annualized.
    # With a single daily return, this is the only way to express 1d vol.
    oil["oil_next_day_return"] = oil["oil_return"].shift(-1)
    oil["oil_fwd_rvol_1d"]    = oil["oil_next_day_return"].abs() * OIL_ANN_DAILY
    oil["oil_direction_label"] = np.where(
        oil["oil_next_day_return"] > 0, "up", "down"
    )

    # Backward-looking volatility features (.shift(1) prevents same-day leakage)
    oil["oil_vol_5d"]  = (
        oil["oil_return"].rolling(5,  min_periods=3).std().shift(1) * OIL_ANN_DAILY
    )
    oil["oil_vol_20d"] = (
        oil["oil_return"].rolling(20, min_periods=10).std().shift(1) * OIL_ANN_DAILY
    )

    for ticker, name, col in [
        (VIX_TICKER,  "VIX",  "vix_close"),
        (GOLD_TICKER, "Gold", "gold_close"),
        (OVX_TICKER,  "OVX",  "ovx_close"),
        (DXY_TICKER,  "DXY",  "dxy_close"),
    ]:
        px = _download(ticker, start, end, name=name)
        if px is not None:
            tmp = px[["dt", "Close"]].rename(
                columns={"dt": "trade_date", "Close": col}
            )
            oil = oil.merge(tmp, on="trade_date", how="left")
            oil[col] = oil[col].ffill()

    if "gold_close" in oil.columns:
        oil["gold_oil_ratio"] = oil["gold_close"] / oil["oil_close"]
    if "ovx_close" in oil.columns:
        oil["ovx_change"] = oil["ovx_close"].pct_change()

    print(f"\n  Daily oil table: {len(oil):,} rows")
    print(f"  oil_fwd_rvol_1d:  "
          f"mean={oil['oil_fwd_rvol_1d'].dropna().mean():.4f}  "
          f"std={oil['oil_fwd_rvol_1d'].dropna().std():.4f}")
    return oil


def build_oil_hourly(start: str, end: str):
    hourly_start = max(
        pd.Timestamp(start),
        pd.Timestamp.today().normalize() - pd.Timedelta(days=729),
    ).strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"  Downloading hourly oil: {hourly_start} -> {end}")
    print(f"  Annualization: sqrt(252 x 23h) = {OIL_ANN_HOURLY:.1f}")
    print(f"{'='*60}")

    raw = _download(BRENT_TICKER, hourly_start, end, interval="1h",
                    name="Brent hourly")
    if raw is None:
        raw = _download(WTI_TICKER, hourly_start, end, interval="1h",
                        name="WTI hourly")
    if raw is None:
        print("  No hourly data — oil_fwd_rvol_4h will be NaN")
        return None

    h = raw.rename(columns={"dt": "bar_datetime"}).sort_values(
        "bar_datetime"
    ).reset_index(drop=True)

    h["hourly_return"] = h["Close"].pct_change()

    # MAML target: std of next 4 hourly returns, annualized with sqrt(252*23)
    shifted = pd.concat(
        [h["hourly_return"].shift(-k) for k in range(1, 5)], axis=1
    )
    h["oil_fwd_rvol_4h"] = shifted.std(axis=1, ddof=1) * OIL_ANN_HOURLY

    h["oil_return_1h_next"] = h["Close"].shift(-1) / h["Close"] - 1
    h["oil_return_4h_next"] = h["Close"].shift(-4) / h["Close"] - 1

    print(f"  Hourly oil table: {len(h):,} bars")
    print(f"  oil_fwd_rvol_4h:  "
          f"mean={h['oil_fwd_rvol_4h'].dropna().mean():.4f}")
    return h


def merge_with_oil(
    events: pd.DataFrame,
    oil_daily: pd.DataFrame,
    oil_hourly,
    label: str
) -> pd.DataFrame:
    print(f"\n  Merging {label}: {len(events):,} events")
    df = events.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).astype(
        "datetime64[us]"
    )

    # Drop S&P columns — replacing with oil
    sp_cols = [
        "next_day_return", "volatility_5d", "volatility_20d",
        "return_direction_label", "abs_return", "sp500_close",
        "return_1h_next", "return_4h_next",
    ]
    df = df.drop(columns=[c for c in sp_cols if c in df.columns],
                 errors="ignore")

    # Daily oil merge
    skip = {"oil_return", "trade_date"}
    oil_cols = ["trade_date"] + [
        c for c in oil_daily.columns if c not in skip
    ]
    oil_daily["trade_date"] = oil_daily["trade_date"].astype("datetime64[us]")

    merged = pd.merge_asof(
        df.sort_values("date"),
        oil_daily[oil_cols].sort_values("trade_date"),
        left_on="date", right_on="trade_date",
        direction="forward", tolerance=pd.Timedelta("4D"),
    )
    before = len(merged)
    merged = merged.dropna(subset=["oil_fwd_rvol_1d"])
    if (dropped := before - len(merged)):
        print(f"    Dropped {dropped:,} events (no oil daily match)")
    merged = merged.drop(columns=["trade_date"], errors="ignore")

    # Hourly oil merge for oil_fwd_rvol_4h
    if oil_hourly is not None and not oil_hourly.empty:
        print("    Merging hourly oil for oil_fwd_rvol_4h...")

        # Check if event_datetime has real time component from v3 fetcher
        has_time = False
        if "event_datetime" in merged.columns:
            dt_parsed = pd.to_datetime(merged["event_datetime"],
                                       errors="coerce")
            n_unique_dt    = dt_parsed.nunique()
            n_unique_dates = pd.to_datetime(
                merged["date"]
            ).dt.date.nunique()
            has_time = n_unique_dt > n_unique_dates

        if has_time:
            merged["event_datetime"] = pd.to_datetime(
                merged["event_datetime"], errors="coerce"
            )
            print("    Using event_datetime from v3 fetcher  ✓")
        elif "snapshot_hour" in merged.columns:
            merged["event_datetime"] = (
                pd.to_datetime(merged["date"])
                + pd.to_timedelta(
                    pd.to_numeric(
                        merged["snapshot_hour"], errors="coerce"
                    ).fillna(12).astype(int), unit="h"
                )
            )
            print("    Using snapshot_hour fallback")
        else:
            merged["event_datetime"] = pd.to_datetime(merged["date"])
            print("    WARNING: No sub-daily timestamps")

        hcols = [
            "bar_datetime", "oil_fwd_rvol_4h",
            "oil_return_1h_next", "oil_return_4h_next"
        ]
        hm = oil_hourly[hcols].copy().astype(
            {"bar_datetime": "datetime64[us]"}
        )
        merged["event_datetime"] = merged["event_datetime"].astype(
            "datetime64[us]"
        )

        merged = pd.merge_asof(
            merged.sort_values("event_datetime"),
            hm.sort_values("bar_datetime"),
            left_on="event_datetime", right_on="bar_datetime",
            direction="forward", tolerance=pd.Timedelta("4h"),
        )
        merged = merged.drop(columns=["bar_datetime"], errors="ignore")

        n_hit = merged["oil_fwd_rvol_4h"].notna().sum()
        print(f"    oil_fwd_rvol_4h matched: {n_hit:,}/{len(merged):,} "
              f"({n_hit/len(merged)*100:.1f}%)")

        unique_per_date = merged.groupby(
            pd.to_datetime(merged["date"]).dt.date
        )["oil_fwd_rvol_4h"].nunique()
        mean_u = unique_per_date.mean()
        print(f"    Unique 4h targets per date: mean={mean_u:.1f}  "
              f"({'✓ event-level' if mean_u > 1.5 else '⚠ still daily'})")

    print(f"    Final: {len(merged):,} events")
    return merged


# Add MAML task structure
def add_maml_structure(df: pd.DataFrame, label: str) -> pd.DataFrame:
    print(f"\n  Building MAML tasks for {label}...")

    df["oil_region"] = df["primary_country"].map(assign_oil_region)
    df["maml_task"]  = df.apply(
        lambda r: build_maml_task_label(
            r["task_category"], r["oil_region"]
        ), axis=1
    )

    stats = df.groupby("maml_task").agg(
        n  = ("event_id", "count"),
        nd = ("date", lambda x: pd.to_datetime(x).dt.date.nunique()),
    )
    viable = stats[
        (stats["n"] >= MIN_EVENTS_PER_TASK) &
        (stats["nd"] >= MIN_DATES_PER_TASK)
    ]
    print(f"    Viable tasks: {len(viable)} / {len(stats)}")
    for task in viable.sort_values("n", ascending=False).index:
        print(f"      {task:<45} {viable.loc[task,'n']:>5} events  "
              f"{viable.loc[task,'nd']:>3} dates")

    df = df[df["maml_task"].isin(viable.index)].copy()

    df["date"]  = pd.to_datetime(df["date"])
    df["split"] = "train"
    df.loc[df["date"] >= VAL_START,  "split"] = "val"
    df.loc[df["date"] >= TEST_START, "split"] = "test"

    print("\n    Splits:")
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        if len(sub):
            print(f"      {split:<6} {len(sub):>6,}  "
                  f"{sub['date'].min().date()} -> {sub['date'].max().date()}")
        else:
            print(f"      {split:<6}      0 events")

    return df


# Task summary
def write_summary(datasets: list, path: str) -> None:
    lines = [
        "=" * 70,
        "  MAML TASK STRUCTURE SUMMARY",
        "=" * 70,
        "",
        "  MLP PRETRAINING TARGET: oil_fwd_rvol_1d",
        "    → abs(next_day_return) * sqrt(252)",
        "    → Trained on daily dataset (2022-2024)",
        "",
        "  MAML TARGET: oil_fwd_rvol_4h",
        "    → std of next 4 hourly returns * sqrt(252*23)",
        "    → Trained on intraday dataset (2024-2026)",
        "",
        "  FEATURES (both datasets):",
        "    goldstein_scale, avg_tone, num_mentions, num_sources,",
        "    oil_close, oil_vol_5d, oil_vol_20d,",
        "    vix_close, ovx_close, dxy_close, gold_oil_ratio",
    ]

    for name, df in datasets:
        lines += [
            "",
            "─" * 70,
            f"  {name}",
            f"  Events: {len(df):,}  |  "
            f"Dates: {df['date'].min().date()} -> "
            f"{df['date'].max().date()}",
            f"  Tasks: {df['maml_task'].nunique()}",
            "",
            "  Task breakdown:",
        ]
        for task in sorted(df["maml_task"].unique()):
            sub = df[df["maml_task"] == task]
            splits = sub["split"].value_counts().to_dict()
            s = " / ".join(
                f"{k}={splits.get(k,0)}" for k in ["train","val","test"]
            )
            lines.append(
                f"    {task:<45} {len(sub):>5}  ({s})"
            )

        lines += ["", "  Target columns:"]
        for col in ["oil_fwd_rvol_1d", "oil_fwd_rvol_4h"]:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(vals):
                    lines.append(
                        f"    {col:<30} mean={vals.mean():.4f}  "
                        f"std={vals.std():.4f}  n={len(vals):,}"
                    )

        feats = [c for c in df.columns if c in [
            "goldstein_scale", "avg_tone", "num_mentions", "num_sources",
            "oil_vol_5d", "oil_vol_20d", "vix_close", "ovx_close",
            "gold_oil_ratio", "dxy_close", "oil_close",
        ]]
        lines += ["", "  Feature columns:", f"    {feats}"]

    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)
    print(f"\n  Summary → {path}")
    print(text)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  MAML DATASET BUILDER v1")
    print("  MLP target: oil_fwd_rvol_1d  |  MAML target: oil_fwd_rvol_4h")
    print("=" * 70)

    print(f"\nLoading {INPUT_DAILY}...")
    df_daily = pd.read_csv(INPUT_DAILY)
    print(f"  {len(df_daily):,} events  "
          f"{df_daily['date'].min()} -> {df_daily['date'].max()}")

    print(f"\nLoading {INPUT_INTRADAY}...")
    df_intraday = pd.read_csv(INPUT_INTRADAY)
    print(f"  {len(df_intraday):,} events  "
          f"{df_intraday['date'].min()} -> {df_intraday['date'].max()}")

    all_dates = list(df_daily["date"]) + list(df_intraday["date"])
    date_min  = pd.Timestamp(min(all_dates))
    date_max  = pd.Timestamp(max(all_dates))
    start = (date_min - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    end   = (date_max + pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    oil_daily  = build_oil_daily(start, end)
    oil_hourly = build_oil_hourly(start, end)

    merged_daily    = merge_with_oil(
        df_daily, oil_daily, oil_hourly=None, label="DAILY"
    )
    maml_daily      = add_maml_structure(merged_daily, "DAILY")

    merged_intraday = merge_with_oil(
        df_intraday, oil_daily, oil_hourly, label="INTRADAY"
    )
    maml_intraday   = add_maml_structure(merged_intraday, "INTRADAY")

    daily_path    = f"{OUTPUT_DIR}/dataset_maml_daily.csv"
    intraday_path = f"{OUTPUT_DIR}/dataset_maml_intraday.csv"
    summary_path  = f"{OUTPUT_DIR}/maml_task_summary.txt"

    maml_daily.to_csv(daily_path, index=False)
    print(f"\nSaved: {daily_path}  ({len(maml_daily):,} events)")

    maml_intraday.to_csv(intraday_path, index=False)
    print(f"Saved: {intraday_path}  ({len(maml_intraday):,} events)")

    write_summary(
        [
            ("DAILY (MLP pretraining, 2022-2024)", maml_daily),
            ("INTRADAY (MAML, 2024-2026)",         maml_intraday),
        ],
        summary_path,
    )

    print(f"\n{'='*70}")
    print("  DONE")
    print("  1. Train MLP on dataset_maml_daily.csv")
    print("     Target: oil_fwd_rvol_1d")
    print("  2. Train MAML on dataset_maml_intraday.csv")
    print("     Target: oil_fwd_rvol_4h")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()