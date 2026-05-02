"""
GDELT 2.0 Political Events Pipeline (shared by v1 and v2 models)

Setup:
    python3 -m venv venv
    source venv/bin/activate
    pip install requests pandas numpy yfinance tqdm pyarrow torch torchvision

Run:
    python3 gdelt_pipeline.py
"""

import io
import sys
import time
import zipfile
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning,      module="yfinance")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="yfinance")
warnings.filterwarnings("ignore", category=FutureWarning,      module="pandas")



@dataclass
class PipelineConfig:
    
    start_date: str = "2022-01-01"
    end_date:   str = field(default_factory=lambda: date.today().strftime("%Y-%m-%d"))

    output_dir:     str = "gdelt_output"
    checkpoint_dir: str = "gdelt_output/checkpoints"
    final_output:   str = "gdelt_output/political_events_gdelt.csv"
    sp500_output:   str = "gdelt_output/sp500_returns.csv"
    log_file:       str = "gdelt_output/pipeline.log"

    max_workers: int = 20
    batch_size:  int = 200

    request_timeout: int   = 25
    retry_attempts:  int   = 3
    retry_delay:     float = 1.5

    ticker: str = "^GSPC"

    # 10+ keeps events reported by multiple independent sources.
    # Lower values let single-source diplomatic noise through.
    min_mentions: int  = 10
    us_only:      bool = False

    # Goldstein +7 = major peace agreements, defence pacts, significant deals.
    # 0.0 disables the filter (keeps all cooperation events).
    coop_goldstein_min: float = 7.0

    # 4 snapshots per trading day:
    #   0400 UTC = Asian/European overnight
    #   0900 UTC = European morning / US pre-market
    #   1400 UTC = US market open (9am ET)
    #   2000 UTC = US close/after-hours (4pm ET)
    snapshot_hours: tuple = (4, 9, 14, 20)


def setup_logging(cfg: PipelineConfig) -> logging.Logger:
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("gdelt_pipeline")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh  = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh  = logging.FileHandler(cfg.log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


# GDELT 2.0 schema: 58 TSV columns, no header in file
GDELT_COLS: list[str] = [
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",
    "NumArticles", "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat", "Actor1Geo_Long",
    "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code", "Actor2Geo_Lat", "Actor2Geo_Long",
    "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_ADM2Code", "ActionGeo_Lat", "ActionGeo_Long",
    "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL",
]

KEEP_COLS: list[str] = [
    "GLOBALEVENTID", "SQLDATE",
    "Actor1Name", "Actor1CountryCode",
    "Actor2Name", "Actor2CountryCode",
    "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale",
    "NumMentions", "NumSources", "NumArticles", "AvgTone",
    "ActionGeo_FullName", "ActionGeo_CountryCode",
    "SOURCEURL",
]


CAMEO_TASK_MAP: dict[str, str] = {
    "01": "cooperation_diplomacy", "010": "cooperation_diplomacy",
    "011": "cooperation_diplomacy", "012": "cooperation_diplomacy",
    "013": "cooperation_diplomacy", "014": "cooperation_diplomacy",
    "02": "cooperation_diplomacy", "020": "cooperation_diplomacy",
    "021": "cooperation_diplomacy", "022": "cooperation_diplomacy",
    "023": "cooperation_diplomacy",
    "03": "cooperation_diplomacy", "030": "cooperation_diplomacy",
    "031": "cooperation_diplomacy", "032": "cooperation_diplomacy",
    "033": "cooperation_diplomacy", "034": "cooperation_diplomacy",
    "04": "cooperation_diplomacy", "040": "cooperation_diplomacy",
    "041": "cooperation_diplomacy", "042": "cooperation_diplomacy",
    "043": "cooperation_diplomacy", "044": "cooperation_diplomacy",
    "045": "cooperation_diplomacy", "046": "cooperation_diplomacy",
    "05": "cooperation_diplomacy", "050": "cooperation_diplomacy",
    "051": "cooperation_diplomacy", "052": "cooperation_diplomacy",
    "053": "cooperation_diplomacy", "054": "cooperation_diplomacy",
    "055": "cooperation_diplomacy", "056": "cooperation_diplomacy",
    "057": "cooperation_diplomacy",
    "06": "cooperation_diplomacy", "060": "cooperation_diplomacy",
    # 061 / 0611-0616 = trade/economic sanctions → sanctions_trade, not cooperation
    "061": "sanctions_trade", "0611": "sanctions_trade", "0612": "sanctions_trade",
    "0613": "sanctions_trade", "0614": "sanctions_trade",
    "0615": "sanctions_trade", "0616": "sanctions_trade",
    "062": "cooperation_diplomacy", "063": "cooperation_diplomacy",
    "064": "cooperation_diplomacy",
    "07": "cooperation_diplomacy", "070": "cooperation_diplomacy",
    "071": "cooperation_diplomacy", "072": "cooperation_diplomacy",
    "073": "cooperation_diplomacy", "074": "cooperation_diplomacy",
    "075": "cooperation_diplomacy",
    "08": "cooperation_diplomacy", "080": "cooperation_diplomacy",
    "081": "cooperation_diplomacy", "082": "cooperation_diplomacy",
    "083": "cooperation_diplomacy", "084": "cooperation_diplomacy",
    "085": "cooperation_diplomacy", "086": "cooperation_diplomacy",
    "09": "cooperation_diplomacy", "090": "cooperation_diplomacy",
    "091": "cooperation_diplomacy", "092": "cooperation_diplomacy",
    "093": "cooperation_diplomacy", "094": "cooperation_diplomacy",

    "10": "policy_statement", "100": "policy_statement",
    "101": "policy_statement", "102": "policy_statement",
    "103": "policy_statement", "104": "policy_statement",
    "105": "policy_statement", "106": "policy_statement",
    "107": "policy_statement",
    "11": "policy_statement", "110": "policy_statement",
    "111": "policy_statement", "112": "policy_statement",
    "1121": "policy_statement", "1122": "policy_statement",
    "1123": "policy_statement", "1124": "policy_statement",
    "1125": "policy_statement",
    "113": "policy_statement", "114": "policy_statement",
    "115": "policy_statement", "116": "policy_statement",
    "12": "policy_statement", "120": "policy_statement",
    "121": "policy_statement", "122": "policy_statement",
    "13": "policy_statement", "130": "policy_statement",
    "131": "policy_statement", "132": "policy_statement",
    "133": "policy_statement", "134": "policy_statement",
    "135": "policy_statement", "136": "policy_statement",
    "137": "policy_statement", "138": "policy_statement",
    "139": "policy_statement",

    "14": "political_instability", "140": "political_instability",
    "141": "political_instability", "142": "political_instability",
    "143": "political_instability", "144": "political_instability",
    "145": "political_instability", "146": "political_instability",

    "15": "coercion", "150": "coercion", "151": "coercion",
    "152": "coercion", "153": "coercion", "154": "coercion",
    "155": "coercion", "156": "coercion",

    "16": "diplomatic_tension", "160": "diplomatic_tension",
    "161": "diplomatic_tension", "162": "diplomatic_tension",
    # 163/164 = impose embargo/sanctions → sanctions_trade, not diplomatic_tension
    "163": "sanctions_trade", "1631": "sanctions_trade", "1632": "sanctions_trade",
    "1633": "sanctions_trade", "1634": "sanctions_trade",
    "164": "sanctions_trade", "1641": "sanctions_trade", "1642": "sanctions_trade",
    "1643": "sanctions_trade", "1644": "sanctions_trade",
    "165": "diplomatic_tension", "166": "diplomatic_tension",
    "17": "diplomatic_tension", "170": "diplomatic_tension",
    "171": "diplomatic_tension", "172": "diplomatic_tension",
    "173": "diplomatic_tension", "174": "diplomatic_tension",
    "175": "diplomatic_tension", "176": "diplomatic_tension",

    "18": "military_conflict", "180": "military_conflict",
    "181": "military_conflict", "182": "military_conflict",
    "183": "military_conflict",
    "19": "military_conflict", "190": "military_conflict",
    "191": "military_conflict", "192": "military_conflict",
    "193": "military_conflict", "194": "military_conflict",
    "195": "military_conflict", "196": "military_conflict",
    "20": "military_conflict", "200": "military_conflict",
    "201": "military_conflict", "202": "military_conflict",
    "203": "military_conflict", "204": "military_conflict",
}


def _build_prefix_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for code, category in CAMEO_TASK_MAP.items():
        lookup[code] = category
        for length in (2, 3, 4):
            prefix = code[:length]
            if prefix not in lookup:
                lookup[prefix] = category
    return lookup

_CAMEO_PREFIX_LOOKUP: dict[str, str] = _build_prefix_lookup()


def _assign_task_categories(df: pd.DataFrame) -> pd.Series:
    """Vectorized CAMEO → task_category. Falls back 4-char → 3-char → 2-char → other_political."""
    ec  = df["EventCode"].astype(str).str.strip()
    ebc = df["EventBaseCode"].astype(str).str.strip()
    result = pd.Series("other_political", index=df.index)

    for length in (4, 3, 2):
        unknown = result == "other_political"
        if not unknown.any():
            break
        mapped = ec[unknown].str[:length].map(_CAMEO_PREFIX_LOOKUP)
        result[unknown] = mapped.fillna("other_political")

        unknown = result == "other_political"
        if not unknown.any():
            break
        mapped = ebc[unknown].str[:length].map(_CAMEO_PREFIX_LOOKUP)
        result[unknown] = mapped.fillna("other_political")

    return result


# Countries whose events meaningfully move the S&P 500
# IMPORTANT: GDELT uses FIPS-2 codes, NOT ISO-3.
# The previous version used ISO-3 (CHN, RUS, DEU...) which caused the
# geography filter to miss most non-US events — only US/USA matched.
# Reference: https://www.gdeltproject.org/data/lookups/FIPS.country.txt
HIGH_IMPACT_COUNTRIES: frozenset[str] = frozenset({
    "US", "USA",  # United States  (both forms appear in GDELT)
    "CH",         # China          (ISO: CHN)
    "RS",         # Russia         (ISO: RUS)
    "GM",         # Germany        (ISO: DEU)
    "UK",         # United Kingdom (ISO: GBR)
    "JA",         # Japan          (ISO: JPN)
    "FR",         # France         (ISO: FRA)
    "IR",         # Iran           (ISO: IRN)
    "KN",         # North Korea    (ISO: PRK)
    "IS",         # Israel         (ISO: ISR)
    "SA",         # Saudi Arabia   (ISO: SAU)
    "UP",         # Ukraine        (ISO: UKR)
    "TW",         # Taiwan         (ISO: TWN)
    "IN",         # India          (ISO: IND)
    "BR",         # Brazil         (ISO: BRA)
    "TU",         # Turkey         (ISO: TUR)
    "KS",         # South Korea    (ISO: KOR)
    "MX",         # Mexico         (ISO: MEX)
})


def generate_snapshot_urls(cfg: PipelineConfig) -> list[dict]:
    """Generates (trading day × snapshot hour) URLs. Returns list of {date, hour, url} dicts."""
    start   = datetime.strptime(cfg.start_date, "%Y-%m-%d")
    end     = datetime.strptime(cfg.end_date,   "%Y-%m-%d")
    entries: list[dict] = []
    current = start

    while current <= end:
        if current.weekday() < 5:
            for hour in cfg.snapshot_hours:
                ts  = current.strftime("%Y%m%d") + f"{hour:02d}0000"
                url = f"http://data.gdeltproject.org/gdeltv2/{ts}.export.CSV.zip"
                entries.append({
                    "date": current.strftime("%Y-%m-%d"),
                    "hour": hour,
                    "url" : url,
                })
        current += timedelta(days=1)

    return entries


def fetch_one_file(url_info: dict, cfg: PipelineConfig) -> Optional[pd.DataFrame]:
    """
    Downloads one GDELT zip, parses TSV, applies CAMEO + geography filters.

    min_mentions filter is applied post-dedup in deduplicate_and_clean() so we
    see the highest mention count version of each event before filtering.
    """
    url        = url_info["url"]
    event_date = url_info["date"]

    for attempt in range(cfg.retry_attempts):
        try:
            response = requests.get(
                url,
                timeout=cfg.request_timeout,
                headers={"User-Agent": "Academic-Research-NTU-CCDS"},
            )
            if response.status_code == 404:
                return None
            if response.status_code != 200:
                time.sleep(cfg.retry_delay * (attempt + 1))
                continue

            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                with zf.open(zf.namelist()[0]) as f:
                    df = pd.read_csv(
                        f, sep="\t", header=None, names=GDELT_COLS,
                        dtype=str, on_bad_lines="skip", low_memory=False,
                    )

            df = df[[c for c in KEEP_COLS if c in df.columns]].copy()

            # Filter 1: political CAMEO codes
            df["EventCode"]     = df["EventCode"].astype(str).str.strip()
            df["EventBaseCode"] = df["EventBaseCode"].astype(str).str.strip()
            all_codes = set(_CAMEO_PREFIX_LOOKUP.keys())
            mask_political = (
                df["EventCode"].isin(all_codes)
                | df["EventBaseCode"].isin(all_codes)
                | df["EventCode"].str[:4].isin(all_codes)
                | df["EventCode"].str[:3].isin(all_codes)
                | df["EventCode"].str[:2].isin(all_codes)
            )

            # Filter 2: US or high-impact country
            mask_us = (
                df["Actor1CountryCode"].isin({"US", "USA"})
                | df["Actor2CountryCode"].isin({"US", "USA"})
                | df["ActionGeo_CountryCode"].isin({"US", "USA"})
            )
            mask_impact = (
                df["Actor1CountryCode"].isin(HIGH_IMPACT_COUNTRIES)
                | df["Actor2CountryCode"].isin(HIGH_IMPACT_COUNTRIES)
            )

            if cfg.us_only:
                df = df[mask_political & mask_us]
            else:
                df = df[mask_political & (mask_us | mask_impact)]

            if df.empty:
                return None

            df["date"] = event_date
            for col in ["NumMentions", "GoldsteinScale", "AvgTone",
                        "NumSources", "NumArticles"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            return df

        except zipfile.BadZipFile:
            return None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            time.sleep(cfg.retry_delay * (attempt + 1))
        except Exception as exc:
            logging.getLogger("gdelt_pipeline").debug(
                f"Unexpected error on {url}: {exc}", exc_info=True
            )
            time.sleep(cfg.retry_delay)

    return None


def fetch_gdelt_parallel(
    entries: list[dict],
    cfg: PipelineConfig,
    log: logging.Logger,
) -> pd.DataFrame:
    """Downloads all GDELT snapshots concurrently with batch checkpointing."""
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    batches = [entries[i:i + cfg.batch_size]
               for i in range(0, len(entries), cfg.batch_size)]
    all_frames:   list[pd.DataFrame] = []
    total_events  = 0
    total_failed  = 0
    is_tty        = sys.stdout.isatty()

    log.info(
        f"Parallel download: {len(entries)} snapshots | "
        f"{len(batches)} batches | {cfg.max_workers} workers"
    )

    pbar = tqdm(
        total=len(entries), desc="GDELT snapshots", unit="file",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        disable=not is_tty,
    )

    for batch_idx, batch in enumerate(batches):
        ckpt_path = Path(cfg.checkpoint_dir) / f"batch_{batch_idx:04d}.parquet"

        if ckpt_path.exists():
            df_ckpt = pd.read_parquet(ckpt_path)
            all_frames.append(df_ckpt)
            total_events += len(df_ckpt)
            pbar.update(len(batch))
            pbar.set_postfix({"events": f"{total_events:,}", "status": "resumed"})
            continue

        batch_frames: list[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            future_map = {executor.submit(fetch_one_file, u, cfg): u for u in batch}
            for future in as_completed(future_map):
                pbar.update(1)
                try:
                    result = future.result()
                    if result is not None and not result.empty:
                        batch_frames.append(result)
                        total_events += len(result)
                    else:
                        total_failed += 1
                except Exception as exc:
                    total_failed += 1
                    log.debug(f"Future exception: {exc}")
                pbar.set_postfix({
                    "events": f"{total_events:,}",
                    "failed": total_failed,
                    "batch" : f"{batch_idx + 1}/{len(batches)}",
                })

        if batch_frames:
            batch_df = pd.concat(batch_frames, ignore_index=True)
            batch_df.to_parquet(ckpt_path, index=False)
            all_frames.append(batch_df)
            log.info(
                f"Batch {batch_idx + 1}/{len(batches)} saved "
                f"+{len(batch_df):,} (total: {total_events:,})"
            )
        else:
            log.warning(f"Batch {batch_idx + 1} returned no data")

    pbar.close()

    if not all_frames:
        log.error("No data collected. Check connectivity:")
        log.error("  curl -I http://data.gdeltproject.org/gdeltv2/20240103120000.export.CSV.zip")
        return pd.DataFrame()

    raw_df = pd.concat(all_frames, ignore_index=True)
    log.info(f"Raw events: {len(raw_df):,} (before dedup + min_mentions filter)")
    return raw_df


# ─────────────────────────────────────────────────────────────────────────────
def deduplicate_and_clean(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    log: logging.Logger,
) -> pd.DataFrame:
    """Deduplicates by GLOBALEVENTID, applies min_mentions, assigns task_category and primary_country."""
    if df.empty:
        return df

    log.info("Deduplicating...")
    df["NumMentions"] = pd.to_numeric(df["NumMentions"], errors="coerce").fillna(0)
    df = df.sort_values("NumMentions", ascending=False)
    before = len(df)
    df = df.drop_duplicates(subset=["GLOBALEVENTID"], keep="first")
    log.info(f"Dedup: {before:,} -> {len(df):,} unique events")

    df = df[df["NumMentions"] >= cfg.min_mentions]
    log.info(f"After min_mentions={cfg.min_mentions}: {len(df):,} events")

    # Assign before Goldstein filter so we can target cooperation_diplomacy specifically
    df["task_category"] = _assign_task_categories(df)

    # cooperation_diplomacy at min_mentions=10 still captures huge volumes of
    # routine events (standard bilateral meetings, UN procedural votes, minor
    # aid transfers) with near-zero S&P impact. Keep only events with
    # |GoldsteinScale| >= cfg.coop_goldstein_min (default 7.0 = major peace
    # deals, defence pacts, significant agreements). Other categories (conflict,
    # coercion) are already rare and high-signal, so no Goldstein filter applied.
    if cfg.coop_goldstein_min > 0.0:
        df["GoldsteinScale"] = pd.to_numeric(df["GoldsteinScale"], errors="coerce")
        before_gold = len(df)
        coop_mask  = df["task_category"] == "cooperation_diplomacy"
        gold_mask  = df["GoldsteinScale"].abs() >= cfg.coop_goldstein_min
        df = df[~coop_mask | gold_mask]
        dropped_gold = before_gold - len(df)
        log.info(
            f"Goldstein filter (coop |scale|>={cfg.coop_goldstein_min}): "
            f"dropped {dropped_gold:,}, remaining {len(df):,}"
        )

    def _clean(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return (df[col].astype(str).str.strip()
                .replace({"nan": np.nan, "None": np.nan, "": np.nan}))

    df["primary_country"] = (
        _clean("ActionGeo_CountryCode")
        .fillna(_clean("Actor1CountryCode"))
        .fillna(_clean("Actor2CountryCode"))
        .fillna("UNKNOWN")
    )
    # GDELT geo fields sometimes write "USA" instead of FIPS-2 "US" — normalise
    df["primary_country"] = df["primary_country"].replace({"USA": "US"})

    rename_map = {
        "GLOBALEVENTID"        : "event_id",
        "SQLDATE"              : "sql_date",
        "Actor1Name"           : "actor1",
        "Actor1CountryCode"    : "actor1_country",
        "Actor2Name"           : "actor2",
        "Actor2CountryCode"    : "actor2_country",
        "EventCode"            : "event_code",
        "EventBaseCode"        : "event_base_code",
        "EventRootCode"        : "event_root_code",
        "QuadClass"            : "quad_class",
        "GoldsteinScale"       : "goldstein_scale",
        "NumMentions"          : "num_mentions",
        "NumSources"           : "num_sources",
        "NumArticles"          : "num_articles",
        "AvgTone"              : "avg_tone",
        "ActionGeo_FullName"   : "location",
        "ActionGeo_CountryCode": "action_country",
        "SOURCEURL"            : "source_url",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    quad_labels = {
        "1": "verbal_cooperation", "2": "material_cooperation",
        "3": "verbal_conflict",    "4": "material_conflict",
    }
    if "quad_class" in df.columns:
        df["quad_class_label"] = df["quad_class"].astype(str).map(quad_labels).fillna("unknown")

    log.info(f"Clean dataset: {len(df):,} events")
    return df


def align_with_sp500(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    log: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Downloads S&P 500, computes returns, merges with events.
    Returns (merged_df, sp500_daily_df). sp500_daily_df is reused by save_outputs().

    Volatility .shift(1) is applied before merge so at event date T, volatility_5d/20d
    reflect T-1 and earlier only — without this, the rolling window includes T's own
    return (derived from same-day close), producing same-day leakage.

    merge_asof(direction='forward') maps event T → T+1 return (no look-ahead).
    """
    if df.empty:
        return df, pd.DataFrame()

    log.info(f"Downloading S&P 500 ({cfg.ticker})...")

    date_min = df["date"].min()
    date_max = df["date"].max()
    start = (date_min - timedelta(days=30)).strftime("%Y-%m-%d")
    end   = (date_max + timedelta(days=10)).strftime("%Y-%m-%d")

    px_daily = yf.download(cfg.ticker, start=start, end=end,
                           interval="1d", progress=False, auto_adjust=True)
    if px_daily.empty:
        log.error("Failed to download S&P 500 daily data")
        return df, pd.DataFrame()

    if isinstance(px_daily.columns, pd.MultiIndex):
        px_daily.columns = [c[0] for c in px_daily.columns]

    px_daily = px_daily.reset_index()
    px_daily.rename(columns={px_daily.columns[0]: "trade_date"}, inplace=True)
    px_daily["trade_date"] = pd.to_datetime(px_daily["trade_date"]).dt.tz_localize(None)

    px_daily["daily_return"]    = px_daily["Close"].pct_change()
    px_daily["next_day_return"] = px_daily["daily_return"].shift(-1)
    px_daily["sp500_close"]     = px_daily["Close"]
    px_daily["abs_return"]      = px_daily["next_day_return"].abs()

    px_daily["volatility_5d"]  = (
        px_daily["daily_return"].rolling(5).std()  * np.sqrt(252)
    ).shift(1)
    px_daily["volatility_20d"] = (
        px_daily["daily_return"].rolling(20).std() * np.sqrt(252)
    ).shift(1)

    px_daily["return_direction_label"] = np.where(
        px_daily["next_day_return"] > 0, "positive", "negative"
    )

    log.info(f"Downloaded {len(px_daily):,} daily bars")

    # yfinance hard limit: 1h data only available for the last 730 days
    hourly_start = max(
        pd.Timestamp(start),
        pd.Timestamp.today().normalize() - pd.Timedelta(days=729),
    ).strftime("%Y-%m-%d")
    log.info(f"Hourly data window: {hourly_start} -> {end} (clamped to 730-day limit)")
    try:
        px_hourly = yf.download(cfg.ticker, start=hourly_start, end=end,
                                interval="1h", progress=False, auto_adjust=True)
        if not px_hourly.empty:
            if isinstance(px_hourly.columns, pd.MultiIndex):
                px_hourly.columns = [c[0] for c in px_hourly.columns]
            px_hourly = px_hourly.reset_index()
            px_hourly.rename(columns={px_hourly.columns[0]: "datetime"}, inplace=True)
            px_hourly["datetime"]  = pd.to_datetime(px_hourly["datetime"]).dt.tz_localize(None)
            px_hourly["date_only"] = px_hourly["datetime"].dt.date

            intraday_rows = []
            for day, grp in px_hourly.groupby("date_only"):
                grp     = grp.sort_values("datetime").reset_index(drop=True)
                open_px = grp.iloc[0]["Open"]
                r1h     = (grp.iloc[1]["Close"] - open_px) / open_px if len(grp) > 1 else None
                r4h     = (grp.iloc[3]["Close"] - open_px) / open_px if len(grp) > 3 else None
                intraday_rows.append({"trade_date": pd.Timestamp(day),
                                      "return_1h": r1h, "return_4h": r4h})

            intraday_df = pd.DataFrame(intraday_rows)
            intraday_df["return_1h_next"] = intraday_df["return_1h"].shift(-1)
            intraday_df["return_4h_next"] = intraday_df["return_4h"].shift(-1)
            intraday_df["trade_date"]     = pd.to_datetime(intraday_df["trade_date"])
            px_daily = px_daily.merge(
                intraday_df[["trade_date", "return_1h_next", "return_4h_next"]],
                on="trade_date", how="left",
            )
            log.info("1h and 4h intraday returns added")
        else:
            log.warning("Hourly data empty — skipping 1h/4h returns")
    except Exception as e:
        log.warning(f"Hourly data failed ({e}) — skipping 1h/4h returns")

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    market_cols = [
        "trade_date", "next_day_return",
        "volatility_5d", "volatility_20d",
        "return_direction_label",
        "abs_return", "sp500_close",
    ]
    if "return_1h_next" in px_daily.columns:
        market_cols += ["return_1h_next", "return_4h_next"]

    # Normalise both keys to the same datetime resolution (pandas 2+ requires exact match)
    df["date"]             = df["date"].astype("datetime64[us]")
    px_daily["trade_date"] = px_daily["trade_date"].astype("datetime64[us]")

    merged = pd.merge_asof(
        df.sort_values("date"),
        px_daily[market_cols].sort_values("trade_date"),
        left_on="date", right_on="trade_date",
        direction="forward",            # event T -> T+1 return
        tolerance=pd.Timedelta("4D"),   # allow 4-day gap for holiday weeks
    )

    before  = len(merged)
    merged  = merged.dropna(subset=["next_day_return"])
    dropped = before - len(merged)
    if dropped:
        log.info(f"Dropped {dropped:,} events with no matching market data")

    merged = merged.drop(columns=["trade_date"], errors="ignore")
    log.info(f"Market-aligned events: {len(merged):,}")

    return merged, px_daily


def save_outputs(
    df: pd.DataFrame,
    sp500_cache: pd.DataFrame,
    cfg: PipelineConfig,
    log: logging.Logger,
) -> None:
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.final_output, index=False)
    log.info(f"Saved {len(df):,} events -> {cfg.final_output}")

    if not sp500_cache.empty:
        sp500_cache.to_csv(cfg.sp500_output, index=False)
        log.info(f"Saved S&P 500 prices -> {cfg.sp500_output}")
    else:
        log.warning("S&P 500 cache empty — skipping sp500 CSV")


def save_split_datasets(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    log: logging.Logger,
) -> None:
    """
    Splits final_df into two non-overlapping datasets:
      next_day_only.csv  - all events, intraday columns dropped, used for main MAML training.
      with_intraday.csv  - only events where return_1h_next is not null (last ~730 days),
                           keeps 1h/4h returns, used for intraday adaptation experiments.
    Split is on return_1h_next.notna() — no event appears in both files.
    """
    intraday_cols = ["return_1h_next", "return_4h_next"]
    has_intraday  = all(c in df.columns for c in intraday_cols)

    out_daily    = cfg.final_output.replace(".csv", "_next_day_only.csv")
    out_intraday = cfg.final_output.replace(".csv", "_with_intraday.csv")

    if has_intraday:
        intraday_mask = df["return_1h_next"].notna()
        df_intraday   = df[intraday_mask].copy()
        df_daily      = df[~intraday_mask].drop(columns=intraday_cols, errors="ignore").copy()

        df_daily.to_csv(out_daily, index=False)
        log.info(
            f"Dataset 1 (next_day_only)  : {len(df_daily):,} events "
            f"({df_daily['date'].min().date()} -> {df_daily['date'].max().date()}) "
            f"-> {out_daily}"
        )

        df_intraday.to_csv(out_intraday, index=False)
        log.info(
            f"Dataset 2 (with_intraday)  : {len(df_intraday):,} events "
            f"({df_intraday['date'].min().date()} -> {df_intraday['date'].max().date()}) "
            f"-> {out_intraday}"
        )
        print(f"\n  Dataset 1 (next_day_only)  : {len(df_daily):,} events  -> {out_daily}")
        print(f"  Dataset 2 (with_intraday)  : {len(df_intraday):,} events  -> {out_intraday}")
        print(f"  No duplicate events between datasets (split on 1h return availability)")
    else:
        # Hourly data unavailable — save everything as daily-only
        df.to_csv(out_daily, index=False)
        log.warning(
            f"No intraday data available — saved all {len(df):,} events "
            f"as next_day_only -> {out_daily}"
        )
        print(f"\n  Dataset 1 (next_day_only)  : {len(df):,} events  -> {out_daily}")
        print(f"  Dataset 2 (with_intraday)  : skipped (hourly data unavailable)")


def print_task_summary(df: pd.DataFrame, log: logging.Logger) -> None:
    """MAML task viability report. MAML needs ~5-15 support + ~5-15 query = ~20-30 minimum per task."""
    print("\n" + "=" * 70)
    print("  META-LEARNING TASK VIABILITY REPORT")
    print("=" * 70)
    print(f"  Total events : {len(df):,}")
    print(f"  Date range   : {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  Unique dates : {df['date'].dt.date.nunique():,}")

    print("\n  -- Task Categories --")
    for task, n in df["task_category"].value_counts().items():
        tag = "VIABLE  " if n >= 50 else ("MARGINAL" if n >= 20 else "TOO FEW ")
        print(f"  {task:<30} {n:>7,}  [{tag}]")

    print("\n  -- Top Countries --")
    for country, n in df["primary_country"].value_counts().head(15).items():
        print(f"  {country:<12} {n:>7,}  {'[OK]' if n >= 30 else '[LOW]'}")

    print("\n  -- Task x Country Combos (MAML tasks) --")
    tc     = df.groupby(["task_category", "primary_country"]).size()
    viable = tc[tc >= 20].sort_values(ascending=False)
    print(f"  Viable combos (>=20 samples): {len(viable)}")
    for (task, country), n in viable.head(12).items():
        print(f"  {task:<30} x {country:<6} = {n:>5,}")

    print("\n  -- Market Return Statistics --")
    if "next_day_return" in df.columns:
        r = df["next_day_return"]
        print(f"  next_day_return  mean : {r.mean()*100:+.4f}%")
        print(f"  next_day_return  std  : {r.std()*100:.4f}%")
        print(f"  next_day_return  min  : {r.min()*100:+.4f}%")
        print(f"  next_day_return  max  : {r.max()*100:+.4f}%")

    if "return_direction_label" in df.columns:
        pos = (df["return_direction_label"] == "positive").mean() * 100
        print(f"  Positive direction   : {pos:.1f}%  [TARGET LABEL — not a feature]")

    if "return_1h_next" in df.columns:
        r1 = df["return_1h_next"].dropna()
        print(f"  return_1h_next  mean : {r1.mean()*100:+.4f}%  (n={len(r1):,})")
    if "return_4h_next" in df.columns:
        r4 = df["return_4h_next"].dropna()
        print(f"  return_4h_next  mean : {r4.mean()*100:+.4f}%  (n={len(r4):,})")

    if "goldstein_scale" in df.columns:
        g = df["goldstein_scale"].dropna()
        print(f"\n  Goldstein scale  mean : {g.mean():+.3f}  std: {g.std():.3f}")
    if "avg_tone" in df.columns:
        t = df["avg_tone"].dropna()
        print(f"  AvgTone          mean : {t.mean():+.3f}")

    # MAML is sensitive to task imbalance. If cooperative events vastly outnumber
    # conflict events, the meta-learner under-trains on conflict tasks — the ones
    # most likely to produce large S&P moves. Flag if either side is >3x the other.
    print("\n  -- Class Balance (Cooperative vs Conflict) --")
    COOPERATIVE_TASKS = {"cooperation_diplomacy", "policy_statement"}
    CONFLICT_TASKS    = {
        "military_conflict", "sanctions_trade",
        "diplomatic_tension", "coercion", "political_instability",
    }

    counts      = df["task_category"].value_counts()
    n_coop      = int(counts[counts.index.isin(COOPERATIVE_TASKS)].sum())
    n_conflict  = int(counts[counts.index.isin(CONFLICT_TASKS)].sum())
    n_other     = int(counts[~counts.index.isin(COOPERATIVE_TASKS | CONFLICT_TASKS)].sum())
    total_known = n_coop + n_conflict

    if total_known > 0:
        pct_coop     = n_coop    / total_known * 100
        pct_conflict = n_conflict / total_known * 100
        ratio        = (n_coop / n_conflict) if n_conflict > 0 else float("inf")

        print(f"  Cooperative events : {n_coop:>7,}  ({pct_coop:.1f}%)")
        print(f"  Conflict events    : {n_conflict:>7,}  ({pct_conflict:.1f}%)")
        print(f"  other_political    : {n_other:>7,}")
        print(f"  Coop : Conflict    : {ratio:.2f} : 1")

        # Per-task breakdown within each group
        print()
        for task in sorted(COOPERATIVE_TASKS):
            n = int(counts.get(task, 0))
            print(f"    [coop]     {task:<30} {n:>6,}")
        for task in sorted(CONFLICT_TASKS):
            n = int(counts.get(task, 0))
            print(f"    [conflict] {task:<30} {n:>6,}")

        # Imbalance warning
        print()
        if ratio > 3.0:
            print(f"  [WARNING] Cooperative events are {ratio:.1f}x more than conflict.")
            print(f"  MAML will under-train on conflict tasks.")
            print(f"  Consider: weighted task sampling in your TaskSampler,")
            print(f"  or note this explicitly in your methodology section.")
        elif ratio < (1 / 3.0):
            inv = 1 / ratio if ratio > 0 else float("inf")
            print(f"  [WARNING] Conflict events are {inv:.1f}x more than cooperative.")
            print(f"  MAML will under-train on cooperative tasks.")
            print(f"  Consider: weighted task sampling in your TaskSampler.")
        else:
            print(f"  [OK] Balance within 3:1 threshold — no reweighting needed.")
    else:
        print("  [ERROR] No events with known task categories found.")

    print("\n  -- Sanity Checks --")
    print(f"  Null next_day_return : {df['next_day_return'].isnull().sum():,} (should be 0)")
    print(f"  Duplicate event_ids  : {df['event_id'].duplicated().sum():,} (should be 0)")
    apd = df.groupby(df["date"].dt.date).size()
    print(f"  Avg events per day   : {apd.mean():.1f}")
    print(f"  Max events one day   : {apd.max():,}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTIVITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
def connectivity_check(log: logging.Logger) -> bool:
    print("\nConnectivity check...")
    test_url = "http://data.gdeltproject.org/gdeltv2/20240103120000.export.CSV.zip"
    try:
        r = requests.head(test_url, timeout=10)
        if r.status_code == 200:
            print("  [OK] GDELT reachable")
        else:
            print(f"  [FAIL] GDELT HTTP {r.status_code}")
            return False
    except Exception as e:
        print(f"  [FAIL] GDELT unreachable: {e}")
        print("  On NTU network? Try home WiFi or hotspot.")
        return False

    try:
        test = yf.download("^GSPC", period="2d", progress=False)
        if not test.empty:
            print("  [OK] yfinance reachable")
        else:
            print("  [FAIL] yfinance returned empty")
            return False
    except Exception as e:
        print(f"  [FAIL] yfinance: {e}")
        return False

    print("  All checks passed\n")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(cfg: Optional[PipelineConfig] = None) -> None:
    """
    Pipeline entry point. Accepts an optional PipelineConfig for unit tests
    or custom runs. If None, uses default config (end_date = today).
    """
    if cfg is None:
        cfg = PipelineConfig()  # end_date = today

    log        = setup_logging(cfg)
    wall_start = time.time()
    today      = date.today().strftime("%Y-%m-%d")

    if cfg.end_date > today:
        log.warning(
            f"end_date {cfg.end_date} is future. "
            f"yfinance silently truncates to today ({today}). "
            f"GDELT snapshots for future dates will 404 (expected, not an error)."
        )

    print("=" * 70)
    print("  GDELT 2.0 POLITICAL EVENTS PIPELINE  v2")
    print(f"  Date range    : {cfg.start_date} -> {cfg.end_date}  (today: {today})")
    print(f"  Snapshots/day : {len(cfg.snapshot_hours)}  "
          f"({', '.join(f'{h:02d}:00 UTC' for h in cfg.snapshot_hours)})")
    print(f"  Workers       : {cfg.max_workers}")
    print(f"  Output        : {cfg.final_output}")
    print("=" * 70)

    s = datetime.strptime(cfg.start_date, "%Y-%m-%d")
    e = datetime.strptime(min(cfg.end_date, today), "%Y-%m-%d")
    trading_days = sum(1 for i in range((e - s).days + 1)
                       if (s + timedelta(days=i)).weekday() < 5)
    n_snapshots  = trading_days * len(cfg.snapshot_hours)
    done_batches = (len(list(Path(cfg.checkpoint_dir).glob("batch_*.parquet")))
                    if Path(cfg.checkpoint_dir).exists() else 0)
    done_snaps   = done_batches * cfg.batch_size

    print(f"\n  Trading days   : {trading_days:,}")
    print(f"  Total snapshots: {n_snapshots:,}")
    print(f"  Checkpointed   : ~{done_snaps:,}")
    print(f"  Remaining      : ~{max(0, n_snapshots - done_snaps):,}")
    print(f"\n  Estimated runtime: ~3-5 hours full run (M4 Pro, 20 workers)")
    print(f"  Note: MPS GPU does NOT help — this is network I/O.")

    if not connectivity_check(log):
        print("Pipeline aborted — fix connectivity first.")
        sys.exit(1)

    entries  = generate_snapshot_urls(cfg)
    log.info(f"Generated {len(entries):,} snapshot URLs")

    raw_df   = fetch_gdelt_parallel(entries, cfg, log)
    if raw_df.empty:
        log.error("No events collected — aborting.")
        sys.exit(1)

    clean_df              = deduplicate_and_clean(raw_df, cfg, log)
    final_df, sp500_cache = align_with_sp500(clean_df, cfg, log)
    save_outputs(final_df, sp500_cache, cfg, log)
    save_split_datasets(final_df, cfg, log)
    print_task_summary(final_df, log)

    elapsed = time.time() - wall_start
    print(f"\nComplete in {elapsed / 60:.1f} min")
    print(f"Full dataset : {cfg.final_output}  ({len(final_df):,} events)\n")


if __name__ == "__main__":
    main()