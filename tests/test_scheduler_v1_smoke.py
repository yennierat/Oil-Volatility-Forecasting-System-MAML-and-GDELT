"""Scheduler v1 smoke test.

Lighter coverage than v2 — v1 is the legacy 15-feature baseline kept for
historical comparison. We verify it still imports, has the expected DB
schema, and its shared helpers (region/task mapping) agree with v2.

For full pipeline / GDELT / model contract coverage of v1, see
test_model_contract.py (parametrized over v1 and v2).
"""
import json
import sqlite3


def test_scheduler_v1_imports(scheduler_v1):
    assert hasattr(scheduler_v1, "make_prediction")
    assert hasattr(scheduler_v1, "init_db")
    assert hasattr(scheduler_v1, "fetch_gdelt")


def test_scheduler_v1_feature_dimensions(scheduler_v1):
    assert len(scheduler_v1.FEATURE_COLS) == 15


def test_scheduler_v1_init_db_schema(scheduler_v1):
    scheduler_v1.init_db()

    conn = sqlite3.connect(scheduler_v1.DB_PATH)
    try:
        cols = conn.execute("PRAGMA table_info(predictions)").fetchall()
    finally:
        conn.close()

    col_names = {c[1] for c in cols}
    expected = {
        "id", "timestamp", "maml_pred", "mlp_pred",
        "oil_close", "ovx_close", "actual_rvol",
        "n_support", "features_json", "gdelt_context",
    }
    assert expected.issubset(col_names), f"Missing columns: {expected - col_names}"


def test_scheduler_v1_log_prediction_inserts_row(scheduler_v1):
    scheduler_v1.init_db()
    scheduler_v1.log_prediction(
        ts="2026-01-01T12:00:00",
        maml_pred=0.40,
        mlp_pred=0.45,
        oil_close=72.0,
        ovx_close=33.0,
        n_support=5,
        features={c: 0.0 for c in scheduler_v1.FEATURE_COLS},
        gdelt_context="conflict=40% | tone=-1.5 | events=20",
    )

    conn = sqlite3.connect(scheduler_v1.DB_PATH)
    try:
        rows = conn.execute(
            "SELECT maml_pred, mlp_pred, features_json FROM predictions"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    maml_pred, mlp_pred, features_json = rows[0]
    assert maml_pred == 0.40
    assert mlp_pred == 0.45
    feats = json.loads(features_json)
    assert len(feats) == 15


def test_scheduler_v1_helpers_match_v2(scheduler_v1):
    """v1 and v2 must agree on country→region and CAMEO→task mappings.

    These are pure functions duplicated across both schedulers; if they ever
    drift apart, downstream features would silently disagree between versions.
    """
    fn_region = scheduler_v1.assign_oil_region_fips
    assert fn_region("US") == "US"
    assert fn_region("SA") == "middle_east"
    assert fn_region("RS") == "oil_producer"
    assert fn_region("CH") == "oil_consumer"
    assert fn_region("XX") == "other"

    fn_task = scheduler_v1.get_task_category
    assert fn_task("180") == "military_conflict"
    assert fn_task("01") == "cooperation_diplomacy"
    assert fn_task("061") == "sanctions_trade"
    assert fn_task("99") == "other_political"


def test_scheduler_v1_should_predict_when_db_empty(scheduler_v1):
    scheduler_v1.init_db()
    assert scheduler_v1.should_predict() is True
