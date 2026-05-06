"""Scheduler bootstrap tests.

Confirms scheduler_v2 can be imported without entering its main loop, that
its DB schema is created correctly, and that its pure helpers map known
inputs to expected categories.
"""
import json
import sqlite3


def test_scheduler_v2_imports(scheduler_v2):
    """If we got here without the session fixture timing out, import worked."""
    assert hasattr(scheduler_v2, "make_prediction")
    assert hasattr(scheduler_v2, "init_db")
    assert hasattr(scheduler_v2, "fetch_gdelt")


def test_scheduler_v2_feature_dimensions(scheduler_v2):
    assert scheduler_v2.INPUT_DIM == 23
    assert len(scheduler_v2.FEATURE_COLS) == 23


def test_init_db_creates_predictions_table(scheduler_v2):
    scheduler_v2.init_db()

    conn = sqlite3.connect(scheduler_v2.DB_PATH)
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


def test_log_prediction_inserts_row(scheduler_v2):
    scheduler_v2.init_db()
    scheduler_v2.log_prediction(
        ts="2026-01-01T12:00:00",
        maml_pred=0.45,
        mlp_pred=0.50,
        oil_close=75.0,
        ovx_close=35.0,
        n_support=10,
        features={c: 0.0 for c in scheduler_v2.FEATURE_COLS},
        gdelt_context="conflict=50% | tone=-2.0 | events=24",
    )

    conn = sqlite3.connect(scheduler_v2.DB_PATH)
    try:
        rows = conn.execute(
            "SELECT maml_pred, mlp_pred, oil_close, ovx_close, features_json "
            "FROM predictions"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    maml_pred, mlp_pred, oil_close, ovx_close, features_json = rows[0]
    assert maml_pred == 0.45
    assert mlp_pred == 0.50
    assert oil_close == 75.0
    assert ovx_close == 35.0
    feats = json.loads(features_json)
    assert len(feats) == 23


def test_assign_oil_region_fips(scheduler_v2):
    fn = scheduler_v2.assign_oil_region_fips
    assert fn("US") == "US"
    assert fn("USA") == "US"
    assert fn("SA") == "middle_east"     # Saudi Arabia
    assert fn("IR") == "middle_east"     # Iran
    assert fn("RS") == "oil_producer"    # Russia
    assert fn("VE") == "oil_producer"    # Venezuela
    assert fn("CH") == "oil_consumer"    # China
    assert fn("JA") == "oil_consumer"    # Japan
    assert fn("XX") == "other"
    assert fn("") == "other"
    assert fn(None) == "other"


def test_get_task_category(scheduler_v2):
    fn = scheduler_v2.get_task_category
    assert fn("180") == "military_conflict"          # falls back to "18"
    assert fn("18") == "military_conflict"
    assert fn("01") == "cooperation_diplomacy"
    assert fn("061") == "sanctions_trade"
    assert fn("99") == "other_political"
    assert fn("") == "other_political"


def test_should_predict_when_db_empty(scheduler_v2):
    """Fresh DB should always trigger a prediction on first cycle."""
    scheduler_v2.init_db()
    assert scheduler_v2.should_predict() is True


def test_should_predict_false_when_recent_prediction_exists(scheduler_v2):
    """A prediction made <1h ago must not trigger another."""
    from datetime import datetime, timedelta

    scheduler_v2.init_db()
    recent_ts = (datetime.utcnow() - timedelta(minutes=30)).isoformat()
    scheduler_v2.log_prediction(
        ts=recent_ts,
        maml_pred=0.40, mlp_pred=0.45,
        oil_close=75.0, ovx_close=35.0,
        n_support=5,
        features={c: 0.0 for c in scheduler_v2.FEATURE_COLS},
    )
    assert scheduler_v2.should_predict() is False


def test_should_predict_true_after_1_hour(scheduler_v2):
    """A prediction made >1h ago must trigger another."""
    from datetime import datetime, timedelta

    scheduler_v2.init_db()
    old_ts = (datetime.utcnow() - timedelta(hours=2)).isoformat()
    scheduler_v2.log_prediction(
        ts=old_ts,
        maml_pred=0.40, mlp_pred=0.45,
        oil_close=75.0, ovx_close=35.0,
        n_support=5,
        features={c: 0.0 for c in scheduler_v2.FEATURE_COLS},
    )
    assert scheduler_v2.should_predict() is True


def test_get_support_set_uses_only_resolved_actuals(scheduler_v2, monkeypatch):
    """Predictions with NULL actual_rvol must be excluded from the support set.

    Inserts 5 resolved + 3 unresolved predictions, disables seed loading,
    then asserts get_support_set returns exactly 5 examples.
    """
    import sqlite3

    scheduler_v2.init_db()

    # Force the seed-loading path to fail so we test the live-only branch
    monkeypatch.setattr(scheduler_v2, "SEED_PATH", "/nonexistent/seed.csv")

    feats = {c: 0.0 for c in scheduler_v2.FEATURE_COLS}

    # 5 resolved predictions (will get actual_rvol set below)
    for hour in range(5):
        scheduler_v2.log_prediction(
            ts=f"2026-01-01T{hour:02d}:00:00",
            maml_pred=0.40, mlp_pred=0.45,
            oil_close=75.0, ovx_close=35.0,
            n_support=0, features=feats,
        )
    # 3 unresolved predictions (actual_rvol stays NULL)
    for hour in range(5, 8):
        scheduler_v2.log_prediction(
            ts=f"2026-01-01T{hour:02d}:00:00",
            maml_pred=0.40, mlp_pred=0.45,
            oil_close=75.0, ovx_close=35.0,
            n_support=0, features=feats,
        )

    # Mark the first 5 rows as resolved
    conn = sqlite3.connect(scheduler_v2.DB_PATH)
    conn.execute("UPDATE predictions SET actual_rvol = ? WHERE id <= 5", (0.30,))
    conn.commit()
    conn.close()

    support, n_support = scheduler_v2.get_support_set()

    assert n_support == 5, (
        f"Expected only the 5 resolved predictions, got {n_support}"
    )
    assert support is not None, "Support set should not be None with 5 resolved actuals"
    sup_X, sup_y = support
    assert sup_X.shape[0] == 5
    assert sup_y.shape[0] == 5
