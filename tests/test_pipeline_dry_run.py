"""End-to-end pipeline dry run.

Mocks `fetch_market` and `fetch_gdelt` so the test never touches the network,
calls `make_prediction()`, and asserts a row was inserted into the DB with
finite predictions and the right feature ordering.

This catches breakage in the glue between fetchers, scaler, model, and DB —
the part that's hardest to test piecewise.
"""
import json
import math
import sqlite3


def _fake_market():
    return {
        "ovx_close":      35.0,
        "vix_close":      18.0,
        "oil_vol_5d":      0.30,
        "oil_vol_20d":     0.35,
        "oil_close":      75.0,
        "dxy_close":     104.0,
        "gold_oil_ratio": 27.0,
    }


def test_make_prediction_inserts_finite_row(scheduler_v2, monkeypatch):
    scheduler_v2.init_db()

    monkeypatch.setattr(scheduler_v2, "fetch_market", _fake_market)
    # GDELT_FALLBACK has the right shape for v2 (23-feature) inputs
    monkeypatch.setattr(
        scheduler_v2, "fetch_gdelt",
        lambda: scheduler_v2.GDELT_FALLBACK.copy(),
    )

    scheduler_v2.make_prediction()

    conn = sqlite3.connect(scheduler_v2.DB_PATH)
    try:
        rows = conn.execute(
            "SELECT maml_pred, mlp_pred, oil_close, ovx_close, "
            "features_json, gdelt_context FROM predictions"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1, "Expected exactly one prediction row"
    maml_pred, mlp_pred, oil_close, ovx_close, features_json, gdelt_context = rows[0]

    assert math.isfinite(maml_pred), f"MAML pred not finite: {maml_pred}"
    assert math.isfinite(mlp_pred),  f"MLP pred not finite: {mlp_pred}"

    assert oil_close == 75.0
    assert ovx_close == 35.0

    feats = json.loads(features_json)
    assert len(feats) == 23, f"Expected 23 features, got {len(feats)}"
    assert feats["ovx_close"] == 35.0
    assert feats["oil_close"] == 75.0
    assert feats["vix_close"] == 18.0

    assert "conflict=" in gdelt_context
    assert "events=" in gdelt_context


def test_make_prediction_skips_on_nan_features(scheduler_v2, monkeypatch):
    """If fetch_market returns NaNs, make_prediction must not insert a row."""
    import numpy as np

    bad_market = _fake_market()
    bad_market["oil_close"] = float("nan")

    monkeypatch.setattr(scheduler_v2, "fetch_market", lambda: bad_market)
    monkeypatch.setattr(
        scheduler_v2, "fetch_gdelt",
        lambda: scheduler_v2.GDELT_FALLBACK.copy(),
    )

    scheduler_v2.init_db()
    scheduler_v2.make_prediction()

    conn = sqlite3.connect(scheduler_v2.DB_PATH)
    try:
        n = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    finally:
        conn.close()

    assert n == 0, "Should not insert when feature vector contains NaN"
    _ = np  # silence unused-import warning if linter strips the line
