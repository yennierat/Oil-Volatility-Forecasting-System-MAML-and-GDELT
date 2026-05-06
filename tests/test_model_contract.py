"""Model contract smoke tests.

Verifies that every saved (scaler, model) pair on disk can be loaded with the
versions of joblib / sklearn / torch declared in requirements.txt, and that a
forward pass through the model produces a finite scalar output.

This is the single highest-value test in the suite: it catches version drift
in PyTorch and scikit-learn, which is the most common cause of a project
working last week and breaking today.
"""
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent


class OilVolatilityMLP(nn.Module):
    """Architecture matching scheduler_v1 (input_dim=15) and scheduler_v2 (input_dim=23)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48), nn.BatchNorm1d(48), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(48, 32),        nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(32, 16),        nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


@pytest.mark.parametrize(
    "version,input_dim,scaler_name,model_name",
    [
        ("v1", 15, "feature_scaler.pkl",     "maml_trained.pth"),
        ("v1", 15, "feature_scaler.pkl",     "mlp_pretrained.pth"),
        ("v2", 23, "feature_scaler_v2.pkl",  "maml_trained_v2.pth"),
        ("v2", 23, "feature_scaler_v2.pkl",  "mlp_pretrained_v2.pth"),
    ],
)
def test_model_loads_and_produces_finite_prediction(
    version, input_dim, scaler_name, model_name
):
    scaler_path = ROOT / "models" / version / scaler_name
    model_path = ROOT / "models" / version / model_name

    assert scaler_path.exists(), f"Scaler not found at {scaler_path}"
    assert model_path.exists(), f"Model not found at {model_path}"

    scaler = joblib.load(scaler_path)
    if hasattr(scaler, "n_features_in_"):
        assert scaler.n_features_in_ == input_dim, (
            f"Scaler expects {scaler.n_features_in_} features, "
            f"version={version} expects {input_dim}"
        )

    model = OilVolatilityMLP(input_dim=input_dim)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    rng = np.random.default_rng(42)
    raw_X = rng.uniform(-1, 1, size=(1, input_dim)).astype(np.float32)
    x_scaled = torch.tensor(scaler.transform(raw_X), dtype=torch.float32)

    with torch.no_grad():
        out = model(x_scaled)

    assert out.shape == (1, 1), f"Expected output shape (1,1), got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), f"Non-finite output: {out}"

    pred_log_rvol = float(out.item())
    pred_rvol = float(np.expm1(pred_log_rvol))
    # Realistic forward-realised vol for crude oil is ~0.1 to ~5. Wide bound to
    # tolerate the synthetic input — we mostly want to catch NaN / inf / huge.
    assert -1 < pred_rvol < 100, f"Unrealistic rvol prediction: {pred_rvol}"
