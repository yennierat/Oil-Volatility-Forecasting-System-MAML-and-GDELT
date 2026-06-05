MAML Oil Price Volatility Prediction
=====================================

This project implements a MAML (Model-Agnostic Meta-Learning) approach to
forecasting 4-hour realized oil price volatility using GDELT political event
features combined with market data (OVX, VIX, Brent crude, DXY, gold).

Two model versions are included:
  v1 — 15 features (7 market + 8 GDELT global aggregates)
  v2 — 23 features (v1 features + 8 per-region GDELT aggregates for
                    Middle East and Oil Producers)

The live system makes hourly predictions during OVX market hours, adapts
using recent resolved predictions as a support set, and includes a paper
trading module that simulates straddle positions based on predicted volatility.


FOLDER STRUCTURE
================
code/                       Offline ML pipeline (training + evaluation)
  gdelt_pipeline.py            Fetches raw GDELT events + aligns with market data
  build_maml_dataset_v1.py     Builds the v1 (15-feature) MAML dataset
  build_maml_dataset_v2.py     Builds the v2 (23-feature) MAML dataset
  get_seed_data.py             Builds v1 seed CSV for live MAML adaptation
  get_seed_data_v2.py          Builds v2 seed CSV for live MAML adaptation
  supervised_learning.py       v1 MLP pretraining
  maml_training.py             v1 MAML meta-training
  evaluation_v1.py             v1 offline evaluation
  supervised_learning_v3.py    v2 MLP pretraining
  MAML_training_v2.py          v2 MAML meta-training
  evaluation_v2.py             v2 offline evaluation

  live_deployment/          Live prediction system
    app_v1.py                  v1 Streamlit dashboard + paper trading dashboard
    app_v2.py                  v2 Streamlit dashboard + paper trading dashboard
    scheduler_v1.py            v1 background prediction scheduler (1-hour cycle)
    scheduler_v2.py            v2 background prediction scheduler (1-hour cycle)
    predictions.db             v1 logged predictions + trades (SQLite)
    predictions_v2.db          v2 logged predictions + trades (SQLite)
    predictions_log.csv        Older CSV log (legacy)
    scheduler_v1.log           v1 scheduler runtime log (auto-rotated, gitignored)
    scheduler_v2.log           v2 scheduler runtime log (auto-rotated, gitignored)

tests/                      pytest test suite (~25 tests, verified passing in Docker)
  conftest.py                  Shared fixtures (scheduler import, DB redirection)
  test_model_contract.py       Model load + forward-pass smoke tests (v1 and v2)
  test_scheduler_bootstrap.py  v2 scheduler import, DB schema, helpers
  test_scheduler_v1_smoke.py   v1 scheduler import, DB schema, helpers
  test_pipeline_dry_run.py     End-to-end prediction cycle (mocked fetchers)
  test_gdelt.py                GDELT parsing (synthetic + opt-in live API test)

data/                       Generated CSVs (gitignored -- rebuild via offline pipeline)
  dataset_daily.csv               Raw daily political events       [via gdelt_pipeline.py]
  dataset_intraday.csv            Raw intraday political events    [via gdelt_pipeline.py]
  dataset_maml_daily.csv          v1 MLP pretraining dataset       [via build_maml_dataset_v1.py]
  dataset_maml_intraday.csv       v1 MAML training dataset         [via build_maml_dataset_v1.py]
  dataset_maml_daily_v2.csv       v2 MLP pretraining dataset       [via build_maml_dataset_v2.py]
  dataset_maml_intraday_v2.csv    v2 MAML training dataset         [via build_maml_dataset_v2.py]
  seed_v1.csv                     v1 seed for live MAML support    [via get_seed_data.py]
  seed_v2.csv                     v2 seed for live MAML support    [via get_seed_data_v2.py]

models/                     Pre-trained model artifacts
  v1/
    mlp_pretrained.pth           v1 pretrained MLP backbone
    maml_trained.pth             v1 trained MAML model
    feature_scaler.pkl           v1 feature scaler
  v2/
    mlp_pretrained_v2.pth        v2 pretrained MLP backbone
    maml_trained_v2.pth          v2 trained MAML model
    feature_scaler_v2.pkl        v2 feature scaler

requirements.txt            Python runtime dependencies
requirements-dev.txt        Dev/test dependencies (pytest, ruff)
pytest.ini                  pytest configuration
Dockerfile                  Reproducible CPU-only test environment
.dockerignore               Docker build-context exclusions
README.txt                  This file


SETUP
=====
1. Create and activate a virtual environment:
     python -m venv .venv
     # Windows:  .venv\Scripts\activate
     # macOS/Linux:  source .venv/bin/activate

2. Install dependencies:
     pip install -r requirements.txt


HOW TO RUN — OFFLINE PIPELINE
==============================
All offline scripts run from the code/ directory. They reference data via "../data/"
and models via "../models/v1/" or "../models/v2/".

  cd code

To skip retraining and just evaluate the included pre-trained models:

   python evaluation_v1.py
   python evaluation_v2.py

   Outputs go to code/eval_outputs_v1/ and code/eval_outputs_v2/.

To re-run the full pipeline from scratch:

   1. Fetch raw GDELT data (slow, ~3-5 hours):
        python gdelt_pipeline.py

   2. Build MAML datasets:
        python build_maml_dataset_v1.py    # builds 15-feature dataset
        python build_maml_dataset_v2.py    # builds 23-feature dataset

   3. Pretrain MLPs:
        python supervised_learning.py      # v1 MLP
        python supervised_learning_v3.py   # v2 MLP

   4. Train MAML:
        python maml_training.py            # v1 MAML
        python MAML_training_v2.py         # v2 MAML

   5. Evaluate:
        python evaluation_v1.py
        python evaluation_v2.py


HOW TO RUN — LIVE DEPLOYMENT
=============================
Live deployment scripts run from code/live_deployment/. They reference data via
"../../data/" and models via "../../models/v1/" or "../../models/v2/".

  cd code/live_deployment

Streamlit dashboards (interactive UI for live predictions + paper trading):

   streamlit run app_v1.py
   streamlit run app_v2.py

Background prediction schedulers (auto-prediction every 1 hour during OVX hours):

   python scheduler_v1.py
   python scheduler_v2.py

Both schedulers and dashboards write to predictions.db and predictions_v2.db
respectively. Each prediction row includes the MAML and MLP forecasts, the
realized vol computed 4 hours later, and paper trading columns (trade_direction,
trade_size, trade_pnl).

Scheduler market hours:
- Runs on weekdays only (OVX calculates Mon–Fri)
- Friday cutoff: no predictions after 19:00 UTC (market closes at 22:00 UTC,
  latest usable hourly bar is 21:00, so last valid 4h window starts at 18:00 UTC)
- Brief pause 02:15–02:30 UTC daily (GDELT export gap)

Each scheduler also writes to a rotating log file (scheduler_v1.log /
scheduler_v2.log) in the same folder. Files rotate at 10 MB and keep 5 backups
(~50 MB cap per scheduler). Log files are gitignored.

To follow live output:
   Get-Content scheduler_v2.log -Wait      (PowerShell)
   tail -f scheduler_v2.log                (macOS / Linux)

Verbosity is controlled by `logger.setLevel(...)` near the top of each
scheduler. INFO is the default; switch to DEBUG for per-cycle "not due yet"
chatter, or WARNING to see only problems.


PAPER TRADING
=============
Both schedulers automatically place simulated straddle trades on each prediction:

  BUY straddle  — when MAML predicted vol > 75th percentile of resolved actuals
                   (expecting a volatility spike)
  SELL straddle — when MAML predicted vol < 25th percentile of resolved actuals
                   (expecting vol to stay low)
  No trade      — when predicted vol falls within the 25th–75th percentile range

Thresholds default to (p25=0.2, p75=0.5) until 100 resolved predictions exist,
then update dynamically from the DB.

Trade sizing: 2% of current capital per trade (starting capital: $100,000).

P&L model:
  BUY:  trade_size × (actual_rvol / maml_pred − 1), capped at −trade_size
  SELL: trade_size × (1 − actual_rvol / maml_pred), capped at −2×trade_size

Trade results are stored in the predictions DB and displayed on the Streamlit
dashboard (Cumulative P&L chart, trade bar chart, trade log table, Sharpe ratio).


TESTING
=======
The project includes a pytest suite (~25 tests) covering model loading,
scheduler bootstrap, end-to-end pipeline, and GDELT parsing.

Local run:
  pip install -r requirements-dev.txt
  pytest

Docker run (reproducible CPU-only environment):
  docker build -t maml-oil .
  docker run maml-oil

To include the live GDELT API contract test:
  pytest -m network


LINTING
=======
The project uses ruff for linting and formatting:
  ruff check .       # find issues
  ruff check --fix . # auto-fix safe issues
  ruff format .      # opinionated formatting


NOTES
=====
- gdelt_pipeline.py and the live components need network access (GDELT + yfinance).
- The pretrained models in models/v1/ and models/v2/ were trained on the CSVs in
  data/. You can evaluate immediately without retraining.
- evaluation_v1.py uses GARCH(1,1) as one of the baselines (requires the 'arch'
  package, already in requirements.txt).
- All training scripts are reproducible (SEED=42).
- MAML adaptation at inference uses inner_lr=0.01 and 5 gradient steps,
  matching the training configuration.
