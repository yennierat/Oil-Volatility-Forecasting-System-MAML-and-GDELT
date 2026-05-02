MAML Oil Price Volatility Prediction
=====================================

This bundle contains the source code, data, trained models, and live deployment
components for a MAML (Model-Agnostic Meta-Learning) approach to oil price
volatility prediction using GDELT political event features.

Two model versions are included:
  v1 — 15 features (7 market + 8 GDELT global aggregates)
  v2 — 23 features (v1 features + 8 per-region GDELT aggregates for
                    Middle East and Oil Producers)


FOLDER STRUCTURE
================
code/                       Offline ML pipeline (training + evaluation)
  gdelt_pipeline.py            Fetches raw GDELT events + aligns with S&P 500
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
    app_v1.py                  v1 Streamlit dashboard
    app_v2.py                  v2 Streamlit dashboard
    scheduler_v1.py            v1 background prediction scheduler (30-min cycle)
    scheduler_v2.py            v2 background prediction scheduler (30-min cycle)
    NLP_sentiment.py           FinBERT-based headline → feature bridge
    predictions.db             v1 logged predictions (SQLite)
    predictions_v2.db          v2 logged predictions (SQLite)
    predictions_log.csv        Older CSV log (legacy)

data/                       Pre-built input CSVs
  dataset_daily.csv               Raw daily political events
  dataset_intraday.csv            Raw intraday political events
  dataset_maml_daily.csv          v1 MLP pretraining dataset
  dataset_maml_intraday.csv       v1 MAML training dataset
  dataset_maml_daily_v2.csv       v2 MLP pretraining dataset
  dataset_maml_intraday_v2.csv    v2 MAML training dataset
  seed_v1.csv                     v1 seed for live MAML support set
  seed_v2.csv                     v2 seed for live MAML support set

models/                     Pre-trained model artifacts
  v1/
    mlp_pretrained.pth           v1 pretrained MLP backbone
    maml_trained.pth             v1 trained MAML model
    feature_scaler.pkl           v1 feature scaler
  v2/
    mlp_pretrained_v2.pth        v2 pretrained MLP backbone
    maml_trained_v2.pth          v2 trained MAML model
    feature_scaler_v2.pkl        v2 feature scaler

requirements.txt            Python dependencies
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

Streamlit dashboards (interactive UI for live predictions):

   streamlit run app_v1.py
   streamlit run app_v2.py

Background prediction schedulers (auto-prediction every 30 min during OVX hours):

   python scheduler_v1.py
   python scheduler_v2.py

Both schedulers and dashboards write to predictions.db and predictions_v2.db
respectively. The included .db files contain previously logged predictions for
inspection; they will be appended to (or created if deleted) on the next run.


NOTES
=====
- gdelt_pipeline.py and the live components need network access (GDELT + yfinance).
- The pretrained models in models/v1/ and models/v2/ were trained on the CSVs in
  data/. You can evaluate immediately without retraining.
- evaluation_v1.py uses GARCH(1,1) as one of the baselines (requires the 'arch'
  package, already in requirements.txt).
- NLP_sentiment.py loads FinBERT (ProsusAI/finbert) on first call — this triggers
  a one-time download of ~440MB from HuggingFace.
- All training scripts are reproducible (SEED=42).
