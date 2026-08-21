# 🦞 LOBSTER Market Microstructure Predictor
### Capstone Project — NASDAQ LOB Direction Prediction & Smart Order Execution

---

## Project Overview

This capstone builds a **production-grade ML pipeline** that:
1. Loads LOBSTER academic-grade NASDAQ orderbook reconstructions
2. Engineers microstructure features from 10-level bid/ask snapshots
3. Trains a **CNN-LSTM** to predict price direction (UP / FLAT / DOWN) over the next 10 events
4. Uses predictions to build an **ML-directed order slicer** that reduces slippage vs TWAP/VWAP

**Target metrics:** ≥61% accuracy · ≥0.59 F1-macro · ~9.5% slippage reduction vs TWAP

---

## Project Structure

```
LOBSTER_Capstone/
│
├── 00_setup_and_data_download.ipynb    ← Install deps, validate LOBSTER files
├── 01_eda_and_preprocessing.ipynb      ← EDA, cleaning, visualisations
├── 02_feature_engineering.ipynb        ← 42-feature matrix, sequences, scaler
├── 03_baseline_models.ipynb            ← XGBoost + LSTM-only baselines
├── 04_cnn_lstm_model.ipynb             ← Main CNN-LSTM model training
├── 05_evaluation_and_comparison.ipynb  ← All-model comparison, confusion matrices
├── 06_order_execution_optimizer.ipynb  ← Slippage sim: TWAP vs VWAP vs ML
├── 07_streamlit_dashboard.py           ← Interactive dashboard
│
├── utils/
│   ├── lobster_loader.py               ← Load & validate LOBSTER CSVs
│   ├── feature_builder.py              ← Feature engineering & sequence builder
│   └── slippage_simulator.py           ← TWAP / VWAP / ML-directed execution
│
├── data/                               ← Created at runtime (gitignored)
│   ├── AAPL_*_message_10.csv           ← YOUR LOBSTER FILES GO HERE
│   ├── AAPL_*_orderbook_10.csv
│   ├── lobster_clean.parquet
│   ├── lobster_features.parquet
│   ├── X_*_seq.npy / y_*_seq.npy
│   ├── scaler.joblib
│   ├── cnn_lstm_model.pt
│   └── ...
│
└── README.md
```

---

## Notebook Run Order

```
NB 00 → NB 01 → NB 02 → NB 03 → NB 04 → NB 05 → NB 06
                                                     ↓
                                        streamlit run 07_streamlit_dashboard.py
```

Each notebook saves its outputs to `data/` for the next notebook to consume.

---

## Data Setup

1. Download LOBSTER sample data from: https://lobsterdata.com/info/DataSamples.php
2. Place both files in `data/`:
   - `AAPL_2012-06-21_34200000_57600000_message_10.csv`
   - `AAPL_2012-06-21_34200000_57600000_orderbook_10.csv`
3. Update `DATA_CFG` paths in `NB 00` if needed

---

## Model Architecture

```
Input: (Batch, T=20, F=42)   ← 20-event window, 42 features
    │
┌───▼─────────────────────────────┐
│  CNN Block × 3                   │
│  Conv1d(F→64) → BN → ReLU       │
│  Conv1d(64→128) → BN → ReLU     │
│  Conv1d(128→256) → BN → ReLU    │
└───────────────┬─────────────────┘
                │  (B, 256, T)
┌───────────────▼─────────────────┐
│  Bidirectional LSTM (2 layers)   │
│  hidden=128 → output=256         │
└───────────────┬─────────────────┘
                │  last hidden state
┌───────────────▼─────────────────┐
│  FC(256→128) → ReLU → Dropout   │
│  FC(128→64)  → ReLU → Dropout   │
│  FC(64→3)    → Softmax           │
└─────────────────────────────────┘
Output: P(DOWN), P(FLAT), P(UP)
```

---

## Feature Set (42 total)

| Group | Columns | Count |
|-------|---------|-------|
| Ask prices | AskP1 … AskP10 | 10 |
| Ask sizes  | AskS1 … AskS10 | 10 |
| Bid prices | BidP1 … BidP10 | 10 |
| Bid sizes  | BidS1 … BidS10 | 10 |
| Order imbalance (L1) | imbalance_l1 | 1 |
| Spread | spread_bps | 1 |

---

## Dependencies

```
pandas numpy matplotlib seaborn scikit-learn xgboost torch streamlit tqdm joblib
```

Install:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost torch streamlit tqdm joblib
```

---

## References

- Kercheval & Zhang (2015): *Modelling high-frequency limit order book dynamics with support vector machines*
- Kolm, Turiel & Westray (2021): *Deep order flow imbalance: Extracting alpha at multiple horizons from the limit order book*
- LOBSTER Data: https://lobsterdata.com
