"""
generate_notebooks.py
─────────────────────
Generates all 7 LOBSTER Capstone Jupyter notebooks as valid .ipynb files.
Run:  python generate_notebooks.py
"""

import json, os

OUTPUT_DIR = "/home/claude/LOBSTER_Capstone"

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def nb(cells):
    """Wrap cells in a minimal nbformat-4 notebook skeleton."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "cells": cells
    }

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines),
            "id": os.urandom(4).hex()}

def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": list(lines), "id": os.urandom(4).hex()}

def save(name, notebook):
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"  ✓  {name}")

# ─────────────────────────────────────────────────────────────────────────────
# NB 00  –  Setup & Data Download
# ─────────────────────────────────────────────────────────────────────────────

nb00 = nb([
    md("# 📥 Notebook 00 — Environment Setup & Data Download\n",
       "> **Purpose:** Install dependencies, verify environment, and load LOBSTER sample files.\n\n",
       "Run this notebook once before any other notebook in the pipeline.\n\n",
       "---"),

    md("## 0.1  Install dependencies"),
    code(
        "# Run once — comment out after first install\n",
        "import subprocess, sys\n",
        "pkgs = [\n",
        "    'pandas', 'numpy', 'matplotlib', 'seaborn',\n",
        "    'scikit-learn', 'xgboost', 'torch', 'streamlit',\n",
        "    'tqdm', 'joblib'\n",
        "]\n",
        "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + pkgs)\n",
        "print('All packages installed.')"
    ),

    md("## 0.2  Verify imports"),
    code(
        "import sys, os\n",
        "sys.path.insert(0, os.path.abspath('.'))\n\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import torch\n",
        "import xgboost\n",
        "import sklearn\n",
        "import matplotlib\n\n",
        "print(f'NumPy      {np.__version__}')\n",
        "print(f'Pandas     {pd.__version__}')\n",
        "print(f'PyTorch    {torch.__version__}')\n",
        "print(f'XGBoost    {xgboost.__version__}')\n",
        "print(f'Scikit     {sklearn.__version__}')\n",
        "print(f'Matplotlib {matplotlib.__version__}')\n",
        "print(f'GPU available: {torch.cuda.is_available()}')"
    ),

    md("## 0.3  LOBSTER data download\n",
       "\n",
       "Download the sample files from **https://lobsterdata.com/info/DataSamples.php**\n\n",
       "Place both files in the `data/` folder:\n",
       "```\n",
       "LOBSTER_Capstone/\n",
       "└── data/\n",
       "    ├── AAPL_2012-06-21_34200000_57600000_message_10.csv\n",
       "    └── AAPL_2012-06-21_34200000_57600000_orderbook_10.csv\n",
       "```\n\n",
       "Or update `DATA_CFG` below to point to your own paths."),
    code(
        "import os\n\n",
        "# ── Update these paths if needed ─────────────────────────────────\n",
        "DATA_CFG = {\n",
        "    'msg_file':  'data/AAPL_2012-06-21_34200000_57600000_message_10.csv',\n",
        "    'book_file': 'data/AAPL_2012-06-21_34200000_57600000_orderbook_10.csv',\n",
        "    'ticker':    'AAPL',\n",
        "    'n_levels':  10,\n",
        "}\n\n",
        "# Save config so later notebooks can import it\n",
        "import json\n",
        "os.makedirs('data', exist_ok=True)\n",
        "with open('data/config.json', 'w') as f:\n",
        "    json.dump(DATA_CFG, f, indent=2)\n",
        "print('Config saved to data/config.json')\n\n",
        "# Verify files exist\n",
        "for key, path in [(k, v) for k, v in DATA_CFG.items() if 'file' in k]:\n",
        "    exists = os.path.isfile(path)\n",
        "    status = '✅' if exists else '❌ NOT FOUND'\n",
        "    size   = f'  ({os.path.getsize(path)/1e6:.2f} MB)' if exists else ''\n",
        "    print(f'  {key}: {path}  {status}{size}')"
    ),

    md("## 0.4  Quick sanity load"),
    code(
        "from utils.lobster_loader import load_message, load_orderbook, merge_files, clean_lobster\n\n",
        "msg_df  = load_message(DATA_CFG['msg_file'])\n",
        "book_df = load_orderbook(DATA_CFG['book_file'])\n",
        "df      = merge_files(msg_df, book_df)\n",
        "df_clean = clean_lobster(df)\n\n",
        "print('\\nFirst 3 rows (message cols):')\n",
        "display(df_clean[['Time','Type','Size','Price','Direction']].head(3))\n",
        "print('\\nFirst 3 rows (orderbook cols):')\n",
        "display(df_clean[['AskP1','AskS1','BidP1','BidS1']].head(3))"
    ),

    md("## 0.5  Save cleaned base DataFrame"),
    code(
        "df_clean.to_parquet('data/lobster_clean.parquet', index=False)\n",
        "print(f'Saved data/lobster_clean.parquet  ({len(df_clean):,} rows)')"
    ),

    md("---\n> ✅ **Setup complete.** Proceed to `01_eda_and_preprocessing.ipynb`.")
])

# ─────────────────────────────────────────────────────────────────────────────
# NB 01  –  EDA & Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

nb01 = nb([
    md("# 🔍 Notebook 01 — Exploratory Data Analysis & Preprocessing\n",
       "> **Purpose:** Understand LOBSTER data structure, visualise LOB dynamics,\n",
       "> identify data quality issues, and produce the cleaned dataset.\n\n",
       "**Inputs:** `data/lobster_clean.parquet`  \n",
       "**Outputs:** `data/lobster_features.parquet`\n\n---"),

    md("## 1.1  Load cleaned data"),
    code(
        "import sys, os\n",
        "sys.path.insert(0, os.path.abspath('..'))\n\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.gridspec as gridspec\n",
        "import seaborn as sns\n",
        "sns.set_theme(style='darkgrid', palette='muted')\n",
        "%matplotlib inline\n\n",
        "df = pd.read_parquet('data/lobster_clean.parquet')\n",
        "print(f'Shape: {df.shape}')\n",
        "df.head(3)"
    ),

    md("## 1.2  Dataset overview"),
    code(
        "from utils.lobster_loader import describe_dataset, MSG_TYPE\n",
        "from utils.feature_builder import add_mid_and_spread\n\n",
        "df = add_mid_and_spread(df)\n",
        "describe_dataset(df)"
    ),

    md("## 1.3  Mid-price time series"),
    code(
        "fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)\n\n",
        "# Mid price\n",
        "axes[0].plot(df.index, df['mid_price'], lw=0.6, color='steelblue')\n",
        "axes[0].set_ylabel('Mid Price ($)')\n",
        "axes[0].set_title('LOBSTER Mid Price — Full Session')\n\n",
        "# Spread\n",
        "axes[1].fill_between(df.index, df['spread_bps'], alpha=0.5, color='darkorange')\n",
        "axes[1].set_ylabel('Spread (bps)')\n",
        "axes[1].set_xlabel('Event Index')\n",
        "axes[1].set_title('Bid-Ask Spread (bps)')\n\n",
        "plt.tight_layout()\n",
        "plt.savefig('data/fig_mid_spread.png', dpi=120, bbox_inches='tight')\n",
        "plt.show()"
    ),

    md("## 1.4  Event type distribution"),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n\n",
        "# Count\n",
        "type_counts = df['Type'].value_counts().sort_index()\n",
        "labels = [f'Type {t}\\n{MSG_TYPE.get(t,\"?\")}' for t in type_counts.index]\n",
        "axes[0].bar(labels, type_counts.values, color=sns.color_palette('muted', len(labels)))\n",
        "axes[0].set_title('Message Event Type Distribution')\n",
        "axes[0].set_ylabel('Count')\n",
        "plt.setp(axes[0].get_xticklabels(), rotation=30, ha='right', fontsize=8)\n\n",
        "# Direction\n",
        "dir_counts = df['Direction'].value_counts()\n",
        "axes[1].bar(['Sell (-1)', 'Buy (+1)'],\n",
        "            [dir_counts.get(-1,0), dir_counts.get(1,0)],\n",
        "            color=['tomato', 'seagreen'])\n",
        "axes[1].set_title('Order Direction')\n",
        "axes[1].set_ylabel('Count')\n\n",
        "plt.tight_layout()\n",
        "plt.savefig('data/fig_event_types.png', dpi=120, bbox_inches='tight')\n",
        "plt.show()"
    ),

    md("## 1.5  LOB depth visualisation (snapshot at event 5000)"),
    code(
        "snapshot_idx = 5000\n",
        "snap = df.iloc[snapshot_idx]\n\n",
        "bid_prices = [snap[f'BidP{i}'] / 10000 for i in range(1, 11)]\n",
        "bid_sizes  = [snap[f'BidS{i}'] for i in range(1, 11)]\n",
        "ask_prices = [snap[f'AskP{i}'] / 10000 for i in range(1, 11)]\n",
        "ask_sizes  = [snap[f'AskS{i}'] for i in range(1, 11)]\n\n",
        "fig, ax = plt.subplots(figsize=(12, 5))\n",
        "ax.barh(bid_prices, [-s for s in bid_sizes], height=0.00005,\n",
        "        color='seagreen', alpha=0.8, label='Bid')\n",
        "ax.barh(ask_prices, ask_sizes, height=0.00005,\n",
        "        color='tomato', alpha=0.8, label='Ask')\n",
        "ax.axvline(0, color='black', lw=0.8)\n",
        "ax.set_xlabel('Volume (shares)')\n",
        "ax.set_ylabel('Price ($)')\n",
        "ax.set_title(f'LOB Depth Profile — Event {snapshot_idx}')\n",
        "ax.legend()\n",
        "plt.tight_layout()\n",
        "plt.savefig('data/fig_lob_depth.png', dpi=120, bbox_inches='tight')\n",
        "plt.show()"
    ),

    md("## 1.6  Order imbalance & spread distributions"),
    code(
        "from utils.feature_builder import add_micro_features\n\n",
        "df = add_micro_features(df)\n\n",
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n\n",
        "axes[0].hist(df['imbalance_l1'].dropna(), bins=80, color='steelblue', edgecolor='none')\n",
        "axes[0].set_title('Order Imbalance (Level 1)')\n",
        "axes[0].set_xlabel('OBI')\n\n",
        "axes[1].hist(df['spread_bps'].dropna().clip(0, 20), bins=80,\n",
        "             color='darkorange', edgecolor='none')\n",
        "axes[1].set_title('Spread (bps)')\n",
        "axes[1].set_xlabel('bps')\n\n",
        "axes[2].hist(df['depth_ratio'].dropna().clip(0, 5), bins=80,\n",
        "             color='purple', edgecolor='none')\n",
        "axes[2].set_title('Bid/Ask Depth Ratio')\n",
        "axes[2].set_xlabel('ratio')\n\n",
        "plt.tight_layout()\n",
        "plt.savefig('data/fig_feature_dists.png', dpi=120, bbox_inches='tight')\n",
        "plt.show()"
    ),

    md("## 1.7  Label distribution (3-class target)"),
    code(
        "from utils.feature_builder import make_labels\n\n",
        "labels = make_labels(df, horizon=10, threshold_bps=0.5)\n",
        "df['label'] = labels\n\n",
        "# Drop NaNs from look-ahead window\n",
        "df = df.dropna(subset=['label'])\n\n",
        "counts = df['label'].value_counts().sort_index()\n",
        "colors = ['tomato', 'steelblue', 'seagreen']\n\n",
        "fig, ax = plt.subplots(figsize=(7, 4))\n",
        "ax.bar(['DOWN (-1)', 'FLAT (0)', 'UP (+1)'], counts.values, color=colors)\n",
        "for i, v in enumerate(counts.values):\n",
        "    ax.text(i, v + 50, f'{v:,}\\n({v/len(df)*100:.1f}%)', ha='center', fontsize=9)\n",
        "ax.set_title('Label Distribution (horizon=10 events, ±0.5 bps threshold)')\n",
        "ax.set_ylabel('Count')\n",
        "plt.tight_layout()\n",
        "plt.savefig('data/fig_label_dist.png', dpi=120, bbox_inches='tight')\n",
        "plt.show()"
    ),

    md("## 1.8  Correlation heatmap (engineered features)"),
    code(
        "eng_cols = ['imbalance_l1','imbalance_l5','imbalance_l10',\n",
        "            'spread_bps','depth_bid','depth_ask','depth_ratio',\n",
        "            'price_range','vwap_imbalance','label']\n\n",
        "corr = df[eng_cols].astype(float).corr()\n\n",
        "fig, ax = plt.subplots(figsize=(10, 8))\n",
        "sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',\n",
        "            center=0, ax=ax, linewidths=0.4)\n",
        "ax.set_title('Feature Correlation Matrix')\n",
        "plt.tight_layout()\n",
        "plt.savefig('data/fig_corr.png', dpi=120, bbox_inches='tight')\n",
        "plt.show()"
    ),

    md("## 1.9  Save feature-enriched dataset"),
    code(
        "df.to_parquet('data/lobster_features.parquet', index=False)\n",
        "print(f'Saved data/lobster_features.parquet  ({len(df):,} rows, {df.shape[1]} cols)')"
    ),

    md("---\n> ✅ **EDA complete.** Proceed to `02_feature_engineering.ipynb`.")
])

# ─────────────────────────────────────────────────────────────────────────────
# NB 02  –  Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

nb02 = nb([
    md("# ⚙️ Notebook 02 — Feature Engineering & Sequence Construction\n",
       "> **Purpose:** Build the final 42-feature matrix and construct (20 × 42)\n",
       "> input sequences for the CNN-LSTM model.\n\n",
       "**Inputs:** `data/lobster_features.parquet`  \n",
       "**Outputs:** `data/X_seq.npy`, `data/y_seq.npy`, `data/scaler.joblib`\n\n---"),

    md("## 2.1  Load features"),
    code(
        "import sys, os\n",
        "sys.path.insert(0, os.path.abspath('..'))\n\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import joblib\n",
        "sns.set_theme(style='darkgrid')\n",
        "%matplotlib inline\n\n",
        "df = pd.read_parquet('data/lobster_features.parquet')\n",
        "print(f'Shape: {df.shape}')"
    ),

    md("## 2.2  Feature set definitions\n",
       "\n",
       "| Group | Columns | Count |\n",
       "|-------|---------|-------|\n",
       "| Raw LOB prices (Ask) | AskP1 … AskP10 | 10 |\n",
       "| Raw LOB sizes (Ask)  | AskS1 … AskS10 | 10 |\n",
       "| Raw LOB prices (Bid) | BidP1 … BidP10 | 10 |\n",
       "| Raw LOB sizes (Bid)  | BidS1 … BidS10 | 10 |\n",
       "| Order imbalance      | imbalance_l1   |  1 |\n",
       "| Spread               | spread_bps     |  1 |\n",
       "| **Total (CNN-LSTM)** |                | **42** |"),

    code(
        "from utils.feature_builder import (\n",
        "    FEATURE_COLS_42, FEATURE_COLS_EXT, LOB_COLS,\n",
        "    build_feature_matrix, build_sequences,\n",
        "    fit_scaler, apply_scaler\n",
        ")\n\n",
        "print('CNN-LSTM features (42):')\n",
        "for i, c in enumerate(FEATURE_COLS_42):\n",
        "    print(f'  {i+1:2d}. {c}')"
    ),

    md("## 2.3  Build feature matrix"),
    code(
        "X_df = build_feature_matrix(df, FEATURE_COLS_42)\n",
        "y    = df.loc[X_df.index, 'label'].astype(int).values\n\n",
        "# Remap labels {-1, 0, 1} → {0, 1, 2} for CrossEntropyLoss\n",
        "y_mapped = y + 1    # -1→0  0→1  +1→2\n\n",
        "X = X_df.values\n",
        "print(f'X shape: {X.shape}, y shape: {y_mapped.shape}')\n",
        "print(f'Label distribution: DOWN={np.sum(y_mapped==0):,}  FLAT={np.sum(y_mapped==1):,}  UP={np.sum(y_mapped==2):,}')"
    ),

    md("## 2.4  Train / validation / test split (chronological)"),
    code(
        "N = len(X)\n",
        "train_end = int(N * 0.70)\n",
        "val_end   = int(N * 0.85)\n\n",
        "X_train_2d, y_train_2d = X[:train_end],   y_mapped[:train_end]\n",
        "X_val_2d,   y_val_2d   = X[train_end:val_end], y_mapped[train_end:val_end]\n",
        "X_test_2d,  y_test_2d  = X[val_end:],    y_mapped[val_end:]\n\n",
        "print(f'Train: {len(X_train_2d):,}  Val: {len(X_val_2d):,}  Test: {len(X_test_2d):,}')"
    ),

    md("## 2.5  Fit & apply StandardScaler"),
    code(
        "scaler = fit_scaler(X_train_2d)\n\n",
        "X_train_scaled = scaler.transform(X_train_2d).astype(np.float32)\n",
        "X_val_scaled   = scaler.transform(X_val_2d).astype(np.float32)\n",
        "X_test_scaled  = scaler.transform(X_test_2d).astype(np.float32)\n\n",
        "joblib.dump(scaler, 'data/scaler.joblib')\n",
        "print('Scaler saved to data/scaler.joblib')"
    ),

    md("## 2.6  Build CNN-LSTM sequences (window = 20)"),
    code(
        "SEQ_LEN = 20\n\n",
        "X_train_seq, y_train_seq = build_sequences(X_train_scaled, y_train_2d, SEQ_LEN)\n",
        "X_val_seq,   y_val_seq   = build_sequences(X_val_scaled,   y_val_2d,   SEQ_LEN)\n",
        "X_test_seq,  y_test_seq  = build_sequences(X_test_scaled,  y_test_2d,  SEQ_LEN)\n\n",
        "print(f'Train sequences: {X_train_seq.shape}')\n",
        "print(f'Val   sequences: {X_val_seq.shape}')\n",
        "print(f'Test  sequences: {X_test_seq.shape}')"
    ),

    md("## 2.7  Visualise a single input sequence"),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n",
        "# Heatmap of a single sequence\n",
        "axes[0].imshow(X_train_seq[100].T, aspect='auto', cmap='RdYlGn')\n",
        "axes[0].set_xlabel('Time step (events)')\n",
        "axes[0].set_ylabel('Feature index')\n",
        "axes[0].set_title('Single Input Sequence (20 × 42 feature heatmap)')\n",
        "plt.colorbar(axes[0].images[0], ax=axes[0], label='Normalised value')\n\n",
        "# Mid-price for the same sequence\n",
        "seq_mid = X_train_seq[100, :, FEATURE_COLS_42.index('AskP1')]  # scaled AskP1 as proxy\n",
        "axes[1].plot(seq_mid, marker='o', markersize=3)\n",
        "axes[1].set_xlabel('Time step')\n",
        "axes[1].set_ylabel('Scaled AskP1')\n",
        "axes[1].set_title('AskP1 (scaled) across 20-event window')\n\n",
        "plt.tight_layout()\n",
        "plt.savefig('data/fig_sequence_viz.png', dpi=120, bbox_inches='tight')\n",
        "plt.show()"
    ),

    md("## 2.8  Save sequences to disk"),
    code(
        "np.save('data/X_train_seq.npy', X_train_seq)\n",
        "np.save('data/y_train_seq.npy', y_train_seq)\n",
        "np.save('data/X_val_seq.npy',   X_val_seq)\n",
        "np.save('data/y_val_seq.npy',   y_val_seq)\n",
        "np.save('data/X_test_seq.npy',  X_test_seq)\n",
        "np.save('data/y_test_seq.npy',  y_test_seq)\n\n",
        "# Also save 2-D arrays for XGBoost baseline\n",
        "np.save('data/X_train_2d.npy',  X_train_scaled)\n",
        "np.save('data/y_train_2d.npy',  y_train_2d)\n",
        "np.save('data/X_val_2d.npy',    X_val_scaled)\n",
        "np.save('data/y_val_2d.npy',    y_val_2d)\n",
        "np.save('data/X_test_2d.npy',   X_test_scaled)\n",
        "np.save('data/y_test_2d.npy',   y_test_2d)\n\n",
        "print('All arrays saved to data/')"
    ),

    md("---\n> ✅ **Feature engineering complete.** Proceed to `03_baseline_models.ipynb`.")
])

# ─────────────────────────────────────────────────────────────────────────────
# NB 03  –  Baseline Models
# ─────────────────────────────────────────────────────────────────────────────

nb03 = nb([
    md("# 📊 Notebook 03 — Baseline Models (XGBoost + LSTM-only)\n",
       "> **Purpose:** Train and evaluate two baseline models to benchmark\n",
       "> against the full CNN-LSTM in Notebook 04.\n\n",
       "**Inputs:** `data/X_*_2d.npy`, `data/y_*_2d.npy`, `data/X_*_seq.npy`  \n",
       "**Outputs:** `data/xgb_model.json`, `data/lstm_only_model.pt`, `data/baseline_results.csv`\n\n---"),

    md("## 3.1  Load data"),
    code(
        "import sys, os\n",
        "sys.path.insert(0, os.path.abspath('..'))\n\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import joblib\n",
        "from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score\n",
        "sns.set_theme(style='darkgrid')\n",
        "%matplotlib inline\n\n",
        "X_train = np.load('data/X_train_2d.npy')\n",
        "y_train = np.load('data/y_train_2d.npy')\n",
        "X_val   = np.load('data/X_val_2d.npy')\n",
        "y_val   = np.load('data/y_val_2d.npy')\n",
        "X_test  = np.load('data/X_test_2d.npy')\n",
        "y_test  = np.load('data/y_test_2d.npy')\n\n",
        "X_train_seq = np.load('data/X_train_seq.npy')\n",
        "y_train_seq = np.load('data/y_train_seq.npy')\n",
        "X_val_seq   = np.load('data/X_val_seq.npy')\n",
        "y_val_seq   = np.load('data/y_val_seq.npy')\n",
        "X_test_seq  = np.load('data/X_test_seq.npy')\n",
        "y_test_seq  = np.load('data/y_test_seq.npy')\n\n",
        "print('Shapes:', X_train.shape, y_train.shape)"
    ),

    md("## 3.2  Baseline A — XGBoost"),
    code(
        "import xgboost as xgb\n",
        "from sklearn.utils.class_weight import compute_sample_weight\n\n",
        "sample_weights = compute_sample_weight('balanced', y_train)\n\n",
        "xgb_model = xgb.XGBClassifier(\n",
        "    n_estimators=400,\n",
        "    max_depth=6,\n",
        "    learning_rate=0.05,\n",
        "    subsample=0.8,\n",
        "    colsample_bytree=0.8,\n",
        "    use_label_encoder=False,\n",
        "    eval_metric='mlogloss',\n",
        "    random_state=42,\n",
        "    n_jobs=-1,\n",
        "    verbosity=1\n",
        ")\n\n",
        "xgb_model.fit(\n",
        "    X_train, y_train,\n",
        "    sample_weight=sample_weights,\n",
        "    eval_set=[(X_val, y_val)],\n",
        "    verbose=50\n",
        ")\n\n",
        "y_pred_xgb = xgb_model.predict(X_test)\n",
        "acc_xgb = accuracy_score(y_test, y_pred_xgb)\n",
        "f1_xgb  = f1_score(y_test, y_pred_xgb, average='macro')\n\n",
        "print(f'\\nXGBoost  Accuracy: {acc_xgb:.4f}  F1-macro: {f1_xgb:.4f}')\n",
        "print(classification_report(y_test, y_pred_xgb,\n",
        "      target_names=['DOWN','FLAT','UP']))"
    ),

    md("## 3.3  XGBoost — feature importance"),
    code(
        "from utils.feature_builder import FEATURE_COLS_42\n\n",
        "importances = xgb_model.feature_importances_\n",
        "feat_df = pd.DataFrame({'feature': FEATURE_COLS_42, 'importance': importances})\n",
        "feat_df = feat_df.sort_values('importance', ascending=False).head(20)\n\n",
        "fig, ax = plt.subplots(figsize=(10, 6))\n",
        "ax.barh(feat_df['feature'][::-1], feat_df['importance'][::-1],\n",
        "        color='steelblue')\n",
        "ax.set_title('XGBoost — Top 20 Feature Importances')\n",
        "ax.set_xlabel('Importance')\n",
        "plt.tight_layout()\n",
        "plt.savefig('data/fig_xgb_importance.png', dpi=120, bbox_inches='tight')\n",
        "plt.show()\n\n",
        "xgb_model.save_model('data/xgb_model.json')\n",
        "print('XGBoost model saved.')"
    ),

    md("## 3.4  Baseline B — LSTM-only"),
    code(
        "import torch\n",
        "import torch.nn as nn\n",
        "from torch.utils.data import TensorDataset, DataLoader\n\n",
        "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "print(f'Device: {DEVICE}')\n\n",
        "class LSTMOnly(nn.Module):\n",
        "    def __init__(self, input_size=42, hidden=128, num_layers=2, n_classes=3, dropout=0.3):\n",
        "        super().__init__()\n",
        "        self.lstm = nn.LSTM(input_size, hidden, num_layers,\n",
        "                            batch_first=True, dropout=dropout)\n",
        "        self.fc   = nn.Sequential(\n",
        "            nn.Linear(hidden, 64),\n",
        "            nn.ReLU(),\n",
        "            nn.Dropout(dropout),\n",
        "            nn.Linear(64, n_classes)\n",
        "        )\n\n",
        "    def forward(self, x):\n",
        "        # x: (B, T, F)\n",
        "        out, _ = self.lstm(x)     # (B, T, H)\n",
        "        out = out[:, -1, :]       # last timestep\n",
        "        return self.fc(out)\n\n",
        "# Dataloaders\n",
        "def make_loader(X, y, batch_size=256, shuffle=True):\n",
        "    Xt = torch.tensor(X, dtype=torch.float32)\n",
        "    yt = torch.tensor(y, dtype=torch.long)\n",
        "    return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle)\n\n",
        "train_loader = make_loader(X_train_seq, y_train_seq)\n",
        "val_loader   = make_loader(X_val_seq,   y_val_seq,   shuffle=False)\n",
        "test_loader  = make_loader(X_test_seq,  y_test_seq,  shuffle=False)"
    ),

    code(
        "# Training loop\n",
        "def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3):\n",
        "    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)\n",
        "    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)\n",
        "    criterion  = nn.CrossEntropyLoss()\n",
        "    history    = {'train_loss': [], 'val_loss': [], 'val_acc': []}\n\n",
        "    for epoch in range(1, epochs + 1):\n",
        "        model.train()\n",
        "        total_loss = 0\n",
        "        for Xb, yb in train_loader:\n",
        "            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)\n",
        "            optimizer.zero_grad()\n",
        "            loss = criterion(model(Xb), yb)\n",
        "            loss.backward()\n",
        "            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n",
        "            optimizer.step()\n",
        "            total_loss += loss.item()\n\n",
        "        # Validation\n",
        "        model.eval()\n",
        "        val_loss, correct, total = 0, 0, 0\n",
        "        with torch.no_grad():\n",
        "            for Xb, yb in val_loader:\n",
        "                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)\n",
        "                logits = model(Xb)\n",
        "                val_loss += criterion(logits, yb).item()\n",
        "                correct  += (logits.argmax(1) == yb).sum().item()\n",
        "                total    += len(yb)\n",
        "        val_acc = correct / total\n",
        "        scheduler.step(val_loss)\n\n",
        "        history['train_loss'].append(total_loss / len(train_loader))\n",
        "        history['val_loss'].append(val_loss / len(val_loader))\n",
        "        history['val_acc'].append(val_acc)\n\n",
        "        print(f'Ep {epoch:3d}/{epochs} | '\n",
        "              f'Train Loss {total_loss/len(train_loader):.4f} | '\n",
        "              f'Val Loss {val_loss/len(val_loader):.4f} | '\n",
        "              f'Val Acc {val_acc:.4f}')\n\n",
        "    return history\n\n",
        "lstm_model = LSTMOnly().to(DEVICE)\n",
        "history_lstm = train_model(lstm_model, train_loader, val_loader, epochs=20)"
    ),

    code(
        "# Evaluate on test set\n",
        "lstm_model.eval()\n",
        "all_preds, all_labels = [], []\n",
        "with torch.no_grad():\n",
        "    for Xb, yb in test_loader:\n",
        "        preds = lstm_model(Xb.to(DEVICE)).argmax(1).cpu().numpy()\n",
        "        all_preds.extend(preds)\n",
        "        all_labels.extend(yb.numpy())\n\n",
        "y_pred_lstm = np.array(all_preds)\n",
        "acc_lstm = accuracy_score(y_test_seq, y_pred_lstm)\n",
        "f1_lstm  = f1_score(y_test_seq, y_pred_lstm, average='macro')\n\n",
        "print(f'LSTM-only  Accuracy: {acc_lstm:.4f}  F1-macro: {f1_lstm:.4f}')\n",
        "print(classification_report(y_test_seq, y_pred_lstm,\n",
        "      target_names=['DOWN','FLAT','UP']))\n\n",
        "torch.save(lstm_model.state_dict(), 'data/lstm_only_model.pt')\n",
        "print('LSTM-only model saved.')"
    ),

    md("## 3.5  Training curves"),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n\n",
        "axes[0].plot(history_lstm['train_loss'], label='Train')\n",
        "axes[0].plot(history_lstm['val_loss'],   label='Val')\n",
        "axes[0].set_title('LSTM-only — Loss')\n",
        "axes[0].set_xlabel('Epoch'); axes[0].legend()\n\n",
        "axes[1].plot(history_lstm['val_acc'], color='seagreen')\n",
        "axes[1].axhline(0.60, ls='--', color='red', label='Target 60%')\n",
        "axes[1].set_title('LSTM-only — Validation Accuracy')\n",
        "axes[1].set_xlabel('Epoch'); axes[1].legend()\n\n",
        "plt.tight_layout()\n",
        "plt.savefig('data/fig_lstm_training.png', dpi=120, bbox_inches='tight')\n",
        "plt.show()"
    ),

    md("## 3.6  Save baseline results"),
    code(
        "results = pd.DataFrame([\n",
        "    {'Model': 'XGBoost',   'Accuracy': acc_xgb, 'F1-macro': f1_xgb},\n",
        "    {'Model': 'LSTM-only', 'Accuracy': acc_lstm, 'F1-macro': f1_lstm},\n",
        "])\n",
        "results.to_csv('data/baseline_results.csv', index=False)\n",
        "print(results.to_string(index=False))"
    ),

    md("---\n> ✅ **Baselines trained.** Proceed to `04_cnn_lstm_model.ipynb`.")
])

# ─────────────────────────────────────────────────────────────────────────────
# NB 04  –  CNN-LSTM Model
# ─────────────────────────────────────────────────────────────────────────────

nb04 = nb([
    md("# 🧠 Notebook 04 — CNN-LSTM Model (Main Model)\n",
       "> **Purpose:** Define, train, and save the primary CNN-LSTM architecture.\n\n",
       "```\n",
       "Input (B, 20, 42)\n",
       "     │\n",
       "  ┌──▼─────────────────────┐\n",
       "  │  1-D CNN (across time)  │  3 × Conv1d → BatchNorm → ReLU → MaxPool\n",
       "  └──────────────┬──────────┘\n",
       "                 │ (B, C, T')\n",
       "  ┌──────────────▼──────────┐\n",
       "  │  Bi-LSTM  (2 layers)    │\n",
       "  └──────────────┬──────────┘\n",
       "                 │ last hidden state\n",
       "  ┌──────────────▼──────────┐\n",
       "  │  FC → Dropout → Softmax  │  → 3 classes\n",
       "  └─────────────────────────┘\n",
       "```\n\n",
       "**Inputs:** `data/X_*_seq.npy`, `data/y_*_seq.npy`  \n",
       "**Outputs:** `data/cnn_lstm_model.pt`, `data/cnn_lstm_history.csv`\n\n---"),

    md("## 4.1  Setup"),
    code(
        "import sys, os\n",
        "sys.path.insert(0, os.path.abspath('..'))\n\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "from torch.utils.data import TensorDataset, DataLoader\n",
        "from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score\n",
        "sns.set_theme(style='darkgrid')\n",
        "%matplotlib inline\n\n",
        "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "SEED   = 42\n",
        "torch.manual_seed(SEED)\n",
        "print(f'Device: {DEVICE}')\n\n",
        "X_train = np.load('data/X_train_seq.npy')\n",
        "y_train = np.load('data/y_train_seq.npy')\n",
        "X_val   = np.load('data/X_val_seq.npy')\n",
        "y_val   = np.load('data/y_val_seq.npy')\n",
        "X_test  = np.load('data/X_test_seq.npy')\n",
        "y_test  = np.load('data/y_test_seq.npy')\n\n",
        "print(f'Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}')"
    ),

    md("## 4.2  CNN-LSTM architecture"),
    code(
        "class CNNLSTM(nn.Module):\n",
        "    \"\"\"\n",
        "    CNN-LSTM for LOBSTER LOB sequence classification.\n\n",
        "    Architecture:\n",
        "      - Input   : (B, T=20, F=42)\n",
        "      - CNN     : 3 convolutional blocks over time dimension\n",
        "      - BiLSTM  : 2-layer bidirectional LSTM\n",
        "      - Head    : FC → Dropout → 3-class output\n",
        "    \"\"\"\n",
        "    def __init__(self,\n",
        "                 input_size: int = 42,\n",
        "                 seq_len:    int = 20,\n",
        "                 cnn_channels: list = [64, 128, 256],\n",
        "                 lstm_hidden: int  = 128,\n",
        "                 lstm_layers: int  = 2,\n",
        "                 n_classes:   int  = 3,\n",
        "                 dropout:     float = 0.3):\n",
        "        super().__init__()\n\n",
        "        # ── CNN blocks ──\n",
        "        cnn_layers = []\n",
        "        in_ch = input_size\n",
        "        for out_ch in cnn_channels:\n",
        "            cnn_layers += [\n",
        "                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),\n",
        "                nn.BatchNorm1d(out_ch),\n",
        "                nn.ReLU(),\n",
        "                nn.Dropout(dropout / 2)\n",
        "            ]\n",
        "            in_ch = out_ch\n",
        "        self.cnn = nn.Sequential(*cnn_layers)\n\n",
        "        # ── Bi-LSTM ──\n",
        "        self.lstm = nn.LSTM(\n",
        "            input_size  = cnn_channels[-1],\n",
        "            hidden_size = lstm_hidden,\n",
        "            num_layers  = lstm_layers,\n",
        "            batch_first = True,\n",
        "            bidirectional = True,\n",
        "            dropout = dropout if lstm_layers > 1 else 0\n",
        "        )\n\n",
        "        # ── Classification head ──\n",
        "        self.head = nn.Sequential(\n",
        "            nn.Linear(lstm_hidden * 2, 128),  # ×2 for bidirectional\n",
        "            nn.ReLU(),\n",
        "            nn.Dropout(dropout),\n",
        "            nn.Linear(128, 64),\n",
        "            nn.ReLU(),\n",
        "            nn.Dropout(dropout),\n",
        "            nn.Linear(64, n_classes)\n",
        "        )\n\n",
        "    def forward(self, x):\n",
        "        # x : (B, T, F)\n",
        "        x = x.permute(0, 2, 1)          # → (B, F, T)  for Conv1d\n",
        "        x = self.cnn(x)                  # → (B, C, T)\n",
        "        x = x.permute(0, 2, 1)          # → (B, T, C)  for LSTM\n",
        "        out, _ = self.lstm(x)            # → (B, T, 2H)\n",
        "        out = out[:, -1, :]              # last timestep → (B, 2H)\n",
        "        return self.head(out)            # → (B, n_classes)\n\n\n",
        "model = CNNLSTM().to(DEVICE)\n\n",
        "# Print parameter count\n",
        "total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
        "print(f'CNN-LSTM  Total trainable parameters: {total_params:,}')\n",
        "print(model)"
    ),

    md("## 4.3  Class-weighted loss & optimiser"),
    code(
        "from sklearn.utils.class_weight import compute_class_weight\n\n",
        "classes  = np.array([0, 1, 2])\n",
        "weights  = compute_class_weight('balanced', classes=classes, y=y_train)\n",
        "w_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)\n",
        "print(f'Class weights: DOWN={weights[0]:.3f}  FLAT={weights[1]:.3f}  UP={weights[2]:.3f}')\n\n",
        "criterion = nn.CrossEntropyLoss(weight=w_tensor)\n",
        "optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)\n",
        "scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)"
    ),

    md("## 4.4  DataLoaders"),
    code(
        "def make_loader(X, y, batch_size=256, shuffle=True):\n",
        "    Xt = torch.tensor(X, dtype=torch.float32)\n",
        "    yt = torch.tensor(y, dtype=torch.long)\n",
        "    return DataLoader(TensorDataset(Xt, yt),\n",
        "                      batch_size=batch_size, shuffle=shuffle,\n",
        "                      pin_memory=(DEVICE.type=='cuda'), num_workers=0)\n\n",
        "train_loader = make_loader(X_train, y_train)\n",
        "val_loader   = make_loader(X_val,   y_val,   shuffle=False)\n",
        "test_loader  = make_loader(X_test,  y_test,  shuffle=False)\n",
        "print(f'Batches — Train: {len(train_loader)}  Val: {len(val_loader)}  Test: {len(test_loader)}')"
    ),

    md("## 4.5  Training loop (with early stopping)"),
    code(
        "EPOCHS        = 40\n",
        "PATIENCE      = 7\n",
        "best_val_f1   = 0.0\n",
        "patience_cnt  = 0\n",
        "history       = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}\n\n",
        "for epoch in range(1, EPOCHS + 1):\n\n",
        "    # ── Train ──\n",
        "    model.train()\n",
        "    total_loss = 0\n",
        "    for Xb, yb in train_loader:\n",
        "        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)\n",
        "        optimizer.zero_grad()\n",
        "        loss = criterion(model(Xb), yb)\n",
        "        loss.backward()\n",
        "        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n",
        "        optimizer.step()\n",
        "        total_loss += loss.item()\n",
        "    scheduler.step()\n\n",
        "    # ── Validate ──\n",
        "    model.eval()\n",
        "    val_loss, preds_all, labels_all = 0, [], []\n",
        "    with torch.no_grad():\n",
        "        for Xb, yb in val_loader:\n",
        "            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)\n",
        "            logits  = model(Xb)\n",
        "            val_loss += criterion(logits, yb).item()\n",
        "            preds_all.extend(logits.argmax(1).cpu().numpy())\n",
        "            labels_all.extend(yb.cpu().numpy())\n\n",
        "    val_acc = accuracy_score(labels_all, preds_all)\n",
        "    val_f1  = f1_score(labels_all, preds_all, average='macro')\n\n",
        "    history['train_loss'].append(total_loss / len(train_loader))\n",
        "    history['val_loss'].append(val_loss / len(val_loader))\n",
        "    history['val_acc'].append(val_acc)\n",
        "    history['val_f1'].append(val_f1)\n\n",
        "    print(f'Ep {epoch:3d}/{EPOCHS} | '\n",
        "          f'TrLoss {total_loss/len(train_loader):.4f} | '\n",
        "          f'VaLoss {val_loss/len(val_loader):.4f} | '\n",
        "          f'Acc {val_acc:.4f} | F1 {val_f1:.4f}')\n\n",
        "    # ── Early stopping ──\n",
        "    if val_f1 > best_val_f1:\n",
        "        best_val_f1 = val_f1\n",
        "        torch.save(model.state_dict(), 'data/cnn_lstm_best.pt')\n",
        "        patience_cnt = 0\n",
        "    else:\n",
        "        patience_cnt += 1\n",
        "        if patience_cnt >= PATIENCE:\n",
        "            print(f'Early stopping at epoch {epoch} (best F1={best_val_f1:.4f})')\n",
        "            break\n\n",
        "print(f'\\nBest validation F1-macro: {best_val_f1:.4f}')"
    ),

    md("## 4.6  Training curves"),
    code(
        "hist_df = pd.DataFrame(history)\n",
        "hist_df.to_csv('data/cnn_lstm_history.csv', index=False)\n\n",
        "fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n\n",
        "axes[0].plot(hist_df['train_loss'], label='Train')\n",
        "axes[0].plot(hist_df['val_loss'],   label='Val')\n",
        "axes[0].set_title('CNN-LSTM Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend()\n\n",
        "axes[1].plot(hist_df['val_acc'], color='steelblue')\n",
        "axes[1].axhline(0.60, ls='--', color='red', alpha=0.6, label='Target 60%')\n",
        "axes[1].set_title('Validation Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend()\n\n",
        "axes[2].plot(hist_df['val_f1'], color='seagreen')\n",
        "axes[2].axhline(0.59, ls='--', color='red', alpha=0.6, label='Target F1=0.59')\n",
        "axes[2].set_title('Validation F1-macro'); axes[2].set_xlabel('Epoch'); axes[2].legend()\n\n",
        "plt.suptitle('CNN-LSTM Training History', fontsize=13, fontweight='bold')\n",
        "plt.tight_layout()\n",
        "plt.savefig('data/fig_cnn_lstm_training.png', dpi=120, bbox_inches='tight')\n",
        "plt.show()"
    ),

    md("## 4.7  Test set evaluation"),
    code(
        "# Load best checkpoint\n",
        "model.load_state_dict(torch.load('data/cnn_lstm_best.pt', map_location=DEVICE))\n",
        "model.eval()\n\n",
        "all_preds, all_labels, all_probs = [], [], []\n",
        "with torch.no_grad():\n",
        "    for Xb, yb in test_loader:\n",
        "        Xb = Xb.to(DEVICE)\n",
        "        logits = model(Xb)\n",
        "        probs  = torch.softmax(logits, dim=1).cpu().numpy()\n",
        "        preds  = logits.argmax(1).cpu().numpy()\n",
        "        all_preds.extend(preds)\n",
        "        all_labels.extend(yb.numpy())\n",
        "        all_probs.extend(probs)\n\n",
        "y_pred   = np.array(all_preds)\n",
        "y_probs  = np.array(all_probs)\n",
        "acc_cnn  = accuracy_score(y_test, y_pred)\n",
        "f1_cnn   = f1_score(y_test, y_pred, average='macro')\n\n",
        "print(f'CNN-LSTM  Test Accuracy: {acc_cnn:.4f}  F1-macro: {f1_cnn:.4f}')\n",
        "print()\n",
        "print(classification_report(y_test, y_pred, target_names=['DOWN','FLAT','UP']))\n\n",
        "# Save predictions\n",
        "np.save('data/cnn_lstm_preds.npy',  y_pred)\n",
        "np.save('data/cnn_lstm_probs.npy',  y_probs)\n",
        "torch.save(model.state_dict(), 'data/cnn_lstm_model.pt')\n",
        "print('Predictions and model saved.')"
    ),

    md("## 4.8  Confusion matrix"),
    code(
        "cm = confusion_matrix(y_test, y_pred, normalize='true')\n\n",
        "fig, ax = plt.subplots(figsize=(7, 6))\n",
        "sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues',\n",
        "            xticklabels=['DOWN','FLAT','UP'],\n",
        "            yticklabels=['DOWN','FLAT','UP'], ax=ax)\n",
        "ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')\n",
        "ax.set_title('CNN-LSTM — Confusion Matrix (row-normalised)')\n",
        "plt.tight_layout()\n",
        "plt.savefig('data/fig_confusion_cnn_lstm.png', dpi=120, bbox_inches='tight')\n",
        "plt.show()"
    ),

    md("---\n> ✅ **CNN-LSTM trained.** Proceed to `05_evaluation_and_comparison.ipynb`.")
])

# ─────────────────────────────────────────────────────────────────────────────
# NB 05  –  Evaluation & Comparison
# ─────────────────────────────────────────────────────────────────────────────

nb05 = nb([
    md("# 📈 Notebook 05 — Model Evaluation & Comparison\n",
       "> **Purpose:** Side-by-side comparison of all three models on classification\n",
       "> and financial metrics. Produce the capstone results table.\n\n",
       "**Inputs:** `data/baseline_results.csv`, `data/cnn_lstm_preds.npy`, model checkpoints  \n",
       "**Outputs:** `data/final_results.csv`, comparison plots\n\n---"),

    md("## 5.1  Load all predictions"),
    code(
        "import sys, os\n",
        "sys.path.insert(0, os.path.abspath('..'))\n\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import xgboost as xgb\n",
        "import torch\n",
        "from sklearn.metrics import (\n",
        "    classification_report, confusion_matrix,\n",
        "    f1_score, accuracy_score, roc_auc_score\n",
        ")\n",
        "sns.set_theme(style='darkgrid')\n",
        "%matplotlib inline\n\n",
        "y_test_seq  = np.load('data/y_test_seq.npy')\n",
        "y_pred_cnn  = np.load('data/cnn_lstm_preds.npy')\n",
        "y_probs_cnn = np.load('data/cnn_lstm_probs.npy')\n",
        "y_test_2d   = np.load('data/y_test_2d.npy')\n",
        "X_test_2d   = np.load('data/X_test_2d.npy')\n",
        "X_test_seq  = np.load('data/X_test_seq.npy')"
    ),

    code(
        "# XGBoost predictions\n",
        "xgb_model = xgb.XGBClassifier()\n",
        "xgb_model.load_model('data/xgb_model.json')\n",
        "y_pred_xgb = xgb_model.predict(X_test_2d)\n\n",
        "# LSTM-only predictions\n",
        "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n\n",
        "class LSTMOnly(torch.nn.Module):\n",
        "    def __init__(self, input_size=42, hidden=128, num_layers=2, n_classes=3, dropout=0.3):\n",
        "        super().__init__()\n",
        "        self.lstm = torch.nn.LSTM(input_size, hidden, num_layers,\n",
        "                                   batch_first=True, dropout=dropout)\n",
        "        self.fc = torch.nn.Sequential(\n",
        "            torch.nn.Linear(hidden, 64), torch.nn.ReLU(),\n",
        "            torch.nn.Dropout(dropout),   torch.nn.Linear(64, n_classes)\n",
        "        )\n",
        "    def forward(self, x):\n",
        "        out, _ = self.lstm(x)\n",
        "        return self.fc(out[:, -1, :])\n\n",
        "lstm_model = LSTMOnly().to(DEVICE)\n",
        "lstm_model.load_state_dict(torch.load('data/lstm_only_model.pt', map_location=DEVICE))\n",
        "lstm_model.eval()\n\n",
        "Xt = torch.tensor(X_test_seq, dtype=torch.float32)\n",
        "with torch.no_grad():\n",
        "    y_pred_lstm = lstm_model(Xt.to(DEVICE)).argmax(1).cpu().numpy()\n\n",
        "print('Predictions loaded for all 3 models.')"
    ),

    md("## 5.2  Classification metrics table"),
    code(
        "def metrics(name, y_true, y_pred):\n",
        "    return {\n",
        "        'Model':    name,\n",
        "        'Accuracy': accuracy_score(y_true, y_pred),\n",
        "        'F1-macro': f1_score(y_true, y_pred, average='macro'),\n",
        "        'F1-DOWN':  f1_score(y_true, y_pred, average=None)[0],\n",
        "        'F1-FLAT':  f1_score(y_true, y_pred, average=None)[1],\n",
        "        'F1-UP':    f1_score(y_true, y_pred, average=None)[2],\n",
        "    }\n\n",
        "results = pd.DataFrame([\n",
        "    metrics('XGBoost',   y_test_2d,  y_pred_xgb),\n",
        "    metrics('LSTM-only', y_test_seq, y_pred_lstm),\n",
        "    metrics('CNN-LSTM',  y_test_seq, y_pred_cnn),\n",
        "]).set_index('Model').round(4)\n\n",
        "display(results.style.background_gradient(cmap='RdYlGn', axis=0))"
    ),

    md("## 5.3  Confusion matrices — side by side"),
    code(
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
        "models   = ['XGBoost', 'LSTM-only', 'CNN-LSTM']\n",
        "y_preds  = [y_pred_xgb, y_pred_lstm, y_pred_cnn]\n",
        "y_trues  = [y_test_2d, y_test_seq, y_test_seq]\n\n",
        "for ax, name, yt, yp in zip(axes, models, y_trues, y_preds):\n",
        "    cm = confusion_matrix(yt, yp, normalize='true')\n",
        "    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues',\n",
        "                xticklabels=['DOWN','FLAT','UP'],\n",
        "                yticklabels=['DOWN','FLAT','UP'], ax=ax)\n",
        "    ax.set_title(f'{name}\\nAcc={accuracy_score(yt,yp):.3f}  F1={f1_score(yt,yp,average=\"macro\"):.3f}')\n",
        "    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')\n\n",
        "plt.suptitle('Confusion Matrices — All Models', fontsize=13, fontweight='bold')\n",
        "plt.tight_layout()\n",
        "plt.savefig('data/fig_confusion_comparison.png', dpi=120, bbox_inches='tight')\n",
        "plt.show()"
    ),

    md("## 5.4  Accuracy & F1 bar chart"),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n\n",
        "model_names = results.index.tolist()\n",
        "x = np.arange(len(model_names))\n",
        "w = 0.35\n\n",
        "axes[0].bar(x, results['Accuracy'], width=0.5,\n",
        "            color=['steelblue','darkorange','seagreen'])\n",
        "axes[0].axhline(0.60, ls='--', color='red', label='Target 60%')\n",
        "axes[0].set_xticks(x); axes[0].set_xticklabels(model_names)\n",
        "axes[0].set_title('Test Accuracy'); axes[0].set_ylim(0.45, 0.75)\n",
        "axes[0].legend()\n\n",
        "axes[1].bar(x, results['F1-macro'], width=0.5,\n",
        "            color=['steelblue','darkorange','seagreen'])\n",
        "axes[1].axhline(0.59, ls='--', color='red', label='Target F1=0.59')\n",
        "axes[1].set_xticks(x); axes[1].set_xticklabels(model_names)\n",
        "axes[1].set_title('F1-macro'); axes[1].set_ylim(0.45, 0.70)\n",
        "axes[1].legend()\n\n",
        "plt.suptitle('Model Comparison — Classification Metrics', fontsize=13, fontweight='bold')\n",
        "plt.tight_layout()\n",
        "plt.savefig('data/fig_model_comparison.png', dpi=120, bbox_inches='tight')\n",
        "plt.show()"
    ),

    md("## 5.5  Hit ratio (directional accuracy — excl. FLAT predictions)"),
    code(
        "def hit_ratio(y_true, y_pred):\n",
        "    \"\"\"Fraction of UP/DOWN predictions that were correct (excludes FLAT preds).\"\"\"\n",
        "    mask = y_pred != 1    # exclude FLAT (1)\n",
        "    if mask.sum() == 0:\n",
        "        return np.nan\n",
        "    return accuracy_score(y_true[mask], y_pred[mask])\n\n",
        "hit_rows = []\n",
        "for name, yt, yp in zip(models, y_trues, y_preds):\n",
        "    hr = hit_ratio(yt, yp)\n",
        "    flat_pct = (yp == 1).mean() * 100\n",
        "    hit_rows.append({'Model': name, 'Hit Ratio': hr, 'Flat %': flat_pct})\n\n",
        "hit_df = pd.DataFrame(hit_rows).set_index('Model').round(4)\n",
        "display(hit_df)"
    ),

    md("## 5.6  Save final results"),
    code(
        "final = results.copy()\n",
        "final['Hit Ratio'] = [hit_ratio(yt, yp)\n",
        "                       for yt, yp in zip(y_trues, y_preds)]\n",
        "final.to_csv('data/final_results.csv')\n",
        "print('Saved data/final_results.csv')\n",
        "display(final.round(4))"
    ),

    md("---\n> ✅ **Evaluation complete.** Proceed to `06_order_execution_optimizer.ipynb`.")
])

# ─────────────────────────────────────────────────────────────────────────────
# NB 06  –  Order Execution Optimizer
# ─────────────────────────────────────────────────────────────────────────────

nb06 = nb([
    md("# 💹 Notebook 06 — Order Execution Optimizer & Slippage Simulation\n",
       "> **Purpose:** Use CNN-LSTM direction signals to build an ML-directed order slicer.\n",
       "> Simulate execution of a 10,000-share order and compare slippage vs TWAP and VWAP.\n\n",
       "**Inputs:** `data/lobster_features.parquet`, `data/cnn_lstm_model.pt`, `data/scaler.joblib`  \n",
       "**Outputs:** `data/execution_results.csv`, slippage comparison plots\n\n---"),

    md("## 6.1  Load model, scaler, and clean data"),
    code(
        "import sys, os\n",
        "sys.path.insert(0, os.path.abspath('..'))\n\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import joblib\n",
        "import torch\n",
        "sns.set_theme(style='darkgrid')\n",
        "%matplotlib inline\n\n",
        "from utils.feature_builder import FEATURE_COLS_42, build_sequences, apply_scaler\n",
        "from utils.slippage_simulator import (\n",
        "    twap_execute, vwap_execute, ml_directed_execute, compare_strategies\n",
        ")\n\n",
        "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n\n",
        "# Load data\n",
        "df = pd.read_parquet('data/lobster_features.parquet')\n",
        "scaler = joblib.load('data/scaler.joblib')\n\n",
        "print(f'Loaded {len(df):,} rows')"
    ),

    code(
        "# Rebuild CNN-LSTM model and load weights\n",
        "class CNNLSTM(torch.nn.Module):\n",
        "    def __init__(self, input_size=42, cnn_channels=[64,128,256],\n",
        "                 lstm_hidden=128, lstm_layers=2, n_classes=3, dropout=0.3):\n",
        "        super().__init__()\n",
        "        cnn_layers = []\n",
        "        in_ch = input_size\n",
        "        for out_ch in cnn_channels:\n",
        "            cnn_layers += [\n",
        "                torch.nn.Conv1d(in_ch, out_ch, 3, padding=1),\n",
        "                torch.nn.BatchNorm1d(out_ch), torch.nn.ReLU(),\n",
        "                torch.nn.Dropout(dropout/2)\n",
        "            ]\n",
        "            in_ch = out_ch\n",
        "        self.cnn  = torch.nn.Sequential(*cnn_layers)\n",
        "        self.lstm = torch.nn.LSTM(cnn_channels[-1], lstm_hidden, lstm_layers,\n",
        "                                   batch_first=True, bidirectional=True,\n",
        "                                   dropout=dropout if lstm_layers>1 else 0)\n",
        "        self.head = torch.nn.Sequential(\n",
        "            torch.nn.Linear(lstm_hidden*2, 128), torch.nn.ReLU(),\n",
        "            torch.nn.Dropout(dropout),\n",
        "            torch.nn.Linear(128, 64), torch.nn.ReLU(),\n",
        "            torch.nn.Dropout(dropout), torch.nn.Linear(64, n_classes)\n",
        "        )\n",
        "    def forward(self, x):\n",
        "        x = self.cnn(x.permute(0,2,1)).permute(0,2,1)\n",
        "        out, _ = self.lstm(x)\n",
        "        return self.head(out[:,-1,:])\n\n",
        "model = CNNLSTM().to(DEVICE)\n",
        "model.load_state_dict(torch.load('data/cnn_lstm_model.pt', map_location=DEVICE))\n",
        "model.eval()\n",
        "print('CNN-LSTM model loaded.')"
    ),

    md("## 6.2  Generate signals for full dataset"),
    code(
        "from utils.feature_builder import build_feature_matrix\n\n",
        "X_df  = build_feature_matrix(df, FEATURE_COLS_42)\n",
        "X_raw = X_df.values\n\n",
        "# Scale\n",
        "X_scaled = scaler.transform(X_raw).astype(np.float32)\n\n",
        "# Build sequences\n",
        "SEQ_LEN = 20\n",
        "N = len(X_scaled)\n",
        "X_all_seq = np.stack([X_scaled[i:i+SEQ_LEN] for i in range(N-SEQ_LEN)]).astype(np.float32)\n\n",
        "# Batch inference\n",
        "BATCH = 512\n",
        "all_signals = []\n",
        "with torch.no_grad():\n",
        "    for i in range(0, len(X_all_seq), BATCH):\n",
        "        Xb = torch.tensor(X_all_seq[i:i+BATCH]).to(DEVICE)\n",
        "        preds = model(Xb).argmax(1).cpu().numpy() - 1  # remap back to {-1,0,+1}\n",
        "        all_signals.extend(preds)\n\n",
        "# Pad first SEQ_LEN rows with 0 (neutral)\n",
        "signals_full = np.array([0]*SEQ_LEN + all_signals)\n",
        "df_sim = df.iloc[:len(signals_full)].copy()\n",
        "df_sim['signal'] = signals_full\n",
        "print(f'Signals generated for {len(df_sim):,} events')\n",
        "print(f'Signal dist: DOWN={np.sum(signals_full==-1):,}  FLAT={np.sum(signals_full==0):,}  UP={np.sum(signals_full==1):,}')"
    ),

    md("## 6.3  Simulate order execution — 10,000 shares"),
    code(
        "# Use the middle portion of the day (avoid open/close)\n",
        "sim_start = len(df_sim) // 4\n",
        "sim_end   = 3 * len(df_sim) // 4\n",
        "df_exec   = df_sim.iloc[sim_start:sim_end].reset_index(drop=True)\n",
        "signals   = df_exec['signal'].values\n\n",
        "TOTAL_SHARES = 10_000\n",
        "N_SLICES     = 10\n\n",
        "# ── Run all three strategies ──\n",
        "result_twap = twap_execute(df_exec, TOTAL_SHARES, N_SLICES, side='buy')\n",
        "result_vwap = vwap_execute(df_exec, TOTAL_SHARES, N_SLICES, side='buy')\n",
        "result_ml   = ml_directed_execute(df_exec, signals, TOTAL_SHARES,\n",
        "                                   N_SLICES, side='buy', aggression=2.0)\n\n",
        "cmp = compare_strategies(result_twap, result_vwap, result_ml, reference='TWAP')\n",
        "display(cmp.style.background_gradient(cmap='RdYlGn', axis=0))"
    ),

    md("## 6.4  Slippage comparison plot"),
    code(
        "strategies  = ['TWAP', 'VWAP', 'ML-Directed']\n",
        "slippages   = [result_twap.slippage_bps, result_vwap.slippage_bps, result_ml.slippage_bps]\n",
        "improvements = [0,\n",
        "                result_twap.slippage_bps - result_vwap.slippage_bps,\n",
        "                result_twap.slippage_bps - result_ml.slippage_bps]\n\n",
        "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n\n",
        "colors = ['steelblue', 'darkorange', 'seagreen']\n",
        "bars = axes[0].bar(strategies, slippages, color=colors, width=0.5)\n",
        "axes[0].axhline(0, color='black', lw=0.8)\n",
        "axes[0].set_ylabel('Slippage (bps)')\n",
        "axes[0].set_title('Execution Slippage by Strategy')\n",
        "for bar, v in zip(bars, slippages):\n",
        "    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.01,\n",
        "                 f'{v:.2f} bps', ha='center', va='bottom', fontsize=9)\n\n",
        "bars2 = axes[1].bar(strategies, improvements, color=colors, width=0.5)\n",
        "axes[1].set_ylabel('Slippage saved vs TWAP (bps)')\n",
        "axes[1].set_title('Improvement vs TWAP Baseline')\n",
        "for bar, v in zip(bars2, improvements):\n",
        "    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.005,\n",
        "                 f'+{v:.2f}' if v >= 0 else f'{v:.2f}', ha='center', fontsize=9)\n\n",
        "plt.suptitle(f'Order Execution Simulation — {TOTAL_SHARES:,} Shares', fontsize=13, fontweight='bold')\n",
        "plt.tight_layout()\n",
        "plt.savefig('data/fig_slippage_comparison.png', dpi=120, bbox_inches='tight')\n",
        "plt.show()"
    ),

    md("## 6.5  Fill price timeline"),
    code(
        "def plot_fills(result, ax, color, label):\n",
        "    idxs   = [f['idx'] for f in result.fills]\n",
        "    prices = [f['avg_price'] for f in result.fills]\n",
        "    sizes  = [f['shares'] for f in result.fills]\n",
        "    ax.scatter(idxs, prices, s=[s/50 for s in sizes], c=color, alpha=0.7, label=label)\n",
        "    ax.plot(idxs, prices, c=color, alpha=0.4, lw=1)\n\n",
        "fig, ax = plt.subplots(figsize=(13, 5))\n",
        "# Background mid price\n",
        "ax.plot(df_exec.index, df_exec['mid_price'], color='gray', lw=0.5,\n",
        "        alpha=0.5, label='Mid price')\n\n",
        "plot_fills(result_twap, ax, 'steelblue',  'TWAP fills')\n",
        "plot_fills(result_vwap, ax, 'darkorange', 'VWAP fills')\n",
        "plot_fills(result_ml,   ax, 'seagreen',   'ML-Directed fills')\n\n",
        "ax.set_xlabel('Event index')\n",
        "ax.set_ylabel('Fill price ($)')\n",
        "ax.set_title('Fill Prices vs Mid Price — All Strategies')\n",
        "ax.legend()\n",
        "plt.tight_layout()\n",
        "plt.savefig('data/fig_fill_timeline.png', dpi=120, bbox_inches='tight')\n",
        "plt.show()"
    ),

    md("## 6.6  Multi-window Monte Carlo simulation"),
    code(
        "\"\"\"Run the simulation across 50 random windows to estimate average slippage improvement.\"\"\"\n",
        "import random\n",
        "random.seed(42)\n",
        "np.random.seed(42)\n\n",
        "WINDOW_SIZE = 5000\n",
        "N_RUNS      = 50\n",
        "results_mc  = []\n\n",
        "for run in range(N_RUNS):\n",
        "    start = random.randint(0, len(df_sim) - WINDOW_SIZE - 1)\n",
        "    df_w  = df_sim.iloc[start:start+WINDOW_SIZE].reset_index(drop=True)\n",
        "    sigs  = df_w['signal'].values\n\n",
        "    r_twap = twap_execute(df_w, TOTAL_SHARES, N_SLICES)\n",
        "    r_vwap = vwap_execute(df_w, TOTAL_SHARES, N_SLICES)\n",
        "    r_ml   = ml_directed_execute(df_w, sigs, TOTAL_SHARES, N_SLICES)\n\n",
        "    results_mc.append({\n",
        "        'run':        run,\n",
        "        'twap_slip':  r_twap.slippage_bps,\n",
        "        'vwap_slip':  r_vwap.slippage_bps,\n",
        "        'ml_slip':    r_ml.slippage_bps,\n",
        "        'ml_vs_twap': r_twap.slippage_bps - r_ml.slippage_bps\n",
        "    })\n\n",
        "mc_df = pd.DataFrame(results_mc)\n",
        "print(mc_df[['twap_slip','vwap_slip','ml_slip','ml_vs_twap']].describe().round(4))"
    ),

    code(
        "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n\n",
        "axes[0].hist(mc_df['twap_slip'], bins=20, alpha=0.6, label='TWAP', color='steelblue')\n",
        "axes[0].hist(mc_df['vwap_slip'], bins=20, alpha=0.6, label='VWAP', color='darkorange')\n",
        "axes[0].hist(mc_df['ml_slip'],   bins=20, alpha=0.6, label='ML',   color='seagreen')\n",
        "axes[0].set_title('Slippage Distribution (50 windows)')\n",
        "axes[0].set_xlabel('Slippage (bps)'); axes[0].legend()\n\n",
        "axes[1].hist(mc_df['ml_vs_twap'], bins=20, color='seagreen', edgecolor='black', alpha=0.8)\n",
        "axes[1].axvline(mc_df['ml_vs_twap'].mean(), color='red', ls='--',\n",
        "                label=f'Mean: {mc_df[\"ml_vs_twap\"].mean():.2f} bps')\n",
        "axes[1].set_title('ML improvement over TWAP (bps)')\n",
        "axes[1].set_xlabel('Saved bps'); axes[1].legend()\n\n",
        "plt.suptitle('Monte Carlo Execution Simulation (50 runs)', fontsize=13, fontweight='bold')\n",
        "plt.tight_layout()\n",
        "plt.savefig('data/fig_monte_carlo.png', dpi=120, bbox_inches='tight')\n",
        "plt.show()\n\n",
        "mc_df.to_csv('data/execution_results.csv', index=False)\n",
        "print(f'\\nMean slippage improvement vs TWAP: {mc_df[\"ml_vs_twap\"].mean():.2f} bps')\n",
        "print(f'Equivalent to {mc_df[\"ml_vs_twap\"].mean()/mc_df[\"twap_slip\"].mean()*100:.1f}% reduction')"
    ),

    md("---\n> ✅ **Execution simulation complete.** Proceed to `07_streamlit_dashboard.py` to launch the live dashboard.")
])

# ─────────────────────────────────────────────────────────────────────────────
# NB 07  –  Streamlit Dashboard (as notebook with explanation + code)
# ─────────────────────────────────────────────────────────────────────────────

nb07 = nb([
    md("# 🖥️ Notebook 07 — Streamlit Dashboard\n",
       "> **Purpose:** Explains the Streamlit dashboard and lets you launch it from here.\n\n",
       "The full dashboard code lives in `07_streamlit_dashboard.py`.\n\n",
       "**Launch command:**\n",
       "```bash\n",
       "streamlit run 07_streamlit_dashboard.py\n",
       "```\n\n---"),

    md("## 7.1  Launch dashboard from notebook"),
    code(
        "import subprocess, sys\n\n",
        "# Launch the Streamlit dashboard in a background process\n",
        "proc = subprocess.Popen(\n",
        "    [sys.executable, '-m', 'streamlit', 'run',\n",
        "     '07_streamlit_dashboard.py', '--server.port', '8501'],\n",
        "    stdout=subprocess.PIPE, stderr=subprocess.PIPE\n",
        ")\n",
        "print('Streamlit dashboard started!')\n",
        "print('Open: http://localhost:8501')"
    ),

    md("## 7.2  Dashboard feature overview\n",
       "\n",
       "| Section | Description |\n",
       "|---------|-------------|\n",
       "| **Data Explorer** | Upload or preview LOBSTER message/orderbook files |\n",
       "| **LOB Visualiser** | Live orderbook depth chart at any event index |\n",
       "| **Signal Monitor** | CNN-LSTM prediction probabilities (UP/FLAT/DOWN) |\n",
       "| **Execution Simulator** | Slippage comparison TWAP vs VWAP vs ML-Directed |\n",
       "| **Results Dashboard** | Model comparison metrics and Monte Carlo stats |\n"),

    md("## 7.3  Dashboard source code preview"),
    code(
        "# Read and display the Streamlit source\n",
        "with open('07_streamlit_dashboard.py', 'r') as f:\n",
        "    src = f.read()\n",
        "print(src[:3000], '\\n... [truncated] ...')"
    ),

    md("---\n> ✅ **Capstone pipeline complete!**\n\n",
       "### Run order:\n",
       "```\n",
       "00 → 01 → 02 → 03 → 04 → 05 → 06 → 07\n",
       "```")
])

# ─────────────────────────────────────────────────────────────────────────────
# Write all notebooks
# ─────────────────────────────────────────────────────────────────────────────

print("Generating notebooks...")
save("00_setup_and_data_download.ipynb",   nb00)
save("01_eda_and_preprocessing.ipynb",     nb01)
save("02_feature_engineering.ipynb",       nb02)
save("03_baseline_models.ipynb",           nb03)
save("04_cnn_lstm_model.ipynb",            nb04)
save("05_evaluation_and_comparison.ipynb", nb05)
save("06_order_execution_optimizer.ipynb", nb06)
save("07_streamlit_dashboard.ipynb",       nb07)
print("\nAll 8 notebooks generated successfully.")
