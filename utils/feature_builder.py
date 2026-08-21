"""
feature_builder.py
──────────────────
Feature engineering from LOBSTER merged DataFrame.

Features produced (42 total):
  ─ Raw LOB (40):  AskP1..AskP10, AskS1..AskS10,
                   BidP1..BidP10, BidS1..BidS10
  ─ Engineered (2): order_imbalance, spread_bps

Sequence builder: stacks T consecutive rows → (T, 42) arrays for CNN-LSTM.
Label builder:    mid-price direction over next K events → {-1, 0, +1}.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

N_LEVELS   = 10
PRICE_TICK = 10000          # LOBSTER stores prices × 10000


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mid-price & Spread
# ─────────────────────────────────────────────────────────────────────────────

def add_mid_and_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Add mid_price (USD) and spread_bps columns."""
    df = df.copy()
    df["mid_price"]  = (df["AskP1"] + df["BidP1"]) / 2 / PRICE_TICK
    df["spread_bps"] = (df["AskP1"] - df["BidP1"]) / (df["AskP1"] + df["BidP1"]) * 20_000
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Order Book Imbalance  (OBI)
# ─────────────────────────────────────────────────────────────────────────────

def order_imbalance(df: pd.DataFrame, levels: int = 1) -> pd.Series:
    """
    Compute Order Book Imbalance across `levels` price levels.

    OBI = (sum_bid_vol - sum_ask_vol) / (sum_bid_vol + sum_ask_vol)
    Range: [-1, +1];  +1 = fully bid-side;  -1 = fully ask-side.
    """
    bid_vol = sum(df[f"BidS{i}"] for i in range(1, levels + 1))
    ask_vol = sum(df[f"AskS{i}"] for i in range(1, levels + 1))
    total   = bid_vol + ask_vol
    return (bid_vol - ask_vol) / total.replace(0, np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Additional Microstructure Features
# ─────────────────────────────────────────────────────────────────────────────

def add_micro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered microstructure features to the DataFrame.

    Added columns:
        imbalance_l1   : OBI at level 1 only
        imbalance_l5   : OBI across levels 1-5
        imbalance_l10  : OBI across all 10 levels
        spread_bps     : bid-ask spread in basis points
        depth_bid      : total bid-side volume (10 levels)
        depth_ask      : total ask-side volume (10 levels)
        depth_ratio    : depth_bid / depth_ask
        price_range    : (AskP10 - BidP10) / mid_price in bps  (10-level range)
        vwap_bid       : volume-weighted avg bid price (USD)
        vwap_ask       : volume-weighted avg ask price (USD)
        vwap_imbalance : (vwap_bid - mid) / spread_bps  (signed pressure)
    """
    df = df.copy()

    if "mid_price" not in df.columns:
        df = add_mid_and_spread(df)

    # ── Imbalance at multiple depths ──
    df["imbalance_l1"]  = order_imbalance(df, levels=1)
    df["imbalance_l5"]  = order_imbalance(df, levels=5)
    df["imbalance_l10"] = order_imbalance(df, levels=10)

    # ── Depth ──
    df["depth_bid"]   = sum(df[f"BidS{i}"] for i in range(1, N_LEVELS + 1))
    df["depth_ask"]   = sum(df[f"AskS{i}"] for i in range(1, N_LEVELS + 1))
    df["depth_ratio"] = df["depth_bid"] / df["depth_ask"].replace(0, np.nan)

    # ── 10-level price range ──
    df["price_range"] = ((df["AskP10"] - df["BidP10"]) /
                         (df["mid_price"] * PRICE_TICK)) * 10_000   # bps

    # ── VWAP bid / ask (1 level for speed; extend as needed) ──
    bid_notional = sum(df[f"BidP{i}"] * df[f"BidS{i}"] for i in range(1, N_LEVELS + 1))
    ask_notional = sum(df[f"AskP{i}"] * df[f"AskS{i}"] for i in range(1, N_LEVELS + 1))
    df["vwap_bid"] = bid_notional / df["depth_bid"].replace(0, np.nan) / PRICE_TICK
    df["vwap_ask"] = ask_notional / df["depth_ask"].replace(0, np.nan) / PRICE_TICK

    spread_usd = df["spread_bps"] * df["mid_price"] / 10_000
    df["vwap_imbalance"] = (df["vwap_bid"] - df["mid_price"]) / spread_usd.replace(0, np.nan)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Label Generation
# ─────────────────────────────────────────────────────────────────────────────

def make_labels(df: pd.DataFrame,
                horizon: int = 10,
                threshold_bps: float = 0.5) -> pd.Series:
    """
    Compute 3-class direction label.

    Label = sign( mid_price[t+horizon] - mid_price[t] )
    with a ±threshold_bps dead-band to define FLAT.

    Parameters
    ----------
    df             : DataFrame with mid_price column
    horizon        : number of events to look ahead (default 10)
    threshold_bps  : if |Δmid| < threshold_bps bps → FLAT (0)
                     else UP (+1) or DOWN (-1)

    Returns
    -------
    pd.Series  int8, values in {-1, 0, +1}
        -1 : DOWN
         0 : FLAT
        +1 : UP
    """
    assert "mid_price" in df.columns, "Run add_mid_and_spread() first."

    future_mid  = df["mid_price"].shift(-horizon)
    delta_bps   = (future_mid - df["mid_price"]) / df["mid_price"] * 10_000

    labels = pd.Series(0, index=df.index, dtype="int8", name="label")
    labels[delta_bps >  threshold_bps] =  1
    labels[delta_bps < -threshold_bps] = -1

    dist = labels.value_counts().sort_index()
    print(f"[make_labels] horizon={horizon}, threshold=±{threshold_bps} bps")
    print(f"  DOWN(-1): {dist.get(-1,0):,}  FLAT(0): {dist.get(0,0):,}  UP(+1): {dist.get(1,0):,}")
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# 5. Feature Matrix Assembly
# ─────────────────────────────────────────────────────────────────────────────

# The 40 raw LOB columns
LOB_COLS = (
    [f"AskP{i}" for i in range(1, N_LEVELS + 1)] +
    [f"AskS{i}" for i in range(1, N_LEVELS + 1)] +
    [f"BidP{i}" for i in range(1, N_LEVELS + 1)] +
    [f"BidS{i}" for i in range(1, N_LEVELS + 1)]
)

# The 2 baseline engineered features (for XGBoost / baseline models)
BASE_ENG_COLS = ["imbalance_l1", "spread_bps"]

# Full 42-feature set (CNN-LSTM)
FEATURE_COLS_42 = LOB_COLS + BASE_ENG_COLS

# Extended feature set (for XGBoost / richer baselines)
FEATURE_COLS_EXT = LOB_COLS + [
    "imbalance_l1", "imbalance_l5", "imbalance_l10",
    "spread_bps", "depth_bid", "depth_ask", "depth_ratio",
    "price_range", "vwap_bid", "vwap_ask", "vwap_imbalance",
]


def build_feature_matrix(df: pd.DataFrame,
                         feature_cols: list = FEATURE_COLS_42) -> pd.DataFrame:
    """
    Extract and forward-fill feature matrix, drop any remaining NaNs.
    Returns float32 DataFrame.
    """
    X = df[feature_cols].copy().astype("float32")
    X = X.replace([np.inf, -np.inf], np.nan).ffill().dropna()
    print(f"[build_feature_matrix] Shape: {X.shape}  NaNs: {X.isna().sum().sum()}")
    return X


# ─────────────────────────────────────────────────────────────────────────────
# 6. Sequence Builder for CNN-LSTM
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(X: np.ndarray,
                    y: np.ndarray,
                    seq_len: int = 20) -> tuple:
    """
    Slide a window of length seq_len over (X, y) to produce
    3-D input tensors for CNN-LSTM.

    Parameters
    ----------
    X       : (N, F) feature array
    y       : (N,)  label array
    seq_len : window size T (default 20)

    Returns
    -------
    X_seq : (N-T, T, F)  float32
    y_seq : (N-T,)       int64
    """
    N, F = X.shape
    n_seq = N - seq_len

    X_seq = np.empty((n_seq, seq_len, F), dtype=np.float32)
    y_seq = np.empty(n_seq, dtype=np.int64)

    for i in range(n_seq):
        X_seq[i] = X[i : i + seq_len]
        y_seq[i] = y[i + seq_len]

    print(f"[build_sequences] X_seq={X_seq.shape}  y_seq={y_seq.shape}")
    return X_seq, y_seq


# ─────────────────────────────────────────────────────────────────────────────
# 7. Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def fit_scaler(X_train_2d: np.ndarray) -> StandardScaler:
    """Fit StandardScaler on 2-D training data (N×F)."""
    scaler = StandardScaler()
    scaler.fit(X_train_2d)
    return scaler


def apply_scaler(scaler: StandardScaler, X_3d: np.ndarray) -> np.ndarray:
    """
    Apply a fitted scaler to a 3-D sequence array (N, T, F).
    Reshapes → scales → reshapes back.
    """
    N, T, F = X_3d.shape
    X_2d = X_3d.reshape(-1, F)
    X_scaled = scaler.transform(X_2d).reshape(N, T, F)
    return X_scaled.astype(np.float32)
