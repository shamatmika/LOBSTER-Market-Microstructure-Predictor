"""
07_streamlit_dashboard.py
─────────────────────────
LOBSTER Market Microstructure Predictor — Interactive Dashboard

Launch:
    streamlit run 07_streamlit_dashboard.py

Requires all upstream notebooks (00–06) to have been run first.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import json
import torch
import streamlit as st

from utils.feature_builder import (
    FEATURE_COLS_42, add_mid_and_spread, add_micro_features,
    make_labels, build_feature_matrix
)
from utils.slippage_simulator import (
    twap_execute, vwap_execute, ml_directed_execute, compare_strategies
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LOBSTER Microstructure Predictor",
    page_icon="🦞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CNN-LSTM definition (must match NB04)
# ─────────────────────────────────────────────────────────────────────────────
class CNNLSTM(torch.nn.Module):
    def __init__(self, input_size=42, cnn_channels=[64, 128, 256],
                 lstm_hidden=128, lstm_layers=2, n_classes=3, dropout=0.3):
        super().__init__()
        cnn_layers = []
        in_ch = input_size
        for out_ch in cnn_channels:
            cnn_layers += [
                torch.nn.Conv1d(in_ch, out_ch, 3, padding=1),
                torch.nn.BatchNorm1d(out_ch),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout / 2),
            ]
            in_ch = out_ch
        self.cnn  = torch.nn.Sequential(*cnn_layers)
        self.lstm = torch.nn.LSTM(cnn_channels[-1], lstm_hidden, lstm_layers,
                                   batch_first=True, bidirectional=True,
                                   dropout=dropout if lstm_layers > 1 else 0)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(lstm_hidden * 2, 128), torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, n_classes),
        )
    def forward(self, x):
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# ─────────────────────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "data/cnn_lstm_model.pt"
    if not os.path.exists(model_path):
        return None
    model = CNNLSTM()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_scaler():
    path = "data/scaler.joblib"
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_data
def load_data():
    path = "data/lobster_features.parquet"
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)

@st.cache_data
def load_results():
    path = "data/final_results.csv"
    return pd.read_csv(path, index_col=0) if os.path.exists(path) else None

@st.cache_data
def load_mc_results():
    path = "data/execution_results.csv"
    return pd.read_csv(path) if os.path.exists(path) else None


# ─────────────────────────────────────────────────────────────────────────────
# Signal generator
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def generate_signals(_df, _model, _scaler, seq_len=20, batch_size=512):
    X_df     = build_feature_matrix(_df, FEATURE_COLS_42)
    X_scaled = _scaler.transform(X_df.values).astype(np.float32)
    N        = len(X_scaled)
    X_seq    = np.stack([X_scaled[i:i + seq_len] for i in range(N - seq_len)])

    signals = []
    with torch.no_grad():
        for i in range(0, len(X_seq), batch_size):
            Xb = torch.tensor(X_seq[i:i + batch_size])
            logits = _model(Xb)
            probs  = torch.softmax(logits, dim=1).numpy()
            preds  = logits.argmax(1).numpy() - 1   # → {-1, 0, +1}
            signals.append((preds, probs))

    all_preds = np.concatenate([s[0] for s in signals])
    all_probs = np.concatenate([s[1] for s in signals], axis=0)
    full_preds = np.array([0] * seq_len + list(all_preds))
    full_probs = np.vstack([np.full((seq_len, 3), 1/3),
                             all_probs])
    return full_preds, full_probs


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image("https://lobsterdata.com/img/LOBSTERLogo.png",
                 use_column_width=True)
st.sidebar.title("🦞 LOBSTER Predictor")
st.sidebar.markdown("**NASDAQ Microstructure ML Pipeline**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Data Explorer",
     "📉 LOB Visualiser",
     "🤖 Signal Monitor",
     "💹 Execution Simulator",
     "🏆 Model Results"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Pipeline status:**")

artifacts = {
    "lobster_features.parquet": "Features",
    "cnn_lstm_model.pt":        "CNN-LSTM",
    "scaler.joblib":            "Scaler",
    "final_results.csv":        "Results",
    "execution_results.csv":    "MC Sim",
}
for fname, label in artifacts.items():
    exists = os.path.exists(f"data/{fname}")
    icon   = "✅" if exists else "❌"
    st.sidebar.write(f"{icon} {label}")


# ─────────────────────────────────────────────────────────────────────────────
# Load all assets
# ─────────────────────────────────────────────────────────────────────────────
df     = load_data()
model  = load_model()
scaler = load_scaler()


# ─────────────────────────────────────────────────────────────────────────────
# ── Page 1: Data Explorer ────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Data Explorer":
    st.title("📊 LOBSTER Data Explorer")
    st.markdown("> Browse message events, orderbook snapshots, and key statistics.")

    if df is None:
        st.error("⚠️ `data/lobster_features.parquet` not found. Run NB 00–01 first.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    mid = df["mid_price"]
    col1.metric("Total Events",       f"{len(df):,}")
    col2.metric("Mid Price Range",    f"${mid.min():.2f} – ${mid.max():.2f}")
    col3.metric("Mean Spread",        f"{df['spread_bps'].mean():.2f} bps")
    col4.metric("Mean OBI (L1)",      f"{df['imbalance_l1'].mean():.3f}")

    st.markdown("---")

    st.subheader("Mid Price & Spread")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(df.index, df["mid_price"], lw=0.5, color="steelblue")
    axes[0].set_ylabel("Mid Price ($)")
    axes[1].fill_between(df.index, df["spread_bps"], alpha=0.5, color="darkorange")
    axes[1].set_ylabel("Spread (bps)")
    axes[1].set_xlabel("Event Index")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.subheader("Raw Data Sample")
    cols_to_show = ["Time", "Type", "Size", "Price", "Direction",
                    "mid_price", "spread_bps", "imbalance_l1"]
    st.dataframe(df[cols_to_show].head(200), use_container_width=True)

    st.subheader("Label Distribution")
    if "label" in df.columns:
        counts = df["label"].value_counts().sort_index()
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.bar(["DOWN", "FLAT", "UP"], counts.values,
                color=["tomato", "steelblue", "seagreen"])
        ax2.set_ylabel("Count")
        ax2.set_title("3-Class Label Distribution (horizon=10, ±0.5 bps)")
        plt.tight_layout()
        st.pyplot(fig2)


# ─────────────────────────────────────────────────────────────────────────────
# ── Page 2: LOB Visualiser ───────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📉 LOB Visualiser":
    st.title("📉 Limit Order Book Depth Visualiser")
    st.markdown("> Inspect the full 10-level bid/ask profile at any event index.")

    if df is None:
        st.error("Run NB 00–01 first."); st.stop()

    idx = st.slider("Event index", 0, len(df) - 1, len(df) // 2, step=10)
    snap = df.iloc[idx]

    bid_prices = [snap[f"BidP{i}"] / 10000 for i in range(1, 11)]
    bid_sizes  = [snap[f"BidS{i}"] for i in range(1, 11)]
    ask_prices = [snap[f"AskP{i}"] / 10000 for i in range(1, 11)]
    ask_sizes  = [snap[f"AskS{i}"] for i in range(1, 11)]
    mid_p      = (snap["AskP1"] + snap["BidP1"]) / 2 / 10000

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Bid", f"${snap['BidP1']/10000:.4f}", f"Size: {int(snap['BidS1'])}")
    col2.metric("Mid Price", f"${mid_p:.4f}")
    col3.metric("Best Ask", f"${snap['AskP1']/10000:.4f}", f"Size: {int(snap['AskS1'])}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(bid_prices, [-s for s in bid_sizes],
            height=0.00004, color="seagreen", alpha=0.85, label="Bid")
    ax.barh(ask_prices, ask_sizes,
            height=0.00004, color="tomato", alpha=0.85, label="Ask")
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(mid_p, color="gold", ls="--", lw=1, label=f"Mid ${mid_p:.4f}")
    ax.set_xlabel("Volume (shares)")
    ax.set_ylabel("Price ($)")
    ax.set_title(f"LOB Depth — Event {idx} | Spread: {snap['spread_bps']:.2f} bps")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.subheader("Level-by-level breakdown")
    level_data = pd.DataFrame({
        "Level":     list(range(1, 11)),
        "Bid Price": [f"${snap[f'BidP{i}']/10000:.4f}" for i in range(1, 11)],
        "Bid Size":  [int(snap[f"BidS{i}"]) for i in range(1, 11)],
        "Ask Price": [f"${snap[f'AskP{i}']/10000:.4f}" for i in range(1, 11)],
        "Ask Size":  [int(snap[f"AskS{i}"]) for i in range(1, 11)],
    })
    st.dataframe(level_data, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── Page 3: Signal Monitor ───────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Signal Monitor":
    st.title("🤖 CNN-LSTM Signal Monitor")
    st.markdown("> Real-time direction predictions from the trained CNN-LSTM model.")

    if df is None or model is None or scaler is None:
        st.error("Run NB 00–04 first."); st.stop()

    with st.spinner("Generating signals…"):
        signals, probs = generate_signals(df, model, scaler)

    # Attach to df
    df_sig = df.copy()
    df_sig["signal"] = signals
    df_sig[["prob_down", "prob_flat", "prob_up"]] = probs

    # Controls
    window = st.slider("Display window (events)", 200, 2000, 500, step=100)
    start_pct = st.slider("Start position (%)", 0, 90, 20)
    start = int(len(df_sig) * start_pct / 100)
    end   = start + window
    df_w  = df_sig.iloc[start:end]

    col1, col2, col3 = st.columns(3)
    col1.metric("UP signals",   f"{(df_w['signal']==1).sum():,}  ({(df_w['signal']==1).mean()*100:.1f}%)")
    col2.metric("FLAT signals", f"{(df_w['signal']==0).sum():,}  ({(df_w['signal']==0).mean()*100:.1f}%)")
    col3.metric("DOWN signals", f"{(df_w['signal']==-1).sum():,}  ({(df_w['signal']==-1).mean()*100:.1f}%)")

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    axes[0].plot(df_w.index, df_w["mid_price"], lw=0.8, color="steelblue")
    # Colour background by signal
    for i in df_w.index:
        sig = df_w.loc[i, "signal"]
        if sig == 1:
            axes[0].axvspan(i, i+1, alpha=0.15, color="seagreen")
        elif sig == -1:
            axes[0].axvspan(i, i+1, alpha=0.15, color="tomato")
    axes[0].set_ylabel("Mid Price ($)")
    axes[0].set_title("Mid Price with UP (green) / DOWN (red) signals")

    axes[1].stackplot(df_w.index,
                      df_w["prob_down"], df_w["prob_flat"], df_w["prob_up"],
                      labels=["DOWN", "FLAT", "UP"],
                      colors=["tomato", "steelblue", "seagreen"], alpha=0.75)
    axes[1].set_ylabel("Probability")
    axes[1].set_title("CNN-LSTM Class Probabilities")
    axes[1].legend(loc="upper right")

    axes[2].scatter(df_w.index, df_w["signal"], s=3,
                    c=df_w["signal"].map({-1: "tomato", 0: "steelblue", 1: "seagreen"}))
    axes[2].set_yticks([-1, 0, 1])
    axes[2].set_yticklabels(["DOWN", "FLAT", "UP"])
    axes[2].set_ylabel("Signal")
    axes[2].set_xlabel("Event Index")
    axes[2].set_title("Raw Direction Signal")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── Page 4: Execution Simulator ──────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💹 Execution Simulator":
    st.title("💹 Order Execution Simulator")
    st.markdown("> Simulate execution of a large order and compare slippage across strategies.")

    if df is None or model is None or scaler is None:
        st.error("Run NB 00–04 first."); st.stop()

    col1, col2, col3 = st.columns(3)
    total_shares = col1.number_input("Total shares to execute", 1000, 100_000, 10_000, step=1000)
    n_slices     = col2.slider("Number of child orders", 5, 20, 10)
    aggression   = col3.slider("ML aggression", 0.5, 4.0, 2.0, step=0.5)

    side = st.radio("Order side", ["buy", "sell"], horizontal=True)

    if st.button("🚀 Run Simulation", type="primary"):
        with st.spinner("Generating signals and simulating execution…"):
            signals, _ = generate_signals(df, model, scaler)
            df_sim = df.copy()
            df_sim["signal"] = signals

            # Use mid-session window
            s = len(df_sim) // 4
            e = 3 * len(df_sim) // 4
            df_exec = df_sim.iloc[s:e].reset_index(drop=True)
            sigs    = df_exec["signal"].values

            r_twap = twap_execute(df_exec, total_shares, n_slices, side)
            r_vwap = vwap_execute(df_exec, total_shares, n_slices, side)
            r_ml   = ml_directed_execute(df_exec, sigs, total_shares, n_slices, side, aggression)

        cmp = compare_strategies(r_twap, r_vwap, r_ml)
        st.subheader("Slippage Comparison")
        st.dataframe(cmp.style.background_gradient(cmap="RdYlGn", axis=0),
                     use_container_width=True)

        # Bar chart
        strategies = ["TWAP", "VWAP", "ML-Directed"]
        slippages  = [r_twap.slippage_bps, r_vwap.slippage_bps, r_ml.slippage_bps]
        colors     = ["steelblue", "darkorange", "seagreen"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(strategies, slippages, color=colors, width=0.5)
        for i, v in enumerate(slippages):
            axes[0].text(i, v * 1.02, f"{v:.2f} bps", ha="center", fontsize=9)
        axes[0].set_ylabel("Slippage (bps)")
        axes[0].set_title("Execution Slippage")

        improvements = [0,
                        r_twap.slippage_bps - r_vwap.slippage_bps,
                        r_twap.slippage_bps - r_ml.slippage_bps]
        axes[1].bar(strategies, improvements, color=colors, width=0.5)
        axes[1].axhline(0, color="black", lw=0.8)
        axes[1].set_ylabel("Saved vs TWAP (bps)")
        axes[1].set_title("Improvement over TWAP")

        plt.suptitle(f"Execution Simulation — {total_shares:,} shares", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("TWAP Slippage",   f"{r_twap.slippage_bps:.2f} bps")
        col2.metric("VWAP Slippage",   f"{r_vwap.slippage_bps:.2f} bps",
                    delta=f"{r_twap.slippage_bps - r_vwap.slippage_bps:.2f} bps saved")
        col3.metric("ML Slippage",     f"{r_ml.slippage_bps:.2f} bps",
                    delta=f"{r_twap.slippage_bps - r_ml.slippage_bps:.2f} bps saved")


# ─────────────────────────────────────────────────────────────────────────────
# ── Page 5: Model Results ────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🏆 Model Results":
    st.title("🏆 Model Results & Capstone Summary")

    results = load_results()
    mc_df   = load_mc_results()

    if results is not None:
        st.subheader("Classification Metrics")
        st.dataframe(results.style.background_gradient(cmap="RdYlGn", axis=0),
                     use_container_width=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        model_names = results.index.tolist()
        x  = np.arange(len(model_names))
        colors = ["steelblue", "darkorange", "seagreen"]

        axes[0].bar(x, results["Accuracy"], color=colors, width=0.5)
        axes[0].axhline(0.60, ls="--", color="red", label="Target 60%")
        axes[0].set_xticks(x); axes[0].set_xticklabels(model_names)
        axes[0].set_title("Accuracy"); axes[0].legend()

        axes[1].bar(x, results["F1-macro"], color=colors, width=0.5)
        axes[1].axhline(0.59, ls="--", color="red", label="Target F1=0.59")
        axes[1].set_xticks(x); axes[1].set_xticklabels(model_names)
        axes[1].set_title("F1-macro"); axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    else:
        st.warning("Run NB 05 to generate results.")

    if mc_df is not None:
        st.subheader("Monte Carlo Execution Results (50 runs)")
        summary = mc_df[["twap_slip", "vwap_slip", "ml_slip", "ml_vs_twap"]].describe().round(4)
        st.dataframe(summary, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg TWAP Slippage",   f"{mc_df['twap_slip'].mean():.2f} bps")
        col2.metric("Avg ML Slippage",     f"{mc_df['ml_slip'].mean():.2f} bps")
        pct_imp = mc_df["ml_vs_twap"].mean() / abs(mc_df["twap_slip"].mean()) * 100
        col3.metric("ML Improvement",      f"{pct_imp:.1f}%",
                    delta=f"{mc_df['ml_vs_twap'].mean():.2f} bps avg saved")

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.hist(mc_df["ml_vs_twap"], bins=20, color="seagreen", edgecolor="black", alpha=0.8)
        ax2.axvline(mc_df["ml_vs_twap"].mean(), color="red", ls="--",
                    label=f"Mean: {mc_df['ml_vs_twap'].mean():.2f} bps")
        ax2.set_title("ML vs TWAP: Slippage Improvement Distribution (50 Monte Carlo runs)")
        ax2.set_xlabel("Saved bps"); ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
    else:
        st.warning("Run NB 06 to generate execution results.")

    st.markdown("---")
    st.subheader("📚 Literature References")
    st.markdown("""
| Paper | Contribution |
|-------|-------------|
| Kercheval & Zhang (2015) | SVM on LOBSTER for LOB direction prediction |
| Kolm et al. (2021) | Deep learning on LOBSTER message streams |
| This Work | CNN-LSTM + execution-aware slippage simulation |
    """)
