"""
lobster_loader.py
─────────────────
Utilities for loading and validating LOBSTER message + orderbook CSV files.

LOBSTER file format reference:
  message_10.csv  → [Time, Type, OrderID, Size, Price, Direction]
  orderbook_10.csv → [AskP1, AskS1, BidP1, BidS1, ..., AskP10, AskS10, BidP10, BidS10]
                      (40 columns, 10 price levels each side)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path


# ── Message event type codes ──────────────────────────────────────────────────
MSG_TYPE = {
    1: "New limit order",
    2: "Cancellation (partial)",
    3: "Deletion (full cancel)",
    4: "Execution (visible)",
    5: "Execution (hidden)",
    6: "Cross trade",
    7: "Trading halt / resume",
}

DUMMY_PRICE  = -9.99e9   # LOBSTER sentinel for inactive levels
N_LEVELS     = 10


def _build_ob_columns(n_levels: int = N_LEVELS) -> list:
    """Generate the 40-column orderbook header for n_levels."""
    cols = []
    for i in range(1, n_levels + 1):
        cols += [f"AskP{i}", f"AskS{i}", f"BidP{i}", f"BidS{i}"]
    return cols


def load_message(path: str) -> pd.DataFrame:
    """
    Load a LOBSTER message file.

    Parameters
    ----------
    path : str
        Path to *_message_10.csv

    Returns
    -------
    pd.DataFrame with columns:
        Time (float64 nanoseconds since midnight),
        Type (int), OrderID (int),
        Size (int), Price (int, ×10000),
        Direction (int: +1 buy / -1 sell)
    """
    cols = ["Time", "Type", "OrderID", "Size", "Price", "Direction"]
    df = pd.read_csv(path, header=None, names=cols)

    # Validate
    assert df.shape[1] == 6, f"Expected 6 columns, got {df.shape[1]}"
    df["Time"]      = df["Time"].astype(float)
    df["Type"]      = df["Type"].astype(int)
    df["OrderID"]   = df["OrderID"].astype(int)
    df["Size"]      = df["Size"].astype(int)
    df["Price"]     = df["Price"].astype(int)
    df["Direction"] = df["Direction"].astype(int)

    print(f"[load_message] {len(df):,} events loaded from {Path(path).name}")
    print(f"  Event types: { {k: MSG_TYPE[k] for k in sorted(df['Type'].unique()) if k in MSG_TYPE} }")
    return df


def load_orderbook(path: str, n_levels: int = N_LEVELS) -> pd.DataFrame:
    """
    Load a LOBSTER orderbook file.

    Parameters
    ----------
    path : str
        Path to *_orderbook_10.csv
    n_levels : int
        Number of price levels (default 10)

    Returns
    -------
    pd.DataFrame with 40 columns (Ask/Bid Price/Size × 10 levels).
    Prices are in raw LOBSTER units (×10000); divide by 10000 for USD.
    """
    cols = _build_ob_columns(n_levels)
    df = pd.read_csv(path, header=None, names=cols)

    assert df.shape[1] == 4 * n_levels, \
        f"Expected {4*n_levels} columns, got {df.shape[1]}"

    print(f"[load_orderbook] {len(df):,} snapshots loaded from {Path(path).name}")
    return df


def merge_files(msg_df: pd.DataFrame, book_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align message events with orderbook snapshots.

    LOBSTER guarantees 1-to-1 row correspondence (each message event
    produces exactly one orderbook snapshot), so we simply concatenate
    column-wise after verifying row counts match.

    Parameters
    ----------
    msg_df  : DataFrame from load_message()
    book_df : DataFrame from load_orderbook()

    Returns
    -------
    Combined DataFrame with all 46 columns.
    """
    assert len(msg_df) == len(book_df), (
        f"Row mismatch: message={len(msg_df)}, orderbook={len(book_df)}. "
        "Ensure both files are from the same LOBSTER export."
    )
    df = pd.concat([msg_df.reset_index(drop=True),
                    book_df.reset_index(drop=True)], axis=1)
    print(f"[merge_files] Merged DataFrame shape: {df.shape}")
    return df


def clean_lobster(df: pd.DataFrame, remove_halts: bool = True) -> pd.DataFrame:
    """
    Remove LOBSTER artefacts:
      1. Dummy levels (BidP = DUMMY_PRICE sentinel)
      2. Zero-size best bid/ask
      3. Trading halt events (Type=7) if remove_halts=True
      4. Crossed books (BidP1 >= AskP1)

    Parameters
    ----------
    df : merged DataFrame from merge_files()
    remove_halts : bool
        Drop Type=7 rows (trading halt / resume events)

    Returns
    -------
    Cleaned DataFrame.
    """
    n_raw = len(df)

    # 1. Remove halt events
    if remove_halts:
        df = df[df["Type"] != 7]

    # 2. Remove rows where best bid/ask size is zero or dummy
    df = df[(df["BidS1"] > 0) & (df["AskS1"] > 0)]
    df = df[(df["BidP1"] > DUMMY_PRICE) & (df["AskP1"] > DUMMY_PRICE)]

    # 3. Remove crossed books (data integrity)
    df = df[df["AskP1"] > df["BidP1"]]

    n_clean = len(df)
    print(f"[clean_lobster] Removed {n_raw - n_clean:,} rows "
          f"({(n_raw-n_clean)/n_raw*100:.1f}%). Clean rows: {n_clean:,}")
    return df.reset_index(drop=True)


def describe_dataset(df: pd.DataFrame) -> None:
    """Print a concise summary of the merged LOBSTER dataset."""
    mid = (df["AskP1"] + df["BidP1"]) / 2 / 10000
    spread_bp = (df["AskP1"] - df["BidP1"]) / (df["AskP1"] + df["BidP1"]) * 20000

    t_start = df["Time"].iloc[0]  / 1e9
    t_end   = df["Time"].iloc[-1] / 1e9
    duration_min = (t_end - t_start) / 60

    print("=" * 55)
    print("  LOBSTER Dataset Summary")
    print("=" * 55)
    print(f"  Total events     : {len(df):,}")
    print(f"  Time span        : {duration_min:.1f} minutes")
    print(f"  Mid price range  : ${mid.min():.2f} – ${mid.max():.2f}")
    print(f"  Mean spread      : {spread_bp.mean():.2f} bps")
    print(f"  Event type dist  :")
    for t, cnt in df["Type"].value_counts().sort_index().items():
        label = MSG_TYPE.get(t, "Unknown")
        print(f"    Type {t} ({label}): {cnt:,}")
    print("=" * 55)
