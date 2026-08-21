"""
slippage_simulator.py
─────────────────────
Execution simulation layer: TWAP, VWAP, and ML-Directed order slicers.

Usage
-----
from utils.slippage_simulator import (
    twap_execute, vwap_execute, ml_directed_execute,
    compute_slippage, compare_strategies
)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    strategy:        str
    total_shares:    int
    avg_fill_price:  float
    arrival_price:   float
    slippage_bps:    float          # (avg_fill - arrival) / arrival × 10000
    num_slices:      int
    fills:           List[dict] = field(default_factory=list)

    @property
    def slippage_pct(self) -> float:
        return self.slippage_bps / 100


# ─────────────────────────────────────────────────────────────────────────────
# LOB liquidity extraction
# ─────────────────────────────────────────────────────────────────────────────

def _available_liquidity(row: pd.Series, side: str, n_levels: int = 10) -> List[tuple]:
    """
    Extract (price, size) pairs from the orderbook snapshot for `side`.

    Parameters
    ----------
    row     : a single row of the merged LOBSTER DataFrame
    side    : 'buy'  → sweep ask side
              'sell' → sweep bid side
    n_levels: number of price levels to consider

    Returns
    -------
    List of (price_usd, size) tuples sorted by priority
      (ascending price for buys, descending for sells)
    """
    prefix = "Ask" if side == "buy" else "Bid"
    levels = [(row[f"{prefix}P{i}"] / 10000, int(row[f"{prefix}S{i}"]))
              for i in range(1, n_levels + 1)
              if row[f"{prefix}S{i}"] > 0]
    return levels


def _fill_order(levels: List[tuple], shares_needed: int) -> tuple:
    """
    Walk the price levels and fill `shares_needed` shares.

    Returns
    -------
    avg_fill_price : float
    shares_filled  : int
    levels_used    : int
    """
    remaining     = shares_needed
    cost          = 0.0
    total_filled  = 0
    levels_used   = 0

    for price, avail in levels:
        if remaining <= 0:
            break
        fill = min(remaining, avail)
        cost         += fill * price
        total_filled += fill
        remaining    -= fill
        levels_used  += 1

    avg_price = cost / total_filled if total_filled > 0 else np.nan
    return avg_price, total_filled, levels_used


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1 – TWAP  (Time-Weighted Average Price)
# ─────────────────────────────────────────────────────────────────────────────

def twap_execute(df: pd.DataFrame,
                 total_shares: int,
                 n_slices: int = 10,
                 side: str = "buy") -> ExecutionResult:
    """
    Execute `total_shares` uniformly across `n_slices` equally-spaced
    snapshots (time-weighted, ignoring LOB signals).

    Parameters
    ----------
    df           : cleaned LOBSTER merged DataFrame with mid_price
    total_shares : total order size in shares
    n_slices     : how many child orders to split into
    side         : 'buy' or 'sell'

    Returns
    -------
    ExecutionResult
    """
    assert "mid_price" in df.columns
    arrival_price  = df["mid_price"].iloc[0]
    slice_size     = total_shares // n_slices
    remainder      = total_shares % n_slices

    # Evenly spaced indices
    indices = np.linspace(0, len(df) - 1, n_slices, dtype=int)

    fills          = []
    total_cost     = 0.0
    total_filled   = 0

    for k, idx in enumerate(indices):
        row    = df.iloc[idx]
        shares = slice_size + (remainder if k == n_slices - 1 else 0)
        levels = _available_liquidity(row, side)
        avg_p, filled, lvls = _fill_order(levels, shares)
        fills.append({"idx": idx, "shares": filled, "avg_price": avg_p, "levels_used": lvls})
        total_cost   += filled * avg_p
        total_filled += filled

    avg_fill = total_cost / total_filled if total_filled > 0 else np.nan
    slip_bps = (avg_fill - arrival_price) / arrival_price * 10_000
    if side == "sell":
        slip_bps = -slip_bps    # for sells, lower fill = worse

    return ExecutionResult("TWAP", total_shares, avg_fill, arrival_price,
                           slip_bps, n_slices, fills)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2 – VWAP  (Volume-Weighted Average Price proxy)
# ─────────────────────────────────────────────────────────────────────────────

def vwap_execute(df: pd.DataFrame,
                 total_shares: int,
                 n_slices: int = 10,
                 side: str = "buy") -> ExecutionResult:
    """
    VWAP-style execution: allocate more shares to snapshots with
    higher available LOB volume (proxy for historical VWAP profile).

    Parameters
    ----------
    Same as twap_execute.
    """
    assert "mid_price" in df.columns
    arrival_price = df["mid_price"].iloc[0]

    # Evenly spaced indices
    indices = np.linspace(0, len(df) - 1, n_slices, dtype=int)

    # Compute available volume at each slice
    prefix = "Ask" if side == "buy" else "Bid"
    vol_at_slice = np.array([
        df.iloc[i][[f"{prefix}S{j}" for j in range(1, 6)]].sum()
        for i in indices
    ], dtype=float)
    vol_weights = vol_at_slice / vol_at_slice.sum()
    slice_shares = (vol_weights * total_shares).astype(int)
    # Fix rounding
    slice_shares[-1] += total_shares - slice_shares.sum()

    fills        = []
    total_cost   = 0.0
    total_filled = 0

    for k, idx in enumerate(indices):
        row    = df.iloc[idx]
        shares = max(1, slice_shares[k])
        levels = _available_liquidity(row, side)
        avg_p, filled, lvls = _fill_order(levels, shares)
        fills.append({"idx": idx, "shares": filled, "avg_price": avg_p, "levels_used": lvls})
        total_cost   += filled * avg_p
        total_filled += filled

    avg_fill = total_cost / total_filled if total_filled > 0 else np.nan
    slip_bps = (avg_fill - arrival_price) / arrival_price * 10_000
    if side == "sell":
        slip_bps = -slip_bps

    return ExecutionResult("VWAP", total_shares, avg_fill, arrival_price,
                           slip_bps, n_slices, fills)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3 – ML-Directed  (signal-aware execution)
# ─────────────────────────────────────────────────────────────────────────────

def ml_directed_execute(df: pd.DataFrame,
                        signals: np.ndarray,
                        total_shares: int,
                        n_slices: int = 10,
                        side: str = "buy",
                        aggression: float = 2.0) -> ExecutionResult:
    """
    ML-directed order slicer.

    Logic (buy-side):
      - Signal +1 (UP):   price will rise → send LARGER child orders NOW
      - Signal  0 (FLAT): hold to baseline TWAP allocation
      - Signal -1 (DOWN): price will fall → send SMALLER child orders, wait

    Parameters
    ----------
    df          : cleaned LOBSTER DataFrame
    signals     : array of predicted direction labels aligned with df rows
    total_shares: total order size in shares
    n_slices    : number of child orders
    side        : 'buy' or 'sell'
    aggression  : multiplier controlling how strongly signals shift allocation
                  (1.0 = no adjustment, 2.0 = moderate, 3.0 = aggressive)

    Returns
    -------
    ExecutionResult
    """
    assert "mid_price" in df.columns
    assert len(signals) == len(df), "signals must align with df rows"

    arrival_price = df["mid_price"].iloc[0]
    indices       = np.linspace(0, len(df) - 1, n_slices, dtype=int)

    # Pull signal at each slice index
    slice_signals = signals[indices]  # values in {-1, 0, +1}

    # Convert signal to urgency weight
    # buy  → +1 signal (UP) = more urgent = bigger slice
    # sell → -1 signal (DOWN) = more urgent = bigger slice
    if side == "buy":
        urgency = 1.0 + aggression * slice_signals.astype(float)
    else:
        urgency = 1.0 - aggression * slice_signals.astype(float)

    urgency = np.clip(urgency, 0.1, None)            # floor at 10% of base
    weights = urgency / urgency.sum()
    slice_shares = (weights * total_shares).astype(int)
    slice_shares[-1] += total_shares - slice_shares.sum()

    fills        = []
    total_cost   = 0.0
    total_filled = 0

    for k, idx in enumerate(indices):
        row    = df.iloc[idx]
        shares = max(1, slice_shares[k])
        levels = _available_liquidity(row, side)
        avg_p, filled, lvls = _fill_order(levels, shares)
        fills.append({"idx": idx, "shares": filled, "avg_price": avg_p,
                      "signal": int(slice_signals[k]), "levels_used": lvls})
        total_cost   += filled * avg_p
        total_filled += filled

    avg_fill = total_cost / total_filled if total_filled > 0 else np.nan
    slip_bps = (avg_fill - arrival_price) / arrival_price * 10_000
    if side == "sell":
        slip_bps = -slip_bps

    return ExecutionResult("ML-Directed", total_shares, avg_fill, arrival_price,
                           slip_bps, n_slices, fills)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_slippage(result: ExecutionResult) -> dict:
    """Return a flat metrics dict for a single ExecutionResult."""
    return {
        "strategy":       result.strategy,
        "avg_fill_price": round(result.avg_fill_price, 4),
        "arrival_price":  round(result.arrival_price, 4),
        "slippage_bps":   round(result.slippage_bps, 3),
        "slippage_pct":   round(result.slippage_pct, 4),
        "num_slices":     result.num_slices,
        "total_shares":   result.total_shares,
    }


def compare_strategies(*results: ExecutionResult,
                        reference: str = "TWAP") -> pd.DataFrame:
    """
    Build a comparison table and compute slippage reduction vs reference strategy.

    Parameters
    ----------
    *results  : ExecutionResult instances (TWAP, VWAP, ML-Directed, …)
    reference : name of the benchmark strategy (default 'TWAP')

    Returns
    -------
    pd.DataFrame with slippage metrics and vs-reference delta column
    """
    rows = [compute_slippage(r) for r in results]
    df   = pd.DataFrame(rows).set_index("strategy")

    ref_slip = df.loc[reference, "slippage_bps"] if reference in df.index else 0
    df["vs_twap_bps"]     = ref_slip - df["slippage_bps"]
    df["vs_twap_pct_imp"] = df["vs_twap_bps"] / abs(ref_slip) * 100 if ref_slip != 0 else 0

    return df.round(4)
