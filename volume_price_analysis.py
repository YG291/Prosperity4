#!/usr/bin/env python3
"""
volume_price_analysis.py

Determines whether large price movements in TUTORIAL_ROUND_1 are preceded
by large increases in volume (both order-book depth and traded volume).

Usage:
    python volume_price_analysis.py
    python volume_price_analysis.py --lookback 3 --move-z 1.5 --vol-z 1.0
    python volume_price_analysis.py --no-plot
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROUND_DIR = Path("TUTORIAL_ROUND_1")

# Defaults
DEFAULT_LOOKBACK = 5       # how many price steps to look back for a volume spike
DEFAULT_MOVE_Z = 2.0       # z-score threshold to call a price change "large"
DEFAULT_VOL_Z = 1.5        # z-score threshold to call volume "elevated"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_prices() -> pd.DataFrame:
    frames = []
    for path in sorted(ROUND_DIR.glob("prices_*.csv")):
        df = pd.read_csv(path, sep=";")
        frames.append(df)
    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)

    # Total order-book depth: sum of all bid/ask volume columns
    vol_cols = [c for c in prices.columns if "volume" in c]
    prices["book_volume"] = prices[vol_cols].abs().sum(axis=1)

    return prices


def load_trades() -> pd.DataFrame:
    frames = []
    for path in sorted(ROUND_DIR.glob("trades_*.csv")):
        # Infer day from filename (e.g. "trades_round_0_day_-1.csv" -> -1)
        stem = path.stem  # e.g. "trades_round_0_day_-1"
        day = int(stem.split("day_")[-1])
        df = pd.read_csv(path, sep=";")
        df["day"] = day
        frames.append(df)
    trades = pd.concat(frames, ignore_index=True)

    # Aggregate traded volume per product per (day, timestamp)
    traded = (
        trades.groupby(["day", "timestamp", "symbol"])["quantity"]
        .sum()
        .reset_index()
        .rename(columns={"symbol": "product", "quantity": "traded_volume"})
    )
    return traded


# ---------------------------------------------------------------------------
# Per-product analysis
# ---------------------------------------------------------------------------

def analyse_product(
    df: pd.DataFrame,
    lookback: int,
    move_z_thresh: float,
    vol_z_thresh: float,
) -> dict:
    """
    df has columns: step, mid_price, book_volume, traded_volume (sorted by step).
    Returns a summary dict and the annotated dataframe.
    """
    df = df.copy().reset_index(drop=True)

    # Price change magnitude between consecutive steps
    df["price_change"] = df["mid_price"].diff().abs()

    # Rolling mean traded volume over the preceding `lookback` steps (not including current)
    df["pre_traded_vol"] = (
        df["traded_volume"].shift(1).rolling(lookback, min_periods=1).mean()
    )
    df["pre_book_vol"] = (
        df["book_volume"].shift(1).rolling(lookback, min_periods=1).mean()
    )

    # Drop the first row (no prior price change)
    df = df.dropna(subset=["price_change"]).reset_index(drop=True)

    # Z-scores (use ddof=0 to avoid NaN on tiny samples)
    pc_mean, pc_std = df["price_change"].mean(), df["price_change"].std(ddof=0)
    tv_mean, tv_std = df["pre_traded_vol"].mean(), df["pre_traded_vol"].std(ddof=0)
    bv_mean, bv_std = df["pre_book_vol"].mean(), df["pre_book_vol"].std(ddof=0)

    df["price_change_z"] = (df["price_change"] - pc_mean) / (pc_std or 1)
    df["pre_traded_vol_z"] = (df["pre_traded_vol"] - tv_mean) / (tv_std or 1)
    df["pre_book_vol_z"] = (df["pre_book_vol"] - bv_mean) / (bv_std or 1)

    # Flag events
    df["large_move"] = df["price_change_z"] >= move_z_thresh
    df["traded_spike"] = df["pre_traded_vol_z"] >= vol_z_thresh
    df["book_spike"] = df["pre_book_vol_z"] >= vol_z_thresh

    n_large = df["large_move"].sum()
    n_preceded_traded = (df["large_move"] & df["traded_spike"]).sum()
    n_preceded_book = (df["large_move"] & df["book_spike"]).sum()
    n_traded_spikes = df["traded_spike"].sum()
    n_book_spikes = df["book_spike"].sum()

    # Pearson correlation between pre-move volume and subsequent price change
    corr_traded = df["pre_traded_vol"].corr(df["price_change"])
    corr_book = df["pre_book_vol"].corr(df["price_change"])

    summary = {
        "large_moves": int(n_large),
        "preceded_by_traded_spike": int(n_preceded_traded),
        "preceded_by_book_spike": int(n_preceded_book),
        "recall_traded": n_preceded_traded / n_large if n_large else float("nan"),
        "recall_book": n_preceded_book / n_large if n_large else float("nan"),
        "precision_traded": n_preceded_traded / n_traded_spikes if n_traded_spikes else float("nan"),
        "precision_book": n_preceded_book / n_book_spikes if n_book_spikes else float("nan"),
        "corr_traded_vol": corr_traded,
        "corr_book_vol": corr_book,
    }
    return summary, df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_product(product: str, df: pd.DataFrame) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"{product}: Volume vs Price Movement", fontsize=13)

    # -- Price
    ax1.plot(df["step"], df["mid_price"], color="steelblue", linewidth=1, label="Mid price")
    large_moves = df[df["large_move"]]
    ax1.scatter(
        large_moves["step"],
        large_moves["mid_price"],
        color="red",
        s=40,
        zorder=3,
        label=f"Large move (z≥{DEFAULT_MOVE_Z})",
    )
    ax1.set_ylabel("Mid Price")
    ax1.legend(fontsize=8)

    # -- Traded volume
    ax2.bar(df["step"], df["traded_volume"], color="slategray", width=100, label="Traded volume")
    spike_rows = df[df["traded_spike"]]
    ax2.bar(
        spike_rows["step"],
        spike_rows["traded_volume"],
        color="orange",
        width=100,
        label=f"Volume spike (pre-move z≥{DEFAULT_VOL_Z})",
    )
    ax2.set_ylabel("Traded Volume")
    ax2.legend(fontsize=8)

    # -- Price change magnitude
    ax3.bar(df["step"], df["price_change"], color="mediumpurple", width=100, label="|Price change|")
    ax3.set_ylabel("|Price Change|")
    ax3.set_xlabel("Step")
    ax3.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Volume-precedes-price-move analysis.")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK,
                        help=f"Steps to look back for volume spike (default: {DEFAULT_LOOKBACK})")
    parser.add_argument("--move-z", type=float, default=DEFAULT_MOVE_Z,
                        help=f"Z-score threshold for a large price move (default: {DEFAULT_MOVE_Z})")
    parser.add_argument("--vol-z", type=float, default=DEFAULT_VOL_Z,
                        help=f"Z-score threshold for a volume spike (default: {DEFAULT_VOL_Z})")
    parser.add_argument("--no-plot", action="store_true", help="Skip charts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prices = load_prices()
    trades = load_trades()

    # Merge traded volume into prices; missing = 0 (no trades that step)
    merged = prices.merge(trades, on=["day", "timestamp", "product"], how="left")
    merged["traded_volume"] = merged["traded_volume"].fillna(0)

    # Create a global sequential step index per product (across days)
    merged = merged.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    merged["step"] = merged.groupby("product").cumcount()

    print(f"\nSettings: lookback={args.lookback}, move_z={args.move_z}, vol_z={args.vol_z}\n")
    print(f"{'Product':<12} {'LargeMoves':>10} {'Preceded(T)':>12} {'Recall(T)':>10} "
          f"{'Prec(T)':>8} {'Corr(T)':>8} {'Recall(B)':>10} {'Corr(B)':>8}")
    print("-" * 90)

    for product, group in merged.groupby("product"):
        summary, annotated = analyse_product(
            group[["step", "mid_price", "book_volume", "traded_volume"]],
            lookback=args.lookback,
            move_z_thresh=args.move_z,
            vol_z_thresh=args.vol_z,
        )

        print(
            f"{product:<12}"
            f"{summary['large_moves']:>10}"
            f"{summary['preceded_by_traded_spike']:>12}"
            f"{summary['recall_traded']:>10.1%}"
            f"{summary['precision_traded']:>8.1%}"
            f"{summary['corr_traded_vol']:>8.3f}"
            f"{summary['recall_book']:>10.1%}"
            f"{summary['corr_book_vol']:>8.3f}"
        )

        if not args.no_plot:
            plot_product(product, annotated)

    print("\nColumns: Preceded(T)=preceded by traded-volume spike, Recall=fraction of large")
    print("         moves that were preceded, Prec=fraction of spikes that led to large moves,")
    print("         Corr=Pearson correlation of pre-move volume with |price change|.")
    print("         (B) = order-book depth volume variant.")


if __name__ == "__main__":
    main()
