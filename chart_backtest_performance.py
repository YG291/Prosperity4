#!/usr/bin/env python3
"""
Plot backtest performance from the CSV output of imc_local_backtester.py.

Usage:
    python chart_backtest_performance.py backtest_results.csv
    python chart_backtest_performance.py backtest_results.csv --save-prefix run1
    python chart_backtest_performance.py backtest_results.csv --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chart PnL, positions, and trade activity from a backtest CSV."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to backtest_results.csv",
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default="",
        help="If provided, saves PNG files like '<prefix>_pnl.png'.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open chart windows.",
    )
    return parser.parse_args()


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"timestamp", "cash", "mtm", "total_pnl", "trade_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s): {sorted(missing)}. "
            "Expected output from imc_local_backtester.py."
        )

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def save_or_show(fig: plt.Figure, filename: str | None, no_show: bool) -> None:
    if filename:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved {filename}")
    if not no_show:
        plt.show()
    else:
        plt.close(fig)


def plot_total_pnl(df: pd.DataFrame, save_prefix: str, no_show: bool) -> None:
    fig = plt.figure(figsize=(10, 5))
    plt.plot(df["timestamp"], df["total_pnl"])
    plt.xlabel("Timestamp")
    plt.ylabel("Total PnL")
    plt.title("Backtest Total PnL")

    filename = f"{save_prefix}_pnl.png" if save_prefix else None
    save_or_show(fig, filename, no_show)


def plot_cash(df: pd.DataFrame, save_prefix: str, no_show: bool) -> None:
    fig = plt.figure(figsize=(10, 5))
    plt.plot(df["timestamp"], df["cash"])
    plt.xlabel("Timestamp")
    plt.ylabel("Cash")
    plt.title("Backtest Cash")

    filename = f"{save_prefix}_cash.png" if save_prefix else None
    save_or_show(fig, filename, no_show)


def plot_mtm(df: pd.DataFrame, save_prefix: str, no_show: bool) -> None:
    fig = plt.figure(figsize=(10, 5))
    plt.plot(df["timestamp"], df["mtm"])
    plt.xlabel("Timestamp")
    plt.ylabel("Mark-to-Market")
    plt.title("Backtest MTM")

    filename = f"{save_prefix}_mtm.png" if save_prefix else None
    save_or_show(fig, filename, no_show)


def plot_trade_count(df: pd.DataFrame, save_prefix: str, no_show: bool) -> None:
    fig = plt.figure(figsize=(10, 5))
    plt.plot(df["timestamp"], df["trade_count"])
    plt.xlabel("Timestamp")
    plt.ylabel("Trades")
    plt.title("Trades per Step")

    filename = f"{save_prefix}_trades.png" if save_prefix else None
    save_or_show(fig, filename, no_show)


def plot_positions(df: pd.DataFrame, save_prefix: str, no_show: bool) -> None:
    position_cols = [col for col in df.columns if col.startswith("pos_")]
    if not position_cols:
        print("No position columns found, skipping position charts.")
        return

    for col in position_cols:
        fig = plt.figure(figsize=(10, 5))
        plt.plot(df["timestamp"], df[col])
        plt.xlabel("Timestamp")
        plt.ylabel("Position")
        plt.title(f"Position: {col.removeprefix('pos_')}")

        filename = (
            f"{save_prefix}_{col.lower()}.png" if save_prefix else None
        )
        save_or_show(fig, filename, no_show)


def print_summary(df: pd.DataFrame) -> None:
    last = df.iloc[-1]

    print("\n=== Summary ===")
    print(f"Final timestamp: {int(last['timestamp'])}")
    print(f"Final cash: {last['cash']:.2f}")
    print(f"Final MTM: {last['mtm']:.2f}")
    print(f"Final total PnL: {last['total_pnl']:.2f}")
    print(f"Total trades across steps: {int(df['trade_count'].sum())}")

    position_cols = [col for col in df.columns if col.startswith('pos_')]
    if position_cols:
        print("Final positions:")
        for col in position_cols:
            print(f"  {col.removeprefix('pos_')}: {int(last[col])}")


def main() -> None:
    args = parse_args()
    df = load_results(args.csv_path)

    print_summary(df)
    plot_total_pnl(df, args.save_prefix, args.no_show)
    plot_cash(df, args.save_prefix, args.no_show)
    plot_mtm(df, args.save_prefix, args.no_show)
    plot_trade_count(df, args.save_prefix, args.no_show)
    plot_positions(df, args.save_prefix, args.no_show)


if __name__ == "__main__":
    main()
