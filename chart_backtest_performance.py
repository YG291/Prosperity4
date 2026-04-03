#!/usr/bin/env python3
"""
Plot backtest performance from the CSV output of imc_local_backtester.py.

Usage:
    python chart_backtest_performance.py backtest_results.csv
    python chart_backtest_performance.py backtest_results.csv --save-prefix run1
    python chart_backtest_performance.py backtest_results.csv --no-show
    python chart_backtest_performance.py backtest_results.csv --candlestick prices.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chart PnL, positions, and trade activity from a backtest CSV."
    )
    parser.add_argument("csv_path", type=Path, help="Path to backtest_results.csv")
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
    parser.add_argument(
        "--candlestick",
        type=Path,
        metavar="PRICES_CSV",
        default=None,
        help="Path to the semicolon-delimited prices CSV; enables candlestick charts per product.",
    )
    parser.add_argument(
        "--candle-period",
        type=int,
        default=1000,
        help="Timestamp units per candlestick (default: 1000).",
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
    return df.sort_values("timestamp").reset_index(drop=True)


def _save_or_show(fig: plt.Figure, filename: str | None, no_show: bool) -> None:
    if filename:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved {filename}")
    if not no_show:
        plt.show()
    else:
        plt.close(fig)


def _scatter_trades(
    ax: plt.Axes,
    df: pd.DataFrame,
    ts_col: str,
    y_col: str,
    buy_col: str,
    sell_col: str,
) -> list[mpatches.Patch]:
    handles = []
    for col, marker, color, label in (
        (buy_col, "^", "green", "Buy"),
        (sell_col, "v", "red", "Sell"),
    ):
        if col not in df.columns:
            continue
        rows = df[df[col] > 0]
        if rows.empty:
            continue
        ax.scatter(
            rows[ts_col], rows[y_col], marker=marker, color=color, s=60, zorder=3
        )
        handles.append(mpatches.Patch(color=color, label=label))
    return handles


def plot_total_pnl(df: pd.DataFrame, save_prefix: str, no_show: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["timestamp"], df["total_pnl"], label="Total PnL", zorder=2)

    handles = _scatter_trades(
        ax, df, "timestamp", "total_pnl", "buy_count", "sell_count"
    )
    if handles:
        ax.legend(handles=handles)

    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Total PnL")
    ax.set_title("Backtest Total PnL")
    _save_or_show(fig, f"{save_prefix}_pnl.png" if save_prefix else None, no_show)


def plot_cash(df: pd.DataFrame, save_prefix: str, no_show: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["timestamp"], df["cash"])
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Cash")
    ax.set_title("Backtest Cash")
    _save_or_show(fig, f"{save_prefix}_cash.png" if save_prefix else None, no_show)


def plot_mtm(df: pd.DataFrame, save_prefix: str, no_show: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["timestamp"], df["mtm"])
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Mark-to-Market")
    ax.set_title("Backtest MTM")
    _save_or_show(fig, f"{save_prefix}_mtm.png" if save_prefix else None, no_show)


def plot_trade_count(df: pd.DataFrame, save_prefix: str, no_show: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["timestamp"], df["trade_count"])
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Trades")
    ax.set_title("Trades per Step")
    _save_or_show(fig, f"{save_prefix}_trades.png" if save_prefix else None, no_show)


def plot_positions(df: pd.DataFrame, save_prefix: str, no_show: bool) -> None:
    position_cols = [col for col in df.columns if col.startswith("pos_")]
    if not position_cols:
        print("No position columns found, skipping position charts.")
        return

    for col in position_cols:
        product = col.removeprefix("pos_")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["timestamp"], df[col])
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Position")
        ax.set_title(f"Position: {product}")
        filename = f"{save_prefix}_{col.lower()}.png" if save_prefix else None
        _save_or_show(fig, filename, no_show)


def _draw_candles(ax: plt.Axes, ohlc: pd.DataFrame, ts_col: str, width: float) -> None:
    for _, row in ohlc.iterrows():
        ts = row[ts_col]
        color = "green" if row["close"] >= row["open"] else "red"
        body_bottom = min(row["open"], row["close"])
        body_height = abs(row["close"] - row["open"]) or 0.01

        ax.add_patch(
            mpatches.Rectangle(
                (ts - width / 2, body_bottom),
                width,
                body_height,
                color=color,
                zorder=2,
            )
        )
        ax.plot([ts, ts], [row["low"], body_bottom], color=color, linewidth=1, zorder=1)
        ax.plot(
            [ts, ts],
            [body_bottom + body_height, row["high"]],
            color=color,
            linewidth=1,
            zorder=1,
        )


def plot_candlestick(
    prices_csv: Path,
    df_results: pd.DataFrame,
    candle_period: int,
    save_prefix: str,
    no_show: bool,
) -> None:
    prices = pd.read_csv(prices_csv, sep=";")

    if "mid_price" not in prices.columns:
        print("No mid_price column in prices CSV — skipping candlestick charts.")
        return

    prices = prices.sort_values(["product", "timestamp"])

    for product, prod_df in prices.groupby("product"):
        prod_df = prod_df[["timestamp", "mid_price"]].dropna().reset_index(drop=True)
        if prod_df.empty:
            continue

        t0 = prod_df["timestamp"].iloc[0]
        prod_df["bucket"] = (prod_df["timestamp"] - t0) // candle_period

        ohlc = (
            prod_df.groupby("bucket")["mid_price"]
            .agg(open="first", high="max", low="min", close="last")
            .reset_index()
        )
        bucket_start = prod_df.groupby("bucket")["timestamp"].first().reset_index()
        ohlc = ohlc.merge(bucket_start, on="bucket")

        fig, ax = plt.subplots(figsize=(12, 6))
        _draw_candles(ax, ohlc, ts_col="timestamp", width=candle_period * 0.7)

        # Per-product columns take priority; fall back to aggregate for older CSVs.
        buy_col = (
            f"buy_count_{product}"
            if f"buy_count_{product}" in df_results.columns
            else "buy_count"
        )
        sell_col = (
            f"sell_count_{product}"
            if f"sell_count_{product}" in df_results.columns
            else "sell_count"
        )

        price_at = prod_df.set_index("timestamp")["mid_price"]
        handles = []
        for col, marker, color, label in (
            (buy_col, "^", "lime", "Buy"),
            (sell_col, "v", "red", "Sell"),
        ):
            if col not in df_results.columns:
                continue
            trade_ts = df_results[df_results[col] > 0]["timestamp"]
            matched = trade_ts[trade_ts.isin(price_at.index)]
            if matched.empty:
                continue
            ax.scatter(
                matched,
                price_at.loc[matched].values,
                marker=marker,
                color=color,
                s=70,
                zorder=4,
                edgecolors="black",
                linewidths=0.5,
            )
            handles.append(mpatches.Patch(color=color, label=label))

        if handles:
            ax.legend(handles=handles)

        ax.autoscale_view()
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Price")
        ax.set_title(f"Candlestick: {product}  (period={candle_period})")

        filename = (
            f"{save_prefix}_candle_{str(product).lower()}.png" if save_prefix else None
        )
        _save_or_show(fig, filename, no_show)


def print_summary(df: pd.DataFrame) -> None:
    last = df.iloc[-1]
    print("\n=== Summary ===")
    print(f"Final timestamp:  {int(last['timestamp'])}")
    print(f"Final cash:       {last['cash']:.2f}")
    print(f"Final MTM:        {last['mtm']:.2f}")
    print(f"Final total PnL:  {last['total_pnl']:.2f}")
    print(f"Total trades:     {int(df['trade_count'].sum())}")

    position_cols = [col for col in df.columns if col.startswith("pos_")]
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

    if args.candlestick:
        plot_candlestick(
            args.candlestick,
            df,
            args.candle_period,
            args.save_prefix,
            args.no_show,
        )


if __name__ == "__main__":
    main()
