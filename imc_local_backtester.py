from __future__ import annotations

import argparse
import multiprocessing
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import pandas as pd

from datamodel import Listing, Observation, Order, OrderDepth, Trade, TradingState
from trader import Trader

Symbol = str
Product = str
Position = int

SUBMISSION = "SUBMISSION"
DEFAULT_DENOMINATION = "XIRECS"
DAY_PATTERN = re.compile(r"day_(-?\d+)")


@dataclass(frozen=True)
class SessionSpec:
    key: str
    day: int
    price_path: Path
    trade_path: Optional[Path]


@dataclass
class Snapshot:
    day: int
    timestamp: int
    order_depths: Dict[Symbol, OrderDepth]
    mid_prices: Dict[Symbol, float]


@dataclass
class SessionData:
    key: str
    day: int
    price_path: Path
    trade_path: Optional[Path]
    listings: Dict[Symbol, Listing]
    products: List[Product]
    snapshots: List[Snapshot]
    market_trades: Dict[int, Dict[Symbol, List[Trade]]]


class Strategy(Protocol):
    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        """Return (orders_by_symbol, conversions, traderData)."""


def _extract_day(path: Path) -> int:
    match = DAY_PATTERN.search(path.stem)
    return int(match.group(1)) if match else 0


def _session_key_from_price_path(path: Path) -> str:
    name = path.name
    if name.startswith("prices_"):
        return name[len("prices_") :]
    return path.stem


def _to_int_if_present(value: object) -> Optional[int]:
    if pd.isna(value):
        return None
    return int(round(float(value)))


def _normalize_party(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def _discover_sessions(
    price_source: str, trade_source: Optional[str]
) -> List[SessionSpec]:
    if trade_source is not None:
        price_path = Path(price_source)
        trade_path = Path(trade_source)
        if not price_path.exists():
            raise FileNotFoundError(f"Prices CSV not found: {price_path}")
        if not trade_path.exists():
            raise FileNotFoundError(f"Trades CSV not found: {trade_path}")
        return [
            SessionSpec(
                key=_session_key_from_price_path(price_path),
                day=_extract_day(price_path),
                price_path=price_path,
                trade_path=trade_path,
            )
        ]

    source_path = Path(price_source)
    if source_path.is_dir():
        price_files = sorted(source_path.glob("prices_*.csv"))
        if not price_files:
            raise FileNotFoundError(f"No prices_*.csv files found in {source_path}")

        trade_lookup = {
            _session_key_from_price_path(
                path.with_name(path.name.replace("trades_", "prices_", 1))
            ): path
            for path in source_path.glob("trades_*.csv")
        }

        sessions = [
            SessionSpec(
                key=_session_key_from_price_path(price_path),
                day=_extract_day(price_path),
                price_path=price_path,
                trade_path=trade_lookup.get(_session_key_from_price_path(price_path)),
            )
            for price_path in price_files
        ]
        return sorted(sessions, key=lambda session: (session.day, session.key))

    if not source_path.exists():
        raise FileNotFoundError(f"Price source not found: {source_path}")

    inferred_trade_path: Optional[Path] = None
    if source_path.name.startswith("prices_"):
        candidate = source_path.with_name(
            source_path.name.replace("prices_", "trades_", 1)
        )
        if candidate.exists():
            inferred_trade_path = candidate

    return [
        SessionSpec(
            key=_session_key_from_price_path(source_path),
            day=_extract_day(source_path),
            price_path=source_path,
            trade_path=inferred_trade_path,
        )
    ]


def _load_market_trades(
    csv_path: Optional[Path],
) -> Tuple[Dict[int, Dict[str, List[Trade]]], str]:
    if csv_path is None:
        return {}, DEFAULT_DENOMINATION

    df = pd.read_csv(csv_path, sep=";")
    if df.empty:
        return {}, DEFAULT_DENOMINATION

    required = {"timestamp", "symbol", "price", "quantity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in trades CSV {csv_path}: {sorted(missing)}"
        )

    for col in ("buyer", "seller", "currency"):
        if col not in df.columns:
            df[col] = None

    df = df.sort_values(["timestamp", "symbol", "price", "quantity"]).reset_index(
        drop=True
    )

    denomination = DEFAULT_DENOMINATION
    currencies = [str(value) for value in df["currency"].dropna().unique().tolist()]
    if currencies:
        denomination = currencies[0]

    trades_by_timestamp: Dict[int, Dict[str, List[Trade]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in df.itertuples(index=False):
        timestamp = int(row.timestamp)
        symbol = str(row.symbol)
        trades_by_timestamp[timestamp][symbol].append(
            Trade(
                symbol=symbol,
                price=int(round(float(row.price))),
                quantity=int(row.quantity),
                buyer=_normalize_party(row.buyer),
                seller=_normalize_party(row.seller),
                timestamp=timestamp,
            )
        )

    return {
        ts: dict(symbols) for ts, symbols in trades_by_timestamp.items()
    }, denomination


def _load_session(spec: SessionSpec) -> SessionData:
    df = pd.read_csv(spec.price_path, sep=";")

    required = {"timestamp", "product"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in prices CSV {spec.price_path}: {sorted(missing)}"
        )

    if "day" not in df.columns:
        df["day"] = spec.day

    day_values = sorted({int(day) for day in df["day"].dropna().unique().tolist()})
    if len(day_values) > 1:
        raise ValueError(
            f"Expected one day per prices CSV, found {day_values} in {spec.price_path}"
        )
    day = day_values[0] if day_values else spec.day

    df = df.sort_values(["timestamp", "product"]).reset_index(drop=True)
    products = sorted(df["product"].astype(str).unique().tolist())

    market_trades, denomination = _load_market_trades(spec.trade_path)
    listings = {
        product: Listing(symbol=product, product=product, denomination=denomination)
        for product in products
    }

    has_mid = "mid_price" in df.columns
    level_cols: List[
        Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]
    ] = []
    for level in (1, 2, 3):
        bid_price_col = (
            f"bid_price_{level}" if f"bid_price_{level}" in df.columns else None
        )
        bid_volume_col = (
            f"bid_volume_{level}" if f"bid_volume_{level}" in df.columns else None
        )
        ask_price_col = (
            f"ask_price_{level}" if f"ask_price_{level}" in df.columns else None
        )
        ask_volume_col = (
            f"ask_volume_{level}" if f"ask_volume_{level}" in df.columns else None
        )
        level_cols.append(
            (bid_price_col, bid_volume_col, ask_price_col, ask_volume_col)
        )

    snapshots: List[Snapshot] = []
    for timestamp, group in df.groupby("timestamp", sort=True):
        order_depths = {product: OrderDepth() for product in products}
        mid_prices: Dict[str, float] = {}

        for row in group.itertuples(index=False):
            product = str(row.product)
            depth = OrderDepth()

            for (
                bid_price_col,
                bid_volume_col,
                ask_price_col,
                ask_volume_col,
            ) in level_cols:
                if bid_price_col and bid_volume_col:
                    bid_price = _to_int_if_present(getattr(row, bid_price_col))
                    bid_volume = _to_int_if_present(getattr(row, bid_volume_col))
                    if (
                        bid_price is not None
                        and bid_volume is not None
                        and bid_volume != 0
                    ):
                        depth.buy_orders[bid_price] = bid_volume

                if ask_price_col and ask_volume_col:
                    ask_price = _to_int_if_present(getattr(row, ask_price_col))
                    ask_volume = _to_int_if_present(getattr(row, ask_volume_col))
                    if (
                        ask_price is not None
                        and ask_volume is not None
                        and ask_volume != 0
                    ):
                        depth.sell_orders[ask_price] = -abs(ask_volume)

            order_depths[product] = depth

            if has_mid:
                mid_value = getattr(row, "mid_price")
                if not pd.isna(mid_value):
                    mid_prices[product] = float(mid_value)

        snapshots.append(
            Snapshot(
                day=day,
                timestamp=int(timestamp),
                order_depths=order_depths,
                mid_prices=mid_prices,
            )
        )

    return SessionData(
        key=spec.key,
        day=day,
        price_path=spec.price_path,
        trade_path=spec.trade_path,
        listings=listings,
        products=products,
        snapshots=snapshots,
        market_trades=market_trades,
    )


def _merge_trade_dicts(*trade_dicts: Dict[str, List[Trade]]) -> Dict[str, List[Trade]]:
    merged: Dict[str, List[Trade]] = defaultdict(list)
    for trade_dict in trade_dicts:
        for symbol, trades in trade_dict.items():
            merged[symbol].extend(trades)
    return {symbol: trades for symbol, trades in merged.items()}


class Backtester:
    """Simulate Prosperity-style trading using official datamodel objects.

    Notes on the fill model:
    - Orders that cross the visible book trade immediately at the resting book price.
    - Any remainder rests for one iteration only.
    - On the next iteration, historical `market_trades` are used as a proxy for
      bots trading against our resting order. Because the tutorial data does not
      expose the full stream of counterparty orders, this passive-fill step is
      necessarily an approximation.
    """

    def __init__(
        self,
        price_source: str,
        trade_source: Optional[str],
        position_limits: Dict[str, int],
        mark_to_mid: bool = True,
    ) -> None:
        self.sessions = [
            _load_session(spec)
            for spec in _discover_sessions(price_source, trade_source)
        ]
        self.position_limits = position_limits
        self.mark_to_mid = mark_to_mid
        self._timestamp_stride = self._compute_timestamp_stride()

    def _compute_timestamp_stride(self) -> int:
        max_timestamp = 0
        for session in self.sessions:
            if session.snapshots:
                max_timestamp = max(max_timestamp, session.snapshots[-1].timestamp)
        if max_timestamp <= 0:
            return 1_000_000
        return ((max_timestamp // 100) + 2) * 100

    def run(self, strategy: Strategy) -> pd.DataFrame:
        rows: List[dict] = []
        cumulative_realized_pnl = 0.0
        global_step = 0

        for session_index, session in enumerate(self.sessions):
            session_rows, session_final_pnl = self._run_session(
                strategy=strategy,
                session=session,
                session_index=session_index,
                global_step_start=global_step,
                realized_pnl_offset=cumulative_realized_pnl,
            )
            rows.extend(session_rows)
            global_step += len(session.snapshots)
            cumulative_realized_pnl += session_final_pnl

        return pd.DataFrame(rows)

    def _run_session(
        self,
        strategy: Strategy,
        session: SessionData,
        session_index: int,
        global_step_start: int,
        realized_pnl_offset: float,
    ) -> Tuple[List[dict], float]:
        position: Dict[str, int] = {product: 0 for product in session.products}
        for product in self.position_limits:
            position.setdefault(product, 0)

        cash = 0.0
        trader_data = ""
        pending_own_trades: Dict[str, List[Trade]] = {}
        resting_orders: Dict[str, List[Order]] = {}
        rows: List[dict] = []

        for step_index, snapshot in enumerate(session.snapshots):
            market_trades = session.market_trades.get(snapshot.timestamp, {})

            passive_trades, passive_cash_delta = self._fill_resting_orders(
                timestamp=snapshot.timestamp,
                resting_orders=resting_orders,
                market_trades=market_trades,
            )
            cash += passive_cash_delta
            self._apply_trades_to_position(position, passive_trades)

            own_trades_for_state = _merge_trade_dicts(
                pending_own_trades, passive_trades
            )

            state = TradingState(
                traderData=trader_data,
                timestamp=snapshot.timestamp,
                listings=session.listings,
                order_depths=snapshot.order_depths,
                own_trades=own_trades_for_state,
                market_trades=market_trades,
                position=position.copy(),
                observations=Observation({}, {}),
            )

            raw = strategy.run(state)
            if not isinstance(raw, tuple) or len(raw) != 3:
                raise TypeError(
                    "Strategy.run must return (orders_by_symbol, conversions, traderData)."
                )

            raw_orders_by_symbol, _conversions, trader_data = raw
            normalized_orders = self._normalize_orders(raw_orders_by_symbol)
            accepted_orders = self._apply_position_limits(position, normalized_orders)

            aggressive_trades, new_resting_orders, aggressive_cash_delta = (
                self._execute_aggressive_orders(
                    timestamp=snapshot.timestamp,
                    orders_by_symbol=accepted_orders,
                    order_depths=snapshot.order_depths,
                )
            )
            resting_orders = new_resting_orders
            cash += aggressive_cash_delta
            self._apply_trades_to_position(position, aggressive_trades)

            executed_now = _merge_trade_dicts(passive_trades, aggressive_trades)
            mtm = self._mark_to_market(
                position, snapshot.mid_prices, snapshot.order_depths
            )

            row = {
                "session_index": session_index,
                "session_key": session.key,
                "day": session.day,
                "session_timestamp": snapshot.timestamp,
                "timestamp": session_index * self._timestamp_stride
                + snapshot.timestamp,
                "global_step": global_step_start + step_index,
                "cash": cash,
                "mtm": mtm,
                "total_pnl": cash + mtm,
                "cumulative_total_pnl": realized_pnl_offset + cash + mtm,
                "trade_count": sum(len(trades) for trades in executed_now.values()),
                "reported_own_trade_count": sum(
                    len(trades) for trades in own_trades_for_state.values()
                ),
                "aggressive_trade_count": sum(
                    len(trades) for trades in aggressive_trades.values()
                ),
                "passive_trade_count": sum(
                    len(trades) for trades in passive_trades.values()
                ),
                "market_trade_count": sum(
                    len(trades) for trades in market_trades.values()
                ),
            }

            per_product_buys: Dict[str, int] = {}
            per_product_sells: Dict[str, int] = {}
            for product, trades in executed_now.items():
                buy_count = sum(1 for trade in trades if trade.buyer == SUBMISSION)
                sell_count = sum(1 for trade in trades if trade.seller == SUBMISSION)
                per_product_buys[product] = buy_count
                per_product_sells[product] = sell_count

            for product in sorted(position):
                row[f"pos_{product}"] = position.get(product, 0)
                row[f"buy_count_{product}"] = per_product_buys.get(product, 0)
                row[f"sell_count_{product}"] = per_product_sells.get(product, 0)

            rows.append(row)
            pending_own_trades = aggressive_trades

        session_final_pnl = float(rows[-1]["total_pnl"]) if rows else 0.0
        return rows, session_final_pnl

    def _normalize_orders(
        self,
        orders_by_symbol: Optional[Dict[str, List[Order]]],
    ) -> Dict[str, List[Order]]:
        normalized: Dict[str, List[Order]] = defaultdict(list)
        if not orders_by_symbol:
            return {}

        for fallback_symbol, orders in orders_by_symbol.items():
            for order in orders or []:
                symbol = str(getattr(order, "symbol", fallback_symbol))
                normalized[symbol].append(
                    Order(
                        symbol=symbol,
                        price=int(order.price),
                        quantity=int(order.quantity),
                    )
                )

        return dict(normalized)

    def _apply_position_limits(
        self,
        position: Dict[str, int],
        orders_by_symbol: Dict[str, List[Order]],
    ) -> Dict[str, List[Order]]:
        accepted: Dict[str, List[Order]] = {}

        for symbol, orders in orders_by_symbol.items():
            pos = position.get(symbol, 0)
            limit = self.position_limits.get(symbol)

            if limit is None:
                accepted[symbol] = []
                continue

            total_buy = sum(max(order.quantity, 0) for order in orders)
            total_sell = sum(max(-order.quantity, 0) for order in orders)
            buy_ok = pos + total_buy <= limit
            sell_ok = pos - total_sell >= -limit

            filtered: List[Order] = []
            for order in orders:
                if order.quantity > 0 and buy_ok:
                    filtered.append(order)
                elif order.quantity < 0 and sell_ok:
                    filtered.append(order)

            accepted[symbol] = filtered

        return accepted

    def _execute_aggressive_orders(
        self,
        timestamp: int,
        orders_by_symbol: Dict[str, List[Order]],
        order_depths: Dict[str, OrderDepth],
    ) -> Tuple[Dict[str, List[Trade]], Dict[str, List[Order]], float]:
        own_trades: Dict[str, List[Trade]] = {}
        resting_orders: Dict[str, List[Order]] = {}
        cash_delta = 0.0

        for symbol, orders in orders_by_symbol.items():
            depth = order_depths.get(symbol, OrderDepth())
            trades: List[Trade] = []
            pending: List[Order] = []

            asks = sorted(depth.sell_orders.items(), key=lambda level: level[0])
            bids = sorted(
                depth.buy_orders.items(), key=lambda level: level[0], reverse=True
            )

            for order in orders:
                remaining = abs(order.quantity)

                if order.quantity > 0:
                    ask_index = 0
                    while remaining > 0 and ask_index < len(asks):
                        ask_price, ask_volume_signed = asks[ask_index]
                        if ask_price > order.price:
                            break

                        available = abs(ask_volume_signed)
                        if available <= 0:
                            ask_index += 1
                            continue

                        fill = min(remaining, available)
                        trades.append(
                            Trade(
                                symbol=symbol,
                                price=ask_price,
                                quantity=fill,
                                buyer=SUBMISSION,
                                seller="",
                                timestamp=timestamp,
                            )
                        )
                        cash_delta -= fill * ask_price
                        remaining -= fill
                        asks[ask_index] = (ask_price, -(available - fill))
                        if available == fill:
                            ask_index += 1

                    if remaining > 0:
                        pending.append(
                            Order(symbol=symbol, price=order.price, quantity=remaining)
                        )

                else:
                    bid_index = 0
                    while remaining > 0 and bid_index < len(bids):
                        bid_price, bid_volume = bids[bid_index]
                        if bid_price < order.price:
                            break

                        available = bid_volume
                        if available <= 0:
                            bid_index += 1
                            continue

                        fill = min(remaining, available)
                        trades.append(
                            Trade(
                                symbol=symbol,
                                price=bid_price,
                                quantity=fill,
                                buyer="",
                                seller=SUBMISSION,
                                timestamp=timestamp,
                            )
                        )
                        cash_delta += fill * bid_price
                        remaining -= fill
                        bids[bid_index] = (bid_price, available - fill)
                        if available == fill:
                            bid_index += 1

                    if remaining > 0:
                        pending.append(
                            Order(symbol=symbol, price=order.price, quantity=-remaining)
                        )

            own_trades[symbol] = trades
            if pending:
                resting_orders[symbol] = pending

        return own_trades, resting_orders, cash_delta

    def _fill_resting_orders(
        self,
        timestamp: int,
        resting_orders: Dict[str, List[Order]],
        market_trades: Dict[str, List[Trade]],
    ) -> Tuple[Dict[str, List[Trade]], float]:
        own_trades: Dict[str, List[Trade]] = {}
        cash_delta = 0.0

        for symbol, orders in resting_orders.items():
            if not orders:
                continue

            symbol_market_trades = market_trades.get(symbol, [])
            buy_pool = [[trade.price, trade.quantity] for trade in symbol_market_trades]
            sell_pool = [
                [trade.price, trade.quantity] for trade in symbol_market_trades
            ]
            buy_pool.sort(key=lambda entry: entry[0])
            sell_pool.sort(key=lambda entry: entry[0], reverse=True)

            trades: List[Trade] = []

            for order in orders:
                remaining = abs(order.quantity)

                if order.quantity > 0:
                    for entry in buy_pool:
                        if remaining <= 0:
                            break
                        traded_price, traded_qty = entry
                        if traded_price > order.price:
                            break
                        if traded_qty <= 0:
                            continue

                        fill = min(remaining, traded_qty)
                        trades.append(
                            Trade(
                                symbol=symbol,
                                price=order.price,
                                quantity=fill,
                                buyer=SUBMISSION,
                                seller="",
                                timestamp=timestamp,
                            )
                        )
                        cash_delta -= fill * order.price
                        remaining -= fill
                        entry[1] -= fill

                else:
                    for entry in sell_pool:
                        if remaining <= 0:
                            break
                        traded_price, traded_qty = entry
                        if traded_price < order.price:
                            break
                        if traded_qty <= 0:
                            continue

                        fill = min(remaining, traded_qty)
                        trades.append(
                            Trade(
                                symbol=symbol,
                                price=order.price,
                                quantity=fill,
                                buyer="",
                                seller=SUBMISSION,
                                timestamp=timestamp,
                            )
                        )
                        cash_delta += fill * order.price
                        remaining -= fill
                        entry[1] -= fill

            own_trades[symbol] = trades

        return own_trades, cash_delta

    def _apply_trades_to_position(
        self,
        position: Dict[str, int],
        trades_by_symbol: Dict[str, List[Trade]],
    ) -> None:
        for symbol, trades in trades_by_symbol.items():
            for trade in trades:
                if trade.buyer == SUBMISSION:
                    position[symbol] = position.get(symbol, 0) + trade.quantity
                elif trade.seller == SUBMISSION:
                    position[symbol] = position.get(symbol, 0) - trade.quantity

    def _mark_to_market(
        self,
        position: Dict[str, int],
        mid_prices: Dict[str, float],
        order_depths: Dict[str, OrderDepth],
    ) -> float:
        if not self.mark_to_mid:
            return 0.0

        mtm = 0.0
        for symbol, pos in position.items():
            if pos == 0:
                continue

            mid = mid_prices.get(symbol)
            if mid is None:
                depth = order_depths.get(symbol)
                if depth and depth.buy_orders and depth.sell_orders:
                    mid = (max(depth.buy_orders) + min(depth.sell_orders)) / 2
                elif depth and depth.buy_orders:
                    mid = float(max(depth.buy_orders))
                elif depth and depth.sell_orders:
                    mid = float(min(depth.sell_orders))

            if mid is not None:
                mtm += pos * mid

        return mtm


class NaiveFairValueStrategy:
    """Small example strategy that uses the visible top of book only."""

    def __init__(self, edge: int = 0, max_clip: int = 5):
        self.edge = edge
        self.max_clip = max_clip

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result: Dict[Symbol, List[Order]] = {}

        for symbol, depth in state.order_depths.items():
            orders: List[Order] = []
            best_bid = max(depth.buy_orders) if depth.buy_orders else None
            best_ask = min(depth.sell_orders) if depth.sell_orders else None

            if best_bid is None and best_ask is None:
                result[symbol] = orders
                continue

            if best_bid is None:
                fair_value = best_ask
            elif best_ask is None:
                fair_value = best_bid
            else:
                fair_value = int(round((best_bid + best_ask) / 2))

            if best_ask is not None:
                ask_volume = abs(depth.sell_orders[best_ask])
                if best_ask < fair_value - self.edge:
                    orders.append(
                        Order(symbol, best_ask, min(self.max_clip, ask_volume))
                    )

            if best_bid is not None:
                bid_volume = depth.buy_orders[best_bid]
                if best_bid > fair_value + self.edge:
                    orders.append(
                        Order(symbol, best_bid, -min(self.max_clip, bid_volume))
                    )

            result[symbol] = orders

        return result, 0, state.traderData


def _run_backtest_worker(args: Tuple) -> pd.DataFrame:
    price_source, trade_source, limits, strategy_cls, strategy_kwargs, mark_to_mid = (
        args
    )
    engine = Backtester(
        price_source=price_source,
        trade_source=trade_source,
        position_limits=limits,
        mark_to_mid=mark_to_mid,
    )
    strategy = strategy_cls(**strategy_kwargs)
    return engine.run(strategy)


def run_parallel_backtests(
    price_source: str,
    trade_source: Optional[str],
    limits: Dict[str, int],
    strategy_cls,
    param_grid: List[Dict[str, Any]],
    mark_to_mid: bool = True,
    workers: Optional[int] = None,
) -> List[pd.DataFrame]:
    n_workers = workers or multiprocessing.cpu_count()
    args_list = [
        (price_source, trade_source, limits, strategy_cls, kwargs, mark_to_mid)
        for kwargs in param_grid
    ]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        return list(executor.map(_run_backtest_worker, args_list))


def parse_limits(text: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, value = chunk.split("=", 1)
        out[key.strip()] = int(value.strip())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Local backtester for Prosperity tutorial CSVs. "
            "Pass either a prices CSV plus optional trades CSV, or a directory "
            "such as TUTORIAL_ROUND_1 to auto-discover all day files."
        )
    )
    parser.add_argument(
        "price_source",
        help="Prices CSV or directory containing prices_*.csv / trades_*.csv files",
    )
    parser.add_argument(
        "trade_source",
        nargs="?",
        help="Trades CSV. Omit when the first argument is a round directory or when the matching trades file can be inferred.",
    )
    parser.add_argument(
        "--limits",
        required=True,
        help="Position limits, e.g. EMERALDS=20,TOMATOES=20",
    )
    parser.add_argument(
        "--out",
        default="backtest_results.csv",
        help="Where to save per-timestamp results",
    )
    parser.add_argument(
        "--no-mark-to-mid",
        action="store_true",
        help="Disable mark-to-mid inventory valuation in the output PnL.",
    )

    args = parser.parse_args()

    engine = Backtester(
        price_source=args.price_source,
        trade_source=args.trade_source,
        position_limits=parse_limits(args.limits),
        mark_to_mid=not args.no_mark_to_mid,
    )
    results = engine.run(Trader())
    results.to_csv(args.out, index=False)

    if results.empty:
        print("Backtest produced no rows.")
        print(f"Saved results to: {args.out}")
        return

    final_by_session = (
        results.sort_values(["session_index", "session_timestamp"])
        .groupby(["session_index", "session_key", "day"], as_index=False)
        .tail(1)
        .sort_values("session_index")
    )

    print(results.tail(10).to_string(index=False))
    print("\nFinal session PnL:")
    for row in final_by_session.itertuples(index=False):
        print(
            f"  session={row.session_index} key={row.session_key} day={row.day} "
            f"final_total_pnl={float(row.total_pnl):.2f}"
        )

    total_across_sessions = float(final_by_session["total_pnl"].sum())
    print(f"\nCombined final PnL across sessions: {total_across_sessions:.2f}")
    print(f"Saved results to: {args.out}")


if __name__ == "__main__":
    main()
