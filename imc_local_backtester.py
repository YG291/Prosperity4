from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple
from trader import Trader
import pandas as pd

Symbol = str
Product = str
Position = int


@dataclass
class Order:
    symbol: Symbol
    price: int
    quantity: int  # positive = buy, negative = sell


@dataclass
class Trade:
    symbol: Symbol
    price: int
    quantity: int
    buyer: str
    seller: str
    timestamp: int


@dataclass
class OrderDepth:
    buy_orders: Dict[int, int] = field(default_factory=dict)
    sell_orders: Dict[int, int] = field(default_factory=dict)  # negative volumes


@dataclass
class TradingState:
    traderData: str
    timestamp: int
    order_depths: Dict[Symbol, OrderDepth]
    own_trades: Dict[Symbol, List[Trade]]
    market_trades: Dict[Symbol, List[Trade]]
    position: Dict[Product, Position]
    observations: Dict[str, object] = field(default_factory=dict)


class Strategy(Protocol):
    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        """Return (orders_by_symbol, conversions, traderData)."""


def _to_int_if_present(value: object) -> Optional[int]:
    if pd.isna(value):
        return None
    return int(value)


class PriceBookLoader:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path, sep=";")
        self._validate_columns()
        self.df = self.df.sort_values(["timestamp", "product"]).reset_index(drop=True)

    def _validate_columns(self) -> None:
        required = {"timestamp", "product"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in prices CSV: {sorted(missing)}"
            )

    def get_products(self) -> List[str]:
        return sorted(self.df["product"].astype(str).unique().tolist())

    def iter_snapshots(self):
        for timestamp, group in self.df.groupby("timestamp", sort=True):
            order_depths: Dict[Symbol, OrderDepth] = {}
            mids: Dict[str, float] = {}

            for _, row in group.iterrows():
                product = str(row["product"])
                depth = OrderDepth()

                for level in (1, 2, 3):
                    bp = _to_int_if_present(row.get(f"bid_price_{level}"))
                    bv = _to_int_if_present(row.get(f"bid_volume_{level}"))
                    ap = _to_int_if_present(row.get(f"ask_price_{level}"))
                    av = _to_int_if_present(row.get(f"ask_volume_{level}"))

                    if bp is not None and bv is not None and bv != 0:
                        depth.buy_orders[bp] = bv
                    if ap is not None and av is not None and av != 0:
                        depth.sell_orders[ap] = -abs(av)

                order_depths[product] = depth
                if "mid_price" in row and not pd.isna(row["mid_price"]):
                    mids[product] = float(row["mid_price"])

            yield int(timestamp), order_depths, mids


class MarketTradeLoader:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path, sep=";")
        self._validate_columns()
        self.df = self.df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

        for col in ("buyer", "seller"):
            if col not in self.df.columns:
                self.df[col] = ""

    def _validate_columns(self) -> None:
        required = {"timestamp", "symbol", "price", "quantity"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in trades CSV: {sorted(missing)}"
            )

    def trades_at(self, timestamp: int) -> Dict[str, List[Trade]]:
        ts_rows = self.df[self.df["timestamp"] == timestamp]
        out: Dict[str, List[Trade]] = {}

        for _, row in ts_rows.iterrows():
            symbol = str(row["symbol"])
            out.setdefault(symbol, []).append(
                Trade(
                    symbol=symbol,
                    price=int(row["price"]),
                    quantity=int(row["quantity"]),
                    buyer="" if pd.isna(row["buyer"]) else str(row["buyer"]),
                    seller="" if pd.isna(row["seller"]) else str(row["seller"]),
                    timestamp=int(row["timestamp"]),
                )
            )

        return out


class Backtester:
    def __init__(
        self,
        price_csv: str,
        trade_csv: str,
        position_limits: Dict[str, int],
        mark_to_mid: bool = True,
    ) -> None:
        self.price_loader = PriceBookLoader(price_csv)
        self.trade_loader = MarketTradeLoader(trade_csv)
        self.position_limits = position_limits
        self.mark_to_mid = mark_to_mid

    def run(self, strategy: Strategy) -> pd.DataFrame:
        products = self.price_loader.get_products()
        position: Dict[str, int] = {product: 0 for product in products}
        for product in self.position_limits:
            position.setdefault(product, 0)

        cash = 0.0
        trader_data = ""
        own_trades_prev: Dict[str, List[Trade]] = {}
        rows: List[dict] = []

        for timestamp, order_depths, mids in self.price_loader.iter_snapshots():
            market_trades = self.trade_loader.trades_at(timestamp)

            state = TradingState(
                traderData=trader_data,
                timestamp=timestamp,
                order_depths=order_depths,
                own_trades=own_trades_prev,
                market_trades=market_trades,
                position=position.copy(),
                observations={"mid_prices": mids},
            )

            raw = strategy.run(state)
            if isinstance(raw, tuple) and len(raw) == 3:
                orders_by_symbol, _conversions, trader_data = raw
            else:
                raise TypeError(
                    "Strategy.run must return (orders_by_symbol, conversions, traderData)."
                )

            accepted_orders = self._apply_position_limits(position, orders_by_symbol)
            own_trades_now, cash_delta = self._execute_orders(
                timestamp, accepted_orders, order_depths
            )
            cash += cash_delta

            for product, trades in own_trades_now.items():
                for trade in trades:
                    if trade.buyer == "SUBMISSION":
                        position[product] = position.get(product, 0) + trade.quantity
                    elif trade.seller == "SUBMISSION":
                        position[product] = position.get(product, 0) - trade.quantity

            mtm = 0.0
            if self.mark_to_mid:
                for product, pos in position.items():
                    mid = mids.get(product)
                    if mid is not None:
                        mtm += pos * mid

            row = {
                "timestamp": timestamp,
                "cash": cash,
                "mtm": mtm,
                "total_pnl": cash + mtm,
                "trade_count": sum(len(v) for v in own_trades_now.values()),
                "market_trade_count": sum(len(v) for v in market_trades.values()),
            }

            for product in sorted(position):
                row[f"pos_{product}"] = position.get(product, 0)

            rows.append(row)
            own_trades_prev = own_trades_now

        return pd.DataFrame(rows)

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

    def _execute_orders(
        self,
        timestamp: int,
        orders_by_symbol: Dict[str, List[Order]],
        order_depths: Dict[str, OrderDepth],
    ) -> Tuple[Dict[str, List[Trade]], float]:
        """Execute immediately against visible book only.

        Market trades from the trades CSV are passed to the strategy as context
        through state.market_trades, but this engine still only fills your orders
        against the visible order book in the prices CSV.
        """
        own_trades: Dict[str, List[Trade]] = {}
        cash_delta = 0.0

        for symbol, orders in orders_by_symbol.items():
            depth = order_depths.get(symbol, OrderDepth())
            trades: List[Trade] = []

            asks = sorted(depth.sell_orders.items(), key=lambda x: x[0])
            bids = sorted(depth.buy_orders.items(), key=lambda x: x[0], reverse=True)

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
                                buyer="SUBMISSION",
                                seller="BOT",
                                timestamp=timestamp,
                            )
                        )
                        cash_delta -= fill * ask_price
                        remaining -= fill
                        asks[ask_index] = (ask_price, -(available - fill))
                        if available - fill == 0:
                            ask_index += 1
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
                                buyer="BOT",
                                seller="SUBMISSION",
                                timestamp=timestamp,
                            )
                        )
                        cash_delta += fill * bid_price
                        remaining -= fill
                        bids[bid_index] = (bid_price, available - fill)
                        if available - fill == 0:
                            bid_index += 1

            own_trades[symbol] = trades

        return own_trades, cash_delta


class NaiveFairValueStrategy:
    def __init__(self, edge: int = 0, max_clip: int = 5):
        self.edge = edge
        self.max_clip = max_clip

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result: Dict[Symbol, List[Order]] = {}

        for symbol, depth in state.order_depths.items():
            orders: List[Order] = []

            mid = state.observations.get("mid_prices", {}).get(symbol)
            if mid is None and state.market_trades.get(symbol):
                mid = state.market_trades[symbol][-1].price
            if mid is None:
                result[symbol] = orders
                continue

            fair = int(round(mid))

            if depth.sell_orders:
                best_ask = min(depth.sell_orders)
                ask_volume = abs(depth.sell_orders[best_ask])
                if best_ask < fair - self.edge:
                    orders.append(
                        Order(symbol, best_ask, min(self.max_clip, ask_volume))
                    )

            if depth.buy_orders:
                best_bid = max(depth.buy_orders)
                bid_volume = depth.buy_orders[best_bid]
                if best_bid > fair + self.edge:
                    orders.append(
                        Order(symbol, best_bid, -min(self.max_clip, bid_volume))
                    )

            result[symbol] = orders

        return result, 0, state.traderData


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
        description="Local backtester for IMC-style prices + trades CSVs."
    )
    parser.add_argument("price_csv", help="Semicolon-delimited prices CSV")
    parser.add_argument("trade_csv", help="Semicolon-delimited trades CSV")
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
    parser.add_argument("--edge", type=int, default=0, help="Edge for example strategy")
    parser.add_argument(
        "--max-clip", type=int, default=5, help="Max trade size per action"
    )
    args = parser.parse_args()

    limits = parse_limits(args.limits)
    strategy = Trader()
    engine = Backtester(
        price_csv=args.price_csv,
        trade_csv=args.trade_csv,
        position_limits=limits,
    )
    results = engine.run(strategy)
    results.to_csv(args.out, index=False)

    final_pnl = float(results["total_pnl"].iloc[-1]) if not results.empty else 0.0
    print(results.tail(10).to_string(index=False))
    print(f"\nFinal total PnL: {final_pnl:.2f}")
    print(f"Saved results to: {args.out}")


if __name__ == "__main__":
    main()
