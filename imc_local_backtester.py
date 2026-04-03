from __future__ import annotations

# argparse lets us read command-line arguments like:
# python file.py prices.csv trades.csv --limits AMETHYSTS=20,STARFRUIT=20
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout

# dataclass automatically creates __init__, __repr__, etc.
# field(default_factory=...) is used for mutable defaults like dicts.
from dataclasses import dataclass, field
import multiprocessing
import os

# Typing tools for clearer code and strategy interface definitions.
from typing import Any, Dict, List, Optional, Protocol, Tuple

# pandas is used to read and manipulate CSV data.
import pandas as pd

from trader import Trader

# Type aliases to make the code easier to read.
# These do not create new types; they are just descriptive names.
Symbol = str
Product = str
Position = int


@dataclass
class Order:
    # The product / symbol this order is for.
    symbol: Symbol

    # Limit price of the order.
    # For a buy: max price you are willing to pay.
    # For a sell: min price you are willing to accept.
    price: int

    # Signed quantity:
    # positive = buy
    # negative = sell
    quantity: int


@dataclass
class Trade:
    # Product / symbol that traded.
    symbol: Symbol

    # Execution price of the trade.
    price: int

    # Executed quantity.
    # In this simplified model, quantity itself is kept positive,
    # while buyer/seller fields indicate which side we were on.
    quantity: int

    # Name / label of buyer.
    buyer: str

    # Name / label of seller.
    seller: str

    # Timestamp when the trade occurred.
    timestamp: int


@dataclass
class OrderDepth:
    # Outstanding buy orders in the book.
    # Dictionary format: price -> volume
    buy_orders: Dict[int, int] = field(default_factory=dict)

    # Outstanding sell orders in the book.
    # Dictionary format: price -> volume
    # Convention used here: sell volumes are stored as NEGATIVE values.
    sell_orders: Dict[int, int] = field(default_factory=dict)


@dataclass
class TradingState:
    # String used to carry state across iterations.
    # This mimics IMC's traderData idea.
    traderData: str

    # Current time step.
    timestamp: int

    # Order books for each symbol at this timestamp.
    order_depths: Dict[Symbol, OrderDepth]

    # Trades that our own strategy executed in the previous step.
    own_trades: Dict[Symbol, List[Trade]]

    # Market trades from the external trades CSV at this timestamp.
    market_trades: Dict[Symbol, List[Trade]]

    # Current inventory / position per product.
    position: Dict[Product, Position]

    # Extra info container.
    # Here we use it mainly to store mid prices.
    observations: Dict[str, object] = field(default_factory=dict)


class Strategy(Protocol):
    # This defines the required interface for a strategy.
    # Any strategy passed into Backtester.run(...) must implement run(...)
    # with this signature.
    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        """Return (orders_by_symbol, conversions, traderData)."""


def _to_int_if_present(value: object) -> Optional[int]:
    # If the value is missing / NaN, return None.
    if pd.isna(value):
        return None

    # Otherwise convert it to int.
    return int(value)


class PriceBookLoader:
    # Responsible for loading the prices CSV and turning each timestamp
    # into a TradingState-like order book snapshot.
    def __init__(self, csv_path: str):
        # IMC price files are semicolon-delimited, not comma-delimited.
        self.df = pd.read_csv(csv_path, sep=";")

        # Make sure essential columns exist.
        self._validate_columns(self.df)

        # Sort by time and product so iteration is stable and predictable.
        self.df = self.df.sort_values(["timestamp", "product"]).reset_index(drop=True)
        self._products = sorted(self.df["product"].astype(str).unique().tolist())

    def _validate_columns(self, df: pd.DataFrame) -> None:
        # Minimum columns required for this loader to work.
        required = {"timestamp", "product"}

        # Check if any required columns are missing.
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing required columns in prices CSV: {sorted(missing)}"
            )

    def get_products(self) -> List[str]:
        # Return sorted list of all unique products in the price file.
        return self._products

    def iter_snapshots(self):
        # Group rows by timestamp so we can reconstruct one market state
        # per iteration.
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
    # Responsible for loading the trades CSV and letting us query
    # all market trades that occurred at a given timestamp.
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path, sep=";")
        self._validate_columns(df)

        # Sort by time and symbol to make processing predictable.
        df = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

        # Some files may not have buyer/seller columns.
        # If missing, create them as empty strings.
        for col in ("buyer", "seller"):
            if col not in df.columns:
                df[col] = ""

        self._by_timestamp: Dict[int, Dict[str, List[Trade]]] = {}
        for row in df.itertuples(index=False):
            timestamp = int(row.timestamp)
            symbol = str(row.symbol)
            self._by_timestamp.setdefault(timestamp, {}).setdefault(symbol, []).append(
                Trade(
                    symbol=symbol,
                    price=int(row.price),
                    quantity=int(row.quantity),
                    buyer="" if pd.isna(row.buyer) else str(row.buyer),
                    seller="" if pd.isna(row.seller) else str(row.seller),
                    timestamp=timestamp,
                )
            )

    def _validate_columns(self, df: pd.DataFrame) -> None:
        # Minimum columns required for trade replay context.
        required = {"timestamp", "symbol", "price", "quantity"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing required columns in trades CSV: {sorted(missing)}"
            )

    def trades_at(self, timestamp: int) -> Dict[str, List[Trade]]:
        return self._by_timestamp.get(timestamp, {})


class Backtester:
    # Main engine that simulates the strategy on historical data.
    def __init__(
        self,
        price_csv: str,
        trade_csv: str,
        position_limits: Dict[str, int],
        mark_to_mid: bool = True,
        suppress_strategy_output: bool = False,
    ) -> None:
        self.price_loader = PriceBookLoader(price_csv)
        self.trade_loader = MarketTradeLoader(trade_csv)

        # Per-product position limits, e.g. {"AMETHYSTS": 20, "STARFRUIT": 20}
        self.position_limits = position_limits

        # Whether to mark inventory to mid-price each step.
        self.mark_to_mid = mark_to_mid
        self.suppress_strategy_output = suppress_strategy_output

    def run(self, strategy: Strategy) -> pd.DataFrame:
        # Start with all products from the price file at zero position.
        products = self.price_loader.get_products()
        position: Dict[str, int] = {product: 0 for product in products}

        # Also ensure every product listed in position_limits exists in position.
        for product in self.position_limits:
            position.setdefault(product, 0)

        # Cash account: negative when we buy, positive when we sell.
        cash = 0.0

        # traderData carried across steps.
        trader_data = ""

        # Our own trades from the previous step.
        own_trades_prev: Dict[str, List[Trade]] = {}

        # List of result rows that will become the final dataframe.
        rows: List[dict] = []
        rows_append = rows.append
        position_products = sorted(position)

        sink = open(os.devnull, "w", encoding="utf-8") if self.suppress_strategy_output else None

        try:
            # Iterate through each timestamp's order book snapshot.
            for timestamp, order_depths, mids in self.price_loader.iter_snapshots():
                # Get market trades that occurred at this timestamp.
                market_trades = self.trade_loader.trades_at(timestamp)

                # Build the state passed to the strategy.
                state = TradingState(
                    traderData=trader_data,
                    timestamp=timestamp,
                    order_depths=order_depths,
                    own_trades=own_trades_prev,
                    market_trades=market_trades,
                    position=position.copy(),
                    observations={"mid_prices": mids},
                )

                # Run the strategy.
                raw = self._run_strategy(strategy, state, sink)

                # Enforce the expected return format.
                if isinstance(raw, tuple) and len(raw) == 3:
                    orders_by_symbol, _conversions, trader_data = raw
                else:
                    raise TypeError(
                        "Strategy.run must return (orders_by_symbol, conversions, traderData)."
                    )

                # Remove illegal orders that would violate position limits.
                accepted_orders = self._apply_position_limits(position, orders_by_symbol)

                # Execute remaining orders against the visible book.
                own_trades_now, cash_delta = self._execute_orders(
                    timestamp, accepted_orders, order_depths
                )

                # Update cash account with this step's realized executions.
                cash += cash_delta

                # Update positions based on our executed trades.
                for product, trades in own_trades_now.items():
                    for trade in trades:
                        # If we are the buyer, inventory increases.
                        if trade.buyer == "SUBMISSION":
                            position[product] = position.get(product, 0) + trade.quantity

                        # If we are the seller, inventory decreases.
                        elif trade.seller == "SUBMISSION":
                            position[product] = position.get(product, 0) - trade.quantity

                # Mark-to-market value of our current inventory.
                mtm = 0.0
                if self.mark_to_mid:
                    for product, pos in position.items():
                        mid = mids.get(product)
                        if mid is not None:
                            mtm += pos * mid

                # Count buy vs sell executions for this step (aggregate + per-product).
                buy_count = 0
                sell_count = 0
                per_product_buys: Dict[str, int] = {}
                per_product_sells: Dict[str, int] = {}
                trade_count = 0
                for prod, trades in own_trades_now.items():
                    pb = ps = 0
                    trade_count += len(trades)
                    for trade in trades:
                        if trade.buyer == "SUBMISSION":
                            buy_count += 1
                            pb += 1
                        elif trade.seller == "SUBMISSION":
                            sell_count += 1
                            ps += 1
                    per_product_buys[prod] = pb
                    per_product_sells[prod] = ps

                market_trade_count = sum(len(v) for v in market_trades.values())

                # Save one row of results for this timestamp.
                row = {
                    "timestamp": timestamp,
                    "cash": cash,
                    "mtm": mtm,
                    "total_pnl": cash + mtm,
                    "trade_count": trade_count,
                    "buy_count": buy_count,
                    "sell_count": sell_count,
                    "market_trade_count": market_trade_count,
                }

                # Per-product positions and trade counts.
                for product in position_products:
                    row[f"pos_{product}"] = position.get(product, 0)
                    row[f"buy_count_{product}"] = per_product_buys.get(product, 0)
                    row[f"sell_count_{product}"] = per_product_sells.get(product, 0)

                rows_append(row)

                # The trades from this step become own_trades in the next state.
                own_trades_prev = own_trades_now
        finally:
            if sink is not None:
                sink.close()

        # Return results as a dataframe for saving / plotting.
        return pd.DataFrame(rows)

    def _run_strategy(
        self,
        strategy: Strategy,
        state: TradingState,
        sink,
    ):
        if sink is None:
            return strategy.run(state)
        with redirect_stdout(sink), redirect_stderr(sink):
            return strategy.run(state)

    def _apply_position_limits(
        self,
        position: Dict[str, int],
        orders_by_symbol: Dict[str, List[Order]],
    ) -> Dict[str, List[Order]]:
        # This function enforces IMC-style pessimistic position limits:
        # buys and sells are checked separately.
        accepted: Dict[str, List[Order]] = {}

        for symbol, orders in orders_by_symbol.items():
            pos = position.get(symbol, 0)
            limit = self.position_limits.get(symbol)

            # If no limit is specified for a symbol, reject all orders for safety.
            if limit is None:
                accepted[symbol] = []
                continue

            # Total possible buy quantity if all buy orders fully execute.
            total_buy = sum(max(order.quantity, 0) for order in orders)

            # Total possible sell quantity if all sell orders fully execute.
            total_sell = sum(max(-order.quantity, 0) for order in orders)

            # Check long-side legality.
            buy_ok = pos + total_buy <= limit

            # Check short-side legality.
            sell_ok = pos - total_sell >= -limit

            filtered: List[Order] = []

            for order in orders:
                # Keep buy orders only if total buy side is legal.
                if order.quantity > 0 and buy_ok:
                    filtered.append(order)

                # Keep sell orders only if total sell side is legal.
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
        """
        Execute immediately against visible book only.

        Important limitation:
        - We do NOT simulate later bots trading into our resting quotes.
        - We only match our submitted orders against the visible book
          from the prices CSV at the current timestamp.
        """
        own_trades: Dict[str, List[Trade]] = {}
        cash_delta = 0.0

        for symbol, orders in orders_by_symbol.items():
            depth = order_depths.get(symbol, OrderDepth())
            trades: List[Trade] = []

            # Sort asks from lowest to highest price.
            asks = sorted(depth.sell_orders.items(), key=lambda x: x[0])

            # Sort bids from highest to lowest price.
            bids = sorted(depth.buy_orders.items(), key=lambda x: x[0], reverse=True)

            for order in orders:
                # Amount still left to execute.
                remaining = abs(order.quantity)

                # BUY ORDER LOGIC
                if order.quantity > 0:
                    ask_index = 0

                    # Sweep upward through asks while the order can still trade.
                    while remaining > 0 and ask_index < len(asks):
                        ask_price, ask_volume_signed = asks[ask_index]

                        # If the ask price is above our bid limit, stop.
                        if ask_price > order.price:
                            break

                        # Available size at this ask level.
                        available = abs(ask_volume_signed)

                        if available <= 0:
                            ask_index += 1
                            continue

                        # Fill either all remaining quantity or all available size.
                        fill = min(remaining, available)

                        # Record our trade.
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

                        # Buying spends cash.
                        cash_delta -= fill * ask_price

                        # Reduce the remaining quantity in our order.
                        remaining -= fill

                        # Reduce remaining size at that ask level.
                        asks[ask_index] = (ask_price, -(available - fill))

                        # Move to next ask if this level is fully exhausted.
                        if available - fill == 0:
                            ask_index += 1

                # SELL ORDER LOGIC
                else:
                    bid_index = 0

                    # Sweep downward through bids while the order can still trade.
                    while remaining > 0 and bid_index < len(bids):
                        bid_price, bid_volume = bids[bid_index]

                        # If the bid price is below our ask limit, stop.
                        if bid_price < order.price:
                            break

                        # Available size at this bid level.
                        available = bid_volume

                        if available <= 0:
                            bid_index += 1
                            continue

                        # Fill either all remaining quantity or all available size.
                        fill = min(remaining, available)

                        # Record our trade.
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

                        # Selling receives cash.
                        cash_delta += fill * bid_price

                        # Reduce remaining quantity in our sell order.
                        remaining -= fill

                        # Reduce remaining size at that bid level.
                        bids[bid_index] = (bid_price, available - fill)

                        # Move to next bid if this level is fully exhausted.
                        if available - fill == 0:
                            bid_index += 1

            own_trades[symbol] = trades

        return own_trades, cash_delta


class NaiveFairValueStrategy:
    # Very simple example strategy.
    #
    # Logic:
    # - Use mid price as fair value if available.
    # - If no mid price exists, use the last market trade price.
    # - Buy best ask if it is sufficiently below fair value.
    # - Sell best bid if it is sufficiently above fair value.
    def __init__(self, edge: int = 0, max_clip: int = 5):
        # Minimum edge required before trading.
        self.edge = edge

        # Max quantity to trade on any single action.
        self.max_clip = max_clip

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result: Dict[Symbol, List[Order]] = {}

        for symbol, depth in state.order_depths.items():
            orders: List[Order] = []

            # Try to get fair value from the mid price.
            mid = state.observations.get("mid_prices", {}).get(symbol)

            # If mid price is unavailable, fall back to last market trade.
            if mid is None and state.market_trades.get(symbol):
                mid = state.market_trades[symbol][-1].price

            # If still no estimate, skip this symbol.
            if mid is None:
                result[symbol] = orders
                continue

            # Convert fair value estimate to integer.
            fair = int(round(mid))

            # If there are asks in the book, inspect the best ask.
            if depth.sell_orders:
                best_ask = min(depth.sell_orders)
                ask_volume = abs(depth.sell_orders[best_ask])

                # Buy if the ask is cheap enough relative to fair.
                if best_ask < fair - self.edge:
                    orders.append(
                        Order(symbol, best_ask, min(self.max_clip, ask_volume))
                    )

            # If there are bids in the book, inspect the best bid.
            if depth.buy_orders:
                best_bid = max(depth.buy_orders)
                bid_volume = depth.buy_orders[best_bid]

                # Sell if the bid is rich enough relative to fair.
                if best_bid > fair + self.edge:
                    orders.append(
                        Order(symbol, best_bid, -min(self.max_clip, bid_volume))
                    )

            result[symbol] = orders

        # No conversions used in this simple example.
        # traderData is unchanged.
        return result, 0, state.traderData


def _run_backtest_worker(args: Tuple[str, str, Dict[str, int], Any, Dict[str, Any], bool, bool]) -> pd.DataFrame:
    price_csv, trade_csv, limits, strategy_cls, strategy_kwargs, mark_to_mid, suppress_output = args
    engine = Backtester(
        price_csv=price_csv,
        trade_csv=trade_csv,
        position_limits=limits,
        mark_to_mid=mark_to_mid,
        suppress_strategy_output=suppress_output,
    )
    strategy = strategy_cls(**strategy_kwargs)
    return engine.run(strategy)


def run_parallel_backtests(
    price_csv: str,
    trade_csv: str,
    limits: Dict[str, int],
    strategy_cls: Any,
    param_grid: List[Dict[str, Any]],
    mark_to_mid: bool = True,
    suppress_strategy_output: bool = True,
    workers: Optional[int] = None,
) -> List[pd.DataFrame]:
    """Run multiple independent backtests in parallel across CPU cores."""
    if not param_grid:
        return []

    max_workers = workers or multiprocessing.cpu_count()
    args_list = [
        (
            price_csv,
            trade_csv,
            limits,
            strategy_cls,
            strategy_kwargs,
            mark_to_mid,
            suppress_strategy_output,
        )
        for strategy_kwargs in param_grid
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_run_backtest_worker, args_list))


buy_sell_weight = 1 / 5


class CoolTrader:
    def bid(self):
        return 15

    def buy_sell_ratio(self, state: TradingState, product):
        orders = state.order_depths[product]
        buys = orders.buy_orders
        sells = orders.sell_orders
        numbuys = abs(sum(buys.values()))
        numsells = abs(sum(sells.values()))

        # if numbuys == 0:
        #    numbuys = 0.2
        # if numsells == 0:
        #    numsells = 0.2
        # return numbuys / numsells

        total = numbuys + numsells
        if total == 0:
            return 0
        return (numbuys - numsells) / total

    def fair_value(self, state: TradingState, product):
        orders = state.order_depths[product]
        buys = orders.buy_orders
        sells = orders.sell_orders
        buy_component = 0
        sell_component = 0
        total = abs(sum(buys.values())) + abs(sum(sells.values()))
        ratiobuysell = self.buy_sell_ratio(state, product)
        for bprice in buys:
            buy_component += abs(bprice * (buys[bprice] / total))
        for sprice in sells:
            sell_component += abs(sprice * (sells[sprice] / total))
        return (
            buy_sell_weight * buy_component * ratiobuysell
            + (1 - buy_sell_weight) * buy_component
            + sell_component
        )
        # this only weighs the buy component of the price, not the whole price.

    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""
        # MAKE SURE THAT YOU CALL ALL OF YOUR METHODS HERE TO DETERMINE HOW TO BUY/SELL

        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []  # Order(symbol, price, quantity)
            acceptable_price = self.fair_value(
                state, product
            )  # Participant should calculate this value
            print("Acceptable price : " + str(acceptable_price))
            print(
                "Buy Order depth : "
                + str(len(order_depth.buy_orders))
                + ", Sell order depth : "
                + str(len(order_depth.sell_orders))
            )

            buy_buffer = 1
            if len(order_depth.sell_orders) != 0:
                buysorted = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
                for index in range(len(buysorted)):
                    best_ask, best_ask_amount = buysorted[index][0], buysorted[index][1]
                    if int(best_ask) <= acceptable_price - buy_buffer:  # buying
                        print("BUY", str(-best_ask_amount) + "x", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_amount))
                    else:
                        break

            sell_buffer = 1
            if len(order_depth.buy_orders) != 0:
                sellsorted = sorted(
                    order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True
                )
                for index in range(len(sellsorted)):
                    best_bid, best_bid_amount = (
                        sellsorted[index][0],
                        sellsorted[index][1],
                    )
                    if int(best_bid) >= acceptable_price + sell_buffer:  # shorting
                        print("SELL", str(best_bid_amount) + "x", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_amount))
                    else:
                        break

            result[product] = orders

        traderData = ""  # No state needed - we check position directly
        conversions = 0
        return result, conversions, traderData


def parse_limits(text: str) -> Dict[str, int]:
    # Parse strings like:
    # "AMETHYSTS=20,STARFRUIT=20"
    # into:
    # {"AMETHYSTS": 20, "STARFRUIT": 20}
    out: Dict[str, int] = {}

    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue

        key, value = chunk.split("=", 1)
        out[key.strip()] = int(value.strip())

    return out


def main() -> None:
    # Set up command-line interface.
    parser = argparse.ArgumentParser(
        description="Local backtester for IMC-style prices + trades CSVs."
    )

    # Positional argument: price CSV file path.
    parser.add_argument("price_csv", help="Semicolon-delimited prices CSV")

    # Positional argument: trades CSV file path.
    parser.add_argument("trade_csv", help="Semicolon-delimited trades CSV")

    # Required argument: product position limits.
    parser.add_argument(
        "--limits",
        required=True,
        help="Position limits, e.g. AMETHYSTS=20,STARFRUIT=20",
    )

    # Optional output path for backtest result CSV.
    parser.add_argument(
        "--out",
        default="backtest_results.csv",
        help="Where to save per-timestamp results",
    )

    # Optional strategy tuning parameter.
    parser.add_argument("--edge", type=int, default=0, help="Edge for example strategy")

    # Optional strategy tuning parameter.
    parser.add_argument(
        "--max-clip", type=int, default=5, help="Max trade size per action"
    )
    parser.add_argument(
        "--suppress-strategy-output",
        action="store_true",
        help="Mute print-heavy strategy output to improve runtime.",
    )

    args = parser.parse_args()

    # Parse limits into a dictionary.
    limits = parse_limits(args.limits)
    strategy = Trader()

    engine = Backtester(
        price_csv=args.price_csv,
        trade_csv=args.trade_csv,
        position_limits=limits,
        suppress_strategy_output=args.suppress_strategy_output,
    )

    # Run the backtest.
    results = engine.run(strategy)

    # Save results to CSV.
    results.to_csv(args.out, index=False)

    # Print final summary.
    final_pnl = float(results["total_pnl"].iloc[-1]) if not results.empty else 0.0
    print(results.tail(10).to_string(index=False))
    print(f"\nFinal total PnL: {final_pnl:.2f}")
    print(f"Saved results to: {args.out}")


if __name__ == "__main__":
    # Standard Python entry point.
    main()
