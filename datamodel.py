"""Reference data structures used by a Prosperity trading algorithm.

The official Prosperity starter template revolves around one callback:
`Trader.run(state)`. The simulator constructs a `TradingState` object on every
timestamp and expects the strategy to return orders built from the models below.
This file groups those models in one place so the algorithm code can stay small.
"""

import json
from json import JSONEncoder
from typing import Dict, List

import jsonpickle

# Type aliases keep the template close to the wording used in the Prosperity
# guide while still being plain Python primitives at runtime.
Time = int
Symbol = str
Product = str
Position = int
UserId = str
ObservationValue = int


class Listing:
    """Describes a tradable instrument and the currency it is quoted in."""

    def __init__(self, symbol: Symbol, product: Product, denomination: Product):
        self.symbol = symbol
        self.product = product
        self.denomination = denomination


class ConversionObservation:
    """Holds external observations used for conversion-style products.

    Later Prosperity rounds introduce products whose fair value depends on
    external variables such as transport fees and weather-like signals.
    """

    def __init__(
        self,
        bidPrice: float,
        askPrice: float,
        transportFees: float,
        exportTariff: float,
        importTariff: float,
        sunlight: float,
        humidity: float,
    ):
        self.bidPrice = bidPrice
        self.askPrice = askPrice
        self.transportFees = transportFees
        self.exportTariff = exportTariff
        self.importTariff = importTariff
        self.sunlight = sunlight
        self.humidity = humidity


class Observation:
    """Bundles all non-order-book signals available to the strategy."""

    def __init__(
        self,
        plainValueObservations: Dict[Product, ObservationValue],
        conversionObservations: Dict[Product, ConversionObservation],
    ) -> None:
        self.plainValueObservations = plainValueObservations
        self.conversionObservations = conversionObservations

    def __str__(self) -> str:
        return (
            "(plainValueObservations: "
            + jsonpickle.encode(self.plainValueObservations)
            + ", conversionObservations: "
            + jsonpickle.encode(self.conversionObservations)
            + ")"
        )


class Order:
    """One order the strategy wants to send to the exchange."""

    def __init__(self, symbol: Symbol, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

    def __str__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"

    def __repr__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"


class OrderDepth:
    """Snapshot of the current order book for one product.

    `buy_orders` maps bid price -> quantity.
    `sell_orders` maps ask price -> quantity, and in Prosperity logs those ask
    quantities are commonly represented as negative numbers.
    """

    def __init__(
        self,
        buy_orders: Dict[int, int] | None = None,
        sell_orders: Dict[int, int] | None = None,
    ) -> None:
        self.buy_orders = buy_orders or {}
        self.sell_orders = sell_orders or {}


class Trade:
    """Represents a fill that already happened in the market."""

    def __init__(
        self,
        symbol: Symbol,
        price: int,
        quantity: int,
        buyer: UserId = None,
        seller: UserId = None,
        timestamp: int = 0,
    ) -> None:
        self.symbol = symbol
        self.price: int = price
        self.quantity = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp

    def __str__(self) -> str:
        return (
            "("
            + self.symbol
            + ", "
            + str(self.buyer)
            + " << "
            + str(self.seller)
            + ", "
            + str(self.price)
            + ", "
            + str(self.quantity)
            + ", "
            + str(self.timestamp)
            + ")"
        )

    def __repr__(self) -> str:
        return (
            "("
            + self.symbol
            + ", "
            + str(self.buyer)
            + " << "
            + str(self.seller)
            + ", "
            + str(self.price)
            + ", "
            + str(self.quantity)
            + ", "
            + str(self.timestamp)
            + ")"
        )


class TradingState(object):
    """Complete market snapshot passed into `Trader.run` on every tick."""

    def __init__(
        self,
        traderData: str,
        timestamp: Time,
        listings: Dict[Symbol, Listing],
        order_depths: Dict[Symbol, OrderDepth],
        own_trades: Dict[Symbol, List[Trade]],
        market_trades: Dict[Symbol, List[Trade]],
        position: Dict[Product, Position],
        observations: Observation,
    ):
        # `traderData` is the one piece of state the algorithm can persist from
        # one invocation of `run` to the next.
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations

    def toJSON(self):
        """Serialize the state for debugging or logging."""
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)


class ProsperityEncoder(JSONEncoder):
    """JSON encoder that knows how to serialize the lightweight model objects."""

    def default(self, o):
        return o.__dict__
