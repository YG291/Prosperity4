import math
import statistics
import string
from statistics import linear_regression
from typing import List

import jsonpickle

from datamodel import Order, OrderDepth, TradingState, UserId


class Trader:
    def bid(self):
        return 15

    def buy_sell_ratio(self, state: TradingState, product):
        orders = state.order_depths[product]
        buys = orders.buy_orders
        sells = orders.sell_orders
        numbuys = abs(sum(buys.values()))
        numsells = abs(sum(sells.values()))
        if numbuys == 0:
            numbuys = 0.2
        if numsells == 0:
            numsells = 0.2
        return numbuys / numsells

    def fair_value(self, state: TradingState, product):
        orders = state.order_depths[product]
        buys = orders.buy_orders
        sells = orders.sell_orders
        buy_component = 0
        sell_component = 0
        total = abs(sum(buys.values())) + abs(sum(sells.values()))
        for bprice in buys:
            buy_component += abs(bprice * (buys[bprice] / total))
        for sprice in sells:
            sell_component += abs(sprice * (sells[sprice] / total))
        return buy_component + buy_component + sell_component

    def regression(self, product, storage):
        linmodel = linear_regression(
            [int(x) for x in storage[product].keys()], list(storage[product].values())
        )
        return (linmodel.intercept, linmodel.slope)

    def compute_mid(self, storage, bsort, ssort, stamp):
        if not bsort and not ssort:
            return storage[stamp - 100]
        if not bsort:
            return ssort[0][0]
        if not ssort:
            return bsort[0][0]
        return (bsort[0][0] + ssort[0][0]) / 2

    def compute_signed_distance(self, intercept, slope, timestamp, current_price):
        predicted_price = slope * timestamp + intercept
        return (predicted_price - current_price) / math.sqrt(slope**2 + 1)

    def trade_tomatoes(
        self,
        state: TradingState,
        storage,
        result,
        order_depth,
        buysorted,
        sellsorted,
        product="TOMATOES",
    ):
        """
        Precondition:
        - product is TOMATOES
        - if len(storage[product]) < 2
        """
        if len(storage[product]) > 25:
            storage[product].pop(next(iter(storage[product])))
        intercept, slope = self.regression(product, storage)
        prediction = slope * int(state.timestamp) + intercept

        orders: List[Order] = []  # Order(symbol, price, quantity)
        acceptable_price = prediction  # Participant should calculate this value
        buy_threshold = 1
        if len(order_depth.sell_orders) != 0:
            for index in range(len(buysorted)):
                best_ask, best_ask_amount = buysorted[index][0], buysorted[index][1]
                """
                if int(best_ask) <= acceptable_price - buy_buffer:  # buying
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))
                else:
                    break
                """

                # new slope distance logic: buy based on distance a bid is from the line
                # TODO: Add inventory management
                ask_residual = best_ask - prediction

                if slope > 0.5 and ask_residual <= buy_threshold:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))
                else:
                    break

        # follow trends based on trendline, instead of mean reversion
        sell_threshold = 1
        if len(order_depth.buy_orders) != 0:
            for index in range(len(sellsorted)):
                best_bid, best_bid_amount = sellsorted[index][0], sellsorted[index][1]
                bid_residual = best_bid - prediction

                if slope < -0.25 and bid_residual >= -sell_threshold:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
                else:
                    break

        result[product] = orders

    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""
        # MAKE SURE THAT YOU CALL ALL OF YOUR METHODS HERE TO DETERMINE HOW TO BUY/SELL

        # Orders to be placed on exchange matching engine
        result = {}
        storage = (
            jsonpickle.decode(state.traderData)
            if state.traderData
            else {"EMERALDS": dict(), "TOMATOES": dict()}
        )
        for product in state.order_depths:
            if product == "TOMATOES":
                order_depth: OrderDepth = state.order_depths[product]
                buysorted = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
                sellsorted = sorted(
                    order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True
                )
                mid = self.compute_mid(storage, buysorted, sellsorted, state.timestamp)
                storage[product][int(state.timestamp)] = mid

                if len(storage[product]) < 2:
                    break

                self.trade_tomatoes(
                    state, storage, result, order_depth, buysorted, sellsorted
                )

        traderData = jsonpickle.encode(storage)
        # No state needed - we check position directly
        conversions = 0
        return result, conversions, traderData
