from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import statistics
from statistics import linear_regression

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
        return numbuys/numsells


    def fair_value(self, state: TradingState, product):
        orders = state.order_depths[product]
        buys = orders.buy_orders
        sells = orders.sell_orders
        buy_component = 0
        sell_component = 0
        total = abs(sum(buys.values())) + abs(sum(sells.values()))
        for bprice in buys:
            buy_component += abs(bprice * (buys[bprice]/total))
        for sprice in sells:
            sell_component += abs(sprice*(sells[sprice]/total))
        return buy_component + buy_component + sell_component

    def regression(self, product, storage):
        linmodel = linear_regression([int(x) for x in storage[product].keys()], list(storage[product].values()))
        return (linmodel.intercept, linmodel.slope)

    def compute_mid(self, storage, ssort, bsort, stamp):
        if not bsort and not ssort:
            return storage[stamp-100]
        if not bsort:
            return ssort[0][0]
        if not ssort:
            return bsort[0][0]
        return (bsort[0][0]+ssort[0][0])/2

    def update_storage(self, storage, price, quantity, product: str):
        storage['pos'][product][0] += price * quantity
        storage['pos'][product][1] += quantity

    def trade_tomatoes(self, state: TradingState, storage, result, order_depth,
                       buysorted, sellsorted, product = 'TOMATOES'):
        """
        Precondition:
        - product is TOMATOES
        - if len(storage[product]) < 2
        """
        if len(storage[product]) > 150:
            storage[product].pop(next(iter(storage[product])))
        intercept, slope = self.regression(product, storage)
        prediction = slope * int(state.timestamp) + intercept

        orders: List[Order] = []  # Order(symbol, price, quantity)
        acceptable_price = prediction  # Participant should calculate this value

        buy_limit = 15
        buy_buffer = 1
        if len(order_depth.sell_orders) != 0:
            for index in range(len(buysorted)):
                best_ask, best_ask_amount = buysorted[index][0], buysorted[index][1]
                if (int(best_ask) <= acceptable_price - buy_buffer and
                        storage['pos'][product][1] < buy_limit):  # buying
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))
                    self.update_storage(storage, best_ask, -best_ask_amount, product)
                else:
                    break

        sell_buffer = 1
        sell_limit = -15
        if len(order_depth.buy_orders) != 0:
            for index in range(len(sellsorted)):
                best_bid, best_bid_amount = sellsorted[index][0], sellsorted[index][1]
                if (int(best_bid) >= acceptable_price + sell_buffer and
                        storage['pos'][product][1] > sell_limit):  # shorting
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
                    self.update_storage(storage, best_bid, -best_bid_amount, product)
                else:
                    break

        result[product] = orders


    def trade_emeralds(self, state: TradingState, storage, result, order_depth,
                       buysorted, sellsorted, product = 'EMERALDS'):
        """
        Precondition:
        - product is EMERALDS
        - if len(storage[product]) < 2
        """
        orders: List[Order] = []  # Order(symbol, price, quantity)
        current_pos = state.position.get(product, 0)
        middle = 10000  # Participant should calculate this value

        sell_buffer = 1
        sell_limit = -15
        best_ask = sellsorted[0][0]
        if current_pos > sell_limit:
            if best_ask > middle:
                orders.append(Order(product, best_ask - sell_buffer, sell_limit - current_pos))
            else:
                orders.append(Order(product, middle + sell_buffer, sell_limit - current_pos))

        buy_buffer = 1
        buy_limit = 15
        best_bid = buysorted[0][0]
        if current_pos < buy_limit:
            if best_bid < middle:
                orders.append(Order(product, best_bid + buy_buffer, buy_limit - current_pos))
            else:
                orders.append(Order(product, middle - buy_buffer, buy_limit - current_pos))

        result[product] = orders
        print(current_pos)


    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""
        #MAKE SURE THAT YOU CALL ALL OF YOUR METHODS HERE TO DETERMINE HOW TO BUY/SELL

        # Orders to be placed on exchange matching engine
        result = {}
        if state.traderData:
            storage = jsonpickle.decode(state.traderData)
        else:
            storage = {"EMERALDS": dict(), "TOMATOES": dict(),
                       'pos': {"EMERALDS":[0,0],"TOMATOES":[0,0]}}
        for product in state.order_depths:
            if product == 'TOMATOES':
                order_depth: OrderDepth = state.order_depths[product]
                sellsorted = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
                buysorted = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
                mid = self.compute_mid(storage, buysorted, sellsorted, state.timestamp)
                storage[product][int(state.timestamp)] = mid

                if len(storage[product]) < 2:
                    break

                #self.trade_tomatoes(state, storage, result, order_depth, buysorted, sellsorted)
                result[product] = []

            if product == 'EMERALDS':
                order_depth: OrderDepth = state.order_depths[product]
                sellsorted = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
                buysorted = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
                self.trade_emeralds(state, storage, result, order_depth, buysorted, sellsorted)
                #result[product] = []

        traderData = jsonpickle.encode(storage)
        # No state needed - we check position directly
        conversions = 0
        return result, conversions, traderData
