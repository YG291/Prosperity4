from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import statistics
from statistics import linear_regression

class Trader:
    def b_insert(self, val, arr: list, q):
        left, right = 0, len(arr)

        # binary search for insertion index (lower bound)
        while left < right:
            mid = (left + right) // 2
            if arr[mid] < val:
                left = mid + 1
            else:
                right = mid

        # insert q copies at correct position
        return arr[:left] + [val] * q + arr[left:]

    def get_median(self, arr):
        return arr[len(arr) // 2]

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
        current_pos = state.position.get(product, 0)

        buy_buffer = int(abs(prediction-storage["buy"][1]))
        buy_limit = 15
        if len(order_depth.sell_orders) != 0:
            if (current_pos < buy_limit):  # buying
                orders.append(Order(product, prediction - buy_buffer, buy_limit - current_pos))
                self.update_storage(storage, prediction, buy_limit - current_pos, product)
        
        sell_buffer = int(abs(prediction-storage["sell"][1]))
        sell_limit = -15
        if len(order_depth.buy_orders) != 0:
            if (current_pos > sell_limit):  # shorting
                orders.append(Order(product, prediction + sell_buffer, sell_limit - current_pos))
                self.update_storage(storage, prediction + sell_buffer, sell_limit - current_pos, product)
        print('tomatoes: '+ str(current_pos))
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
        middle = int(storage["sort"][1]) 
        sell_buffer = int(abs(middle-storage["sell"][1]))
        sell_limit = -80
        if current_pos > sell_limit:
            orders.append(Order(product, middle + sell_buffer, sell_limit - current_pos))
        buy_buffer = int(abs(middle-storage["buy"][1]))
        buy_limit = 80
        if current_pos < buy_limit:
            orders.append(Order(product, middle - buy_buffer, buy_limit - current_pos))
        print('emerald: '+ str(current_pos))

        result[product] = orders

    def _with_market_orders(self, sorted_list: list[tuple], storage):
        for price, quantity in sorted_list:
            quantity = abs(quantity)
            if len(storage["sort"]) == 0:
                storage["sort"] = [1, price]
                continue
            q, val = (storage["sort"][0], storage["sort"][1]) #total order volume, mean of all order offers
            storage["sort"][0] += quantity
            storage["sort"][1] = (quantity * price + val * q) / (q + quantity)
    
    def _update_best_storage(self, sorted_list, storage, buy_or_sell: str):
        """best as in records highest bids and lowest asks.
        """
        q, val = (storage[buy_or_sell][0], storage[buy_or_sell][1])
        storage[buy_or_sell][0] += sorted_list[0][1]
        storage[buy_or_sell][1] = (sorted_list[0][1]*sorted_list[0][0] + val*q)/(q+sorted_list[0][1])


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
                       'pos': {"EMERALDS":[0,0],"TOMATOES":[0,0]}, "sort": [], "buy": [], "sell": []}
        for product in state.order_depths:
            if product == 'TOMATOES':
                order_depth: OrderDepth = state.order_depths[product]
                if len(storage[product]) < 2:
                    break
                for trade in state.market_trades.get("TOMATOES", []):
                    q, val = (trade.quantity, trade.price)
                    if q < 0:
                        order_depth.sell_orders[val] = q
                    else:
                        order_depth.buy_orders[val] = q
                    # this part adds in the market orders to the current order book
                sellsorted = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
                buysorted = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
                # each index is price, quantity 
                self._with_market_orders(buysorted, storage)
                self._with_market_orders(sellsorted, storage)
                # computes new mean price of market orders + order book orders
                if len(storage["buy"]) == 0:
                    storage["buy"] = [1, buysorted[0][0]]
                else:
                    self._update_best_storage(buysorted, storage, 'buy')
                if len(storage["sell"]) == 0:
                    storage["sell"] = [1, sellsorted[0][0]]
                else:
                    self._update_best_storage(sellsorted, storage, 'sell')
                    # print(storage)
                if not len(storage["sort"]) < 1:
                    self.trade_tomatoes(state, storage, result, order_depth, buysorted, sellsorted)
                #result[product] = []

            if product == 'EMERALDS':
                order_depth: OrderDepth = state.order_depths[product]
                for trade in state.market_trades.get("EMERALDS", []):
                    q, val = (trade.quantity, trade.price)
                    if q < 0:
                        order_depth.sell_orders[val] = q
                    else:
                        order_depth.buy_orders[val] = q
                    # this part adds in the market orders to the current order book
                sellsorted = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
                buysorted = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
                # each index is price, quantity 
                self._with_market_orders(buysorted, storage)
                self._with_market_orders(sellsorted, storage)
                # computes new mean price of market orders + order book orders
                if len(storage["buy"]) == 0:
                    storage["buy"] = [1, buysorted[0][0]]
                else:
                    self._update_best_storage(buysorted, storage, 'buy')
                if len(storage["sell"]) == 0:
                    storage["sell"] = [1, sellsorted[0][0]]
                else:
                    self._update_best_storage(sellsorted, storage, 'sell')
                    # print(storage)
                if not len(storage["sort"]) < 1:
                    self.trade_emeralds(state, storage, result, order_depth, buysorted, sellsorted)
                #result[product] = []

        traderData = jsonpickle.encode(storage)
        # No state needed - we check position directly
        conversions = 0
        return result, conversions, traderData
