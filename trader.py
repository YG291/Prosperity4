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


    def regression(self, product, storage):
        linmodel = linear_regression([int(x) for x in storage[product]['historical'].keys()], list(storage[product]['historical'].values()))
        return (linmodel.intercept, linmodel.slope)

    def compute_mid(self, storage, ssort, bsort, stamp, product):
        if not bsort and not ssort:
            return storage[product]['historical'][stamp-100]
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
        if len(storage[product]['historical']) > 100:
            storage[product]['historical'].pop(next(iter(storage[product]['historical'])))
        intercept, slope = self.regression(product, storage)
        prediction = slope * int(state.timestamp) + intercept

        orders: List[Order] = []  # Order(symbol, price, quantity)
        current_pos = state.position.get(product, 0)

        buy_buffer = int(abs(prediction-storage[product]["buy"][1]))
        buy_limit = 80
        if len(order_depth.sell_orders) != 0:
            if (current_pos < buy_limit):  # buying
                orders.append(Order(product, int(prediction - buy_buffer), buy_limit - current_pos))
                self.update_storage(storage, int(prediction - buy_buffer), buy_limit - current_pos, product)
        
        sell_buffer = int(abs(prediction-storage[product]["sell"][1]))
        sell_limit = -80
        if len(order_depth.buy_orders) != 0:
            if (current_pos > sell_limit):  # shorting
                orders.append(Order(product, int(prediction + sell_buffer), sell_limit - current_pos))
                self.update_storage(storage, int(prediction + sell_buffer), sell_limit - current_pos, product)
                
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
        middle = int(storage[product]["sort"][1]) 
        sell_buffer = int(abs(middle-storage[product]["sell"][1]))
        sell_limit = -80
        if current_pos > sell_limit:
            orders.append(Order(product, middle + sell_buffer, sell_limit - current_pos))
        buy_buffer = int(abs(middle-storage[product]["buy"][1]))
        buy_limit = 80
        if current_pos < buy_limit:
            orders.append(Order(product, middle - buy_buffer, buy_limit - current_pos))

        result[product] = orders

    def _with_market_orders(self, sorted_list: list[tuple], storage, product: str):
        for price, quantity in sorted_list:
            quantity = abs(quantity)
            if len(storage[product]["sort"]) == 0:
                storage[product]["sort"] = [1, price]
                continue
            q, val = (storage[product]["sort"][0], storage[product]["sort"][1]) #total order volume, mean of all order offers
            storage[product]["sort"][0] += quantity
            storage[product]["sort"][1] = (quantity * price + val * q) / (q + quantity)
    
    def _update_best_storage(self, sorted_list, storage, buy_or_sell: str, product):
        """best as in records highest bids and lowest asks.
        """
        q, val = (storage[product][buy_or_sell][0], storage[product][buy_or_sell][1])
        storage[product][buy_or_sell][0] += sorted_list[0][1]
        storage[product][buy_or_sell][1] = (sorted_list[0][1]*sorted_list[0][0] + val*q)/(q+sorted_list[0][1])


    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""
        #MAKE SURE THAT YOU CALL ALL OF YOUR METHODS HERE TO DETERMINE HOW TO BUY/SELL

        # Orders to be placed on exchange matching engine
        result = {}
        if state.traderData:
            storage = jsonpickle.decode(state.traderData)
        else:
            storage = {"EMERALDS": {"historical": {}, "sort": [], "buy": [], "sell": []}, "TOMATOES": {"historical": {}, "sort": [], "buy": [], "sell": []},
                       'pos': {"EMERALDS":[0,0],"TOMATOES":[0,0]}}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            sellsorted = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
            buysorted = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)

            mid = self.compute_mid(storage, buysorted, sellsorted, state.timestamp, product)
            storage[product]["historical"][int(state.timestamp)] = mid
            for trade in state.market_trades.get(product, []):
                q, val = (trade.quantity, trade.price)
                if q < 0:
                    order_depth.sell_orders[val] = q
                else:
                    order_depth.buy_orders[val] = q
            self._with_market_orders(buysorted, storage, product)
            self._with_market_orders(sellsorted, storage, product)

            if len(storage[product]["buy"]) == 0:
                storage[product]["buy"] = [1, buysorted[0][0]]
            else:
                self._update_best_storage(buysorted, storage, 'buy', product)
            if len(storage[product]["sell"]) == 0:
                storage[product]["sell"] = [1, sellsorted[0][0]]
            else:
                self._update_best_storage(sellsorted, storage, 'sell', product)
            
            if product == 'TOMATOES':
                if len(storage[product]['historical']) < 2:
                    continue
                if not len(storage[product]["sort"]) < 1:
                    self.trade_tomatoes(state, storage, result, order_depth, buysorted, sellsorted)
                #result[product] = []
                print(state.position.get(product, 0))

            if product == 'EMERALDS':
                if len(storage[product]['historical']) < 2:
                    continue
                if not len(storage[product]["sort"]) < 1:
                    self.trade_emeralds(state, storage, result, order_depth, buysorted, sellsorted)
                #result[product] = []
                print(state.position.get(product, 0))

        traderData = jsonpickle.encode(storage)
        # No state needed - we check position directly
        conversions = 0
        return result, conversions, traderData
