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
    
    def long_regression(self, product, storage):
        prices = storage['trend'][1]
        x_axis = list(range(len(prices)))
        if len(prices) < 2:
            return prices[-1] if prices else 0, 0
        linmodel = linear_regression(x_axis, prices)
        return (linmodel.intercept, linmodel.slope)

    def regression(self, product, storage):
        prices = list(storage[product]['historical'].values())
        x_axis = list(range(len(prices)))
        if len(prices) < 2:
            return prices[-1] if prices else 0, 0
        linmodel = linear_regression(x_axis, prices)
        return (linmodel.intercept, linmodel.slope)

    def compute_mid(self, storage, ssort, bsort, stamp, product):
        if not bsort and not ssort:
            # Gets the last value saved in the previous tick
            history = storage[product]['historical']
            if history:
                return list(history.values())[-1] 
            return 0 # Fallback for the very first tick
        if not bsort:
            return ssort[0][0]
        if not ssort:
            return bsort[0][0]
        return (bsort[0][0]+ssort[0][0])/2

    def update_storage(self, storage, price, quantity, product: str):
        storage['pos'][product][0] += price * quantity
        storage['pos'][product][1] += quantity

    def GEOM_trade(self, state: TradingState, storage, result, order_depth,
                       buysorted, sellsorted, product, long_dir):
        """
        Precondition:
        - if len(storage[product]) < 2
        """
        if len(storage[product]['historical']) > 20:
            storage[product]['historical'].pop(next(iter(storage[product]['historical'])))
        intercept, slope = self.regression(product, storage)
        current_index = len(storage[product]['historical']) + 30
        prediction = slope * current_index + intercept

        orders: List[Order] = []  # Order(symbol, price, quantity)
        current_pos = state.position.get(product, 0)
        buy_limit = 80
        sell_limit = -80

        if long_dir:
            sell_limit = 0
        else:
            buy_limit = 0

        buy_buffer = 1 #int(abs(prediction-storage[product]["buy"][1]))//4
        if len(order_depth.sell_orders) != 0:
            if (current_pos < buy_limit):  # buying
                orders.append(Order(product, int(prediction - buy_buffer), buy_limit - current_pos))
        
        sell_buffer = 1 #int(abs(prediction-storage[product]["sell"][1]))//4
        if len(order_depth.buy_orders) != 0:
            if (current_pos > sell_limit):  # shorting
                orders.append(Order(product, int(prediction + sell_buffer), sell_limit - current_pos))
                
        result[product] = orders

    def MM_trade(self, state: TradingState, storage, result, order_depth,
                       buysorted, sellsorted, product):
        """
        Precondition:
        - if len(storage[product]) < 2
        """
        orders: List[Order] = []  # Order(symbol, price, quantity)
        current_pos = state.position.get(product, 0)
        middle = int(storage[product]["sort"][1]) 
        sell_buffer = int(abs(middle-storage[product]["sell"][1]))//5
        sell_limit = -80
        if current_pos > sell_limit:
            orders.append(Order(product, middle + sell_buffer, sell_limit - current_pos))
        buy_buffer = int(abs(middle-storage[product]["buy"][1]))//5
        buy_limit = 80
        if current_pos < buy_limit:
            orders.append(Order(product, middle - buy_buffer, buy_limit - current_pos))

        result[product] = orders

    def _with_market_orders(self, sorted_list: list[tuple], storage, product: str):
        for price, quantity in sorted_list:
            quantity = abs(quantity)
            q, val = (storage[product]["sort"][0], storage[product]["sort"][1]) #total order volume, mean of all order offers
            new_q = q + quantity
            if new_q > 0:
                storage[product]["sort"][1] = (quantity * price + val * q) / new_q
                storage[product]["sort"][0] = new_q
    
    def _update_best_storage(self, sorted_list, storage, buy_or_sell: str, product):
        """best as in records highest bids and lowest asks.
        """
        q, val = (storage[product][buy_or_sell][0], storage[product][buy_or_sell][1])
        absq = abs(sorted_list[0][1])
        new_total_q = q + absq
        storage[product][buy_or_sell][0] = new_total_q
        if new_total_q > 0:
            storage[product][buy_or_sell][1] = (absq * sorted_list[0][0] + val * q) / new_total_q
        else:
            storage[product][buy_or_sell][1] = 0


    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""
        #MAKE SURE THAT YOU CALL ALL OF YOUR METHODS HERE TO DETERMINE HOW TO BUY/SELL

        # Orders to be placed on exchange matching engine
        result = {}
        if state.traderData:
            storage = jsonpickle.decode(state.traderData)
        else:
            storage = {"ASH_COATED_OSMIUM": {"historical": {}, "sort": [0,0], "buy": [0,0], "sell": [0,0]}, "INTARIAN_PEPPER_ROOT": {"historical": {}, "sort": [0,0], "buy": [0,0], "sell": [0,0]},
                       'pos': {"ASH_COATED_OSMIUM":[0,0],"INTARIAN_PEPPER_ROOT":[0,0]}, 'trend': [[], None]} #list of vals, buyorsell: bool
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            sellsorted = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
            buysorted = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)

            if not buysorted or not sellsorted:
                continue

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

            self._update_best_storage(buysorted, storage, 'buy', product)
            self._update_best_storage(sellsorted, storage, 'sell', product)
            
            if product == 'INTARIAN_PEPPER_ROOT' and storage['trend'][1] is None:
                if len(storage['trend'][0]) < 100:
                    storage['trend'][0].append(mid)
                else:
                    if storage['trend'][1] is None:
                        intercept, slope = self.long_regression(product, storage)
                        current_index = len(storage['trend'][0]) + 100
                        prediction = slope * current_index + intercept
                        if prediction > storage['trend'][0][len(storage['trend'][1]) - 1]:
                            storage['trend'][1] = True
                        else:
                            storage['trend'][1] = False

            if product == 'INTARIAN_PEPPER_ROOT':
                if len(storage[product]['historical']) < 2:
                    continue
                if not len(storage[product]["sort"]) < 1:
                    self.GEOM_trade(state, storage, result, order_depth, buysorted, sellsorted, product, storage['trend'][1])
                print(product, state.position.get(product, 0))

            if product == 'ASH_COATED_OSMIUM':
                if len(storage[product]['historical']) < 2:
                    continue
                if not len(storage[product]["sort"]) < 1:
                    self.MM_trade(state, storage, result, order_depth, buysorted, sellsorted, product)
                print(product, state.position.get(product, 0))

        traderData = jsonpickle.encode(storage)
        conversions = 0
        return result, conversions, traderData
