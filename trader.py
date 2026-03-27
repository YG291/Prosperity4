from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string


class Trader:

    def bid(self):
        return 15

    def fair_value(self, state: TradingState, product):
        orders = state.order_depths[product]
        buys = orders.buy_orders
        sells = orders.sell_orders
        fair = 0
        total = abs(sum(buys.values())) + abs(sum(sells.values()))
        for bprice in buys:
            fair += abs(bprice * (buys[bprice] / total))
        for sprice in sells:
            fair += abs(sprice*(sells[sprice]/total))
        return fair

    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""
        #MAKE SURE THAT YOU CALL ALL OF YOUR METHODS HERE TO DETERMINE HOW TO BUY/SELL

        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []  # Order(symbol, price, quantity)
            acceptable_price = self.fair_value(state, product)  # Participant should calculate this value
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(
                len(order_depth.buy_orders)) + ", Sell order depth : " + str(
                len(order_depth.sell_orders)))

            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:  # buying
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))

            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:  # shorting
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))

            result[product] = orders

        traderData = ""  # No state needed - we check position directly
        conversions = 0
        return result, conversions, traderData
