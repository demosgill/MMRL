import heapq
import random


"""
Example of a synthetic order book construction
"""
class Order:
    def __init__(self, order_id, side, price, quantity):
        self.order_id = order_id
        self.side = side
        self.price = price
        self.quantity = quantity

    def __lt__(self, other):
        if self.side == 'buy':
            return self.price > other.price  # Max heap for buy orders
        else:
            return self.price < other.price  # Min heap for sell orders


class OrderBook:
    def __init__(self):
        self.buy_orders = []
        self.sell_orders = []
        self.order_count = 0

    def add_order(self, side, price, quantity):
        self.order_count += 1
        order = Order(self.order_count, side, price, quantity)
        if side == 'buy':
            heapq.heappush(self.buy_orders, order)
        else:
            heapq.heappush(self.sell_orders, order)
        self.match_orders()

    def match_orders(self):
        while self.buy_orders and self.sell_orders and self.buy_orders[0].price >= self.sell_orders[0].price:
            buy_order = heapq.heappop(self.buy_orders)
            sell_order = heapq.heappop(self.sell_orders)
            matched_quantity = min(buy_order.quantity, sell_order.quantity)
            buy_order.quantity -= matched_quantity
            sell_order.quantity -= matched_quantity
            print(
                f"Matched {matched_quantity} units at price {sell_order.price} between Buy Order {buy_order.order_id} and Sell Order {sell_order.order_id}")

            if buy_order.quantity > 0:
                heapq.heappush(self.buy_orders, buy_order)
            if sell_order.quantity > 0:
                heapq.heappush(self.sell_orders, sell_order)

    def print_order_book(self):
        print("\nOrder Book:")
        print("Buy Orders:")
        for order in self.buy_orders:
            print(f"ID: {order.order_id}, Price: {order.price}, Quantity: {order.quantity}")
        print("Sell Orders:")
        for order in self.sell_orders:
            print(f"ID: {order.order_id}, Price: {order.price}, Quantity: {order.quantity}")


if __name__ == "__main__":
    order_book = OrderBook()
    sides = ['buy', 'sell']
    for _ in range(10):
        side = random.choice(sides)
        price = random.uniform(90, 110)
        quantity = random.randint(1, 10)
        order_book.add_order(side, price, quantity)
        order_book.print_order_book()

