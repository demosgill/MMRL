import numpy as np
import random


class MarketEnv:
    """
    Create a simple market environment for testing purposes
    """
    def __init__(self, start_price=100, price_step=1):
        self.start_price = start_price
        self.price_step = price_step
        self.reset()


    def reset(self):
        """
        Environment becomes initial state again
        """
        self.price = self.start_price
        self.inventory = 0
        self.profit_loss = 0
        self.iteration = 0
        return self._get_state()

    def _get_state(self):
        """
        Returns the state of things (price, inventory and PnL)
        """
        return (self.price, self.inventory, self.profit_loss)

    def step(self, action):
        """
        Simulates a single time step, updates inventory, PnL, and market price based on the agent' quoting action.
        """
        bid_price = self.price - action * self.price_step
        ask_price = self.price + action * self.price_step
        transaction = random.choice(['buy', 'sell', 'none'])

        if transaction == 'buy':
            self.inventory += 1
            self.profit_loss -= ask_price
        elif transaction == 'sell':
            self.inventory -= 1
            self.profit_loss += bid_price

        self.price += np.random.normal()
        self.profit_loss += self.inventory * (self.price - self.start_price)
        self.iteration += 1

        done = self.iteration >= 100
        reward = self.profit_loss
        state = self._get_state()

        return state, reward, done, {}

    def display(self):
        print(f"Time: {self.time}, Price: {self.price:.2f}, Inventory: {self.inventory}, PnL: {self.profit_loss:.2f}")

