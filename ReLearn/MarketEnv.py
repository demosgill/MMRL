import numpy as np
import random


class MarketEnv:
    """
    Create a very simple LOB
    """
    def __init__(self, start_price=100, price_step=1):
        self.start_price = start_price
        self.price_step = price_step
        self.max_iterations = 100
        self.inventory = 0
        self.profit_loss = 0
        self.iteration = 0
        self.price = 0
        self.reset()


    def reset(self):
        """
        Reset env. vars.
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

    def get_vars(self):
        """
        Helper method for forwarding info fown to learners
        """
        done = self.iteration >= self.max_iterations
        reward = self.profit_loss
        state = self._get_state()

        return state, reward, done, {}

    def step(self, action):
        """
        Simulates a single time step, updates inventory, PnL, and market price based on the agent' quoting action.
        """
        bid_price = self.price - action * self.price_step
        ask_price = self.price + action * self.price_step
        transaction = random.choice(['buy', 'sell', 'none'])

        # If we buy
        if transaction == 'buy':
            self.inventory += 1  # Inventory increases
            self.profit_loss -= ask_price

        # If we sell
        elif transaction == 'sell':
            self.inventory -= 1  # Inventory decreases
            self.profit_loss += bid_price

        # price from iid(0,1)
        self.price += np.random.normal()
        self.profit_loss += self.inventory * (self.price - self.start_price)
        self.iteration += 1

        return self.get_vars()

    def display(self):
        print(f"Price: {self.price:.2f}, Inventory: {self.inventory}, PnL: {self.profit_loss:.2f}")

