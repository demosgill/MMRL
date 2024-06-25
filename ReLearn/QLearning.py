import numpy as np
import random
from collections import defaultdict
from ReLearn.MarketEnv import MarketEnv


class QLearner:
    """
    Using Q-Learning to maximise PnL via optimal quoting: MM Problem
    """
    def __init__(self, env, learning_rate=0.1, discount_factor=0.8, exploration_rate=0.1):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_values = defaultdict(lambda: np.zeros(3))
        self.iterations = 100


    def choose_action(self, state):
        """
        Chooses an action based on an epsilon-greedy policy
        """
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice([0, 1, 2])
        else:
            return np.argmax(self.q_values[state])

    def update_q_table(self, state, action, reward, next_state):
        """
        Updates the Q-values using the Bellman equation (see EQ. 3.12 Sutton' book).
        :param state: Current state of things
        :param action: Action one should take
        :param reward: The reward for taking such action
        :param next_state: Update Q-Matrix
        """
        # Get the best future action
        best_future_action = np.argmax(self.q_values[next_state])

        # Get target
        target = reward + self.discount_factor * self.q_values[next_state][best_future_action]

        # increment
        delta = target - self.q_values[state][action]

        # Update q_values
        self.q_values[state][action] += self.learning_rate * delta

    def train(self):
        """
        Train agent
        """
        for episode in range(self.iterations):
            state = self.env.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state

    def evaluate(self):
        """
        Test agent
        """
        cumulative_pnl = 0
        for episode in range(self.iterations):
            state = self.env.reset()
            done = False
            while not done:
                action = np.argmax(self.q_values[state])
                state, reward, done, _ = self.env.step(action)
            cumulative_pnl += self.env.profit_loss
        average_pnl = cumulative_pnl / self.iterations
        print(f"Average PnL over {self.iterations} evaluation episodes: {average_pnl:.2f}")


def main():
    """
    main function for evaluating a given policy via QLearning
    """
    # Instantiate Market environment Class
    EV = MarketEnv()

    # instantiate the QL Class
    AG = QLearner(EV)

    # Train
    AG.train()

    # Test the agent
    AG.evaluate()


if __name__ == "__main__":
    main()

