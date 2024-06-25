import numpy as np
import random
from collections import defaultdict
from ReLearn.MarketEnv import MarketEnv


class QLearner:
    """
    Using Q-Learning to maximise PnL via optimal quoting: MM Problem
    """
    def __init__(self, env, learning_rate=0.1, discount_factor=0.8, exploration_rate=0.1):
        self.env = env  # OB Simulated environment (use gym for more sophistication or syntheticOB.py)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_values = defaultdict(lambda: np.zeros(3))  # initialized to zeros for each state-action pair.
        self.iterations = 100


    def choose_action(self, state):
        """
        Chooses an action based on an epsilon-greedy policy
        E.g. With probability epsilon (exploration rate), selects a random action from [0, 1, 2]
        Otherwise, selects the action with the highest Q-value for the given state
        """
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice([0, 1, 2])
        else:
            return np.argmax(self.q_values[state])

    def update_q_table(self, state, action, reward, next_state):
        """
        Updates the Q-values using the Bellman equation (see EQ. 3.12 Sutton' book).
        """
        # Find the best action for the next state
        best_future_action = np.argmax(self.q_values[next_state])

        # Calculate the target Q-value
        target = reward + self.discount_factor * self.q_values[next_state][best_future_action]

        # Calculate the temporal difference error
        delta = target - self.q_values[state][action]

        # Update the Q-value for the current state-action pair
        self.q_values[state][action] += self.learning_rate * delta

    def train(self):
        """
        Train agent
        """
        # Training the Q-learning agent over a specified number of iterations
        for i in range(self.iterations):

            # Reset the environment to start a new episode
            state = self.env.reset()
            done = False

            while not done:
                # Choose action using epsilon-greedy policy
                action = self.choose_action(state)

                # Interact with the environment
                next_state, reward, done, _ = self.env.step(action)

                # Update Q-values based on observed transition
                self.update_q_table(state, action, reward, next_state)

                # Define next state
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

