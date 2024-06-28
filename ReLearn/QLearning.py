import numpy as np
import random
from collections import defaultdict
from ReLearn.MarketEnv import MarketEnv


class QLearner:
    """
    Using Q-Learning to maximise PnL via optimal quoting: MM Problem
    """
    def __init__(self, env, learning_rate=0.2, discount_factor=0.5, exploration_rate=0.9):
        self.env = env  # OB Simulated environment (use gym for more sophistication or RealisticMarketEnv.py)
        self.learning_rate = learning_rate  # what extent newly acquired information overrides old information
        self.discount_factor = discount_factor  # importance of future rewards compared to immediate rewards
        self.exploration_rate = exploration_rate  # Defines the tradeoff between exploration and exploitation
        self.q_values = defaultdict(lambda: np.zeros(3))  # initialized to zeros for each state-action pair.
        self.iterations = 100


    def choose_action(self, state):
        """
        Chooses an action based on an epsilon-greedy policy
        Eg: With probability epsilon (exploration rate), selects a random action from [0, 1, 2]
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

            # Reset the environment to start a new iteration
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
        # Evaluating the trained Q-learning agent over a specified number of episodes to measure its performance
        cumulative_pnl = 0
        for i in range(self.iterations):

            # Reset the environment to start a new iteration
            state = self.env.reset()
            done = False

            while not done:

                # Choose action greedily according to learned Q-values
                action = np.argmax(self.q_values[state])

                # Interact with the environment
                state, reward, done, _ = self.env.step(action)

            # Accumulate the total profit and loss over all iterations
            cumulative_pnl += self.env.profit_loss

        # get avg pnl
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

