import numpy as np
import random
from collections import defaultdict
from MarketEnv import MarketEnv

class SARSA:
    """
    SARSA agent for the Market Making problem
    """
    def __init__(self, env, alpha=0.1, gamma=0.8, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(3))


    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1, 2])
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, next_action):
        target = reward + self.gamma * self.q_table[next_state][next_action]
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            done = False

            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.choose_action(next_state)
                self.update_q_table(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action

    def evaluate(self, episodes=100):
        total_pnl = 0
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = np.argmax(self.q_table[state])
                state, reward, done, _ = self.env.step(action)
            total_pnl += self.env.profit_loss
        average_pnl = total_pnl / episodes
        print(f"Average PnL over {episodes} evaluation episodes: {average_pnl:.2f}")

def main():
    # Instantiate Market environment Class
    env = MarketEnv()

    # Instantiate the SARSA Agent
    agent = SARSA(env)

    # Train the agent
    agent.train()

    # Evaluate the agent
    agent.evaluate()

if __name__ == "__main__":
    main()

