Sure, here's a description for your README.md file:

---

## Package Overview

This package provides two main modules designed to aid in understanding and applying fundamental concepts in financial modeling and reinforcement learning.

### Modules

1. **Toy Models Collection**
    - A simple example of how to find an optimal path on a graph using Q-Learning
    - A synthetic construction of an order book.

2. **Reinforcement Learning for Market Making**
    - A simplistic example demonstrating the application of Reinforcement Learning (RL) to maximize Profit and Loss (PnL) in a basic market-making scenario.
    - The module employs two popular RL techniques: **Q-Learning** and **SARSA**.
        - **Q-Learning**: An off-policy RL algorithm that seeks to find the best action to take given the current state.
        - **SARSA**: An on-policy RL algorithm that updates the action-value function based on the action actually taken.
    - This example is designed to be an accessible introduction to using RL in financial contexts, showcasing how these methods can be applied to optimize market-making strategies.

### Getting Started

To get started with the package, clone the repository and install the necessary dependencies. Detailed usage instructions and examples are provided within each module to guide you through the implementation and application of the models and techniques.

### Installation

```bash
git clone https://github.com/demosgill/MMRL
```

### Usage

#### Toy Models Collection

Detailed examples and documentation for each toy model can be found in the `toy_models` directory.

```python
from ToyModels.OptimalPathProblem import  main

# Example of how to use Q-Learning to find the optimal path on a given graph
main()
```

#### Reinforcement Learning for Market Making

```python
from ReLearn.QLearning import QLearner
from ReLearn.SARSA import SARSA
from ReLearn.MarketEnv import MarketEnv

# Example usage (Q Learning)
# Instantiate Market Environment
EV = MarketEnv()

# Instantiate QLearner Agent
QL = QLearner(EV)

# Train and Evaluate
QL.train()
QL.evaluate()


# Example usage (SARSA)
EV = MarketEnv()

# Instantiate SARSA Agent
SR = SARSA(EV)

# Train and Evaluate
SR.train()
SR.evaluate()

```

### Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) for more details.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments

Special thanks to all contributors and the open-source community for their valuable input and support.
