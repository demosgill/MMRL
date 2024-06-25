import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


"""

Reinforcement Learning Toy Problem using Q-Learn

"""

class QLearning:
    def __init__(self):
        self.points_list = [(0, 1), (1, 5), (5, 6), (5, 4), (1, 2), (2, 3), (2, 7)]  # Define points on a 2-D plane
        self.goal_node = 7  # our goal is to reach Number 7 via the best path
        self.matrix_size = 8  # Number of nodes on the graph
        self.gamma = 0.8  # Learning Parameter
        self.initial_state = 1  # Initial point on the graph
        self.Q = np.matrix(np.zeros([self.matrix_size, self.matrix_size]))  # random matrix


    def plot_graph(self):
        # Plot using networkx
        G = nx.Graph()
        G.add_edges_from(self.points_list)
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos)

    def plot_scores(self, scores):
        # Plot scores as a function of iterations
        f, ax = plt.subplots(1,1,figsize=(16,6))
        ax.set_ylabel('Score')
        ax.set_xlabel('Iterations')
        ax.plot(scores)
        plt.show()

    def create_q_learning_matrix(self):
        """
        Creates a reward matrix for taking different paths
        """
        # create matrix x*y
        R = np.matrix(np.ones(shape=(self.matrix_size, self.matrix_size)))
        R *= -1

        # assign zeros to paths and 100 to goal-reaching point
        for point in self.points_list:
            print(point)
            if point[1] == self.goal_node:
                R[point] = 100
            else:
                R[point] = 0

            if point[0] == self.goal_node:
                R[point[::-1]] = 100
            else:
                # reverse of point
                R[point[::-1]] = 0

        # add goal point round trip
        R[self.goal_node, self.goal_node] = 100

        return R

    def available_actions(self, R, state):
        """
        :param R: Q-Learning Matrix with rewards structure
        :return: available actions to take
        """
        current_state_row = R[state,]
        av_act = np.where(current_state_row >= 0)[1]
        return av_act

    def sample_next_action(self, available_act):
        """
        :param available_act: available actions to take
        :return: next action to be taken
        """
        next_action = int(np.random.choice(available_act, 1))
        return next_action

    def update(self, R, current_state, action):
        """
        :param R: State Matrix
        :param current_state: Current state of the payout
        :param action: Action to take
        :return: Return resulting payout
        """

        # Get max index
        max_index = np.where(self.Q[action,] == np.max(self.Q[action,]))[1]

        # Randomly select the next choice
        if max_index.shape[0] > 1:
            max_index = int(np.random.choice(max_index, size=1))
        else:
            max_index = int(max_index)

        # Get the action with the max value / loc
        max_value = self.Q[action, max_index]

        # Update the policy accordingly
        self.Q[current_state, action] = R[current_state, action] + self.gamma * max_value
        print('max_value', R[current_state, action] + self.gamma * max_value)

        # Return maxpayout for optimal path
        if (np.max(self.Q) > 0):
            return (np.sum(self.Q / np.max(self.Q) * 100))
        else:
            return (0)

    def update_rule(self):
        """
        Single step example for updating state
        """
        # Get R Matrix
        R = self.create_q_learning_matrix()

        # Get available actions
        state = 1
        available_act = self.available_actions(R, state)

        # sample next action to take
        action = self.sample_next_action(available_act)

        # Update state
        return self.update(R, self.initial_state, action)


def main():
    """
    Main method for running the toy model example
    """
    # Instantiate
    QL = QLearning()

    # Which node is our objective?
    print(f'Algorithm Aim: Achieve Noce {QL.goal_node}')

    # Plot the graph with paths the algorithms can follow.
    QL.plot_graph()


    # Get R Matrix
    R = QL.create_q_learning_matrix()

    # Training
    scores = []
    for i in range(700):
        current_state = np.random.randint(0, int(QL.Q.shape[0]))
        available_act = QL.available_actions(R, current_state)
        action = QL.sample_next_action(available_act)
        score = QL.update(R, current_state, action)
        scores.append(score)
        print('Score:', str(score))

    print("Trained Q matrix:")
    print(QL.Q / np.max(QL.Q) * 100)

    # Testing
    current_state = 0
    steps = [current_state]

    while current_state != 7:

        next_step_index = np.where(QL.Q[current_state,] == np.max(QL.Q[current_state,]))[1]

        if next_step_index.shape[0] > 1:
            next_step_index = int(np.random.choice(next_step_index, size=1))
        else:
            next_step_index = int(next_step_index)

        steps.append(next_step_index)
        current_state = next_step_index

    print("Most efficient path:")
    print(steps)

    # Plot Results
    QL.plot_scores(scores)


if __name__ == "__main__":
    main()

