import numpy as np
import pprint
import sys
import matplotlib.pyplot as plt

from multiprocessing import Pool
from collections import defaultdict


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


# Global values defined, so that can be used while running using multithreading

s = 5
start_action = 0

start_state = s*s - s
# epsilon
#epsilon = 0

# learning rate
#alpha = 0.1

# experiments
exps = 500

#epsiodes
eps = 500

#gamma
gamma = 0.99

def mutithread(lambda_parameter):
    qlearn = QLearning()
    #qlearn.q_learning(g, start_state, alpha, eps, gamma, epsilon, exps, s)
    qlearn.sarsa_lambda(g, start_state, 0.1, eps, gamma, 0.1, exps, s, lambda_parameter)

class GridworldEnv():
    """
    You are an agent on an s x s grid and your goal is to reach the terminal
    state at the top right corner.
    For example, a 4x4 grid looks as follows:
    o  o  o  T
    o  o  o  o
    o  o  o  o
    x  o  o  o

    x is your position and T is the terminal state.
    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -0.1 at each step until you reach a terminal state.
    """

    def __init__(self, size):
        shape = [size, size]
        self.shape = shape

        nS = np.prod(shape) # The area of the gridworld
        MAX_Y = shape[0]
        MAX_X = shape[1]
        nA = 4  # There are four possible actions
        self.P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex    # s is the current position id. s = y * 4 + x
            y, x = it.multi_index

            self.P[s] = {a : [] for a in range(nA)}

            is_done = lambda s: s == shape[1] - 1
            reward = 5.0 if is_done(s) else -0.1

            # We're stuck in a terminal state
            if is_done(s):
                self.P[s][UP] = [(s, reward, True)]
                self.P[s][RIGHT] = [(s, reward, True)]
                self.P[s][DOWN] = [(s, reward, True)]
                self.P[s][LEFT] = [(s, reward, True)]
            # Not a terminal state, and if the agent ’bump into the wall’, it will stay in the same state
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1
                self.P[s][UP] = [(ns_up, reward, is_done(ns_up))]
                self.P[s][RIGHT] = [(ns_right, reward, is_done(ns_right))]
                self.P[s][DOWN] = [(ns_down, reward, is_done(ns_down))]
                self.P[s][LEFT] = [(ns_left, reward, is_done(ns_left))]
            it.iternext()


    # The possible action has a 0.8 probability of succeeding
    def action_success(self, success_rate = 0.8):
        return np.random.choice(2, 1, p=[1-success_rate, success_rate])[0]

    # If the action fails, any action is chosen uniformly(including the succeeding action)
    def get_action(self, action):
        if self.action_success():
            return action
        else:
            random_action = np.random.choice(4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
            return random_action

    # Given the current position, this function outputs the position after the action.
    def move(self, s, action):
        return self.P[s][action]


class Learning:
    '''

    Class which implements
    the Q learning

    '''

    def get_next_action(self, epsilon, current_state, Q_lookup):

        '''

        Chooses next state using greedy epsilon

        '''
        available_actions = [0, 1, 2, 3]

        if np.random.rand() <= epsilon:
            action_list = np.ones(4)
            all_actions = np.divide( action_list, 4)

        else:
            optimal_action = np.argmax(Q_lookup[current_state])
            all_actions = np.zeros(len(available_actions))
            all_actions[optimal_action] = 1

        return all_actions

    def get_max_action(self, current_state, Q_lookup):
        available_actions = [0, 1, 2, 3]
        optimal_action = np.argmax(Q_lookup[current_state])
        all_actions = np.zeros(len(available_actions))
        all_actions[optimal_action] = 1

        return all_actions

    def plot_graph(self, num_dict, title):

        y_axis = []

        for i in range(len(num_dict)):
            y_axis.append(np.average(num_dict[i]))

        x_axis = np.arange(1, len(num_dict)+1)

        plt.figure(1)

        plt.plot(x_axis, y_axis)
        plt.xlabel('Episode Number')
        plt.ylabel('Average time steps')
        plt.title(title)
        plt.savefig("data/1ep_{}_a.png".format(title))


        #plt.show()

    def plot_graph_qa(self, plot_second, title):

        y_axis = []

        for i in range(len(plot_second)):
            y_axis.append(np.max(plot_second[i]))

        x_axis = np.arange(1, len(plot_second)+1)

        plt.figure(2)

        plt.plot(x_axis, y_axis)
        plt.xlabel('Episode Number')
        plt.ylabel('Maximum Q value')
        plt.title(title)
        plt.savefig("data/1ep_{}_b.png".format(title))


    def q_learning(self, environment, start_state, alpha, eps, gamma, epsilon, exps, s):
        '''

        '''
        available_actions = [0,1,2,3]

        actions = len(available_actions)
        max_episodes = eps
        experiments = exps
        #max_episodes = 10
        #experiments = 10

        terminal_state = s - 1
        start_state = s*s - s

        num_dict = defaultdict(lambda: np.zeros(experiments))
        plot_second = defaultdict(lambda: np.zeros(experiments))

        for experiment in range(experiments):
            Q_lookup = defaultdict(lambda: np.zeros(actions))
            total_reward = 0

            for episode in range(max_episodes):

                current_state = start_state
                num_steps = 0
                while True:
                    # Choose this action using greedy epsilon
                    all_action = self.get_next_action(epsilon, current_state, Q_lookup)
                    next_action = np.random.choice(np.arange(4), p=all_action)

                    # Get them from the domain

                    next_state_information = environment.move(current_state, environment.get_action(next_action))

                    next_state = next_state_information[0][0]
                    reward = next_state_information[0][1]
                    is_terminal = next_state_information[0][2]

                    if is_terminal:
                        reward = 4.9
                    else:
                        pass

                    new_value = reward + gamma*np.max(Q_lookup[next_state])

                    Q_lookup[current_state][next_action] = Q_lookup[current_state][next_action] + \
                                            alpha*(new_value -\
                                            Q_lookup[current_state][next_action])

                    if next_state == terminal_state:
                        num_dict[episode][experiment] = num_steps
                        plot_second[episode][experiment] = \
                        np.max(Q_lookup[start_state])
                        break

                    current_state = next_state
                    num_steps += 1


            print("experiment ", experiment)
            #pprint.pprint(Q_lookup)

        title = "Alpha {} and Epsilon {} and lambda {}".format(alpha, epsilon, lambda_parameter)

        #print("experiment", experiment)
        #pprint.pprint(Q_lookup)
        #print("total reward: ", total_reward)
        #print(num_dict)
        self.plot_graph(num_dict, title)
        #print(plot_second)
        self.plot_graph_qa(plot_second, title)
        #self.together(plot_second, num_dict, title)
        #return plot_second, num_dict, title


    def sarsa_lambda(self, environment, start_state, alpha, eps, gamma, epsilon, exps, s, lambda_parameter):
        '''

        Sarsa lambda

        '''
        available_actions = [0,1,2,3]

        actions = len(available_actions)
        max_episodes = eps
        experiments = exps

        terminal_state = s - 1
        start_state = s*s - s

        num_dict = defaultdict(lambda: np.zeros(experiments))
        plot_second = defaultdict(lambda: np.zeros(experiments))

        for experiment in range(experiments):
            Q_lookup = defaultdict(lambda: np.zeros(actions))
            total_reward = 0

            for episode in range(max_episodes):
                E = defaultdict(lambda: np.zeros(actions))
                current_state = start_state
                num_steps = 0
                while True:
                    # Choose this action using greedy epsilon
                    all_action = self.get_next_action(epsilon, current_state, Q_lookup)
                    next_action = np.random.choice(np.arange(4), p=all_action)

                    # Get them from the domain

                    next_state_information = environment.move(current_state, environment.get_action(next_action))

                    next_state = next_state_information[0][0]
                    reward = next_state_information[0][1]
                    is_terminal = next_state_information[0][2]

                    if is_terminal:
                        reward = 4.9
                    else:
                        pass

                    new_all_action = self.get_next_action(epsilon, next_state, Q_lookup)
                    new_action = np.random.choice(np.arange(4), p=new_all_action)

                    delta = reward + gamma*Q_lookup[next_state][new_action] - Q_lookup[current_state][next_action]

                    E[current_state][next_action] =\
                    E[current_state][next_action] + 1

                    # For up action 0
                    for state in Q_lookup:
                        Q_lookup[state][UP] = Q_lookup[state][UP] + alpha*delta*E[state][UP]
                        E[state][UP] = gamma*lambda_parameter*E[state][UP]

                    # For right action
                    for state in Q_lookup:
                        Q_lookup[state][RIGHT] = Q_lookup[state][RIGHT] +\
                        alpha*delta*E[state][RIGHT]
                        E[state][RIGHT] = gamma*lambda_parameter*E[state][RIGHT]

                    # For left action
                    for state in Q_lookup:
                        Q_lookup[state][LEFT] = Q_lookup[state][LEFT] +\
                        alpha*delta*E[state][LEFT]
                        E[state][LEFT] = gamma*lambda_parameter*E[state][LEFT]

                    # For down action
                    for state in Q_lookup:
                        Q_lookup[state][DOWN] = Q_lookup[state][DOWN] +\
                        alpha*delta*E[state][DOWN]
                        E[state][DOWN] = gamma*lambda_parameter*E[state][DOWN]


                    if next_state == terminal_state:
                        num_dict[episode][experiment] = num_steps
                        plot_second[episode][experiment] =\
                        np.max(Q_lookup[start_state])
                        break

                    current_state = next_state
                    num_steps += 1

            print("experiment ", experiment)
            #pprint.pprint(Q_lookup)

        #print("experiment", experiment)
        #pprint.pprint(Q_lookup)
        #print("total reward: ", total_reward)
        #print(num_dict)
        self.plot_graph(num_dict)
        #print(plot_second)
        #self.plot_graph_qa(plot_second)


if __name__ == "__main__":


    # size

    s = 5
    g = GridworldEnv(s)

    start_state = s*s - s

    # epsilon
    epsilon = 1

    # learning rate
    alpha = 1

    # experiments
    exps = 500
    #epsiodes
    eps = 500
    #gamma
    gamma = 0.99

    #lambda
    lambda_parameter = 0.75

    # size of the grid
    learn = Learning()

    #p = Pool(100)
    #p.map(multithread, [0, 0.25, 0.5, 0.75])

    # Call Q learning

    learn.q_learning(g, start_state, alpha, eps, gamma, 1, exps, s)

    # Sarsa lambda

    #learn.sarsa_lambda(g, start_state, alpha, eps, gamma, epsilon, exps, s, lambda_parameter)


