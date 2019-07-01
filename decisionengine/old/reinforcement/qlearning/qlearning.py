import numpy as np 

class QAgent(object):
    def __init__(self, states, epsilon, gamma, learning_rate, initial_q):
        self.states = {}

        for state in states.keys():
            self.states[state] = {
                'Q': [initial_q] * len(states[state]),
                'N': [0] * len(states[state])
            }
        self.epsilon = epsilon
        self.gamma = gamma 
        self.learning_rate = learning_rate 


    def get_action(self, s):
        if np.random.rand() >= self.epsilon:
            action = self.random_argmax(self.states[s]['Q'])
        else:
            action = np.random.randint(len(self.states[s]['Q']))

        return action

    def reward(self, state, action, new_state, reward):
        self.states[state]['N'][action] += 1
        self.states[state]['Q'][action] = (1-self.learning_rate)*self.states[state]['Q'][action]+self.learning_rate*(reward+self.gamma*self.random_argmax(self.states[new_state]['Q']))
        

    @staticmethod
    def random_argmax(vector):
        """ Argmax that chooses randomly among eligible maximum indices. """
        m = np.amax(vector)
        indices = np.nonzero(vector == m)[0]
        return np.random.choice(indices)

    def print_Q(self):
        print(self.states)