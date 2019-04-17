import numpy as np 

class TableauClassifier(object):
    def __init__(self, epsilon, gamma, learning_rate, initial_q, action_size):
        self.epsilon = epsilon
        self.gamma = gamma 
        self.learning_rate = learning_rate 
        self.action_size = action_size
        self.states = {}
        self.initial_q = initial_q
        self.min_epsilon = 0.1

        self.action_history = []
        self.train_mode = True

    def get_action(self, s):
        if s not in self.states:
            self.states[s] = {
                'Q': [self.initial_q] * self.action_size,
                'N': [0] * self.action_size
            }

        if np.random.rand() >= self.epsilon:
            action = self.random_argmax(self.states[s]['Q'])
        else:
            action = np.random.randint(self.action_size)

        if self.train_mode:
            self.action_history.append((s, action))

        return action

    def reward(self, rewards):
        if not self.train_mode:
            return

        try:
            x = float(rewards)
            rewards = [x] * len(self.action_history)
        except:
            if len(rewards) < len(self.action_history):
                raise Exception('Too few rewards')

        # Update Q
        for ((state, act_idx), reward) in zip(self.action_history, rewards):
            self.states[state]['N'][act_idx] += 1
            n = self.states[state]['N'][act_idx]
            prev_q = self.states[state]['Q'][act_idx]
            self.states[state]['Q'][act_idx] = prev_q + 1.0 / n * (reward - prev_q)
            # self.states[state]['Q'][act_idx] = prev_q + self.learning_rate * (reward - prev_q)
            #self.states[state]['Q'][act_idx] = (1-self.learning_rate)*prev_q+self.learning_rate*(reward+self.gamma*)
            
        self.reset_action_history()
        self.epsilon = (self.epsilon - self.min_epsilon) * self.gamma + self.min_epsilon


    def reset_action_history(self):
        self.action_history = []

    @staticmethod
    def random_argmax(vector):
        """ Argmax that chooses randomly among eligible maximum indices. """
        m = np.amax(vector)
        indices = np.nonzero(vector == m)[0]
        return np.random.choice(indices)
