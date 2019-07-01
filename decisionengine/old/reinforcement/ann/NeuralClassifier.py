import numpy as np
import os
from sklearn import neural_network

try:
    import cPickle as pickle
except:
    import pickle

class NeuralClassifier(object):
    def __init__(self, hidden_size):
        self.single_testcases = True
        self.train_mode = True 
        self.name = 'mlpclassifier'
        self.experience_length = 10000
        self.experience_batch_size = 1000
        self.experience = ExperienceReplay(max_memory=self.experience_length)
        self.episode_history = []
        self.iteration_counter = 0

        if isinstance(hidden_size, tuple):
            self.hidden_size = hidden_size
        else:
            self.hidden_size = (hidden_size,)
        self.model = None
        self.model_fit = False
        self.init_model(True)

    def init_model(self, warm_start=True):
        self.model = neural_network.MLPClassifier(hidden_layer_sizes=self.hidden_size, activation='relu',
                                                      warm_start=warm_start, solver='adam', max_iter=750)
        self.model_fit = False
    
    def get_action(self, s):
        if self.model_fit:
            a = self.model.predict_proba(np.array(s).reshape(1, -1))[0][1]

        else:
            a = np.random.random()

        if self.train_mode:
            self.episode_history.append((s, a))

        return a

    def reward(self, rewards):
        if not self.train_mode:
            return

        try:
            x = float(rewards)
            rewards = [x] * len(self.episode_history)
        except:
            if len(rewards) < len(self.episode_history):
                raise Exception('Too few rewards')

        self.iteration_counter += 1

        for ((state, action), reward) in zip(self.episode_history, rewards):
            self.experience.remember((state, reward))

        self.episode_history = []

        if self.iteration_counter == 1 or self.iteration_counter % 5 == 0:
            self.learn_from_experience()
    
    def learn_from_experience(self):
        experiences = self.experience.get_batch(self.experience_batch_size)
        x, y = zip(*experiences)
        if self.model_fit:
            try:
                self.model.partial_fit(x, y)
            except ValueError:
                self.init_model(warm_start=False)
                self.model.fit(x, y)
                self.model_fit = True
        else:
            self.model.fit(x, y)  # Call fit once to learn classes
            self.model_fit = True


class ExperienceReplay(object):
    def __init__(self, max_memory=5000, discount=0.9):
        self.memory = []
        self.max_memory = max_memory
        self.discount = discount

    def remember(self, experience):
        self.memory.append(experience)

    def get_batch(self, batch_size=10):
        if len(self.memory) > self.max_memory:
            del self.memory[:len(self.memory) - self.max_memory]

        if batch_size < len(self.memory):
            timerank = range(1, len(self.memory) + 1)
            p = timerank / np.sum(timerank, dtype=float)
            batch_idx = np.random.choice(range(len(self.memory)), replace=False, size=batch_size, p=p)
            batch = [self.memory[idx] for idx in batch_idx]
        else:
            batch = self.memory

        return batch