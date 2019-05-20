import random
import operator 
import numpy as np 

class BaseAlgorithm(object):
    def __init__(self, arch,params):
        self.arch = arch
        fault_types = ['abort','delay']
        self.fault_injections = []
        operations = self.arch.get_operations()
        for operation in operations:
            if operation.name in ['a1','a2']:
                continue
            for fault in fault_types: 
                self.fault_injections.append(operation.name + '-' + fault) 

    def get_action(self):
        return 0

    def result(self, result):
        pass
    def reset(self):
        pass

class BanditEpsilonAlgorithm(BaseAlgorithm):
    def __init__(self,arch,params):
        super(BanditEpsilonAlgorithm, self).__init__(arch, params)
        self.epsilon = params[0]
        self.gamma = params[1]
        self.min_epsilon = params[2]
        self.original_epsilon = self.epsilon

        self.name = 'bandit-epsilon'
        self.Q = {}
        self.N = {}
        for action in self.fault_injections:
            self.Q[action] = 0
            self.N[action] = 0 
    def get_action(self):
        if random.random() < self.epsilon:
            action = random.choice(list(self.Q.keys())) 
            self.last_action = action
        else: 
            action = max(self.Q.items(), key=operator.itemgetter(1))[0]
            self.last_action = action
        return action

    def result(self,result):
        action = self.last_action
        self.N[action] += 1
        self.Q[action] = self.Q[action] + (1/self.N[action]) * (result - self.Q[action])
        self.epsilon = (self.epsilon - self.min_epsilon) * self.gamma + self.min_epsilon

    def reset(self):
        self.__init__(self.arch,[self.original_epsilon,self.gamma,self.min_epsilon])

class BanditOptimisticAlgorithm(BaseAlgorithm):
    def __init__(self,arch,params):
        super(BanditOptimisticAlgorithm, self).__init__(arch, params)
        self.optimistic = params[0]

        self.name = 'bandit-optimistic'
        self.Q = {}
        self.N = {}
        for action in self.fault_injections:
            self.Q[action] = self.optimistic
            self.N[action] = 0 
    def get_action(self):
        action = max(self.Q.items(), key=operator.itemgetter(1))[0]
        self.last_action = action
        return action

    def result(self,result):
        action = self.last_action
        self.N[action] += 1
        self.Q[action] = self.Q[action] + (1/self.N[action]) * (result - self.Q[action])

    def reset(self):
        self.__init__(self.arch,[self.optimistic])


class QLearningAlgorithm(BaseAlgorithm):
    def __init__(self, arch, params):
        super(QLearningAlgorithm, self).__init__(arch, params)
        self.epsilon = params[0]
        self.gamma = params[1]
        self.learning_rate = params[2]
        self.initial_q = params[3]

        self.name = 'qlearning'

        self.states = {}
        for operation in arch.get_operations():
            inc_deps = self.arch.get_incoming_dependencies(operation)
            if len(inc_deps) == 0:
                continue
            self.states[operation.name] = {
                'Q': {},
                'N': {}
            }

            for dep in inc_deps:
                self.states[operation.name]['Q'][dep.name] = self.initial_q
                self.states[operation.name]['N'][dep.name] = 0

        self.state = random.choice(list(self.states.keys()))
    def get_action(self):
        if np.random.rand() >= self.epsilon:
            action = self.random_argmax(self.states[self.state]['Q'])
        else:
            action = random.choice(list(self.states[self.state]['Q'].keys()))
        self.next_state = action

        fault_types = ['abort','delay']
        fault_type = random.choice(fault_types)
        action = action + '-' + fault_type
        return action

    def result(self,result):
        if self.next_state not in self.states.keys():
            next_state_actual = random.choice(list(self.states.keys()))
        else:
            next_state_actual = self.next_state

        prev_q = self.states[self.state]['Q'][self.next_state]
        self.states[self.state]['N'][self.next_state] += 1
        self.states[self.state]['Q'][self.next_state] = (1-self.learning_rate)*prev_q+self.learning_rate*(result+self.gamma*self.states[next_state_actual]['Q'][self.random_argmax(self.states[next_state_actual]['Q'])])
        self.state = next_state_actual

    def reset(self):
        self.__init__(self.arch,[self.epsilon,self.gamma,self.learning_rate,self.initial_q])

    @staticmethod
    def random_argmax(vector):
        """ Argmax that chooses randomly among eligible maximum indices. """
        mx_tuple = max(vector.items(),key = lambda x:x[1])
        max_list =[i[0] for i in vector.items() if i[1]==mx_tuple[1]]
        return np.random.choice(max_list)

class RandomAlgorithm(BaseAlgorithm):
    def __init__(self, arch,params):
        super(RandomAlgorithm, self).__init__(arch,params)
        self.name = 'random'

    def get_action(self):
        return random.choice(self.fault_injections)