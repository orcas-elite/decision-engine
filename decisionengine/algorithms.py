import random
import operator 
import numpy as np 
from sklearn import neural_network
from queue import *

class BaseAlgorithm(object):
    def __init__(self, arch,params):
        self.params = params
        self.arch = arch
        fault_types = ['abort','delay']
        self.fault_injections = []
        operations = self.arch.get_operations()
        for operation in operations:
            if operation.name in ['a1','a2']:
                continue
            for fault in fault_types: 
                self.fault_injections.append(operation.name + '-' + fault) 

    def get_experiment(self):
        return 0

    def result(self, result):
        pass
    def reset(self):
        self.__init__(self.arch,self.params)

class BayesianLowAlgorithm(BaseAlgorithm):
    def __init__(self,arch,params):
        super(BayesianLowAlgorithm, self).__init__(arch,params)
        self.name = 'bayes-low'
        self.counters = []
        self.probs = []
        self.fault_injections = []
        self.train_mode = True 
        self.iterations = 0
        self.training_limit = 100

        operations = self.arch.get_operations()
        faults = ['abort','delay']
        for op in operations:
            if op.name not in ['a1','a2']:
                for fault in faults:
                    self.fault_injections.append(op.name + '-' + fault)
                    self.counters.append([0,0])
                    self.probs.append(1/((2*len(operations))-4))

    def get_experiment(self):
        choice = np.random.choice(range(len(self.fault_injections)), 1, replace=True, p=self.probs)
        self.last_injection = self.fault_injections[choice[0]]
        self.last_index = choice[0]

        self.iterations += 1
        if self.iterations >= self.training_limit: 
            self.train_mode = False
        return self.last_injection

    def result(self,result):
        if result > 0:
            self.counters[self.last_index][0] += 1
        self.counters[self.last_index][1] += 1

        if not self.train_mode:
            if self.counters[self.last_index][0] > 0:
                self.probs[self.last_index] = self.counters[self.last_index][0] / self.counters[self.last_index][1]
                sum_probs = sum(self.probs)
                for i in range(len(self.probs)):
                    self.probs[i] = self.probs[i] / sum_probs
class BayesianHighAlgorithm(BaseAlgorithm):
    def __init__(self,arch,params):
        super(BayesianHighAlgorithm, self).__init__(arch,params) 
        self.name = 'bayes-high'
        self.feature_list = []
        feature_init = []
        self.fault_injections = {}
        self.counters = []
        self.probs = []
        self.train_mode = True 
        self.iterations = 0
        self.training_limit = 100

        seen = []
        operations = self.arch.get_operations() 
        faults = ['abort','delay']
        for op in operations:
            if op.name not in ['a1','a2']:
                for fault in faults:
                    features = self.extract_features(op,fault)
                    if features not in feature_init:
                        feature_init.append(features)
                    self.fault_injections[op.name + '-' + fault] = features 
        

        for feature in feature_init:
            self.feature_list.append(feature)
            self.counters.append([0,0])
            self.probs.append(1/len(feature_init)) 
        
    def get_experiment(self):
        choice = np.random.choice(range(len(self.feature_list)), 1, replace=True, p=self.probs) 
        all_matches = [] 
        for key,value in self.fault_injections.items():
            if value == self.feature_list[choice[0]]:
                all_matches.append(key)
        
        self.last_injection = random.choice(all_matches)
        self.last_index = choice[0]

        self.iterations += 1
        if self.iterations >= self.training_limit: 
            self.train_mode = False
        return self.last_injection

    def result(self,result):
        if result > 0:
            self.counters[self.last_index][0] += 1
        self.counters[self.last_index][1] += 1

        if not self.train_mode:
            if self.counters[self.last_index][0] > 0:
                self.probs[self.last_index] = self.counters[self.last_index][0] / self.counters[self.last_index][1]
                sum_probs = sum(self.probs)
                for i in range(len(self.probs)):
                    self.probs[i] = self.probs[i] / sum_probs

    def extract_features(self,op,fault):
        features = [] 
        if fault == 'abort':
            features.append(0)
        else:
            features.append(1) 
        if op.circuitbreaker == None:
            features.append(0)
        else:
            features.append(1)
        features.append(len(op.dependencies)) 

        return features

class BanditEpsilonAlgorithm(BaseAlgorithm):
    def __init__(self,arch,params):
        super(BanditEpsilonAlgorithm, self).__init__(arch, params)
        self.epsilon = params[0]
        self.gamma = params[1]
        self.min_epsilon = params[2]
        self.original_epsilon = self.epsilon

        self.name = 'bandit-eps'
        self.Q = {}
        self.N = {}
        for action in self.fault_injections:
            self.Q[action] = 0
            self.N[action] = 1
    def get_experiment(self):
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
        prev_q = self.Q[action] 
        self.Q[action] = prev_q + (1/self.N[action]) * (result - prev_q)
        self.epsilon = (self.epsilon - self.min_epsilon) * self.gamma + self.min_epsilon

    def reset(self):
        self.__init__(self.arch,[self.original_epsilon,self.gamma,self.min_epsilon])

class BanditOptimisticAlgorithm(BaseAlgorithm):
    def __init__(self,arch,params):
        super(BanditOptimisticAlgorithm, self).__init__(arch, params)
        self.optimistic = params[0]

        self.name = 'bandit-opt'
        self.Q = {}
        self.N = {}
        for action in self.fault_injections:
            self.Q[action] = self.optimistic
            self.N[action] = 1
    def get_experiment(self):
        # TODO: GEt all the max actions, selection randomly from all 
        action = max(self.Q.items(), key=operator.itemgetter(1))[0]
        self.last_action = action
        return action

    def result(self,result):
        action = self.last_action
        prev_q = self.Q[action]
        self.N[action] += 1
        self.Q[action] = prev_q + (1/self.N[action]) * (result - prev_q)


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
    def get_experiment(self):
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


class TableauAlgorithm(BaseAlgorithm):
    def __init__(self,arch,params):
        super(TableauAlgorithm, self).__init__(arch,params)
        self.name = 'tableau'
        self.epsilon = params[0]
        self.gamma = params[1]
        self.initial_q = params[2]
        self.min_epsilon = params[3]
        self.action_size = params[4]

        self.fault_injections = {}
        self.states = {}
        operations = self.arch.get_operations()
        faults = ['abort','delay']
        for operation in operations:
            if operation.name not in ['a1','a2']:
                for fault in faults:
                    features = self.extract_features(operation,fault)
                    self.states[str(features)] = {
                        'Q': [self.initial_q] * self.action_size,
                        # initialize N at 1 to prevent the first result setting Q to 0 
                        'N': [1] * self.action_size
                    }
                    self.fault_injections[operation.name + '-' + fault] = str(features)
                    
    def get_experiment(self):
        queues = []
        for i in range(self.action_size):
            queues.append([])
        for s in self.states:
            if np.random.rand() >= self.epsilon:
                action = self.random_argmax(self.states[s]['Q'])
            else: 
                action = np.random.randint(self.action_size)
            queues[action].append(s)
        
        for i in range(self.action_size):
            if len(queues[i]) != 0:
                self.last_state = random.choice(queues[i])
                self.last_action = i

                matching_injections = []
                for key,value in self.fault_injections.items():
                    if value == self.last_state:
                        matching_injections.append(key)
                fault_injection = random.choice(matching_injections)
                return fault_injection

    def result(self, result):
        # Update Q
        self.states[self.last_state]['N'][self.last_action] += 1
        n = self.states[self.last_state]['N'][self.last_action]
        prev_q = self.states[self.last_state]['Q'][self.last_action]
        self.states[self.last_state]['Q'][self.last_action] = prev_q + 1.0 / n * (result - prev_q)
        # self.states[state]['Q'][act_idx] = prev_q + self.learning_rate * (reward - prev_q)
        #self.states[state]['Q'][act_idx] = (1-self.learning_rate)*prev_q+self.learning_rate*(reward+self.gamma*)
            
        self.epsilon = (self.epsilon - self.min_epsilon) * self.gamma + self.min_epsilon

    def reset(self):
        self.__init__(self.arch,self.params)

    @staticmethod
    def random_argmax(vector):
        """ Argmax that chooses randomly among eligible maximum indices. """
        m = np.amax(vector)
        indices = np.nonzero(vector == m)[0]
        return np.random.choice(indices)

    def extract_features(self,operation,fault):
        features = []

        if operation.circuitbreaker != None:
            features.append(1)
        else:
            features.append(0)
        
        features.append(len(operation.dependencies))

        if fault == 'abort':
            features.append(0)
        else:
            features.append(1)
        return features

class NeuralNetworkAlgorithm(BaseAlgorithm):
    def __init__(self,arch,params):
        super(NeuralNetworkAlgorithm, self).__init__(arch,params)
        self.name = 'ann'
        self.train_mode = True 
        self.experience_length = 10000
        self.experience_batch_size = 1000
        self.experience = ExperienceReplay(max_memory=self.experience_length)
        self.episode_history = []
        self.iteration_counter = 0

        self.hidden_size = params[0]
        self.model = None
        self.model_fit = False
        self.init_model(True)

        self.states = {}
        operations = self.arch.get_operations() 
        faults = ['abort','delay']

        for operation in operations: 
            if operation.name not in ['a1','a2']:
                for fault in faults: 
                    self.states[str(operation.name + '-' + fault)] = self.extract_features(operation, fault)
        

    def init_model(self, warm_start=True):
        self.model = neural_network.MLPClassifier(hidden_layer_sizes=self.hidden_size, activation='relu',
                                                      warm_start=warm_start, solver='adam', max_iter=750)
        self.model_fit = False
    
    def get_experiment(self):
        probs = {}
        for state in self.states.keys():
            features = self.states[state]
            if self.model_fit: 
                p = self.model.predict_proba(np.array(tuple(features)).reshape(1,-1))[0][1]
            else:
                p = np.random.random() 
            probs[state] = p

        total_prob = sum(probs.values())
        keys = [] 
        weights = []
        for key in probs.keys():
            keys.append(key)
            weights.append(probs[key] / total_prob)
        self.last_state = random.choices(keys, weights = weights, k = 1)[0]
        return self.last_state

    def result(self, result):
        if not self.train_mode:
            return


        self.iteration_counter += 1

        features = self.states[self.last_state]

        self.experience.remember((features, result))


        self.states[self.last_state][6] = self.states[self.last_state][5]
        self.states[self.last_state][5] = self.states[self.last_state][4]
        self.states[self.last_state][4] = self.states[self.last_state][3]
        self.states[self.last_state][3] = result

        if self.iteration_counter == 1 or self.iteration_counter % 5 == 0:
            self.learn_from_experience()

    def reset(self):
        self.__init__(self.arch,self.params)

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
            self.model.fit(x,y)  # Call fit once to learn classes
            self.model_fit = True

    def extract_features(self,operation,fault):
        features = []

        if operation.circuitbreaker != None:
            features.append(1)
        else:
            features.append(0)
        
        features.append(len(operation.dependencies))
        features.append(operation.service.instances)
        history = []
        for i in range(4):
            history.append(random.choice([0,1]))

        features.extend(history) 
        if fault == 'abort':
            features.append(0)
        else:
            features.append(1)
        return features 
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

class MLFQAlgorithm(BaseAlgorithm):
    def __init__(self,arch,params):
        super(MLFQAlgorithm, self).__init__(arch,params) 
        self.name = 'mlfq'
        self.queue_count = params[0]
        self.iterations_reset = params[1]

        self.init_queues()
    
    def get_experiment(self):
        self.last_action = None 
        while self.last_action == None:
            q = np.random.choice(range(self.queue_count), 1, replace=True, p=self.probs)[0]
            if not self.queues[q].empty():
                self.last_action = self.queues[q].get() 
                self.last_index = q
                return self.last_action

    def result(self,result):
        if result > 0:
            if self.last_index > 0:
                self.last_index -= 1 
        else:
            if not self.last_index == self.queue_count - 1:
                self.last_index += 1
        
        self.queues[self.last_index].put(self.last_action)

        self.iterations += 1

        if self.iterations == self.iterations_reset: 
            self.init_queues()
    
    def init_queues(self):
        self.iterations = 0
        self.queues = [Queue()] * self.queue_count
        operations = self.arch.get_operations() 
        faults = ['abort','delay']
        for op in operations:
            if op.name not in ['a1','a2']:
                for fault in faults:
                    self.queues[0].put(op.name + '-' + fault)
        self.probs = [] * self.queue_count 
        max_prob = 100 
        total = 0
        for i in range(self.queue_count):
            total = total + max_prob
            self.probs.append(max_prob)
            max_prob = max_prob/2
        
        probs_sum = sum(self.probs)
        for i in range(len(self.probs)):
            self.probs[i] = self.probs[i] / probs_sum
class RandomAlgorithm(BaseAlgorithm):
    def __init__(self, arch,params):
        super(RandomAlgorithm, self).__init__(arch,params)
        self.name = 'random'

    def get_experiment(self):
        return random.choice(self.fault_injections)