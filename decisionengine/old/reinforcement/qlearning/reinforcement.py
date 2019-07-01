import numpy as np
import pandas as pd
import csv 
import json 
import graph 
import random
import statistics
import qlearning as classifier

date = 'experiment-2018-12-02T07-32-48UTC' #for testing
pattern = 'a1false-a2true-b1false-c1false' #for testing

injections=['b1-abort','b1-delay','c1-abort','c1-delay','c2-abort','c2-delay','d1-abort','d1-delay','e1-abort','e1-delay','e2-abort','e2-delay']
#configs=['a1false-a2false-b1false-c1false','a1false-a2false-b1false-c1true','a1false-a2false-b1true-c1false','a1false-a2false-b1true-c1true','a1false-a2true-b1false-c1false','a1false-a2true-b1false-c1true','a1false-a2true-b1true-c1false','a1false-a2true-b1true-c1true','a1true-a2false-b1false-c1false','a1true-a2false-b1false-c1true','a1true-a2false-b1true-c1false','a1true-a2false-b1true-c1true','a1true-a2true-b1false-c1false','a1true-a2true-b1false-c1true','a1true-a2true-b1true-c1false','a1true-a2true-b1true-c1true']

configs = ['a1false-a2true-b1false-c1false']


dep_graph = graph.Graph()

# vertex: [opname, circuitBreaker?, opcount for service, instances service, incoming dependencies]
with open("architecture_model.json", 'r') as f:
    architecture = json.load(f)

    nMicros = len(architecture["microservices"])
    for service in range(0,nMicros):      
        nop = len(architecture["microservices"][service]['operations'])
        for op in range(0,nop):
            cb = True 
            if architecture["microservices"][service]['operations'][op]['circuitBreaker'] == None:
                cb = False
            instances = architecture["microservices"][service]['instances']
            vertex = tuple([architecture["microservices"][service]['operations'][op]['name'], cb, 0, '', None, 0])
            dep_graph.add_vertex(vertex)
            for dep in architecture["microservices"][service]['operations'][op]['dependencies']:
                dep_graph.add_edge([vertex, dep['operation']])

states_unprocessed = {}
counter = {}
# state: [opname, circuitBreaker?, opcount for service, incoming dependencies, faultType]
for vertex in dep_graph.vertices():
    incoming = dep_graph.vertex_incoming(vertex)
    vert_list = list(vertex)
    if vert_list[0] in ['a1','a2','b2']:
        continue
    for item in ['a1','a2','b2']:
        if item in incoming: 
            incoming.remove(item)
    
    vert_list[2] = len(incoming )
    vert_list[3] = 'abort'
    states_unprocessed[tuple(vert_list)] = incoming
    vert_list[3] = 'delay'
    states_unprocessed[tuple(vert_list)] = incoming


#for vertex in dep_graph.vertices():
#    outgoing = dep_graph.vertex_outgoing(vertex)
#    vert_list = list(vertex)
#    if vert_list[0] in ['a1','a2','b2']:
#        continue
#    for item in ['a1','a2','b2']:
#        if item in outgoing:
#            outgoing.remove(item)
#    
#    vert_list[2] = len(dep_graph.vertex_incoming(vertex))
#    vert_list[3] = 'abort'
#    states_unprocessed[tuple(vert_list)] = outgoing 
#    vert_list[3] = 'delay'
#    states_unprocessed[tuple(vert_list)] = outgoing 

# process state data 
states = {}
# operations to operations
for state in states_unprocessed.keys():
    if state[0] + '-' + state[3] not in states.keys():
        states[state[0] + '-' + state[3]] = []
    for state_inc in states_unprocessed[state]:
        if state_inc[0] + '-abort' not in states[state[0] + '-' + state[3]]:
            states[state[0] + '-' + state[3]].append(state_inc[0] + '-abort')
        if state_inc[0] + '-delay' not in states[state[0] + '-' + state[3]]:
            states[state[0] + '-' + state[3]].append(state_inc[0] + '-delay')



# features to features - TODO
#for state_unprocessed in states_unprocessed.keys():
#    state_list = [state_unprocessed[1], state_unprocessed[2]]
#    state = tuple(state_list)
#    if state not in states.keys():
#        states[state] = []
#    for state_inc in states_unprocessed[state_unprocessed]:
#        state_inc_processed = tuple([state_inc[1],state_inc[2]])
#        if state_inc_processed not in states[state]:
#            states[state].append(tuple(state_inc_processed)) 

nTrials = 25
nExperiments = 500
total = nTrials * nExperiments
done = 0

totalRewardsQL = []
totalRewardsRandom = []
for f in range(nTrials):
    classi = classifier.QAgent(states, 0.2, 0.99, 0.66, 5)

    experimentRewardsQL = []
    experimentRewardsRandom = []

    # start in random state
    state = np.random.choice(list(states.keys()))
    for j in range(nExperiments):
        action_index = classi.get_action(state)
        
        total_reward = 0
        random_reward = 0

        injection = state[0] + '-' + state[5]
        df = pd.read_csv('../../experiments/' + date + '/' + pattern + '/' +state + '/response.csv',   usecols=[0,1,2,3,4,5])
        sample = df.sample(n=1)
        threshold = 0.06
        if ((sample.iloc[0,4] > threshold) or (sample.iloc[0,1]==500)):
            total_reward = 1

        
        # random selection
        injection = random.choice(injections)
        df = pd.read_csv('../../experiments/' + date + '/' + pattern + '/' +injection + '/response.csv',   usecols=[0,1,2,3,4,5])
        sample = df.sample(n=1)
        threshold = 0.06
        if ((sample.iloc[0,4] > threshold) or (sample.iloc[0,1]==500)):
            random_reward = 1

            


        new_state = states[state][action_index]
        # reset to random state, if no actions for the new state
        if new_state not in states.keys():
            new_state = np.random.choice(list(states.keys()))

        classi.reward(state, action_index, new_state, float(total_reward))
        state = new_state
        
        experimentRewardsQL.append(total_reward)
        experimentRewardsRandom.append(random_reward)

        done = done + 1 
    
    print(str((done/total)*100) + '%')
    print(statistics.mean(experimentRewardsQL))
    print(statistics.mean(experimentRewardsRandom))
    classi.print_Q()
    totalRewardsQL.append(statistics.mean(experimentRewardsQL))
    totalRewardsRandom.append(statistics.mean(experimentRewardsRandom))


percDF = pd.DataFrame({'QL': totalRewardsQL,'Random': totalRewardsRandom})
percDF.to_csv('experiment_results_double_11.csv')