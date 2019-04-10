import numpy as np
import pandas as pd
import csv 
import json 
import graph 
import NeuralClassifier as classifier
import random
import statistics

date = 'experiment-2018-12-02T07-32-48UTC' #for testing
pattern = 'a1false-a2true-b1false-c1false' #for testing

injections=['b1-abort','b1-delay','c1-abort','c1-delay','c2-abort','c2-delay','d1-abort','d1-delay','e1-abort','e1-delay','e2-abort','e2-delay']
#configs=['a1false-a2false-b1false-c1false','a1false-a2false-b1false-c1true','a1false-a2false-b1true-c1false','a1false-a2false-b1true-c1true','a1false-a2true-b1false-c1false','a1false-a2true-b1false-c1true','a1false-a2true-b1true-c1false','a1false-a2true-b1true-c1true','a1true-a2false-b1false-c1false','a1true-a2false-b1false-c1true','a1true-a2false-b1true-c1false','a1true-a2false-b1true-c1true','a1true-a2true-b1false-c1false','a1true-a2true-b1false-c1true','a1true-a2true-b1true-c1false','a1true-a2true-b1true-c1true']

configs = ['a1false-a2true-b1false-c1false']



# vertex: [opname, circuitBreaker?, opcount for service, instances service, incoming dependencies]
dep_graph = graph.Graph()
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
            vertex = tuple([architecture["microservices"][service]['operations'][op]['name'], cb, nop, instances, 0, '', None, 0])
            dep_graph.add_vertex(vertex)
            for dep in architecture["microservices"][service]['operations'][op]['dependencies']:
                dep_graph.add_edge([vertex, dep['operation']])

states = []
counter = {}
# state: [opname, circuitBreaker?, opcount for service, instances service, incoming dependencies, faultType, [4 sample results], incoming dependencies]
for vertex in dep_graph.vertices():
    incoming = dep_graph.vertex_incoming(vertex)
    vert_list = list(vertex)
    if vert_list[0] in ['a1','a2','b2']:
        continue
    vert_list[4] = incoming 
    vert_list[6] = []
    vert_list[5] = 'abort'
    fileName = '../../experiments/' + date + '/' + pattern + '/' + vert_list[0] + '-' + vert_list[5] + '/response.csv'
    df = pd.read_csv(fileName,usecols=[0,1,2,3,4,5])
    sample = df.sample(n=4)

    threshold = 0.06
    for i in range(4):
        if((sample.iloc[i,4]>threshold) or (sample.iloc[i,1]==500)):
            vert_list[6].append(1)
        else:
            vert_list[6].append(0)
    states.append(tuple(vert_list))
    vert_list[5] = 'delay'
    vert_list[6] = []
    fileName = '../../experiments/' + date + '/' + pattern + '/' + vert_list[0] + '-' + vert_list[5] + '/response.csv'
    df = pd.read_csv(fileName,usecols=[0,1,2,3,4,5])
    sample = df.sample(n=5)

    for i in range(4):
        if((sample.iloc[i,4]>threshold) or (sample.iloc[i,1]==500)):
            vert_list[6].append(1)
        else:
            vert_list[6].append(0)
    states.append(tuple(vert_list))

    counter[vert_list[0] + '-abort'] = 0
    counter[vert_list[0] + '-delay'] = 0

classifier = classifier.NeuralClassifier(12)

nTrials = 5
nExperiments = 1000
total = nTrials * nExperiments
done = 0

totalRewardsNetwork = []
totalRewardsRandom = []
for f in range(nTrials):
    experimentRewardsNetwork = []
    experimentRewardsRandom = []
    for j in range(nExperiments):
        sz = len(states)
        for i in range(sz):
            # preprocess state into [n_states, n_features]
            params = []
            # params[0] = states[6]
            cb_pattern = states[i][1]
            deps = states[i][4]
            op_instance_ratio = states[i][2] / states[i][3]
            params = [
                cb_pattern,
                deps,
                op_instance_ratio
            ]
            #params.extend(states[i][6])

            prob = classifier.get_action(tuple(params))
            list_state = list(states[i])
            list_state[7] = prob 
            # list_state[7] is selection prob
            states[i] = tuple(list_state)

        sorted_states = sorted(states, key = lambda x: x[7], reverse=True)

        total_reward = 0
        random_reward = 0

        episode_len = 5
        for i in range(episode_len):
            # network selection
            state = sorted_states[i]
            injection = state[0] + '-' + state[5]
            df = pd.read_csv('../../experiments/' + date + '/' + pattern + '/' + injection + '/response.csv',   usecols=[0,1,2,3,4,5])
            sample = df.sample(n=1)

            threshold = 0.06
            if ((sample.iloc[0,4] > threshold) or (sample.iloc[0,1]==500)):
                total_reward += 1

            counter[injection] = counter[injection] + 1

            # random selection
            injection = random.choice(injections)
            df = pd.read_csv('../../experiments/' + date + '/' + pattern + '/' + injection + '/response.csv',   usecols=[0,1,2,3,4,5])
            sample = df.sample(n=1)

            threshold = 0.06
            if ((sample.iloc[0,4] > threshold) or (sample.iloc[0,1]==500)):
                random_reward += 1


        classifier.reward(float(total_reward))

        experimentRewardsNetwork.append((total_reward/episode_len)*100)
        experimentRewardsRandom.append((random_reward/episode_len)*100)

        done = done + 1 
        print(str((done/total)*100) + '%')

    totalRewardsNetwork.append(statistics.mean(experimentRewardsNetwork))
    totalRewardsRandom.append(statistics.mean(experimentRewardsRandom))

print(counter)

percDF = pd.DataFrame({'Network': totalRewardsNetwork,'Random': totalRewardsRandom})
percDF.to_csv('experiment_results.csv')