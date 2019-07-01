import queue
import pandas as pd
import numpy as np 

date = 'experiment-2018-12-02T07-32-48UTC' #for testing
pattern = 'a1false-a2true-b1false-c1false' #for testing

injections=['b1-abort','b1-delay','c1-abort','c1-delay','c2-abort','c2-delay','d1-abort','d1-delay','e1-abort','e1-delay','e2-abort','e2-delay']
#configs=['a1false-a2false-b1false-c1false','a1false-a2false-b1false-c1true','a1false-a2false-b1true-c1false','a1false-a2false-b1true-c1true','a1false-a2true-b1false-c1false','a1false-a2true-b1false-c1true','a1false-a2true-b1true-c1false','a1false-a2true-b1true-c1true','a1true-a2false-b1false-c1false','a1true-a2false-b1false-c1true','a1true-a2false-b1true-c1false','a1true-a2false-b1true-c1true','a1true-a2true-b1false-c1false','a1true-a2true-b1false-c1true','a1true-a2true-b1true-c1false','a1true-a2true-b1true-c1true']

configs = ['a1false-a2true-b1false-c1false']



nTrials = 25
nExperiments = 200
total_exp = nTrials * nExperiments
done = 0


reward_learning_total = []
reward_random_total = []
for trial in range(nTrials):
    reward_learning = 0
    reward_random = 0


    n = 4

    queues = []
    probs = []
    max_prob = 100
    total = 0
    for i in range(n):
        q = queue.Queue()
        queues.append(q)

        total = total + max_prob
        probs.append(max_prob)
        max_prob = max_prob/2
    
    for i in range(n):
        probs[i] = probs[i] / total 

    # everyone start in queue 0 
    for injection in injections:
        queues[0].put(injection)

    counter = {}
    for experiment in range(nExperiments):
        # select the queue
        queue_index = np.random.choice(range(n),p=probs)
        while queues[queue_index].qsize() == 0:
            queue_index = np.random.choice(range(n),p=probs)
        # pop the action 
        action = queues[queue_index].get()
        fault_found = 0
        # inject
        threshold = 0.06

        df = pd.read_csv('../experiments/' + date + '/' + pattern + '/' + action + '/response.csv',   usecols=[0,1,2,3,4,5])
        sample = df.sample(n=1)

        if((sample.iloc[0,4]>threshold) or (sample.iloc[0,1]==500)):
            # fault found
            fault_found = 1
            if queue_index == 0:
                queues[queue_index].put(action)
            else: 
                queues[queue_index-1].put(action)
        else:

            if ((queue_index > 0) and(queues[queue_index - 1].qsize() == 0)):
                queues[queue_index - 1].put(action)
            elif (queue_index < (n-1)):
                queues[queue_index + 1].put(action)
            else: 
                queues[queue_index].put(action)
        
        if action in counter:
            counter[action] = counter[action] + 1
        else:
            counter[action] = 1
        
        # RANDOM for comparison
        action = np.random.choice(injections)
        fault_found_random = 0
        df = pd.read_csv('../experiments/' + date + '/' + pattern + '/' + action + '/response.csv',   usecols=[0,1,2,3,4,5])
        sample = df.sample(n=1)

        if((sample.iloc[0,4]>threshold) or (sample.iloc[0,1]==500)):
            # fault found
            fault_found_random = 1

        reward_learning += fault_found 
        reward_random += fault_found_random

        done = done + 1
        print(str((done/total_exp)*100)+'%')
        if done%10 == 0:
            print_len = []
            for f in range(n):
                print_len.append(queues[f].qsize())
            print(print_len)


    print(counter)
    reward_learning_total.append(reward_learning / nExperiments)
    reward_random_total.append(reward_random / nExperiments)

learningDF = pd.DataFrame({'Learning': reward_learning_total,'No learning': reward_random_total})
learningDF.to_csv('multi' + str(n) + '.csv')
        

