import numpy as np
import pandas as pd
import csv 

date = 'experiment-2018-12-02T07-32-48UTC' #for testing
pattern = 'a1false-a2true-b1false-c1false' #for testing

injections=['b1-abort','b1-delay','c1-abort','c1-delay','c2-abort','c2-delay','d1-abort','d1-delay','e1-abort','e1-delay','e2-abort','e2-delay']
#configs=['a1false-a2false-b1false-c1false','a1false-a2false-b1false-c1true','a1false-a2false-b1true-c1false','a1false-a2false-b1true-c1true','a1false-a2true-b1false-c1false','a1false-a2true-b1false-c1true','a1false-a2true-b1true-c1false','a1false-a2true-b1true-c1true','a1true-a2false-b1false-c1false','a1true-a2false-b1false-c1true','a1true-a2false-b1true-c1false','a1true-a2false-b1true-c1true','a1true-a2true-b1false-c1false','a1true-a2true-b1false-c1true','a1true-a2true-b1true-c1false','a1true-a2true-b1true-c1true']

configs = ['a1false-a2true-b1false-c1false']

actions = []

for config in configs:
    for injection in injections:
        actions.append(config + '/' + injection)

trials = 50
nExperiments = 50
total = trials * nExperiments
done = 0

rewardLearningTotal = []
#rewardRandomTotal = []
for trialno in range(trials):
    rewardLearning = 0
    rewardRandom = 0

    estimated=[]
    count=[]

    for a in actions: 
        estimated.append(5)
        count.append(0)
    for i in range(nExperiments):
        # find best action 
        max_index = estimated.index(max(estimated)) 

        act_index = max_index

        # get the reward
        fileName = '../../Experiments/' + date + '/' + actions[act_index] + '/' + 'response.csv'
        resultsDf = pd.read_csv(fileName,usecols=[0,1,2,3,4,5])
        df_elements = resultsDf.sample(n=1)

        threshold = 0.06
        if((df_elements.iloc[0,4]>threshold) or (df_elements.iloc[0,1]==500)):
            reward = 1
            rewardLearning = rewardLearning + 1
        else:
            reward = 0
        

        count[act_index] = count[act_index] + 1
        estimated[act_index] = estimated[act_index] + (1/count[act_index])*(reward - estimated[act_index])

        done = done + 1
        print(str((done/total)*100)+'%')
    #with open('estimates.csv', 'w') as f:
    #    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
    #    for i in range(len(actions)):
    #        row = []
    #        row.append(actions[i])
    #        row.append(estimated[i])
    #        row.append(count[i])
    #        wr.writerow(row)

    rewardLearningTotal.append(rewardLearning)

learningDF = pd.DataFrame({'Learning': rewardLearningTotal})
learningDF.to_csv('bandit-50.csv')
