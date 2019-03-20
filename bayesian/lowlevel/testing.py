import pandas as pd
import numpy as np
import random
import json
import csv
from scipy.stats import ks_2samp
import statistics


ex = 'experiment-2018-12-02T07-32-48UTC' #for testing
patterns = ['a1false-a2false-b1false-c1false','a1false-a2false-b1false-c1true','a1false-a2false-b1true-c1false','a1false-a2false-b1true-c1true','a1false-a2true-b1false-c1false','a1false-a2true-b1false-c1true','a1false-a2true-b1true-c1false','a1false-a2true-b1true-c1true','a1true-a2false-b1false-c1false','a1true-a2false-b1false-c1true','a1true-a2false-b1true-c1false','a1true-a2false-b1true-c1true','a1true-a2true-b1false-c1false','a1true-a2true-b1false-c1true','a1true-a2true-b1true-c1false','a1true-a2true-b1true-c1true']
faultInjection=['b1-abort','b1-delay','c1-abort','c1-delay','c2-abort','c2-delay','d1-abort','d1-delay','e1-abort','e1-delay','e2-abort','e2-delay']


probMatrix=pd.read_csv('probs60nohypothesis.csv')


# fitness proportionate selection

trials = 50
nExperiments = 100

percAllLearning=[]
percAllNoLearning=[]

for trialno in range(trials):
    percentageFailuresLearning = []
    percentageFailuresNoLearning = []
    for i in range(nExperiments):
        # select a random pattern
        pat = random.choice(patterns)

        condprobMatrix = probMatrix.ix[(probMatrix['patterns']==pat)]
        condprobMatrix = condprobMatrix.reset_index(drop=True)

        total_prob = sum(condprobMatrix['probabilities'])
        relative_probs = [p/total_prob for p in condprobMatrix['probabilities']]
        x = random.choices(condprobMatrix.index, weights = relative_probs, k = 1)
        indexSel=x[0]
    
        #select fault to inject using learning
        injectFault =condprobMatrix.iloc[indexSel, 2 ]
    
        #select fault to inject randomly
        randomFault = random.choice(faultInjection)
    
        fileName = nofaultcsvfile='../../Experiments/'+ex+'/'+pat+'/'+injectFault+'/'+'response.csv'

        randomFilename = nofaultcsvfile='../../Experiments/'+ex+'/'+pat+'/'+randomFault+'/'+'response.csv'
    
        resultsDf = pd.read_csv(fileName,usecols=[0,1,2,3,4,5])
        randomDf = pd.read_csv(randomFilename,usecols=[0,1,2,3,4,5])

        # Randomly sample 70% of your dataframe
        #df_elements = resultsDf.sample(frac=0.2)

        df_elements = resultsDf.sample(n=100)
        randomdf_elements = randomDf.sample(n=100)
    
        threshold=0.03
        count=0
        sizeL=len(df_elements.iloc[:,4])
    
        for j in range(0,sizeL):
            if((df_elements.iloc[j,4]>threshold) or (df_elements.iloc[j,1]==500)):
                count=count+1

        countRandom = 0
        for j in range(0,sizeL):
            if((randomdf_elements.iloc[j,4]>threshold) or (randomdf_elements.iloc[j,1]==500)):
                countRandom = countRandom+1
        percentageFailuresLearning.append((count/sizeL)*100)
        percentageFailuresNoLearning.append((countRandom/sizeL)*100)
    print(str((trialno/trials)*100)+'%')
    percAllLearning.append(statistics.mean(percentageFailuresLearning))
    percAllNoLearning.append(statistics.mean(percentageFailuresNoLearning))

percDF = pd.DataFrame({'Learning': percAllLearning,'No learning': percAllNoLearning})
percDF.to_csv('test60nohypothesis.csv')
