import pandas as pd
import numpy as np
import random
import json
import csv
from scipy.stats import ks_2samp
import statistics

# Read JSON data into the architecture variable

# Parse architecture model
with open("architecture_model.json", 'r') as f:
    architecture = json.load(f)
    nMicros = len(architecture["microservices"])
    opdep  = []
    opnames = []
    pa=[]
    for ms in range(0,nMicros):
        #print(architecture["microservices"][ms])
        nop=len(architecture["microservices"][ms]['operations'])
        for opp in range(0,nop):
            opdep.append(len(architecture["microservices"][ms]['operations'][opp]['dependencies']))
            opnames.append(architecture["microservices"][ms]['operations'][opp]['name'])
            if(architecture["microservices"][ms]['operations'][opp]['circuitBreaker']==None):
                pa.append(0)
            else:
                pa.append(1)

# operation name and number of dependencies
archInfo = [[opnames,opdep,pa]]
# put operations with dependencies and patterns (circuitBreaker) into dataFrame
archInfoDf = pd.DataFrame({'ops':opnames,'dependencies':opdep,'patterns':pa})


ex = 'experiment-2018-12-02T07-32-48UTC' #for testing
# example operations with hystrix config settings
exComps=['a1','a2','b1','c1']
# example hystrix configurations for patterns
patterns = ['a1false-a2false-b1false-c1false','a1false-a2false-b1false-c1true','a1false-a2false-b1true-c1false','a1false-a2false-b1true-c1true','a1false-a2true-b1false-c1false','a1false-a2true-b1false-c1true','a1false-a2true-b1true-c1false','a1false-a2true-b1true-c1true','a1true-a2false-b1false-c1false','a1true-a2false-b1false-c1true','a1true-a2false-b1true-c1false','a1true-a2false-b1true-c1true','a1true-a2true-b1false-c1false','a1true-a2true-b1false-c1true','a1true-a2true-b1true-c1false','a1true-a2true-b1true-c1true']
# example faults
faultInjection=['b1-abort','b1-delay','c1-abort','c1-delay','c2-abort','c2-delay','d1-abort','d1-delay','e1-abort','e1-delay','e2-abort','e2-delay']

# dataFrame with all example patterns
patternsDF=pd.DataFrame({'patterns':patterns})
# dataFrame with all example faults
faultInjectionDF=pd.DataFrame({'faults':faultInjection})

# Initialize probabilities from learningHighlevel.py
probMatrix=pd.read_csv('probhighlevel.csv')

#   patterns    dependencies    errorInjected    probabilities
#0    0              2              abort            0
#1    1              2              abort            0
# fitness proportionate selection

# count of trials to check
trials = 50
# count of experiments to check in a trial
nExperiments = 100

# Mean percentages for each trial
percAllLearning=[]
percAllNoLearning=[]

totalDf = pd.read_csv('res60nohypothesis.csv')

total = trials * nExperiments
done = 0

for trialno in range(trials):
    percentageFailuresLearning = []
    percentageFailuresNoLearning = []

    for i in range(nExperiments):
        done = done+1
        print(str((done/total)*100)+'%')
        
        #### SELECTED EXPERIMENT FROM TRAINING
        # Sum the total probabilities from the probMatrix
        total_prob = sum(probMatrix['probabilities'])
        # Adjust to 0-100% scale
        relative_probs = [p/total_prob for p in probMatrix['probabilities']]
        
        # choose 1 random index from probMatrix based on adjusted weights
        x = random.choices(probMatrix.index, weights = relative_probs, k = 1)
        indexSel=x[0]
    
        # Get tuple values (pattern, dependencies, faultType) for the index
        selPat =probMatrix.iloc[indexSel, 1]
        selDep =probMatrix.iloc[indexSel, 2]
        selErrorType =probMatrix.iloc[indexSel, 3]
        
        # Get all example faults
        # posFaults=faultInjectionDF.loc[(faultInjectionDF['faults'])]
        # get matching operations for patterns/dependencies
        posCases = archInfoDf.loc[(archInfoDf['patterns']==selPat)&(archInfoDf['dependencies']==selDep)]
        #poscases
        #      dependencies ops  patterns
        #8          1       c1      0
        
        ##select fault to inject randomly

        # choose random operation
        randomrow = posCases.sample()
        # remove the index column from the random operation
        randomrow = randomrow.reset_index(drop=True)

        # build the experiment by combining operation with error type: 
        # dependencies - ops - patterns - errorType
        # DONE: Changed the index from [0,1] to [0,0] to get the correct operation
        injectcomp=randomrow.iloc[0,0]
        injectFault=str(injectcomp)+'-'+selErrorType

        # choose a random pattern from example patterns
        pat=random.choice(patterns)
        # iterate through example operations
        if injectcomp in exComps:
            # parse bool to string from pattern field in the previously selected experiment
            comPat='false' if randomrow.iloc[0,2]==0 else 'true'
            # get all matching experiments from the example hystrix configurations
            # e.g. a1false returns a1false-a2false-..., a1false-a2true...,...
            rowsComs=patternsDF[patternsDF['patterns'].str.contains(injectcomp+comPat)]
            # randomly select a hystrix configuration of patterns
            learnedPat=rowsComs.sample()
            # drop the index column
            learnedPat = learnedPat.reset_index(drop=True)
            pat=learnedPat.iloc[0,0]
        # select the experiment data file
        #fileName ='../../Experiments/'+ex+'/'+pat+'/'+injectFault+'/'+'response.csv'


        ###### RANDOM EXPERIMENT WITHOUT TRAINING
        # choose a random example pattern
        randomPat = random.choice(patterns)
        # choose a random example fault injection
        randomFF = random.choice(faultInjection)
        # set the random file to check (ex is previously defined experiment run)
        #randomFilename = nofaultcsvfile='../../Experiments/'+ex+'/'+randomPat+'/'+randomFF+'/'+'response.csv'
    
        # Check if any a1 or a2 operations are being tested
        # Skip if yes, as we don't have any data on these (yet)
        if 'a1-' in injectFault:
            continue
        elif 'a2-' in injectFault:
            continue
        elif 'a1-' in randomFF:
            continue
        elif 'a2-' in randomFF:
            continue

        if 'b2-' in injectFault:
            continue
        elif 'b2-' in randomFF:
            continue


        # Load selected and random results
        #resultsDf = pd.read_csv(fileName,usecols=[0,1,2,3,4,5])
        #randomDf = pd.read_csv(randomFilename,usecols=[0,1,2,3,4,5])

        resultsDf = totalDf.loc[(totalDf['patterns'] == pat) & (totalDf['faultinjection'] == injectFault)]
        randomDf = totalDf.loc[(totalDf['patterns'] == randomPat) & (totalDf['faultinjection'] == randomFF)]

        # Randomly sample 100 elements from your dataframe
        df_elements = resultsDf.sample(n=1000)
        randomdf_elements = randomDf.sample(n=1000)

        count=0
        # Count the experiments with errors
        sizeL=len(df_elements.iloc[:,4] == 1)
        # Iterate over samples for selected experiment
        for j in range(0,sizeL):
            # Only count if timeout is > threshold and return code is 500 status code (?)
            if(df_elements.iloc[j,4]):
                count=count+1

        # Check for random selection
        countRandom = 0
        for j in range(0,sizeL):
            if(randomdf_elements.iloc[j,4] == 1):
                countRandom = countRandom+1
        
        ## METRIC
        # Add percentages of faults found to percentageFailures
        percentageFailuresLearning.append((count/sizeL)*100)
        percentageFailuresNoLearning.append((countRandom/sizeL)*100)

    

    print(str((trialno/trials)*100)+'%')
    percAllLearning.append(statistics.mean(percentageFailuresLearning))
    percAllNoLearning.append(statistics.mean(percentageFailuresNoLearning))

    print("Learning: " + str(statistics.mean(percAllLearning)))
    print("NoLearning: " + str(statistics.mean(percAllNoLearning)))

percDF = pd.DataFrame({'Learning': percAllLearning,'No learning': percAllNoLearning})
percDF.to_csv('test.csv')
