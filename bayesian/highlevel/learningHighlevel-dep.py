import pandas as pd
import numpy as np
import random
import json
import csv
from scipy.stats import ks_2samp


# read the processed results
# dependencies | errorInjected | faultRevealed | patterns
df=pd.read_csv('postProcessHighLevel-dep.csv')


# add param combinations (patterns,dependencies,errorInjected) to uniqueParams
# use seen to ensure unique tupels only
# only allow valid combinations! 
seen = set()
uniqueParams=[]
paramsArray=np.array((df['patterns'],df['dependencies'],df['errorInjected']))
print(paramsArray)
for item in paramsArray.T:
    t = tuple(item)
    if t not in seen:
        uniqueParams.append(item)
        seen.add(t)

probs=[]

# uniqueParams: [array([0, 2, 'abort'], dtype=object), array([1, 2, 'abort'], dtype=object)]
# uniqueParams: [[patterns,dependencies,faultType],...]

## initially, all parameter combinations have the same probability of being selected
selP=1/len(uniqueParams)

# populate probs with all uniqueParam tuples
for item in uniqueParams:
    probs.append(selP)
# Initialize dataFrame to print the results later (uniqueParams with headers)
probMatrix = pd.DataFrame(uniqueParams, columns=['patterns','dependencies','errorInjected'])

# Add probabilities column with uniform distribution for probabilities (selP)
probMatrix=probMatrix.assign(probabilities = probs)
print(probMatrix)

done=0
# combs: possible combinations, |patterns| * |errorInjected| * |dependencies| 
# combs = len(uniqueParams) ^ 3 ?
combs=len(probMatrix['patterns'])*len(probMatrix['errorInjected'])*len(probMatrix['dependencies'])

# Iterate over patterns in tupels
for pat in probMatrix['patterns']:
    # Iterate over faultTypes in tupels
    for faultIn in probMatrix['errorInjected']:
        # Iterate over dependencies in tupels
        for dep in probMatrix['dependencies']:
            # count total processed out of n^3
            done=done+1

            print(str((done/combs)*100)+'%')

            # for each faultType and pattern combination, check:
            # Check if combination of pattern, faultType, dependencies exists in experiment data
            total = df.loc[(df['patterns']==pat) & (df['errorInjected']==faultIn) & (df['errorInjected']==faultIn) & (df['dependencies']==dep)]
            if(len(total) > 0):
                # Skip already computed results
                # print(probMatrix.loc[((probMatrix['patterns']==pat) & (probMatrix['errorInjected']==faultIn)& (df['dependencies']==dep))].iloc[0]['probabilities'])
                if (probMatrix.loc[((probMatrix['patterns']==pat) & (probMatrix['errorInjected']==faultIn) & (probMatrix['dependencies'] == dep))].iloc[0]['probabilities'] == selP):
                    # Calculate the probability for pattern, faultType, dependency
                    # Bayesian Inference
                    # |tupel with fault revealed| / |tupel all|
                    pro=len(total[(total['patterns']==pat) & (total['errorInjected']==faultIn) & (total['dependencies']==dep) & (total['faultRevealed']==1)])/len(total)
                    probMatrix.loc[((probMatrix['patterns']==pat) & (probMatrix['errorInjected']==faultIn)& (probMatrix['dependencies']==dep)), 'probabilities']=pro
                    print(pro)
            else:
                # No prior data from experiments: Use uniform distribution
                probMatrix.loc[((probMatrix['patterns']==pat) & (probMatrix['errorInjected']==faultIn)& (df['dependencies']==dep)), 'probabilities']=selP

probMatrix.to_csv('probhighlevel-dep.csv')
