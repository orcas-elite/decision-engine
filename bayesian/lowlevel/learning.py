import pandas as pd
import numpy as np
import random
import json
import csv
from scipy.stats import ks_2samp


#Read JSON data into the architecture variable

with open("architecture_model.json", 'r') as f:
    architecture = json.load(f)
    nMicros = len(architecture["microservices"])
    opdep  = []
    opnames = []
    for ms in range(0,nMicros):
        #print(architecture["microservices"][ms])
        nop=len(architecture["microservices"][ms]['operations'])
        for opp in range(0,nop):
            opdep.append(len(architecture["microservices"][ms]['operations'][opp]['dependencies']))
            opnames.append(architecture["microservices"][ms]['operations'][opp]['name'])
                #if(architecture["microservices"][ms]['operations'][opp]['circuitBreaker']==None):
                #pa.append(0)
                #else:
                #pa.append(1)

# operation name and number of dependencies
archInfo = np.array((opnames,opdep))

patterns = ['a1false-a2false-b1false-c1false','a1false-a2false-b1false-c1true','a1false-a2false-b1true-c1false','a1false-a2false-b1true-c1true','a1false-a2true-b1false-c1false','a1false-a2true-b1false-c1true','a1false-a2true-b1true-c1false','a1false-a2true-b1true-c1true','a1true-a2false-b1false-c1false','a1true-a2false-b1false-c1true','a1true-a2false-b1true-c1false','a1true-a2false-b1true-c1true','a1true-a2true-b1false-c1false','a1true-a2true-b1false-c1true','a1true-a2true-b1true-c1false','a1true-a2true-b1true-c1true']
faultInjection=['b1-abort','b1-delay','c1-abort','c1-delay','c2-abort','c2-delay','d1-abort','d1-delay','e1-abort','e1-delay','e2-abort','e2-delay']

# read the processed results
df=pd.read_csv('allres30.csv')

allparams=[]

# assign equal selecction probabilities to all ccombinations
selp=1/(len(patterns)*len(faultInjection))
for anti in patterns:
    for faultIn in faultInjection:
        pp=[anti,faultIn,selp]
        allparams.append(pp)

# probmatrix is the datastructure that contains the selection probabilities
probMatrix = pd.DataFrame(allparams, columns=['patterns', 'faultinjection', 'probabilities'])

for anti in patterns:
    for faultIn in faultInjection:
        if(len(df[(df['patterns']==anti) & (df['faultinjection']==faultIn)])>0):
            pro=len(df[(df['patterns']==anti) & (df['faultinjection']==faultIn) & (df['hypothesis']==0)])/len(df[(df['patterns']==anti) & (df['faultinjection']==faultIn)])
            probMatrix.loc[((probMatrix['patterns']==anti) & (probMatrix['faultinjection']==faultIn)), 'probabilities']=pro
        else:
            probMatrix.loc[((probMatrix['patterns']==anti) & (probMatrix['faultinjection']==faultIn)), 'probabilities']=selp

probMatrix.to_csv('probabilities30.csv', header=['patterns', 'faultinjection', 'probabilities'])
