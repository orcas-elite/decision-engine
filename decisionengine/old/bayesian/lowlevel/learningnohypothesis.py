import pandas as pd
import numpy as np
import random
import json
import csv
from scipy.stats import ks_2samp



patterns = ['a1false-a2false-b1false-c1false','a1false-a2false-b1false-c1true','a1false-a2false-b1true-c1false','a1false-a2false-b1true-c1true','a1false-a2true-b1false-c1false','a1false-a2true-b1false-c1true','a1false-a2true-b1true-c1false','a1false-a2true-b1true-c1true','a1true-a2false-b1false-c1false','a1true-a2false-b1false-c1true','a1true-a2false-b1true-c1false','a1true-a2false-b1true-c1true','a1true-a2true-b1false-c1false','a1true-a2true-b1false-c1true','a1true-a2true-b1true-c1false','a1true-a2true-b1true-c1true']
faultInjection=['b1-abort','b1-delay','c1-abort','c1-delay','c2-abort','c2-delay','d1-abort','d1-delay','e1-abort','e1-delay','e2-abort','e2-delay']

# read the processed results
df=pd.read_csv('res60nohypothesis.csv')

allparams=[]

# assign equal selecction probabilities to all combinations
combs=len(patterns)*len(faultInjection)
selp=1/combs
for anti in patterns:
    for faultIn in faultInjection:
        pp=[anti,faultIn,selp]
        allparams.append(pp)

# probmatrix is the datastructure that contains the selection probabilities
probMatrix = pd.DataFrame(allparams, columns=['patterns', 'faultinjection', 'probabilities'])

done=0

for anti in patterns:
    for faultIn in faultInjection:
        done=done+1
        print(str((done/combs)*100)+'%')
        if(len(df[(df['patterns']==anti) & (df['faultinjection']==faultIn)])>0):
            pro=len(df[(df['patterns']==anti) & (df['faultinjection']==faultIn) & (df['error']==1)])/len(df[(df['patterns']==anti) & (df['faultinjection']==faultIn)])
            probMatrix.loc[((probMatrix['patterns']==anti) & (probMatrix['faultinjection']==faultIn)), 'probabilities']=pro
        else:
            probMatrix.loc[((probMatrix['patterns']==anti) & (probMatrix['faultinjection']==faultIn)), 'probabilities']=selp

probMatrix.to_csv('probs60nohypothesis.csv', header=['patterns', 'faultinjection', 'probabilities'])
