import pandas as pd
import numpy as np
import random
import json
import csv
from scipy.stats import ks_2samp


#2018-09-16T15:23:41.334202, 500, Internal Server Error, http://chaos-kube:31380/a1, 0.014236000, {"timestamp":"2018-09-16T15:23:41.329+0000","status":500,"error":"Internal Server Error","message":"502 Bad Gateway","path":"/a1"}
#2018-09-16T15:23:41.261085, 200, OK, http://chaos-kube:31380/a2, 0.014724000, Operation a/a2 executed successfully.

experimentFile=['experiment-2018-09-16T15-13-59UTC','experiment-2018-09-19T19-10-21UTC','experiment-2018-09-28T15-05-51UTC','experiment-2018-10-09T18-32-17UTC','experiment-2018-10-21T11-28-15UTC','experiment-2018-11-04T08-32-02UTC','experiment-2018-11-18T10-44-58UTC'] #'experiment-2018-12-02T07-32-48UTC' is used for testing
patterns = ['a1false-a2false-b1false-c1false','a1false-a2false-b1false-c1true','a1false-a2false-b1true-c1false','a1false-a2false-b1true-c1true','a1false-a2true-b1false-c1false','a1false-a2true-b1false-c1true','a1false-a2true-b1true-c1false','a1false-a2true-b1true-c1true','a1true-a2false-b1false-c1false','a1true-a2false-b1false-c1true','a1true-a2false-b1true-c1false','a1true-a2false-b1true-c1true','a1true-a2true-b1false-c1false','a1true-a2true-b1false-c1true','a1true-a2true-b1true-c1false','a1true-a2true-b1true-c1true']
faultInjection=['b1-abort','b1-delay','c1-abort','c1-delay','c2-abort','c2-delay','d1-abort','d1-delay','e1-abort','e1-delay','e2-abort','e2-delay']

 


# total number of combinations
all=len(experimentFile)*len(patterns)*len(faultInjection)

done=0
results=[]
for ex in experimentFile:
    
    for anti in patterns:
        nofaultcsvfile='../../Experiments/'+ex+'/'+anti+'/nofault/'+'response.csv'
        noFaultdf=pd.read_csv(nofaultcsvfile,usecols=[0,1,2,3,4,5])
        
        for faultIn in faultInjection:
            done=done+1
            
            csvfile='../../Experiments/'+ex+'/'+anti+'/'+faultIn+'/'+'response.csv'

            faultdf = pd.read_csv(csvfile,usecols=[0,1,2,3,4,5])
            nofaultlist=[]
            faultlist=[]
            
            # df.iloc[:,4] is response time
            sizeL=len(noFaultdf.iloc[:,4])
            
            # acceptable response time threshold lies between 0.05 and 0.06
            threshold=0.06
            for i in range(0,sizeL):
                # TODO: Does error message 500 count as failure? Fail fast ..
                if((noFaultdf.iloc[i,4]>threshold) or (noFaultdf.iloc[i,1]==500)):
                    nofaultlist.append(0)
                else:
                    nofaultlist.append(1)
            
            sizeL=len(faultdf.iloc[:,4])
                    
            for i in range(0,sizeL):
                if((faultdf.iloc[i,4]>threshold) or (faultdf.iloc[i,1]==500)):
                    faultlist.append(0)
                else:
                    faultlist.append(1)
        
            nsamples =100
            sampleSize=200
            # here we take nsamples of size sampleSize from the same population (file)
            for x in range(nsamples):
                randomfaultlist = random.sample(faultlist, sampleSize)
                randomnofaultlist = random.sample(nofaultlist, sampleSize)
                
                # 2 sample kolmogorov-smirnov test
                # we could experiment with different pValues
                sValue, pValue = ks_2samp(randomnofaultlist, randomfaultlist)
            
                hypothesis=0 # distributions are equal
                # significance level is 0.05. we could experiment with 0.01
                if(pValue<0.03):
                    hypothesis=1 # distributions are different
                results.append([ex,anti,faultIn,hypothesis])
    
            print(str((done/all)*100)+'%')
headers=['experiment','patterns','faultinjection','hypothesis']
dfResults = pd.DataFrame(results, columns=headers)
dfResults.to_csv('allres60.csv')
