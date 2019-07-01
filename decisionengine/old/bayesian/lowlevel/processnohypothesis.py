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
nofaultlist=[]
faultlist=[]

# it is worth exploring different threshold values for response time, bu 0.05 gave the most promising results out of 0.03, 0.05, 0.1, 0.2
threshold=0.06

for ex in experimentFile:
    
    for anti in patterns:
        nofaultcsvfile='../../Experiments/'+ex+'/'+anti+'/nofault/'+'response.csv'
        noFaultdf=pd.read_csv(nofaultcsvfile,usecols=[0,1,2,3,4,5])
        
        # df.iloc[:,4] is response time
        
        sizeL=len(noFaultdf.iloc[:,4])
            
        for i in range(0,sizeL):
            # you could experiment by not cconsider the error message 500
            if((noFaultdf.iloc[i,4]>threshold) or (noFaultdf.iloc[i,1]==500)):
                nofaultlist.append(0)
            else:
                nofaultlist.append(1)
                    
        for faultIn in faultInjection:
            done=done+1
            
            #logfile='Experiments/'+ex+'/'+anti+'/'+faultIn+'/'+'response.log'
            csvfile='../../Experiments/'+ex+'/'+anti+'/'+faultIn+'/'+'response.csv'
            # convert from log to csv -- this has already been done. only once needed
            #in_txt = csv.reader(open(logfile, "r"), delimiter = ',')
            #out_csv = csv.writer(open(csvfile, 'w'))
            #out_csv.writerows(in_txt)

            faultdf = pd.read_csv(csvfile,usecols=[0,1,2,3,4,5])

            sizeL=len(faultdf.iloc[:,4])
                    
            for i in range(0,sizeL):
                if((faultdf.iloc[i,4]>threshold) or (faultdf.iloc[i,1]==500)):
                    results.append([ex,anti,faultIn,1]) #issue
                else:
                    results.append([ex,anti,faultIn,0])
        
            print(str((done/all)*100)+'%')
headers=['experiment','patterns','faultinjection','error']
dfResults = pd.DataFrame(results, columns=headers)
dfResults.to_csv('res60nohypothesis.csv')
