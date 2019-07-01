import pandas as pd
import numpy as np
import random
import json
import csv
from scipy.stats import ks_2samp


######################################
######################################
# Problem: a1/a2 are the only operations with circuit breaker operations
# However there exist no fault injections into either a1/a2
# Thus the pattern value is always zero, as there exists no circuit breaker for any of the other operations
# This vastly reduces the amount of different parameter combinations
######################################

#Read JSON data into the architecture variable

with open("architecture_model.json", 'r') as f:
    architecture = json.load(f)
    nMicros = len(architecture["microservices"])
    opdep  = []
    opnames = []
    pa=[]
    for ms in range(0,nMicros):
        nop=len(architecture["microservices"][ms]['operations'])
        for opp in range(0,nop):
            index = len(opdep)
            opdep.append([])
            for dep in architecture["microservices"][ms]['operations'][opp]['dependencies']:
                opdep[index].append(dep['operation'])
            opnames.append(architecture["microservices"][ms]['operations'][opp]['name'])
            if(architecture["microservices"][ms]['operations'][opp]['circuitBreaker']==None):
                pa.append(0)
            else:
                pa.append(1)

# operation name and number of dependencies
archInfo = np.array((opnames,opdep,pa))


#patterns = ['a1false-a2false-b1false-c1false','a1false-a2false-b1false-c1true','a1false-a2false-b1true-c1false','a1false-a2false-b1true-c1true','a1false-a2true-b1false-c1false','a1false-a2true-b1false-c1true','a1false-a2true-b1true-c1false','a1false-a2true-b1true-c1true','a1true-a2false-b1false-c1false','a1true-a2false-b1false-c1true','a1true-a2false-b1true-c1false','a1true-a2false-b1true-c1true','a1true-a2true-b1false-c1false','a1true-a2true-b1false-c1true','a1true-a2true-b1true-c1false','a1true-a2true-b1true-c1true']
#faultInjection=['b1-abort','b1-delay','c1-abort','c1-delay','c2-abort','c2-delay','d1-abort','d1-delay','e1-abort','e1-delay','e2-abort','e2-delay']

# read the processed results
dfc=pd.read_csv('res60nohypothesis.csv')

df = dfc.sample(100000)
# here we learn based on the number of dependecies and patterns of the component where the fault was injected

#0    experiment-2018-09-16T15-13-59UTC    a1false-a2false-b1false-c1false    b1-abort    0
patterns = []
dependencies = []
errorInjected = []
faultRevealed = []

allparams = []

total = len(df.index)
done = 0
for index, row in df.iterrows():
    # b1-abort -> [b1,abort]
    tem=row['faultinjection'].split('-')
    # append action -> abort
    errorInjected.append(tem[1])
    # find operation b1
    r1 , c1 = np.where(archInfo == tem[0])
    #tem[0] is the operation where the error was injected
    # get dependencies from archInfo: [opnames, opdep, pa] -> opdep[b1]
    dependencies.append(archInfo[1,c1[0]])



    # -> 0 or 1 
    faultRevealed.append(row['error'])
    
    # hystrixConfigs 
    pattemp = row['patterns'].split('-')
    
    # get operations with hystrix configs
    #ops = [x[0:2] for x in pattemp]
    # TODO There doesn't seem to be any circuitBreaker defined for b1/c1 -> results are incorrect when going based off the configs?
    # Manually define instead
    ops = ['a1', 'a2']
    #['a1', 'a2', 'b1', 'c1']
    try:
        idx = ops.index(tem[0])
        # figure out if hystrix is active for current operation
        if ('false' in pattemp[idx]):
            patterns.append(0)
        else:
            patterns.append(1)
    except:
        # assume hystrix is active -> check if circuitBreaker is set for operation
        patterns.append(archInfo[2,c1[0]])

    done = done + 1
    print(str((done/total)*100)+'%')
    

allres = pd.DataFrame({'patterns': patterns,'dependencies': dependencies,'errorInjected':errorInjected, 'faultRevealed':faultRevealed})
allres.to_csv('postProcessHighLevel-dep.csv')
