import pandas as pd 
from pathlib import Path
import os 
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import numpy as np
import matplotlib.pyplot as plt


# Build nested matrix: 
#                   fault                               fault                           fault   fault   ...
#   date [date,date,date,date..,date]      [date,date,date,date,...,date]               ...     ...
#   date            ...                                 ...
#   date            ...
#   ...
#
#


experimentDates = ['experiment-2018-09-16T15-13-59UTC','experiment-2018-09-19T19-10-21UTC','experiment-2018-09-28T15-05-51UTC','experiment-2018-10-09T18-32-17UTC','experiment-2018-10-21T11-28-15UTC','experiment-2018-11-04T08-32-02UTC','experiment-2018-11-18T10-44-58UTC','experiment-2018-12-02T07-32-48UTC']
faultInjections=['b1-abort','b1-delay','c1-abort','c1-delay','c2-abort','c2-delay','d1-abort','d1-delay','e1-abort','e1-delay','e2-abort','e2-delay','nofault']
hystrixConfigs = ["a1false-a2false-b1false-c1false","a1true-a2false-b1false-c1false","a1false-a2true-b1false-c1false", "a1false-a2false-b1true-c1false","a1false-a2false-b1false-c1true","a1true-a2true-b1false-c1false","a1true-a2false-b1true-c1false","a1true-a2false-b1false-c1true","a1false-a2true-b1true-c1false","a1false-a2true-b1false-c1true","a1false-a2false-b1true-c1true","a1true-a2true-b1true-c1false","a1true-a2true-b1false-c1true","a1true-a2false-b1true-c1true","a1false-a2true-b1true-c1true","a1true-a2true-b1true-c1true"]
services=['a1','a2']

total = 2 * 8 * 13 * 16 * 8
done = 0
for service in services:
    results = []
    for config in hystrixConfigs:
        results_config = []
        for date in experimentDates:
            results_date = []
            for fault in faultInjections:
                results_fault = []
                fault1_path = Path('data_service',service,date,config,fault+'.csv')
                for date2 in experimentDates:
                    fault2_path = Path('data_service',service,date2,config,fault+'.csv') 

                    fault1data = pd.read_csv(fault1_path)["responseTime"].values
                    fault2data = pd.read_csv(fault2_path)["responseTime"].values

                    len1 = len(fault1data)
                    len2 = len(fault2data)

                    #sample_size = None
                    # Take the smallest total size
                    # taking the larger with replace=True gives terrible results in regards to density
                    #if len1 > len2:
                    #    sample_size = len2
                    #else:
                    #    sample_size = len1 

                    # fault1data=np.random.choice(fault1data, sample_size,replace=False)
                    # fault2data=np.random.choice(fault2data, sample_size,replace=False)
                    sValue, pValue = ks_2samp(fault1data, fault2data)
                    sig = 0.01
                    if pValue > sig:
                        hyp = 1
                    else:
                        hyp = 0
                    
                    results_fault.append(hyp)

                    sample_size = None
                    if len1 > len2:
                        sample_size = len2
                    else:
                        sample_size = len1 

                    fault1data=np.random.choice(fault1data, sample_size,replace=False)
                    fault2data=np.random.choice(fault2data, sample_size,replace=False)
                    df = pd.DataFrame({date:fault1data, date2:fault2data})
                    ax = df.plot.kde()
                    ax.set_title(pValue)
                    filename = Path('data_service',service,'kolgsmir',config,date,fault,date2 + ".png")
                    filedir = os.path.dirname(filename)
                    if not os.path.exists(filedir):
                        os.makedirs(filedir)
                    plt.savefig(filename)
                    plt.close()
                    done = done+1
                    
                results_date.append(results_fault)
                print(str((done/total)*100)+'%')
            results_config.append(results_date)
                
        results.append(results_config)
    
    size_conf = len(hystrixConfigs)
    for i in range(0,size_conf):
        df = pd.DataFrame(index=experimentDates,columns=faultInjections)
        size_date = len(experimentDates)
        for j in range(0,size_date):
            size_faults = len(faultInjections)
            for f in range(0,size_faults):
                    df.at[experimentDates[j],faultInjections[f]] = [results[i][j][f]]
        df.to_csv("data_service/" + service + "/kolgsmir/" + hystrixConfigs[i] + "/kolgsmir.csv", sep=" ")

