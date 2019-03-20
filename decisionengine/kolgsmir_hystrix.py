import pandas as pd 
from pathlib import Path
import os 
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import numpy as np
import matplotlib.pyplot as plt


# Run kolgsmir test for hystrix configs
# Question: Doe the b1 and c1 configs do anything, as b1 and c1 don't have a circuit breaker?


date = 'experiment-2018-09-28T15-05-51UTC'
faultInjections=['b1-abort','b1-delay','c1-abort','c1-delay','c2-abort','c2-delay','d1-abort','d1-delay','e1-abort','e1-delay','e2-abort','e2-delay','nofault']
hystrixConfigs = ["a1false-a2false-b1false-c1false","a1true-a2false-b1false-c1false","a1false-a2true-b1false-c1false", "a1false-a2false-b1true-c1false","a1false-a2false-b1false-c1true","a1true-a2true-b1false-c1false","a1true-a2false-b1true-c1false","a1true-a2false-b1false-c1true","a1false-a2true-b1true-c1false","a1false-a2true-b1false-c1true","a1false-a2false-b1true-c1true","a1true-a2true-b1true-c1false","a1true-a2true-b1false-c1true","a1true-a2false-b1true-c1true","a1false-a2true-b1true-c1true","a1true-a2true-b1true-c1true"]
services=['a1','a2']

total = len(services) * len(hystrixConfigs) * len(faultInjections) * len(hystrixConfigs)
done = 0
for service in services:
    results = []
    for fault in faultInjections:
        results_fault = []
        for config1 in hystrixConfigs:
            results_config = []
            for config2 in hystrixConfigs:
                fault1_path = Path('data_service',service,date,config1,fault+'.csv')
                fault2_path = Path('data_service',service,date,config2,fault+'.csv') 
                fault1data = pd.read_csv(fault1_path)["responseTime"].values
                fault2data = pd.read_csv(fault2_path)["responseTime"].values
                len1 = len(fault1data)
                len2 = len(fault2data)
                sample_size = None
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
                
                results_config.append(hyp)

                sample_size = None
                if len1 > len2:
                    sample_size = len2
                else:
                    sample_size = len1 
                fault1data=np.random.choice(fault1data, sample_size,replace=False)
                fault2data=np.random.choice(fault2data, sample_size,replace=False)
                df = pd.DataFrame({config1:fault1data, config2:fault2data})
                ax = df.plot.kde()
                ax.set_title(pValue)
                filename = Path('data_service',service,'kolgsmir_configs',config1,fault,config2 + ".png")
                filedir = os.path.dirname(filename)
                if not os.path.exists(filedir):
                    os.makedirs(filedir)
                plt.savefig(filename)
                plt.close()
                

                done = done+1
                print(str((done/total)*100)+'%')
            results_fault.append(results_config)
        results.append(results_fault)

    df = pd.DataFrame(index=hystrixConfigs,columns=faultInjections)    
    size_faults = len(faultInjections)
    for i in range(0,size_faults):
        size_conf = len(hystrixConfigs)
        for j in range(0,size_conf):
            df.at[hystrixConfigs[j],faultInjections[i]] = [results[i][j]]
    df.to_csv("data_service/" + service + "/kolgsmir_configs/kolgsmir_configs.csv", sep=" ")

