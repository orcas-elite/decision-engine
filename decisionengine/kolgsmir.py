import pandas as pd 
from scipy.stats import ks_2samp
import numpy as np
import matplotlib.pyplot as plt
import os.path

# GOAL:
# Run ks test on all possible combinations of 2 samples
#           b1-abort b1-delay c1-abort ... nofault
# b1-abort
# b1-delay
# c1-abort
# ...
# nofault

faultInjection=['b1-abort','b1-delay','c1-abort','c1-delay','c2-abort','c2-delay','d1-abort','d1-delay','e1-abort','e1-delay','e2-abort','e2-delay','nofault']
hystrixConfigs = ["a1false-a2false-b1false-c1false","a1true-a2false-b1false-c1false","a1false-a2true-b1false-c1false", "a1false-a2false-b1true-c1false","a1false-a2false-b1false-c1true","a1true-a2true-b1false-c1false","a1true-a2false-b1true-c1false","a1true-a2false-b1false-c1true","a1false-a2true-b1true-c1false","a1false-a2true-b1false-c1true","a1false-a2false-b1true-c1true","a1true-a2true-b1true-c1false","a1true-a2true-b1false-c1true","a1true-a2false-b1true-c1true","a1false-a2true-b1true-c1true","a1true-a2true-b1true-c1true"]


def plot_data_combined(d1, d2, fault1, fault2, p):
    df = pd.DataFrame({fault1:d1,fault2:d2})
    filename = ''
    # HIST and KDE plot two in one
    if fault1 == 'nofault':
        filename = "data/" + config + "/kolgsmir/" + fault1 + "-" + fault2 + ".png"
    elif fault2 == 'nofault':
        filename = "data/" + config + "/kolgsmir/" + fault2 + "-" + fault1 + ".png"
    else:
        return
    ax = df.plot.hist(bins=60, alpha=0.5)
    ax2 = df.plot.kde(ax=ax, secondary_y=True) 
    ax.set_title(p)
    plt.savefig(filename)
    plt.close()


def plot_data_sep(d1, d2, fault1, fault2, p):
    df = pd.DataFrame({fault1:d1, fault2:d2})
    filename=''
    # separate hist and kde plots
    diagram_types = ['hist', 'kde']
    for diag in diagram_types:
        if not os.path.isdir("data/" + config + "/kolgsmir/" + diag + "/"): 
            os.makedirs("data/" + config + "/kolgsmir/" + diag, mode=0o777)
        if fault1 == 'nofault':
            filename = "data/" + config + "/kolgsmir/" + diag + "/" + fault1 + "-" + fault2 + ".png"
        elif fault2 == 'nofault':
            filename = "data/" + config + "/kolgsmir/" + diag + "/" + fault2 + "-" + fault1 + ".png"
        else:
            return
        ax = None
        if diag == 'hist':
            ax = df.plot.hist(bins=60, alpha=0.5)
        elif diag == 'kde':
            ax = df.plot.kde()
        else:
            return
        ax.set_title(p)
        plt.savefig(filename)
        plt.close()



for config in hystrixConfigs:
    resultsAll = []
    for fault1 in faultInjection:
        resultsFault = []
        for fault2 in faultInjection:
            fault1data=pd.read_csv("data/" + config + "/" + fault1 + ".csv")["responseTime"].values
            fault2data=pd.read_csv("data/" + config + "/" + fault2 + ".csv")["responseTime"].values

            
            sample_size = 150000
            fault1data=np.random.choice(fault1data, sample_size,replace=False)
            fault2data=np.random.choice(fault2data, sample_size,replace=False)

            sValue, pValue = ks_2samp(fault1data, fault2data)
            sig = 0.05
            if pValue > sig:
                hyp = 1
            else:
                hyp = 0
            resultsFault.append(hyp)

            if not os.path.isdir("data/" + config + "/kolgsmir/"): 
                os.makedirs("data/" + config + "/kolgsmir")

            plot_data_combined(fault1data, fault2data, fault1, fault2, pValue)
            plot_data_sep(fault1data, fault2data, fault1, fault2, pValue)

        resultsAll.append(resultsFault)

    df = pd.DataFrame(resultsAll,index=faultInjection,columns=faultInjection)
    df.to_csv("data/" + config + "/kolgsmir/kolgsmir.csv",index=True,header=True,sep=' ')

    

