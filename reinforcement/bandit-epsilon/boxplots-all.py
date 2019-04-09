from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from itertools import cycle
import matplotlib
from matplotlib.pyplot import *
import brewer2mpl
import seaborn as sns
from scipy.stats import ks_2samp
from scipy.stats import mode
import pandas as pd

epsilons = ['01', '03', '05', '07', '09', '10']

resAll = []
for epsilon in epsilons:
    res = pd.read_csv('bandit' + epsilon + '.csv')
    resAll.append(res['Learning'])



fig = plt.figure()#figsize=(4, 2.5))
sns.set_style("whitegrid")
ax = sns.boxplot(data=resAll,sym='',palette="Greys")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticklabels(epsilons)

#ax.xaxis.set_ticklabels([' ',' '])
#ax.set_title(fNumber+' variables')
#ax.set_ylim([-0.1,1.1])

ax.yaxis.set_label_text('Total reward (50d actions)')
formatter = ScalarFormatter(useOffset=False)
ax.yaxis.set_major_formatter(formatter)
fig.savefig('boxplotsbandit-all.pdf', bbox_inches='tight')
