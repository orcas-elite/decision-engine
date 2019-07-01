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
    

queuecount = ['2', '3', '4', '5']


resAll = []
for count in queuecount:
    res = pd.read_csv('multi' + count + '.csv')
    resAll.append(res['Learning'])

resAll.append(pd.read_csv('multi2.csv')['No learning'])



fig = plt.figure()#figsize=(4, 2.5))
sns.set_style("whitegrid")
ax = sns.boxplot(data=resAll,sym='',palette="Greys")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
queuecount.append('random')
ax.xaxis.set_ticklabels(queuecount)

#ax.xaxis.set_ticklabels([' ',' '])
#ax.set_title(fNumber+' variables')
#ax.set_ylim([-0.1,1.1])

ax.yaxis.set_label_text('Percentage fault found')
formatter = ScalarFormatter(useOffset=False)
ax.yaxis.set_major_formatter(formatter)
fig.savefig('multi-all.pdf', bbox_inches='tight')
