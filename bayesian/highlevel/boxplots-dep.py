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
    

res = pd.read_csv('test-dep.csv')

resL = res['Learning']
resNoL = res['No learning']

resAll=[resL,resNoL]
print("Learning"+str(np.median(resL)))
print("no Learning "+str(np.median(resNoL)))
print(ks_2samp(resL,resNoL))

fig = plt.figure(figsize=(4, 2.5))
sns.set_style("whitegrid")

ax = sns.boxplot(data=resAll,sym='',palette="Greys")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticklabels(['Learning','No learning'])
#ax.xaxis.set_ticklabels([' ',' '])
#ax.set_title(fNumber+' variables')
#ax.set_ylim([-0.1,1.1])
ax.yaxis.set_label_text('% faults')
formatter = ScalarFormatter(useOffset=False)
ax.yaxis.set_major_formatter(formatter)
fig.savefig('boxplots-dep.pdf', bbox_inches='tight')
