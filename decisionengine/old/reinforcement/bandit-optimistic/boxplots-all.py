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


actions = 50

resAll = []
res = pd.read_csv('bandit-configs-' + str(actions) + '.csv')
res2 = pd.read_csv('bandit-' + str(actions) + '.csv')

resAll.append(res['Learning'])
resAll.append(res2['Learning'])


fig = plt.figure()#figsize=(4, 2.5))
sns.set_style("whitegrid")
ax = sns.boxplot(data=resAll,sym='',palette="Greys")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticklabels(['all-configs', 'a1false-a2true-b1false-c1false'])

#ax.xaxis.set_ticklabels([' ',' '])
#ax.set_title(fNumber+' variables')
#ax.set_ylim([-0.1,1.1])

ax.yaxis.set_label_text('Total reward (' + str(actions) + ' actions)')
formatter = ScalarFormatter(useOffset=False)
ax.yaxis.set_major_formatter(formatter)
fig.savefig('boxplotsbandit-' + str(actions) + '.pdf', bbox_inches='tight')
