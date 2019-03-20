import abc 
import numpy as np
import experiment
import architecture
import strategy
import os
import pandas as pd
import random


class BayesianLow(strategy.Strategy):
    """ Use low level bayesian learning for experiment selection
    """
    arch = []
    hystrixConfigs = ['a1false-a2false-b1false-c1false','a1false-a2false-b1false-c1true','a1false-a2false-b1true-c1false','a1false-a2false-b1true-c1true','a1false-a2true-b1false-c1false','a1false-a2true-b1false-c1true','a1false-a2true-b1true-c1false','a1false-a2true-b1true-c1true','a1true-a2false-b1false-c1false','a1true-a2false-b1false-c1true','a1true-a2false-b1true-c1false','a1true-a2false-b1true-c1true','a1true-a2true-b1false-c1false','a1true-a2true-b1false-c1true','a1true-a2true-b1true-c1false','a1true-a2true-b1true-c1true']
    faultInjections=['b1-abort','b1-delay','c1-abort','c1-delay','c2-abort','c2-delay','d1-abort','d1-delay','e1-abort','e1-delay','e2-abort','e2-delay']
    probMatrix = None

    def init(self,arch):
        # initialize model
        print("Initializing model for bayesian ..")
        
        # parse the architecture
        self.arch = arch


        self.init_processed()
        # init_nonprocessed()

    def init_processed(self):
        """ Init probMatrix based on learning off processed data
        """
        df = pd.DataFrame(columns=['hystrixConfig','faultInjection','error'])
        
        # populate the df with processed data
        # TODO Reduce this, way too much!
        for config in self.hystrixConfigs:
            for faultIn in self.faultInjections:
                path = 'data/' + config + '/' + faultIn + '.csv'
                # faultDf: index, statusCode, responseTime, error
                faultDf = pd.read_csv(path)
                size = len(faultDf.index)
                for i in range(0,size):
                    df.append([config,faultIn,faultDf.iloc[i,3]])

        # build the probability model based on processed data
        allparams = []

        # assign equal selecction probabilities to all combinations
        combs=len(self.hystrixConfigs)*len(self.faultInjections)
        selp=1/combs
        for config in self.hystrixConfigs:
            for faultIn in self.faultInjections:
                pp=[config,faultIn,selp]
                allparams.append(pp)
        
        self.probMatrix = pd.DataFrame(allparams, columns=['hystrixConfig', 'faultInjection', 'probabilities'])
        for config in self.hystrixConfigs:
            for faultIn in self.faultInjections:
                done=done+1
                print(str((done/combs)*100)+'%')
                if(len(df[(df['hystrixConfig']==config) & (df['faultInjection']==faultIn)])>0):
                    pro=len(df[(df['hystrixConfig']==config) & (df['faultInjection']==faultIn) & (df['error']==1)])/len(df[(df['hystrixConfig']==config) & (df['faultInjection']==faultIn)])
                    self.probMatrix.loc[((self.probMatrix['hystrixConfig']==config) & (self.probMatrix['faultInjection']==faultIn)), 'probabilities']=pro
                else:
                    self.probMatrix.loc[((self.probMatrix['hystrixConfig']==config) & (self.probMatrix['faultInjection']==faultIn)), 'probabilities']=selp
        
        if not os.path.isdir('bayesian_low'):
            os.makedirs('bayesian_low')
        self.probMatrix.to_csv('bayesian_low/probs.csv', header=['hystrixConfig', 'faultInjection', 'probabilities'])
    
    def init_nonprocessed(self):
        """ Init the probMatrix without any learning, same probability for each parameter combination
        """
        allparams = []
        # assign equal selecction probabilities to all combinations
        combs=len(self.hystrixConfigs)*len(self.faultInjections)
        selp=1/combs
        for config in self.hystrixConfigs:
            for faultIn in self.faultInjections:
                pp=[config,faultIn,selp]
                allparams.append(pp)
        
        self.probMatrix = pd.DataFrame(allparams, columns=['hystrixConfig', 'faultInjection', 'probabilities'])
        for config in self.hystrixConfigs:
            for faultIn in self.faultInjections:
                done=done+1
                print(str((done/combs)*100)+'%')
                self.probMatrix.loc[((self.probMatrix['hystrixConfig']==config) & (self.probMatrix['faultInjection']==faultIn)), 'probabilities']=selp
        if not os.path.isdir('bayesian_low'):
            os.makedirs('bayesian_low')
        self.probMatrix.to_csv('bayesian_low/probs.csv', header=['hystrixConfig', 'faultInjection', 'probabilities'])


    def next_experiment(self):
        # return the next experiment

        # select a random pattern
        #pat = random.choice(patterns)

        # Only care about 'a1false-a2false-b1false-c1false' for now! 
        # TODO Figure out whether this actually makes sense .. 
        # When changing: Need to add hystrixConfig parameter to Experiment constructor
        pat = 'a1false-a2false-b1false-c1false'

        condprobMatrix = self.probMatrix.ix[(self.probMatrix['hystrixConfig']==pat)]
        condprobMatrix = condprobMatrix.reset_index(drop=True)

        total_prob = sum(condprobMatrix['probabilities'])
        relative_probs = [p/total_prob for p in condprobMatrix['probabilities']]
        x = random.choices(condprobMatrix.index, weights = relative_probs, k = 1)
        indexSel=x[0]

        #select fault to inject using learning
        injectFault =condprobMatrix.iloc[indexSel, 2 ]
        split = str.split('-')

        # find the operation in the architectural model ..
        # TODO object-oriented approach for this is probably a terrible idea
        op = None
        for service in self.arch.microservices:
            for operation in service.operations:
                if operation.name == split[0]:
                    op = operation
        
        # build the experiment
        exp = experiment.Experiment(op,split[1])
        return exp



    def process_result(self,exp,result):
        # update the probMatrix

        # log the result
        exp.countcsv('bayesian_low/results.csv',result)
