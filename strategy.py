import abc 
import numpy as np
import experiment

class Context:
    """
    Define the interface of interest to clients.
    """

    def __init__(self, strategy):
        self._strategy = strategy

    def next_experiment(self,experiments):
        self._strategy.next_experiment(experiments)
class Strategy(metaclass=abc.ABCMeta):
    """ Declare interface for nextExperiment 
    """
    @abc.abstractmethod
    def next_experiment(self,experiments):
        pass

class RandomSelection(Strategy):
    """ Randomly select experiment 
    """

    def next_experiment(self,experiments):
        index = np.random.randint(0,len(experiments))
        experiments[index].countcsv("experiments.csv")
