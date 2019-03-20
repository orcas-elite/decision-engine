import abc 
import numpy as np
import experiment
import architecture
import csv

class Context:
    """
    Define the interface of interest to clients.
    """

    def __init__(self, strategy):
        self._strategy = strategy

    def next_experiment(self):
        return self._strategy.next_experiment()
    
    def init(self,arch):
        self._strategy.init(arch)

    def process_result(self,exp,result):
        self._strategy.process_result(exp,result)

class Strategy(metaclass=abc.ABCMeta):
    """ Declare interface for nextExperiment 
    """
    @abc.abstractmethod
    def next_experiment(self):
        pass
    @abc.abstractmethod
    def process_result(self,exp,result):
        pass
    @abc.abstractmethod
    def init(self,arch):
        pass









