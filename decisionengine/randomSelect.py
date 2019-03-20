import abc 
import numpy as np
import experiment
import architecture
import csv
import strategy

class RandomSelection(strategy.Strategy):
    """ Randomly select experiment 
    """
    arch = []
    experiments = []
    csvPath = "experiments_random.csv"

    def next_experiment(self):
        index = np.random.randint(0,len(self.experiments))
        return self.experiments[index]

    def init(self,arch):
        # pass the architecture
        self.arch = arch
        self.get_experiments()
        self.init_csv()

    def process_result(self,exp,result):
        # count in csv
        exp.countcsv(self.csvPath,result)

    def get_experiments(self):
        # generate experiments
        actions = ["delay","abort"]
        for ms in self.arch.microservices:
            for op in ms.operations:
                for action in actions:
                    self.experiments.append(experiment.Experiment(op,action)) 

    def init_csv(self):
        ## prepare csv file with count 0 for each experiment
        with open(self.csvPath, "w") as f:
            writer = csv.writer(f)
            for exp in self.experiments:
                line = exp.row()
                line.append("0")
                line.append(str(0))
                writer.writerow(line)
                line = exp.row()
                line.append("1")
                line.append(str(0))
                writer.writerow(line)
