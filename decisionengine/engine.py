from architecture import *
import pandas as pd
from algorithms import *
import time
from multiprocessing.dummy import Pool as ThreadPool 
from functools import partial

# define architectures
def init_architectures():
    patterns = ['a1false-a2false-b1false-c1false','a1true-a2false-b1false-c1false','a1false-a2true-b1false-c1false',    'a1true-a2true-b1false-c1false']

    architectures = []
    for pattern in patterns:
        architectures.append(Architecture('architecture_model.json', pattern))
    return architectures 
# define mocking
class FaultInjector(object):
    def __init__(self,architectures):
        self.cached_results = {}
        self.date = 'experiment-2018-11-04T08-32-02UTC'

        for architecture in architectures:
            patterns = architecture.patterns 
            if patterns not in self.cached_results.keys():
                self.cached_results[patterns] = {}
                fault_types = ['abort','delay']
                for operation in architecture.get_operations():
                    if operation.name not in ['a1','a2']:
                        for fault in fault_types:
                            faultinj = operation.name + '-' + fault
                            path = './data_service/' + 'a1' + '/' + self.date + '/' + patterns + '/' + faultinj +'.csv'
                            results_1 = pd.read_csv(path).sample(n=5000)
                            path = './data_service/' + 'a2' + '/' + self.date + '/' + patterns + '/' + faultinj +'.csv'
                            results_2 = pd.read_csv(path).sample(n=5000)
                            self.cached_results[patterns][faultinj] = {'a1': results_1, 'a2': results_2}
    def lookup_result(self,architecture,faultinjection):
        patterns = architecture.patterns
        faults = 0
        for ip in ['a1','a2']:
            results = self.cached_results[patterns][faultinjection][ip]
            sample = results.sample(n=1)
            threshold = 0.06
            if ((sample.iloc[0,2] > threshold) or (sample.iloc[0,1]==500)):
                faults += 1

        return faults


def main():
    architectures = init_architectures()
    mocker = FaultInjector(architectures)
    archMetrics = {}
    for archi in architectures:
        algorithms = []
        algorithms.append(RandomAlgorithm(archi,[]))
        algorithms.append(BanditEpsilonAlgorithm(archi,[0.5,0.99,0.1]))
        algorithms.append(BanditOptimisticAlgorithm(archi,[5]))
        algorithms.append(QLearningAlgorithm(archi,[0.2, 0.99, 0.66, 5]))

        metrics = {}
        metrics['ER-R1'] = {}
        metrics['ER-R2'] = {}
        metrics['FDR'] = {}
        metrics['WFDR'] = {}
        metrics['TIME'] = {}
        
        pool = ThreadPool(len(algorithms))
        pool.map(partial(run_experiment,archi,metrics,mocker), algorithms)
        pool.close()
        pool.join()
        
        archMetrics[archi.patterns] = metrics

def run_experiment(arch,metrics,mocker,algorithm):

    metrics['ER-R1'][algorithm.name] = {}
    metrics['ER-R2'][algorithm.name] = {}
    metrics['FDR'][algorithm.name] = {}
    metrics['WFDR'][algorithm.name] = {}
    metrics['TIME'][algorithm.name] = {}

    # ER
    runs = 5
    faults_to_find_all = [50]#,100,200,500,1000,5000]
    # FDR AND WFDR
    fault_injections_all = [50]#,100,200,500,1000,5000]

    total_faults_to_find = runs
    total_fault_injections = runs
    for num in faults_to_find_all:
        total_faults_to_find = total_faults_to_find * num 
    for num in fault_injections_all: 
        total_fault_injections = total_fault_injections * num
    total = (total_faults_to_find + total_fault_injections) * 2
    done = 0


    for run in range(runs):
        for faults_to_find in faults_to_find_all:
            if faults_to_find not in metrics['ER-R1'][algorithm.name].keys():
                metrics['ER-R1'][algorithm.name][faults_to_find] = [0]
            else:
                metrics['ER-R1'][algorithm.name][faults_to_find].append(0)
            
            if 'ER-R1' not in metrics['TIME'][algorithm.name].keys():
                metrics['TIME'][algorithm.name]['ER-R1'] = []

            if faults_to_find not in metrics['ER-R2'][algorithm.name].keys():
                metrics['ER-R2'][algorithm.name][faults_to_find] = [0]
            else:
                metrics['ER-R2'][algorithm.name][faults_to_find].append(0)
            
            if 'ER-R2' not in metrics['TIME'][algorithm.name].keys():
                metrics['TIME'][algorithm.name]['ER-R2'] = []

            time_start = int(time.time() * 1000)
            for i in range(faults_to_find):
                fault_injection = algorithm.get_action()
                result = mocker.lookup_result(arch,fault_injection)
                if result > 0:
                    result = 1
                    metrics['ER-R1'][algorithm.name][faults_to_find][-1] += result
                algorithm.result(result)
                done += 1
                if done % 100 == 0:
                    print("Progress (" + algorithm.name + "):" + str(round((done / total) * 100,2)) + "%")
            metrics['TIME'][algorithm.name]['ER-R1'].append(int(time.time() * 1000) - time_start)
            algorithm.reset()

            time_start = int(time.time() * 1000)
            for i in range(faults_to_find):
                fault_injection = algorithm.get_action()
                result = mocker.lookup_result(arch,fault_injection)
                if result > 0:
                    metrics['ER-R2'][algorithm.name][faults_to_find][-1] += result
                algorithm.result(result)
                done += 1
                if done % 100 == 0:
                    print("Progress (" + algorithm.name + "):" + str(round((done / total) * 100,2)) + "%")
            metrics['TIME'][algorithm.name]['ER-R2'].append(int(time.time() * 1000) - time_start)
            algorithm.reset()

        for fault_injections in fault_injections_all: 
            if fault_injections not in metrics['FDR'][algorithm.name].keys():
                metrics['FDR'][algorithm.name][fault_injections] = [0]
            else: 
                metrics['FDR'][algorithm.name][fault_injections].append(0)
            if fault_injections not in metrics['WFDR'][algorithm.name].keys():
                metrics['WFDR'][algorithm.name][fault_injections] = [0]
            else: 
                metrics['WFDR'][algorithm.name][fault_injections].append(0)
            if 'FDR' not in metrics['TIME'][algorithm.name].keys():
                metrics['TIME'][algorithm.name]['FDR'] = []
            if 'WFDR' not in metrics['TIME'][algorithm.name].keys():
                metrics['TIME'][algorithm.name]['WFDR'] = []


            time_start = int(time.time() * 1000)
            for i in range(faults_to_find):
                fault_injection = algorithm.get_action()
                result = mocker.lookup_result(arch,fault_injection)
                if result > 0:
                    result = 1
                    metrics['FDR'][algorithm.name][faults_to_find][-1] += result
                algorithm.result(result)
                done += 1
                if done % 10 == 0:
                    print("Progress (" + algorithm.name + "):" + str((done / total) * 100) + "%")
            metrics['TIME'][algorithm.name]['FDR'].append(int(time.time() * 1000) - time_start) 
            algorithm.reset()

            time_start = int(time.time() * 1000) 
            for i in range(faults_to_find):
                fault_injection = algorithm.get_action()
                result = mocker.lookup_result(arch,fault_injection)
                if result > 0:
                    metrics['WFDR'][algorithm.name][faults_to_find][-1] += result
                algorithm.result(result)
                done += 1
                if done % 10 == 0:
                    print("Progress (" + algorithm.name + "):" + str((done / total) * 100) + "%")
            metrics['TIME'][algorithm.name]['WFDR'].append(int(time.time() * 1000) - time_start)
            algorithm.reset()
    print(metrics)

if __name__ == "__main__":
    main()