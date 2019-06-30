from architecture import *
import pandas as pd
from algorithms import *
import time
from multiprocessing.dummy import Pool as ThreadPool 
from functools import partial
import statistics
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import csv

# define architectures
def init_architectures():
    """
    Initialize the architectures with the circuit breaker patterns. 
    """
    # TODO: Remove the patterns bit and use circuit breakers as specified in architectural_model.json
    patterns = ['a1false-a2false-b1true-c1false', 'a1false-a2false-b1false-c1true','a1false-a2false-b1true-c1true','a1true-a2false-b1false-c1false','a1false-a2true-b1false-c1false']#['a1false-a2false-b1false-c1false','a1true-a2false-b1false-c1false','a1false-a2true-b1false-c1false', 'a1true-a2true-b1false-c1false']

    architectures = []
    for pattern in patterns:
        architectures.append(Architecture('architecture_model.json', pattern))
    return architectures 

class FaultInjector(object):
    """
    Fault injector that returns mocked results
    """
    def __init__(self,architectures):
        """
        Initialize mocker, retrieve data for provided architectures.
        Takes sample with size n=5000 for each fault injection. 
        """
        # TODO: Pass data directory and integration points as parameters. 

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
    def inject_fault(self,architecture,faultinjection):
        """
        Retrieve mocked results of a fault injection. Return sample with n=1.  
        """
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
    # archMetrics is dictionary that tracks metrics for each architectures
    archMetrics = {}

    # perform experiments on all architectures
    for archi in architectures:
        metrics = {}
        metrics['ER-R1'] = {} 
        metrics['ER-R2'] = {}
        metrics['FDR-R1'] = {}
        metrics['FDR-R2'] = {}
        metrics['WFDR-R1'] = {}
        metrics['WFDR-R2'] = {}
        metrics['OPERATIONS-R1'] = {}
        metrics['OPERATIONS-R2'] = {} 
        metrics['TIME_GET'] = {}
        metrics['TIME_RESULT'] = {}
        metrics['INIT_TIME'] = {} 

        algorithms = []
        algorithms.append(RandomAlgorithm(archi,[]))
        algorithms.append(BanditEpsilonAlgorithm(archi,[0.5,0.99,0.1]))
        algorithms.append(BanditOptimisticAlgorithm(archi,[5]))
        algorithms.append(QLearningAlgorithm(archi,[0.2, 0.99, 0.66, 5]))
        algorithms.append(TableauAlgorithm(archi,[0.4, 0.99, 5, 0.1,5]))
        algorithms.append(NeuralNetworkAlgorithm(archi,[12]))
        algorithms.append(BayesianLowAlgorithm(archi,[100]))
        algorithms.append(BayesianHighAlgorithm(archi,[100])) 
        algorithms.append(MLFQAlgorithm(archi,[4,200]))

        # run algorithms in thread pool 
        pool = ThreadPool(len(algorithms))
        pool.map(partial(run_experiment,archi,metrics,mocker), algorithms)
        pool.close()
        pool.join()
        
        archMetrics[archi.patterns] = metrics
        generate_ouptput(metrics,archi)

def run_experiment(arch,metrics,mocker,algorithm):
    """
    Run the experiment for an algorithm and an architecture. 
    """
    # how many times to perform the experiment
    runs = 20
    # how many faults to find for ER metric
    faults_to_find = 150
    # how many experiments to run for FDR / counter / times metrics
    experiments_to_run = 400

    total = (runs * (faults_to_find + experiments_to_run))
    done = 0


    injection_results_R1 = []
    injection_selections_R1 = []
    #injection_results_R2 = []
    #injection_selections_R2 = []
    injection_times_get = []
    injection_times_result = []

    for run in range(runs):
        injection_results_R1.append({'ER': [], 'WFDR': []})
        #injection_results_R2.append({'ER': [], 'WFDR': []})

        injection_selections_R1.append({'ER': {}, 'WFDR': {}})
        #injection_selections_R2.append({'ER': {}, 'WFDR': {}})
        injection_times_get.append([])
        injection_times_result.append([])
        i = 0
        while i < faults_to_find:
            fault_injection = algorithm.get_experiment()
            if fault_injection in injection_selections_R1[run]['ER'].keys():
                injection_selections_R1[run]['ER'][fault_injection] += 1
            else: 
                injection_selections_R1[run]['ER'][fault_injection] = 0 
            result = mocker.inject_fault(arch,fault_injection)
            injection_results_R1[run]['ER'].append(result)

            if result > 0:
                i += 1
                done += 1
                algorithm.enter_result(1)
            else:
                algorithm.enter_result(0)
            if done % 1000 == 0:
                print("Progress (" + algorithm.name + "):" + str(round((done / total) * 100,2)) + "%")
        # reset the algorithm for next metric
        algorithm.reset() 

        #i = 0
        #while i < faults_to_find:
        #    fault_injection = algorithm.get_experiment()
        #    if fault_injection in injection_selections_R2[run]['ER'].keys():
        #        injection_selections_R2[run]['ER'][fault_injection] += 1
        #    else: 
        #        injection_selections_R2[run]['ER'][fault_injection] = 0 
        #    result = mocker.inject_fault(arch,fault_injection)
        #    injection_results_R2[run]['ER'].append(result)
        #    if result > 0:
        #        done += 1
        #        i += 1
        #    algorithm.enter_result(result)
        #    if done % 100 == 0:
        #        print("Progress (" + algorithm.name + "):" + str(round((done / total) * 100,2)) + "%")
        #algorithm.reset() 

        for i in range(experiments_to_run):
            start_time = int(time.time() * 1000)
            fault_injection = algorithm.get_experiment()
            injection_times_get[run].append(int(time.time() * 1000) - start_time)
            if fault_injection in injection_selections_R1[run]['WFDR'].keys():
                injection_selections_R1[run]['WFDR'][fault_injection] += 1
            else: 
                injection_selections_R1[run]['WFDR'][fault_injection] = 0 
            result = mocker.inject_fault(arch,fault_injection)
            injection_results_R1[run]['WFDR'].append(result)
            start_time = int(time.time() * 1000)
            if result > 0:
                algorithm.enter_result(1)
            else:
                algorithm.enter_result(0)
            injection_times_result[run].append(int(time.time() * 1000) - start_time)
            done += 1
            if done % 1000 == 0:
                print("Progress (" + algorithm.name + "):" + str(round((done / total) * 100,2)) + "%")
        algorithm.reset() 

        #for i in range(experiments_to_run):
        #    fault_injection = algorithm.get_experiment()
        #    if fault_injection in injection_selections_R2[run]['WFDR'].keys():
        #        injection_selections_R2[run]['WFDR'][fault_injection] += 1
        #    else: 
        #        injection_selections_R2[run]['WFDR'][fault_injection] = 0 
        #    result = mocker.inject_fault(arch,fault_injection)
        #    injection_results_R2[run]['WFDR'].append(result)
        #    algorithm.enter_result(result)
        #    done += 1
        #    if done % 100 == 0:
        #        print("Progress (" + algorithm.name + "):" + str(round((done / total) * 100,2)) + "%")
        #algorithm.reset() 

    metrics['ER-R1'][algorithm.name] = compute_ER(injection_results_R1,faults_to_find) 
    #metrics['ER-R2'][algorithm.name] = compute_ER(injection_results_R2,faults_to_find)
    metrics['FDR-R1'][algorithm.name] = compute_FDR(injection_results_R1,experiments_to_run)
    #metrics['FDR-R2'][algorithm.name] = compute_FDR(injection_results_R2,experiments_to_run)
    metrics['WFDR-R1'][algorithm.name] = compute_WFDR(injection_results_R1,experiments_to_run) 
    #metrics['WFDR-R2'][algorithm.name] = compute_WFDR(injection_results_R2,experiments_to_run) 
    metrics['OPERATIONS-R1'][algorithm.name] = compute_operations_counter(injection_selections_R1)
    #metrics['OPERATIONS-R2'][algorithm.name] = compute_operations_counter(injection_selections_R2)
    metrics['TIME_GET'][algorithm.name] = compute_time(injection_times_get)
    metrics['TIME_RESULT'][algorithm.name] = compute_time(injection_times_result)

def compute_ER(data,limit):
    """
    Computes the ER data from the raw data of all runs.
    """
    thresholds = [] 
    threshold = 10 
    while threshold <= limit: 
        thresholds.append(threshold)
        threshold += 10
    result = {}
    for threshold in thresholds:
        result[threshold] = []
    for run in data:
        for threshold in thresholds:
            counter = 0 
            len_data = len(run['ER'])
            for i in range(len_data):
                if run['ER'][i] > 0:
                    counter += 1
                if counter == threshold:
                    result[threshold].append(i+1)
                    break 
            

    for threshold in thresholds:
        result[threshold] = statistics.mean(result[threshold]) 

    return result 

def compute_FDR(data,total):
    """
    Compute the FDR data from the raw data of all runs.
    """
    threshold = 10
    thresholds = []
    while threshold <= total:
        thresholds.append(threshold)
        threshold += 10

    result = {}
    for threshold in thresholds:
        result[threshold] = []
    for run in data:
        for threshold in thresholds:
            counter = 0 
            len_data = len(run['WFDR'])
            for i in range(len_data):
                if run['WFDR'][i] > 0:
                    counter += 1
                if i+1 in thresholds:
                    result[i+1].append(counter)
            


    return result 

def compute_WFDR(data,total):
    """
    Compute the WFDR data from the raw data of all runs.
    """
    threshold = 10
    thresholds = []
    while threshold <= total:
        thresholds.append(threshold)
        threshold += 10

    result = {}
    for threshold in thresholds:
        result[threshold] = []
    for run in data:
        for threshold in thresholds:
            counter = 0 
            len_data = len(run['WFDR'])
            for i in range(len_data):
                counter += run['WFDR'][i]
                if i+1 in thresholds:
                    result[i+1].append(counter)
            
    return result 

def compute_time(data):
    """
    Compute the time data from the raw data of all runs.
    """
    total_sum = 0
    total_len = 0
    
    for run in data:
        total_len += len(run)
        total_sum += sum(run) 
    
    return total_sum / total_len

def compute_operations_counter(data):
    """
    Compute the counters data from the raw data of all runs.
    """
    result = []
    for run in range(len(data)):
        result.append(data[run]['WFDR'])
    return result

def generate_ouptput(metrics,archi):
    """
    Generate visualizations and tables for the metrics.
    """
    ER1 = metrics['ER-R1']
    #ER2 = metrics['ER-R2']
    FDR1 = metrics['FDR-R1']
    #FDR2 = metrics['FDR-R2']
    WFDR1 = metrics['WFDR-R1']
    #WFDR2 = metrics['WFDR-R2']
    TIME_GET = metrics['TIME_GET']
    TIME_RESULT = metrics['TIME_RESULT']
    INIT_TIME = metrics['INIT_TIME']
    OPERATIONS1 = metrics['OPERATIONS-R1']
    #OPERATIONS2 = metrics['OPERATIONS-R2']

    visualize_er([ER1],archi)

    visualize_fdr([FDR1],archi,'FDR')
    visualize_fdr([WFDR1],archi,'WFDR')

    visualize_time(TIME_GET, 'average time for get_experiment()')
    visualize_time(TIME_RESULT, 'average time for enter_result)')

    table_counter([OPERATIONS1], archi)


def visualize_er(data,archi):
    """
    Visualize the ER metric in line plots, for both the rate and the total.
    """
    patterns = archi.patterns 


    for i in range(len(data)):
        for algorithm in data[i].keys():
            xvals = sorted(data[i][algorithm].keys()) 
            yvals = [] 
            for key in xvals:
                yvals.append(data[i][algorithm][key]) 

            plt.plot(xvals,yvals,label = algorithm)
        
        plt.title('ER for ' + patterns + ', reward function ' + str(i+1))
        plt.xlabel('faults found')
        plt.ylabel('experiments run') 
        plt.legend(loc='best')
        path = './figures/' + patterns + '-' + str(i+1) + '-ER-line-total.pdf' 
        plt.savefig(path)
        plt.close()
    
    for i in range(len(data)):
        for algorithm in data[i].keys():
            xvals = sorted(data[i][algorithm].keys()) 
            yvals = [] 
            for key in xvals:
                yvals.append(data[i][algorithm][key]/key) 
            plt.plot(xvals,yvals,label = algorithm)
            
        plt.title('ER for ' + patterns + ', reward function ' + str(i+1))
        plt.xlabel('faults found')
        plt.ylabel('ER') 
        plt.legend(loc='best')
        path = './figures/' + patterns + '-' + str(i+1) + '-ER-line-rate.pdf' 
        plt.savefig(path)
        plt.close()

def visualize_fdr(data,archi,yaxis):
    """
    Visualize the FDR metric as line and box plots, for both the rate and total.
    """
    patterns = archi.patterns 

    for i in range(len(data)):  
        for algorithm in data[i].keys():
            xvals = sorted(data[i][algorithm].keys()) 
            yvals = [] 
            for key in xvals:
                yvals.append(statistics.mean(data[i][algorithm][key]))   
            plt.plot(xvals,yvals,label = algorithm) 
        plt.title(yaxis + ' for ' + patterns + ', reward function ' + str(i+1))
        plt.xlabel('experiments run')
        plt.ylabel('faults found') 
        plt.legend(loc='best')
        path = './figures/' + patterns + '-' + str(i+1) + '-' + yaxis + '-line-total.pdf'
        plt.savefig(path)
        plt.close()

        for algorithm in data[i].keys():
            xvals = sorted(data[i][algorithm].keys()) 
            yvals = [] 
            for key in xvals:
                yvals.append(statistics.mean(data[i][algorithm][key])/key)   
            plt.plot(xvals,yvals,label = algorithm) 
        plt.title(yaxis + ' for ' + patterns + ', reward function ' + str(i+1))
        plt.xlabel('experiments run')
        plt.ylabel(yaxis) 
        plt.legend(loc='best')
        path = './figures/' + patterns + '-' + str(i+1) + '-' + yaxis + '-line-rate.pdf'
        plt.savefig(path)
        plt.close()



        labels = sorted(data[i].keys())
        data_all = []
        for algorithm in labels:
            max_key = max(data[i][algorithm].keys())
            data_all.append([x / max_key for x in data[i][algorithm][max_key]]) 

        sns.set_style("whitegrid")
        ax = sns.boxplot(data=data_all,sym='',palette="Greys")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticklabels(labels)
        plt.xticks(rotation=90)
        plt.gcf().subplots_adjust(bottom=0.25)
        ax.yaxis.set_label_text(yaxis + ' for ' + str(max_key) + ' experiments')
        ax.set_title(yaxis + ' for ' + patterns + ', reward function ' + str(i+1))
        formatter = mtick.ScalarFormatter(useOffset=False)
        ax.yaxis.set_major_formatter(formatter)
        path = './figures/' + patterns + '-' + str(i+1) + '-' + yaxis + '-box.pdf'
        plt.savefig(path)
        plt.close()
        
def visualize_time(data,title):
    """
    Visualize the time metric as bar plots.
    """
    xvals = sorted(data.keys())
    yvals = [] 
    for x in xvals:
        yvals.append(data[x])
    
    plt.bar(xvals,yvals) 
    plt.xticks(rotation=90)
    plt.ylabel('ms')
    plt.title(title)
    plt.gcf().subplots_adjust(bottom=0.2)

    if 'experiment' in title:
        path = './figures/times_get.pdf' 
    else: 
        path = './figures/times_result.pdf'
        
    plt.savefig(path)
    plt.close() 

def table_counter(data, archi):
    """
    Construct a table for the counter data.
    """
    patterns = archi.patterns 

    for i in range(len(data)):
        result = {}
        algorithms = sorted(data[i].keys())
        for algorithm in algorithms:
            run_results = {}
            for run in data[i][algorithm]:
                fault_injections = sorted(run.keys())
                for fault_injection in fault_injections:
                    if fault_injection in run_results.keys(): 
                        run_results[fault_injection].append(run[fault_injection])
                    else:
                        run_results[fault_injection]=[run[fault_injection]]
            
            fault_injections = sorted(run_results.keys())
            for fault_injection in fault_injections:
                if fault_injection in result.keys():
                    result[fault_injection].append(statistics.mean(run_results[fault_injection]))
                else:
                    result[fault_injection] = [statistics.mean(run_results[fault_injection])]
        
        csv_columns = ['fault-injection'] + algorithms 

        csv_file = './figures/' + patterns + '-' + str(i+1) + '-counter.csv'
        try:
            with open(csv_file, 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter = ',')
                writer.writerow(csv_columns) 

                fault_injections = sorted(result.keys())
                for fault_injection in fault_injections:
                    row = [fault_injection]
                    for value in result[fault_injection]:
                        row.append(str(round(value,2)))
                    writer.writerow(row)
        except IOError:
            print("I/O error")


if __name__ == "__main__":
    main()