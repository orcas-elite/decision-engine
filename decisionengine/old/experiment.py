import csv
import json
from collections import OrderedDict
class Experiment:
    def __init__(self,operation,action):
        self.title = "chaosExperiment"
        self.version = "1.0.0"
        self.description = "Generated experiment"
        self.ssh = SteadyStateHypothesis()
        self.ssh.add_probe(Probe(operation,action))
        self.method = []
        self.method.append(Method(operation.service.name, operation.name,action))

    def reprJSON(self):
        ## use OrderedDict for hyphen in steady-state-hypothesis
        return OrderedDict([
            ('title',self.title),
            ('version',self.version),
            ('description',self.description),
            ('steady-state-hypothesis', self.ssh.reprJSON()),
            ('method',[self.method[0].reprJSON()])])
    def row(self):
        """ returns [service.name,operation.name,action.name]
        """
        row = [self.ssh.probes[0].provider.arguments.service,self.ssh.probes[0].provider.arguments.operation,self.method[0].provider.arguments.action]
        return row

    def countcsv(self,path,result):
        """ increments the counter of the experiment in the provided csv (assuming the row [service,operation,action,result,count] exists)
        """
        experiments = []
        # Read all data from the csv file.
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader: 
                experiments.append(row)
        exp = self.row()
        exp.append(result)
        line = 0
        found = False
        for row in experiments:
            if row[0] == exp[0] and row[1] == exp[1] and row[2] == exp[2] and row[3] == exp[3]:
                row[4] = str(int(row[4]) + 1)
                line_to_override = {line:row}
                found = True
                continue
            line = line + 1
        if found == False:
            with open(path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(exp)
        else:
            with open(path, 'w') as f:
                writer = csv.writer(f)
                for line, row in enumerate(experiments):
                    data = line_to_override.get(line, row)
                    writer.writerow(data)
        

class SteadyStateHypothesis:
    def __init__(self):
        self.title = "ssh" # ?
        self.probes = [] # ?
    def add_probe(self,probe):
        self.probes.append(probe)
    def reprJSON(self):
        return dict(title=self.title,probes=[self.probes[0].reprJSON()])
class Probe:
    def __init__(self,operation,action):
        self.type="probe"
        self.tolerance = '0'
        self.name = "mock-results"
        self.provider = Provider("python","chaosmock","get_result",Arguments(operation.service.name,operation.name,action,"a1false-a2false-b1false-c1false"))
    def reprJSON(self):
        return dict(type=self.type,tolerance=self.tolerance,name=self.name,provider=self.provider.reprJSON())
class Provider:
    def __init__(self,kind,module,func,args):
        self.type = kind
        self.module = module
        self.func = func
        self.arguments = args
    
    def reprJSON(self):
        return dict(type=self.type,module=self.module,func=self.func,arguments=self.arguments.reprJSON())

class Method:
    def __init__(self,service,operation,action):
        self.type = "action"
        self.name = "print-experiment"
        self.provider = Provider("python","chaosmock","print_experiment",Arguments(service,operation,action,"a1false-a2false-b1false-c1false"))
    def reprJSON(self):
        return dict(type=self.type,name=self.name,provider=self.provider.reprJSON())

class Arguments:
    def __init__(self,service,operation,action,hystrix):
        self.service = service
        self.operation = operation 
        self.action = action 
        self.hystrix = hystrix
    def reprJSON(self):
        return dict(service=self.service,operation=self.operation,action=self.action,hystrix=self.hystrix)