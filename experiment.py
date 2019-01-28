import csv

class Experiment:
    def __init__(self,operation,action):
        self.version = "1.0.0"
        self.description = "Generated experiment"
        self.ssh = SteadyStateHypothesis()
        self.ssh.add_probe(Probe(operation))
        self.method = Method(action) 

    def print(self):
        """ prints a description of the experiment containing service, operation, and action
        """
        print("Service: " + self.ssh.probes[0].provider.url.service.name 
            + ", Operation: " + self.ssh.probes[0].provider.url.name 
            + ", Action: " + self.method.action)

    def row(self):
        """ returns [service.name,operation.name,action.name]
        """
        row = [self.ssh.probes[0].provider.url.service.name,self.ssh.probes[0].provider.url.name,self.method.action]
        return row

    def countcsv(self,path):
        """ increments the counter of the experiment in the provided csv (assuming the row [service,operation,action,count] exists)
        """
        experiments = []
        # Read all data from the csv file.
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader: 
                experiments.append(row)
        exp = self.row()

        line = 0
        for row in experiments:
            if row[0] == exp[0] and row[1] == exp[1] and row[2] == exp[2]:
                row[3] = str(int(row[3]) + 1)
                line_to_override = {line:row}
                continue
            line = line + 1
        
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

class Probe:
    def __init__(self,target):
        self.tolerance = True
        self.type = "probe"
        self.name = "request"
        self.provider = Provider("HTTP",target) # TODO generate target URL

class Provider:
    def __init__(self,kind,target):
        self.type = kind
        if self.type == "HTTP":
            self.timeout = 5
            self.url = target
        # TODO support different providers

class Method:
    def __init__(self,action):
        self.action = action
        # TODO figure out fault injection options