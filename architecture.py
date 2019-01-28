import json

class Architecture:
    def __init__(self, archJson):
        with open(archJson, "r") as f:
            self.arch = json.load(f)
            self.microservices = []
            for ms in self.arch["microservices"]:
                self.microservices.append(Microservice(ms))
            for ms in self.microservices:
                for op in ms.operations:
                    op.init_dependencies(self.microservices)


class Microservice:
    def __init__(self,ms):
        self.name = ms['name']
        self.instances = ms['instances']
        self.patterns = ms['patterns']
        self.capacity = ms['capacity']
        self.operations = []
        for op in ms['operations']:
            self.operations.append(Operation(self,op))

class Pattern:
    def __init__(self,p):
        self.name = p['name']
        self.arguments = []
        for arg in p['arguments']:
            self.arguments.append(arg)

class Operation:
    def __init__(self,ms,operation):
        self.service = ms
        self.name = operation['name']
        self.demand = operation['demand']
        self.circuitbreaker = CircuitBreaker(operation['circuitBreaker'])
        self.dependencies = []
        for dp in operation['dependencies']:
            self.dependencies.append(Dependency(dp))

    def init_dependencies(self,mss):
        for dp in self.dependencies:
            dp.update(mss)

class CircuitBreaker:
    def __init__(self,cb):
        if cb != None:
            self.rollingwindow = cb['rollingWindow'] 
            self.requestvolumethreshold = cb['requestVolumeThreshold']
            self.errorthresholdpercentage = cb['errorThresholdPercentage']
            self.timeout = cb['timeout'] 
            self.sleepwindow = cb['sleepWindow'] 

class Dependency:
    def __init__(self,dp):
        self.service = dp['service'] # TODO get the correct service from the architecture
        self.operation = dp['operation'] # TODO get the correct operation from the architecture 
        self.probability = dp['probability']

    def update(self,mss):
        try:
            self.find_op(self.service,self.operation,mss)
        except Exception as error: 
            print("Caught this exception: " + repr(error))


    def find_op(self,name_ms,name_op,mss):
        for ms in mss:
            if ms.name == name_ms:
                self.service = ms
                for op in ms.operations:
                    if op.name ==name_op:
                        self.operation = op
                        return
                raise Exception("Couldnt find the operation: " + name_op)

        raise Exception("Couldn't find the microservice: " + name_ms)



def new(path):
    """ returns a new Architecture object from the provided model json
    """
    return Architecture(path)