import json
class Architecture:
    def __init__(self, archJson, patterns):
        with open(archJson, "r") as f:
            arch = json.load(f)
            self.services = []
            for ms in arch["microservices"]:
                self.services.append(Microservice(ms))
            for ms in self.services:
                for op in ms.operations:
                    op.init_dependencies(self.services)

        self.patterns = patterns 

        for i in range(len(self.services)):
            for j in range(len(self.services[i].operations)):
                if self.services[i].operations[j].name + 'true' in patterns:
                    cb = CircuitBreaker(arch["microservices"][i]['operations'][j]['circuitBreaker'])
                    
                    ops = []
                    ops.append(self.services[i].operations[j])
                    seen = [self.services[i].operations[j]]

                    while len(ops) > 0:
                        operation = ops.pop()
                        for dep in operation.dependencies:
                            ops.append(dep.operation)
                            seen.append(dep.operation)
                            dep.operation.circuitbreaker = cb 

    def get_operations(self):
        operations = []
        for service in self.services:
            for operation in service.operations: 
                operations.append(operation)
        return operations
    
    def get_incoming_dependencies(self,operation):
        operations = set()
        for service in self.services:
            for op in service.operations:
                for dep in op.dependencies:
                    if dep.operation.name == operation.name and op.name not in ['a1','a2']:
                        operations.add(op)

        return list(operations)
    def to_string(self):
        results = self.patterns + "\n"
        results = results + "Services \n"
        for service in self.services:
            results = results + service.name + "\n"
            for operation in service.operations:
                if operation.circuitbreaker:

                    results = results + operation.name + " " + str(operation.circuitbreaker.rollingwindow) + ", "
                else:
                    results = results + operation.name + ", "
                for dep in operation.dependencies:
                    results = results + dep.operation.name + " "
                results = results + "\n"
        return results
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
        self.circuitbreaker = None # CircuitBreaker(operation['circuitBreaker'])
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
        else:
            self.rollingWindow = None 
            self.requestvolumethreshold = None 
            self.errorthresholdpercentage = None 
            self.timeout = None 
            self.sleepwindow = None
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
                        return op
                raise Exception("Couldnt find the operation: " + name_op)

        raise Exception("Couldn't find the microservice: " + name_ms)


