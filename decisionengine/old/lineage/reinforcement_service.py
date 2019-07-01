import json
import graph 


# build dependency graph from architecture
# TODO: Add weights by weighing features, e.g. circuit breaker
dep_graph = graph.Graph()
with open("architecture_model.json", 'r') as f:
    architecture = json.load(f)

    nMicros = len(architecture["microservices"])
    for service in range(0,nMicros):
        vertex = architecture["microservices"][service]['name']
        dep_graph.add_vertex(vertex)
        
        nop = len(architecture["microservices"][service]['operations'])
        for op in range(0,nop):
            for dep in architecture["microservices"][service]['operations'][op]['dependencies']:
                dep_graph.add_edge({vertex, dep['service']})

print(dep_graph.__str__())