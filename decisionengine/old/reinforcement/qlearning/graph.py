

class Graph(object):
    def __init__(self,graph_dict=None):
        if graph_dict == None:
            graph_dict = {}
        self.__graph_dict = graph_dict 

    def vertices(self):
        return list(self.__graph_dict.keys())
    
    def edges(self):
        return self.__generate_edges()
    
    def add_vertex(self,vertex):
        if vertex not in self.__graph_dict:
            self.__graph_dict[vertex]=[]
    
    def add_edge(self,edge):
        if edge[0] in self.__graph_dict:
            self.__graph_dict[edge[0]].append(edge[1])
        else:
            self.__graph_dict[edge[0]] = [edge[1]]
            print(edge)
        
    def __generate_edges(self):
        edges = []
        for vertex in self.__graph_dict:
            for neighbor in self.__graph_dict[vertex]:
                if {neighbor, vertex} not in edges:
                    edges.append({vertex, neighbor})
        return edges 
    
    def __str__(self):
        res = "vertices: " 
        for k in self.__graph_dict:
            res += str(k) + " " 
        res += "\nedges: "
        for edge in self.__generate_edges(): 
            res += str(edge) + " " 
        return res 

    def vertex_incoming(self,vertex):
        adj_vertices = []
        for key, val in self.__graph_dict.items():
            if key != vertex:
                if vertex[0] in val:
                    adj_vertices.append(key)
        return adj_vertices

    def vertex_outgoing(self,vertex):
        return self.__graph_dict[vertex]