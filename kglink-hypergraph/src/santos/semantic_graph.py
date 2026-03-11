import networkx as nx

class SemanticGraph:

    def __init__(self):

        self.graph = nx.Graph()

    def add_column(self, col, emb):

        self.graph.add_node(col, embedding=emb)

    def add_relation(self, c1, c2, label):

        self.graph.add_edge(c1, c2, relation=label)