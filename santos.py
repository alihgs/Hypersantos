import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter, defaultdict
import itertools
import math

############################################################
# Utility functions
############################################################

def normalize_string(x):
    if isinstance(x, str):
        return x.strip().lower()
    return x

def unique_values(series):
    return set(series.dropna().apply(normalize_string))

############################################################
# Column Semantics
############################################################

class ColumnSemantics:

    def __init__(self):
        self.type_counts = defaultdict(int)
        self.total = 0

    def add(self, type_name):
        self.type_counts[type_name] += 1
        self.total += 1

    def compute_scores(self):

        scores = {}

        for t, count in self.type_counts.items():
            fs = count / self.total

            # granularity score approximation
            gs = 1 / max(1, math.log(count + 1))

            scores[t] = fs * gs

        return scores


def infer_simple_type(value):

    if isinstance(value, str):
        if value.isdigit():
            return "number"
        return "text"

    if isinstance(value, int) or isinstance(value, float):
        return "number"

    return "unknown"


def compute_column_semantics(column):

    cs = ColumnSemantics()

    for v in column.dropna():

        t = infer_simple_type(v)

        cs.add(t)

    return cs.compute_scores()

############################################################
# Relationship Semantics
############################################################

class RelationshipSemantics:

    def __init__(self):
        self.relations = Counter()
        self.total = 0

    def add(self, relation):

        self.relations[relation] += 1
        self.total += 1

    def best_relation(self):

        if self.total == 0:
            return None, 0

        rel, count = self.relations.most_common(1)[0]

        score = count / self.total

        return rel, score


def infer_relation(v1, v2):

    if isinstance(v1, str) and isinstance(v2, str):
        return "text_relation"

    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        return "numeric_relation"

    return None


def compute_relationship_semantics(col1, col2):

    rs = RelationshipSemantics()

    for v1, v2 in zip(col1, col2):

        if pd.isna(v1) or pd.isna(v2):
            continue

        r = infer_relation(v1, v2)

        if r:
            rs.add(r)

    return rs.best_relation()

############################################################
# Semantic Graph
############################################################

class SemanticGraph:

    def __init__(self, table_name):

        self.table_name = table_name
        self.graph = nx.Graph()

    def add_column(self, col_name, semantics):

        self.graph.add_node(col_name, semantics=semantics)

    def add_relationship(self, col1, col2, relation, score):

        self.graph.add_edge(col1, col2,
                            relation=relation,
                            score=score)

    def columns(self):

        return list(self.graph.nodes)

    def relationships(self):

        return list(self.graph.edges(data=True))


def build_semantic_graph(df, table_name):

    sg = SemanticGraph(table_name)

    # nodes
    for col in df.columns:

        cs = compute_column_semantics(df[col])

        sg.add_column(col, cs)

    # edges
    for c1, c2 in itertools.combinations(df.columns, 2):

        rel, score = compute_relationship_semantics(df[c1], df[c2])

        if rel:
            sg.add_relationship(c1, c2, rel, score)

    return sg

############################################################
# Matching functions
############################################################

def column_match(cs_q, cs_t):

    best = 0

    for tq, sq in cs_q.items():

        if tq in cs_t:

            st = cs_t[tq]

            score = sq * st

            if score > best:
                best = score

    return best


def relation_match(rel_q, rel_t):

    if rel_q is None or rel_t is None:
        return 0

    rq, sq = rel_q
    rt, st = rel_t

    if rq == rt:
        return sq * st

    return 0

############################################################
# Pair matching
############################################################

def pair_match(q_graph, t_graph, qc1, qc2, tc1, tc2):

    q_cs1 = q_graph.graph.nodes[qc1]["semantics"]
    q_cs2 = q_graph.graph.nodes[qc2]["semantics"]

    t_cs1 = t_graph.graph.nodes[tc1]["semantics"]
    t_cs2 = t_graph.graph.nodes[tc2]["semantics"]

    col_score1 = column_match(q_cs1, t_cs1)
    col_score2 = column_match(q_cs2, t_cs2)

    if not q_graph.graph.has_edge(qc1, qc2):
        return 0

    if not t_graph.graph.has_edge(tc1, tc2):
        return 0

    q_edge = q_graph.graph[qc1][qc2]
    t_edge = t_graph.graph[tc1][tc2]

    rel_q = (q_edge["relation"], q_edge["score"])
    rel_t = (t_edge["relation"], t_edge["score"])

    rel_score = relation_match(rel_q, rel_t)

    return col_score1 * rel_score * col_score2

############################################################
# Unionability score
############################################################

def unionability_score(q_graph, t_graph):

    score = 0

    for qc1, qc2 in itertools.combinations(q_graph.columns(), 2):

        for tc1, tc2 in itertools.combinations(t_graph.columns(), 2):

            s = pair_match(q_graph, t_graph, qc1, qc2, tc1, tc2)

            score += s

    return score

############################################################
# Data Lake
############################################################

class DataLake:

    def __init__(self):

        self.tables = {}
        self.graphs = {}

    def add_table(self, name, dataframe):

        self.tables[name] = dataframe

        graph = build_semantic_graph(dataframe, name)

        self.graphs[name] = graph

    def search_unionable(self, query_df, k=5):

        query_graph = build_semantic_graph(query_df, "query")

        scores = []

        for name, graph in self.graphs.items():

            s = unionability_score(query_graph, graph)

            scores.append((name, s))

        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:k]

############################################################
# Example usage
############################################################

if __name__ == "__main__":

    dl = DataLake()

    parks = pd.DataFrame({
        "Park Name": ["River Park", "West Lawn"],
        "City": ["Fresno", "Chicago"],
        "Country": ["USA", "USA"]
    })

    films = pd.DataFrame({
        "Park Name": ["Chippewa Park", "Lawler Park"],
        "Park City": ["Cook", "Riverside"],
        "Film": ["Bee Movie", "Coco"]
    })

    people = pd.DataFrame({
        "Person": ["James", "Anthony"],
        "Birthplace": ["Boston", "London"],
        "Country": ["USA", "UK"]
    })

    dl.add_table("parks", parks)
    dl.add_table("films", films)
    dl.add_table("people", people)

    query = parks

    results = dl.search_unionable(query, k=3)

    print("\nTop unionable tables:\n")

    for name, score in results:
        print(name, score)