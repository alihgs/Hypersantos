import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer


# ============================================================
# Utils
# ============================================================

def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def unique_non_empty(series: pd.Series) -> List[str]:
    vals = [norm_text(v) for v in series.dropna().tolist()]
    vals = [v for v in vals if v]
    return list(dict.fromkeys(vals))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def mean_pool(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) == 0:
        return np.zeros((384,), dtype=np.float32)
    return np.mean(vectors, axis=0)


# ============================================================
# Embedding model
# ============================================================

class BertEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.cache: Dict[str, np.ndarray] = {}

    def encode_text(self, text: str) -> np.ndarray:
        text = norm_text(text)
        if text not in self.cache:
            self.cache[text] = self.model.encode(text, normalize_embeddings=True)
        return self.cache[text]

    def encode_many(self, texts: List[str]) -> np.ndarray:
        clean = [norm_text(t) for t in texts]
        missing = [t for t in clean if t not in self.cache]
        if missing:
            embs = self.model.encode(missing, normalize_embeddings=True)
            for t, e in zip(missing, embs):
                self.cache[t] = e
        return np.array([self.cache[t] for t in clean])


# ============================================================
# SANTOS+BERT semantic objects
# ============================================================

@dataclass
class ColumnInfo:
    name: str
    header_emb: np.ndarray
    values_emb: np.ndarray
    column_emb: np.ndarray


@dataclass
class RelationInfo:
    label: str
    score: float
    emb: np.ndarray


class SemanticGraph:
    def __init__(self, table_name: str):
        self.table_name = table_name
        self.graph = nx.Graph()

    def add_column(self, col_name: str, info: ColumnInfo) -> None:
        self.graph.add_node(col_name, info=info)

    def add_relation(self, c1: str, c2: str, rel: RelationInfo) -> None:
        self.graph.add_edge(c1, c2, rel=rel)

    def columns(self) -> List[str]:
        return list(self.graph.nodes)

    def has_relation(self, c1: str, c2: str) -> bool:
        return self.graph.has_edge(c1, c2)

    def relation(self, c1: str, c2: str) -> RelationInfo:
        return self.graph[c1][c2]["rel"]

    def column_info(self, c: str) -> ColumnInfo:
        return self.graph.nodes[c]["info"]


# ============================================================
# Column semantics with BERT
# ============================================================

def build_column_info(df: pd.DataFrame, col: str, encoder: BertEncoder) -> ColumnInfo:
    header_emb = encoder.encode_text(col)

    values = unique_non_empty(df[col])
    if values:
        value_embs = encoder.encode_many(values)
        values_emb = mean_pool(value_embs)
    else:
        values_emb = np.zeros_like(header_emb)

    # mix header + values
    column_emb = 0.4 * header_emb + 0.6 * values_emb
    denom = np.linalg.norm(column_emb)
    if denom > 0:
        column_emb = column_emb / denom

    return ColumnInfo(
        name=col,
        header_emb=header_emb,
        values_emb=values_emb,
        column_emb=column_emb,
    )


def column_match(q_col: ColumnInfo, t_col: ColumnInfo) -> float:
    # combine header similarity and value similarity
    s_header = cosine(q_col.header_emb, t_col.header_emb)
    s_values = cosine(q_col.values_emb, t_col.values_emb)
    s_column = cosine(q_col.column_emb, t_col.column_emb)

    # weighted combination
    score = 0.2 * s_header + 0.3 * s_values + 0.5 * s_column
    return max(0.0, score)


# ============================================================
# Relationship semantics with BERT
# ============================================================

def infer_relation_label(
    df: pd.DataFrame,
    c1: str,
    c2: str,
    fd_threshold: float = 0.9,
) -> Tuple[str, float]:
    """
    Heuristique simple pour approximer une relation.
    Tu peux remplacer ça par une vraie extraction relationnelle plus tard.
    """
    x = df[c1].dropna()
    y = df[c2].dropna()
    if len(x) == 0 or len(y) == 0:
        return "unknown_relation", 0.0

    # align rows valides
    pairs = df[[c1, c2]].dropna()
    if len(pairs) == 0:
        return "unknown_relation", 0.0

    # functional dependency approx: c1 -> c2
    grouped_12 = pairs.groupby(c1)[c2].nunique()
    grouped_21 = pairs.groupby(c2)[c1].nunique()

    fd_12 = float((grouped_12 <= 1).mean()) if len(grouped_12) else 0.0
    fd_21 = float((grouped_21 <= 1).mean()) if len(grouped_21) else 0.0

    # lexical hints
    name1 = norm_text(c1)
    name2 = norm_text(c2)

    if "city" in name2 and ("country" in name1 or "state" in name1):
        return "located_in_reverse", 0.7
    if "country" in name2 and "city" in name1:
        return "located_in", 0.8
    if "director" in name2 and "film" in name1:
        return "directed_by", 0.9
    if "phone" in name2:
        return "has_phone", 0.9

    if fd_12 >= fd_threshold:
        return f"{name1}_determines_{name2}", fd_12
    if fd_21 >= fd_threshold:
        return f"{name2}_determines_{name1}", fd_21

    return "associated_with", max(fd_12, fd_21, 0.3)


def build_relation_info(
    df: pd.DataFrame,
    c1: str,
    c2: str,
    encoder: BertEncoder,
) -> RelationInfo:
    label, score = infer_relation_label(df, c1, c2)
    emb = encoder.encode_text(label)
    return RelationInfo(label=label, score=score, emb=emb)


def relation_match(q_rel: RelationInfo, t_rel: RelationInfo) -> float:
    sim = cosine(q_rel.emb, t_rel.emb)
    return max(0.0, sim) * q_rel.score * t_rel.score


# ============================================================
# Build semantic graph
# ============================================================

def build_semantic_graph(df: pd.DataFrame, table_name: str, encoder: BertEncoder) -> SemanticGraph:
    sg = SemanticGraph(table_name)

    for col in df.columns:
        info = build_column_info(df, col, encoder)
        sg.add_column(col, info)

    for c1, c2 in itertools.combinations(df.columns, 2):
        rel = build_relation_info(df, c1, c2, encoder)
        if rel.score > 0:
            sg.add_relation(c1, c2, rel)

    return sg


# ============================================================
# SANTOS pair score and table score
# Inspired by pairMatch and unionability score in the paper
# ============================================================

def pair_match(
    q_graph: SemanticGraph,
    t_graph: SemanticGraph,
    qc1: str,
    qc2: str,
    tc1: str,
    tc2: str,
) -> float:
    q1 = q_graph.column_info(qc1)
    q2 = q_graph.column_info(qc2)
    t1 = t_graph.column_info(tc1)
    t2 = t_graph.column_info(tc2)

    col1 = column_match(q1, t1)
    col2 = column_match(q2, t2)

    if not q_graph.has_relation(qc1, qc2) or not t_graph.has_relation(tc1, tc2):
        return 0.0

    rel_q = q_graph.relation(qc1, qc2)
    rel_t = t_graph.relation(tc1, tc2)
    rel = relation_match(rel_q, rel_t)

    return col1 * rel * col2


def best_alignment_score(
    q_graph: SemanticGraph,
    t_graph: SemanticGraph,
    threshold: float = 0.35,
) -> float:
    """
    Pour chaque paire de colonnes du query, on cherche la meilleure paire
    dans la table candidate. Approximation simple du matching SANTOS.
    """
    total = 0.0

    q_cols = q_graph.columns()
    t_cols = t_graph.columns()

    for qc1, qc2 in itertools.combinations(q_cols, 2):
        best = 0.0
        for tc1, tc2 in itertools.combinations(t_cols, 2):
            s1 = pair_match(q_graph, t_graph, qc1, qc2, tc1, tc2)
            s2 = pair_match(q_graph, t_graph, qc1, qc2, tc2, tc1)
            best = max(best, s1, s2)

        if best >= threshold:
            total += best

    return total


# ============================================================
# Search engine
# ============================================================

class SantosBert:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = BertEncoder(model_name=model_name)
        self.tables: Dict[str, pd.DataFrame] = {}
        self.graphs: Dict[str, SemanticGraph] = {}

    def add_table(self, name: str, df: pd.DataFrame) -> None:
        self.tables[name] = df.copy()
        self.graphs[name] = build_semantic_graph(df, name, self.encoder)

    def search(self, query_df: pd.DataFrame, k: int = 5) -> List[Tuple[str, float]]:
        q_graph = build_semantic_graph(query_df, "query", self.encoder)

        scores = []
        for name, g in self.graphs.items():
            score = best_alignment_score(q_graph, g)
            scores.append((name, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    santos = SantosBert()

    parks = pd.DataFrame({
        "Park Name": ["River Park", "West Lawn Park", "Lawler Park"],
        "City": ["Fresno", "Chicago", "Riverside"],
        "Country": ["USA", "USA", "USA"],
        "Supervisor": ["Vera Onate", "Paul Veliotis", "Anna Reed"],
    })

    films = pd.DataFrame({
        "Park Name": ["Chippewa Park", "Lawler Park", "Union Park"],
        "Park City": ["Cook", "Riverside", "Chicago"],
        "Film Title": ["Bee Movie", "Coco", "Moana"],
        "Film Director": ["Simon J. Smith", "Adrian Molina", "Ron Clements"],
    })

    people = pd.DataFrame({
        "Person": ["James Taylor", "Anthony Pelissier", "Akram Afif"],
        "Birthplace": ["Boston", "Barnet", "Doha"],
        "Country": ["USA", "UK", "Qatar"],
        "Occupation": ["Singer", "Film Director", "Football Player"],
    })

    gardens = pd.DataFrame({
        "Garden": ["Union Garden", "Maple Garden", "West Lawn Park"],
        "Town": ["Chicago", "Fresno", "Chicago"],
        "Nation": ["USA", "USA", "USA"],
        "Manager": ["Tom Lee", "Sara May", "Paul Veliotis"],
    })

    santos.add_table("parks", parks)
    santos.add_table("films", films)
    santos.add_table("people", people)
    santos.add_table("gardens", gardens)

    query = pd.DataFrame({
        "Park": ["Union Park", "Lawler Park"],
        "City": ["Chicago", "Riverside"],
        "Country": ["USA", "USA"],
        "Lead": ["Tom Lee", "Anna Reed"],
    })

    results = santos.search(query, k=4)

    print("Top unionable tables:")
    for name, score in results:
        print(f"{name:10s} -> {score:.4f}")