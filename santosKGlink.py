import itertools
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

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


def is_numeric_like(text: str) -> bool:
    text = norm_text(text)
    if not text:
        return False
    text = text.replace(",", "").replace(" ", "")
    return bool(re.fullmatch(r"[-+]?\d*\.?\d+", text))


def is_date_like(text: str) -> bool:
    text = norm_text(text)
    if not text:
        return False
    patterns = [
        r"\d{4}-\d{2}-\d{2}",
        r"\d{4}/\d{2}/\d{2}",
        r"\d{2}/\d{2}/\d{4}",
        r"\d{2}-\d{2}-\d{4}",
        r"\d{4}",
    ]
    return any(re.fullmatch(p, text) for p in patterns)


def jaccard_tokens(a: str, b: str) -> float:
    ta = set(norm_text(a).split())
    tb = set(norm_text(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


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
# Lightweight KG structures (KGLink-inspired)
# ============================================================

@dataclass
class KGEntity:
    entity_id: str
    label: str
    types: List[str]
    neighbors: List[str] = field(default_factory=list)


class SimpleKnowledgeGraph:
    """
    Mini KG local pour démonstration / intégration SANTOS.
    Remplaçable plus tard par Wikidata/DBpedia/Elastic BM25.
    """
    def __init__(self):
        self.entities: Dict[str, KGEntity] = {}
        self.label_to_ids: Dict[str, List[str]] = {}

    def add_entity(self, entity: KGEntity) -> None:
        self.entities[entity.entity_id] = entity
        key = norm_text(entity.label)
        self.label_to_ids.setdefault(key, []).append(entity.entity_id)

    def get_entity(self, entity_id: str) -> KGEntity:
        return self.entities[entity_id]

    def find_candidates(self, mention: str, top_k: int = 5) -> List[Tuple[KGEntity, float]]:
        """
        Approximation légère d'entity linking.
        Score = mix exact match / containment / jaccard token.
        """
        mention_n = norm_text(mention)
        if not mention_n:
            return []

        scored: List[Tuple[KGEntity, float]] = []

        for ent in self.entities.values():
            lbl = norm_text(ent.label)

            score = 0.0
            if mention_n == lbl:
                score = 1.0
            elif mention_n in lbl or lbl in mention_n:
                score = 0.85
            else:
                score = jaccard_tokens(mention_n, lbl)

            if score > 0:
                scored.append((ent, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def build_demo_kg() -> SimpleKnowledgeGraph:
    """
    Petit KG d'exemple.
    Tu peux enrichir facilement cette base.
    """
    kg = SimpleKnowledgeGraph()

    entities = [
        KGEntity("E1", "Chicago", ["city", "location"], ["illinois", "usa"]),
        KGEntity("E2", "Fresno", ["city", "location"], ["california", "usa"]),
        KGEntity("E3", "Riverside", ["city", "location"], ["california", "usa"]),
        KGEntity("E4", "USA", ["country", "location"], ["north america"]),
        KGEntity("E5", "Qatar", ["country", "location"], ["asia"]),
        KGEntity("E6", "James Taylor", ["person", "artist", "singer"], ["usa", "music"]),
        KGEntity("E7", "Akram Afif", ["person", "athlete", "football player"], ["qatar", "football"]),
        KGEntity("E8", "Anthony Pelissier", ["person", "film director"], ["film"]),
        KGEntity("E9", "Moana", ["film"], ["animation", "disney"]),
        KGEntity("E10", "Coco", ["film"], ["animation", "disney"]),
        KGEntity("E11", "Bee Movie", ["film"], ["animation"]),
        KGEntity("E12", "Tom Lee", ["person"], []),
        KGEntity("E13", "Anna Reed", ["person"], []),
        KGEntity("E14", "Paul Veliotis", ["person"], []),
        KGEntity("E15", "Vera Onate", ["person"], []),
        KGEntity("E16", "Boston", ["city", "location"], ["usa"]),
        KGEntity("E17", "Doha", ["city", "location"], ["qatar"]),
        KGEntity("E18", "Barnet", ["city", "location"], ["uk"]),
        KGEntity("E19", "UK", ["country", "location"], ["europe"]),
        KGEntity("E20", "Singer", ["occupation"], ["music"]),
        KGEntity("E21", "Film Director", ["occupation"], ["film"]),
        KGEntity("E22", "Football Player", ["occupation"], ["sport"]),
    ]

    for e in entities:
        kg.add_entity(e)

    return kg


# ============================================================
# KGLink-style semantic objects
# ============================================================

@dataclass
class ColumnInfo:
    name: str
    header_emb: np.ndarray
    values_emb: np.ndarray
    type_emb: np.ndarray
    column_emb: np.ndarray
    candidate_types: List[Tuple[str, float]]
    linked_entities: List[Tuple[str, str, float]]  # (cell, entity_label, score)
    coverage: float


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
# KGLink-inspired column semantics
# ============================================================

def infer_numeric_candidate_types(series: pd.Series) -> List[Tuple[str, float]]:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        return []

    out = []
    out.append(("numeric", 1.0))

    mn = float(vals.mean())
    std = float(vals.std()) if len(vals) > 1 else 0.0

    if np.allclose(vals, vals.astype(int), atol=1e-6):
        out.append(("integer", 0.9))
    else:
        out.append(("real number", 0.8))

    if mn >= 1800 and mn <= 2100 and std < 100:
        out.append(("year", 0.85))
    if mn >= 0 and mn <= 150 and std < 50:
        out.append(("measurement", 0.55))

    return out


def extract_candidate_types_from_entities(
    linked_entities: List[Tuple[KGEntity, float]],
    kg: SimpleKnowledgeGraph,
) -> List[Tuple[str, float]]:
    """
    Version simplifiée inspirée de KGLink :
    - on collecte les types des entités liées
    - on ajoute aussi des voisins 1-hop comme types contextuels faibles
    - on score par somme pondérée
    """
    type_scores: Dict[str, float] = {}

    for ent, link_score in linked_entities:
        for t in ent.types:
            type_scores[t] = type_scores.get(t, 0.0) + 1.0 * link_score

        # information contextuelle légère via voisins 1-hop
        for nb in ent.neighbors:
            nb_n = norm_text(nb)
            if nb_n not in {"person", "date"}:
                type_scores[nb_n] = type_scores.get(nb_n, 0.0) + 0.25 * link_score

    ranked = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:5]


def embed_candidate_types(
    candidate_types: List[Tuple[str, float]],
    encoder: BertEncoder,
    dim_fallback_from: np.ndarray,
) -> np.ndarray:
    if not candidate_types:
        return np.zeros_like(dim_fallback_from)

    texts = [t for t, _ in candidate_types]
    weights = np.array([w for _, w in candidate_types], dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)

    embs = encoder.encode_many(texts)
    type_emb = np.average(embs, axis=0, weights=weights)

    denom = np.linalg.norm(type_emb)
    if denom > 0:
        type_emb = type_emb / denom
    return type_emb


def build_column_info(
    df: pd.DataFrame,
    col: str,
    encoder: BertEncoder,
    kg: SimpleKnowledgeGraph,
    kg_weight: float = 0.65,
    text_weight: float = 0.35,
) -> ColumnInfo:
    """
    Remplace la version BERT simple par une version KGLink-inspired.

    Représentation finale :
        column_emb = kg_weight * type_emb + text_weight * text_emb

    où :
        text_emb = mix(header_emb, values_emb)
        type_emb = embedding des candidate types extraits du KG
    """
    header_emb = encoder.encode_text(col)

    values = unique_non_empty(df[col])
    if values:
        value_embs = encoder.encode_many(values)
        values_emb = mean_pool(value_embs)
    else:
        values_emb = np.zeros_like(header_emb)

    # représentation textuelle de secours
    text_emb = 0.4 * header_emb + 0.6 * values_emb
    denom = np.linalg.norm(text_emb)
    if denom > 0:
        text_emb = text_emb / denom

    linked_entities: List[Tuple[KGEntity, float]] = []
    linked_entities_debug: List[Tuple[str, str, float]] = []

    # Cas numérique : pas de linking KG exact, on génère des types numériques
    numeric_ratio = pd.to_numeric(df[col], errors="coerce").notna().mean()

    if numeric_ratio >= 0.8:
        candidate_types = infer_numeric_candidate_types(df[col])
        type_emb = embed_candidate_types(candidate_types, encoder, header_emb)

        # combine KG-type semantics + textual fallback
        column_emb = kg_weight * type_emb + text_weight * text_emb
        denom = np.linalg.norm(column_emb)
        if denom > 0:
            column_emb = column_emb / denom

        return ColumnInfo(
            name=col,
            header_emb=header_emb,
            values_emb=values_emb,
            type_emb=type_emb,
            column_emb=column_emb,
            candidate_types=candidate_types,
            linked_entities=[],
            coverage=0.0,
        )

    # Linking cellule -> KG (version légère)
    matched_cells = 0
    for cell in values[:50]:  # borne simple
        if is_numeric_like(cell) or is_date_like(cell):
            continue

        candidates = kg.find_candidates(cell, top_k=3)
        if not candidates:
            continue

        best_ent, best_score = candidates[0]
        linked_entities.append((best_ent, best_score))
        linked_entities_debug.append((cell, best_ent.label, float(best_score)))
        matched_cells += 1

    coverage = matched_cells / max(1, len(values[:50]))

    candidate_types = extract_candidate_types_from_entities(linked_entities, kg)
    type_emb = embed_candidate_types(candidate_types, encoder, header_emb)

    # fallback si rien trouvé
    if len(candidate_types) == 0 or np.linalg.norm(type_emb) == 0:
        column_emb = text_emb
        type_emb = np.zeros_like(text_emb)
    else:
        column_emb = kg_weight * type_emb + text_weight * text_emb
        denom = np.linalg.norm(column_emb)
        if denom > 0:
            column_emb = column_emb / denom

    return ColumnInfo(
        name=col,
        header_emb=header_emb,
        values_emb=values_emb,
        type_emb=type_emb,
        column_emb=column_emb,
        candidate_types=candidate_types,
        linked_entities=linked_entities_debug,
        coverage=coverage,
    )


def column_match(q_col: ColumnInfo, t_col: ColumnInfo) -> float:
    """
    Similarité de colonnes orientée KGLink :
    - type similarity majoritaire
    - text similarity en soutien
    - bonus si types candidats se recoupent
    """
    s_header = cosine(q_col.header_emb, t_col.header_emb)
    s_values = cosine(q_col.values_emb, t_col.values_emb)
    s_type = cosine(q_col.type_emb, t_col.type_emb)
    s_column = cosine(q_col.column_emb, t_col.column_emb)

    q_types = {t for t, _ in q_col.candidate_types}
    t_types = {t for t, _ in t_col.candidate_types}
    overlap_bonus = 0.0
    if q_types and t_types:
        overlap_bonus = len(q_types & t_types) / len(q_types | t_types)

    # On donne plus d'importance à la sémantique de type
    score = (
        0.10 * s_header +
        0.15 * s_values +
        0.45 * s_type +
        0.25 * s_column +
        0.05 * overlap_bonus
    )

    # Bonus de couverture KG si les deux colonnes sont bien liées
    score *= (0.9 + 0.1 * min(1.0, q_col.coverage + t_col.coverage))

    return max(0.0, min(1.0, score))


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
    """
    x = df[c1].dropna()
    y = df[c2].dropna()
    if len(x) == 0 or len(y) == 0:
        return "unknown_relation", 0.0

    pairs = df[[c1, c2]].dropna()
    if len(pairs) == 0:
        return "unknown_relation", 0.0

    grouped_12 = pairs.groupby(c1)[c2].nunique()
    grouped_21 = pairs.groupby(c2)[c1].nunique()

    fd_12 = float((grouped_12 <= 1).mean()) if len(grouped_12) else 0.0
    fd_21 = float((grouped_21 <= 1).mean()) if len(grouped_21) else 0.0

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
    if "occupation" in name2 and ("person" in name1 or "name" in name1):
        return "has_occupation", 0.85

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

def build_semantic_graph(
    df: pd.DataFrame,
    table_name: str,
    encoder: BertEncoder,
    kg: SimpleKnowledgeGraph,
) -> SemanticGraph:
    sg = SemanticGraph(table_name)

    for col in df.columns:
        info = build_column_info(df, col, encoder, kg)
        sg.add_column(col, info)

    for c1, c2 in itertools.combinations(df.columns, 2):
        rel = build_relation_info(df, c1, c2, encoder)
        if rel.score > 0:
            sg.add_relation(c1, c2, rel)

    return sg


# ============================================================
# SANTOS pair score and table score
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
    Approximation simple du matching SANTOS.
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

class SantosKGLink:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        kg: Optional[SimpleKnowledgeGraph] = None,
    ):
        self.encoder = BertEncoder(model_name=model_name)
        self.kg = kg if kg is not None else build_demo_kg()
        self.tables: Dict[str, pd.DataFrame] = {}
        self.graphs: Dict[str, SemanticGraph] = {}

    def add_table(self, name: str, df: pd.DataFrame) -> None:
        self.tables[name] = df.copy()
        self.graphs[name] = build_semantic_graph(df, name, self.encoder, self.kg)

    def search(self, query_df: pd.DataFrame, k: int = 5) -> List[Tuple[str, float]]:
        q_graph = build_semantic_graph(query_df, "query", self.encoder, self.kg)

        scores = []
        for name, g in self.graphs.items():
            score = best_alignment_score(q_graph, g)
            scores.append((name, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def explain_column(self, df: pd.DataFrame, col: str) -> Dict:
        info = build_column_info(df, col, self.encoder, self.kg)
        return {
            "column": col,
            "candidate_types": info.candidate_types,
            "linked_entities": info.linked_entities,
            "coverage": info.coverage,
        }


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    santos = SantosKGLink()

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

    years = pd.DataFrame({
        "Player": ["A", "B", "C"],
        "Year": [1998, 2001, 2010],
        "Country": ["USA", "USA", "Qatar"],
    })

    santos.add_table("parks", parks)
    santos.add_table("films", films)
    santos.add_table("people", people)
    santos.add_table("gardens", gardens)
    santos.add_table("years", years)

    query = pd.DataFrame({
        "Park": ["Union Park", "Lawler Park"],
        "City": ["Chicago", "Riverside"],
        "Country": ["USA", "USA"],
        "Lead": ["Tom Lee", "Anna Reed"],
    })

    print("=== Explain query columns ===")
    for c in query.columns:
        print(c, "->", santos.explain_column(query, c))

    print("\n=== Top unionable tables ===")
    results = santos.search(query, k=5)
    for name, score in results:
        print(f"{name:10s} -> {score:.4f}")