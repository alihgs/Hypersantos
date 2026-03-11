import itertools
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

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
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def mean_pool(vectors: np.ndarray, fallback_dim: int = 384) -> np.ndarray:
    if len(vectors) == 0:
        return np.zeros((fallback_dim,), dtype=np.float32)
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


# ============================================================
# Embedding model
# ============================================================

class TextEncoder:
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
# KG backend abstraction
# ============================================================

@dataclass
class KGEntity:
    entity_id: str
    label: str
    types: List[str]
    neighbors: List[str] = field(default_factory=list)
    predicates: List[str] = field(default_factory=list)


class KGBackend:
    """
    Interface abstraite pour brancher plus tard :
    - un BM25 réel sur Wikidata
    - un index Elasticsearch
    - un service d'entity linking
    """
    def link_cell_mentions(self, mentions: List[str], top_k: int = 5) -> Dict[str, List[Tuple[KGEntity, float]]]:
        raise NotImplementedError

    def get_one_hop_neighbors(self, entity: KGEntity) -> List[str]:
        raise NotImplementedError


class DemoKGBackend(KGBackend):
    """
    Backend simplifié pour test local.
    Remplaçable plus tard par vrai BM25 + Wikidata.
    """
    def __init__(self):
        self.entities: Dict[str, KGEntity] = {}
        self._build_demo_kg()

    def _add(self, e: KGEntity) -> None:
        self.entities[e.entity_id] = e

    def _build_demo_kg(self) -> None:
        demo_entities = [
            KGEntity("E1", "Chicago", ["city", "location"], ["illinois", "usa"], ["located_in"]),
            KGEntity("E2", "Fresno", ["city", "location"], ["california", "usa"], ["located_in"]),
            KGEntity("E3", "Riverside", ["city", "location"], ["california", "usa"], ["located_in"]),
            KGEntity("E4", "USA", ["country", "location"], ["north america"], ["part_of"]),
            KGEntity("E5", "Qatar", ["country", "location"], ["asia"], ["part_of"]),
            KGEntity("E6", "Messi", ["person", "athlete", "football player"], ["psg", "argentina"], ["plays_for"]),
            KGEntity("E7", "Akram Afif", ["person", "athlete", "football player"], ["psg", "qatar"], ["plays_for"]),
            KGEntity("E8", "PSG", ["team", "sports team"], ["football", "france"], ["has_player"]),
            KGEntity("E9", "Coco", ["film"], ["animation", "disney"], ["directed_by"]),
            KGEntity("E10", "Moana", ["film"], ["animation", "disney"], ["directed_by"]),
            KGEntity("E11", "Film Director", ["occupation"], ["film"], []),
            KGEntity("E12", "Singer", ["occupation"], ["music"], []),
            KGEntity("E13", "Doha", ["city", "location"], ["qatar"], ["located_in"]),
            KGEntity("E14", "Boston", ["city", "location"], ["usa"], ["located_in"]),
        ]
        for e in demo_entities:
            self._add(e)

    def _score(self, mention: str, label: str) -> float:
        m = norm_text(mention)
        l = norm_text(label)
        if not m or not l:
            return 0.0
        if m == l:
            return 1.0
        if m in l or l in m:
            return 0.85
        mt = set(m.split())
        lt = set(l.split())
        if not mt or not lt:
            return 0.0
        return len(mt & lt) / len(mt | lt)

    def link_cell_mentions(self, mentions: List[str], top_k: int = 5) -> Dict[str, List[Tuple[KGEntity, float]]]:
        out: Dict[str, List[Tuple[KGEntity, float]]] = {}
        for mention in mentions:
            scored = []
            for ent in self.entities.values():
                s = self._score(mention, ent.label)
                if s > 0:
                    scored.append((ent, s))
            scored.sort(key=lambda x: x[1], reverse=True)
            out[mention] = scored[:top_k]
        return out

    def get_one_hop_neighbors(self, entity: KGEntity) -> List[str]:
        return entity.neighbors


# ============================================================
# KGLink-like output structures
# ============================================================

@dataclass
class ColumnInfo:
    name: str
    header_emb: np.ndarray
    values_emb: np.ndarray
    type_emb: np.ndarray
    feature_emb: np.ndarray
    column_emb: np.ndarray
    candidate_types: List[Tuple[str, float]]
    linked_entities: List[Tuple[str, str, float]]
    coverage: float


@dataclass
class RelationInfo:
    label: str
    score: float
    emb: np.ndarray


@dataclass
class HyperRelation:
    columns: Tuple[str, ...]
    label: str
    score: float
    emb: np.ndarray
    support: int
    uniqueness: float
    coverage: float


# ============================================================
# KGLink encoder
# ============================================================

class KGLinkEncoder:
    """
    Version intégrable dans SANTOS/Hypergraph.

    Cette classe suit la logique du papier :
    - linking des cellules vers un KG
    - types candidats
    - feature vector à partir du KG
    - représentation de colonne

    Mais sans la partie training multi-task du papier.
    """
    def __init__(
        self,
        text_encoder: TextEncoder,
        kg_backend: KGBackend,
        top_k_entities: int = 3,
        top_k_types: int = 5,
    ):
        self.text_encoder = text_encoder
        self.kg_backend = kg_backend
        self.top_k_entities = top_k_entities
        self.top_k_types = top_k_types

    def _infer_numeric_types(self, series: pd.Series) -> List[Tuple[str, float]]:
        vals = pd.to_numeric(series, errors="coerce").dropna()
        if len(vals) == 0:
            return []

        out = [("numeric", 1.0)]
        mn = float(vals.mean())
        std = float(vals.std()) if len(vals) > 1 else 0.0

        if np.allclose(vals, vals.astype(int), atol=1e-6):
            out.append(("integer", 0.9))
        else:
            out.append(("real number", 0.8))

        if 1800 <= mn <= 2100 and std < 100:
            out.append(("year", 0.85))
        if 0 <= mn <= 150 and std < 50:
            out.append(("measurement", 0.55))

        return out

    def _build_feature_sequence(self, best_entities: List[KGEntity]) -> List[str]:
        """
        Approximation de la feature sequence du papier :
        entité + voisins 1-hop + éventuellement prédicats.
        """
        seq = []
        for ent in best_entities[:3]:
            seq.append(ent.label)
            seq.extend(ent.predicates[:3])
            seq.extend(self.kg_backend.get_one_hop_neighbors(ent)[:5])
        return [norm_text(x) for x in seq if norm_text(x)]

    def _extract_candidate_types(
        self,
        linked_entities: List[Tuple[KGEntity, float]]
    ) -> List[Tuple[str, float]]:
        type_scores: Dict[str, float] = {}

        for ent, score in linked_entities:
            for t in ent.types:
                type_scores[t] = type_scores.get(t, 0.0) + score

            # context léger via 1-hop neighbors
            for nb in ent.neighbors:
                nb = norm_text(nb)
                if nb and nb not in {"person", "date"}:
                    type_scores[nb] = type_scores.get(nb, 0.0) + 0.2 * score

        ranked = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[: self.top_k_types]

    def _embed_weighted_labels(
        self,
        labels: List[Tuple[str, float]],
        fallback_dim_from: np.ndarray
    ) -> np.ndarray:
        if not labels:
            return np.zeros_like(fallback_dim_from)

        texts = [t for t, _ in labels]
        weights = np.array([w for _, w in labels], dtype=np.float32)
        weights = weights / (weights.sum() + 1e-8)

        embs = self.text_encoder.encode_many(texts)
        out = np.average(embs, axis=0, weights=weights)

        denom = np.linalg.norm(out)
        if denom > 0:
            out = out / denom
        return out

    def encode_column(self, df: pd.DataFrame, col: str) -> ColumnInfo:
        header_emb = self.text_encoder.encode_text(col)
        values = unique_non_empty(df[col])

        if values:
            values_emb = mean_pool(self.text_encoder.encode_many(values))
        else:
            values_emb = np.zeros_like(header_emb)

        text_emb = 0.4 * header_emb + 0.6 * values_emb
        denom = np.linalg.norm(text_emb)
        if denom > 0:
            text_emb = text_emb / denom

        numeric_ratio = pd.to_numeric(df[col], errors="coerce").notna().mean()

        # Cas numérique
        if numeric_ratio >= 0.8:
            candidate_types = self._infer_numeric_types(df[col])
            type_emb = self._embed_weighted_labels(candidate_types, header_emb)
            feature_emb = type_emb.copy()

            column_emb = 0.55 * type_emb + 0.45 * text_emb
            denom = np.linalg.norm(column_emb)
            if denom > 0:
                column_emb = column_emb / denom

            return ColumnInfo(
                name=col,
                header_emb=header_emb,
                values_emb=values_emb,
                type_emb=type_emb,
                feature_emb=feature_emb,
                column_emb=column_emb,
                candidate_types=candidate_types,
                linked_entities=[],
                coverage=0.0,
            )

        # Entity linking
        non_special_values = [
            v for v in values[:50]
            if not is_numeric_like(v) and not is_date_like(v)
        ]

        link_results = self.kg_backend.link_cell_mentions(non_special_values, top_k=self.top_k_entities)

        best_entities: List[KGEntity] = []
        linked_entities: List[Tuple[KGEntity, float]] = []
        debug_links: List[Tuple[str, str, float]] = []

        matched_cells = 0
        for cell in non_special_values:
            candidates = link_results.get(cell, [])
            if not candidates:
                continue
            ent, score = candidates[0]
            best_entities.append(ent)
            linked_entities.append((ent, score))
            debug_links.append((cell, ent.label, float(score)))
            matched_cells += 1

        coverage = matched_cells / max(1, len(non_special_values))

        candidate_types = self._extract_candidate_types(linked_entities)
        type_emb = self._embed_weighted_labels(candidate_types, header_emb)

        # Feature vector KG
        feature_tokens = self._build_feature_sequence(best_entities)
        if feature_tokens:
            feature_emb = mean_pool(self.text_encoder.encode_many(feature_tokens))
            denom = np.linalg.norm(feature_emb)
            if denom > 0:
                feature_emb = feature_emb / denom
        else:
            feature_emb = np.zeros_like(header_emb)

        if np.linalg.norm(type_emb) == 0 and np.linalg.norm(feature_emb) == 0:
            column_emb = text_emb
        else:
            column_emb = 0.45 * type_emb + 0.25 * feature_emb + 0.30 * text_emb
            denom = np.linalg.norm(column_emb)
            if denom > 0:
                column_emb = column_emb / denom

        return ColumnInfo(
            name=col,
            header_emb=header_emb,
            values_emb=values_emb,
            type_emb=type_emb,
            feature_emb=feature_emb,
            column_emb=column_emb,
            candidate_types=candidate_types,
            linked_entities=debug_links,
            coverage=coverage,
        )


# ============================================================
# Semantic graph + hypergraph
# ============================================================

class SemanticGraph:
    def __init__(self, table_name: str):
        self.table_name = table_name
        self.graph = nx.Graph()
        self.hyperedges: List[HyperRelation] = []

    def add_column(self, col_name: str, info: ColumnInfo) -> None:
        self.graph.add_node(col_name, info=info)

    def add_relation(self, c1: str, c2: str, rel: RelationInfo) -> None:
        self.graph.add_edge(c1, c2, rel=rel)

    def add_hyper_relation(self, hrel: HyperRelation) -> None:
        self.hyperedges.append(hrel)

    def columns(self) -> List[str]:
        return list(self.graph.nodes)

    def has_relation(self, c1: str, c2: str) -> bool:
        return self.graph.has_edge(c1, c2)

    def relation(self, c1: str, c2: str) -> RelationInfo:
        return self.graph[c1][c2]["rel"]

    def column_info(self, c: str) -> ColumnInfo:
        return self.graph.nodes[c]["info"]


# ============================================================
# Column matching
# ============================================================

def column_match(q_col: ColumnInfo, t_col: ColumnInfo) -> float:
    s_header = cosine(q_col.header_emb, t_col.header_emb)
    s_values = cosine(q_col.values_emb, t_col.values_emb)
    s_type = cosine(q_col.type_emb, t_col.type_emb)
    s_feature = cosine(q_col.feature_emb, t_col.feature_emb)
    s_column = cosine(q_col.column_emb, t_col.column_emb)

    q_types = {t for t, _ in q_col.candidate_types}
    t_types = {t for t, _ in t_col.candidate_types}
    overlap_bonus = 0.0
    if q_types and t_types:
        overlap_bonus = len(q_types & t_types) / len(q_types | t_types)

    score = (
        0.08 * s_header +
        0.10 * s_values +
        0.32 * s_type +
        0.20 * s_feature +
        0.25 * s_column +
        0.05 * overlap_bonus
    )

    score *= (0.9 + 0.1 * min(1.0, q_col.coverage + t_col.coverage))
    return max(0.0, min(1.0, score))


# ============================================================
# Binary relation layer
# ============================================================

def infer_relation_label(
    df: pd.DataFrame,
    c1: str,
    c2: str,
    fd_threshold: float = 0.9,
) -> Tuple[str, float]:
    pairs = df[[c1, c2]].dropna()
    if len(pairs) == 0:
        return "unknown_relation", 0.0

    grouped_12 = pairs.groupby(c1)[c2].nunique()
    grouped_21 = pairs.groupby(c2)[c1].nunique()

    fd_12 = float((grouped_12 <= 1).mean()) if len(grouped_12) else 0.0
    fd_21 = float((grouped_21 <= 1).mean()) if len(grouped_21) else 0.0

    name1 = norm_text(c1)
    name2 = norm_text(c2)

    if "country" in name2 and "city" in name1:
        return "located_in", 0.8
    if "director" in name2 and "film" in name1:
        return "directed_by", 0.9
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
    encoder: TextEncoder,
) -> RelationInfo:
    label, score = infer_relation_label(df, c1, c2)
    emb = encoder.encode_text(label)
    return RelationInfo(label=label, score=score, emb=emb)


def relation_match(q_rel: RelationInfo, t_rel: RelationInfo) -> float:
    sim = cosine(q_rel.emb, t_rel.emb)
    return max(0.0, sim) * q_rel.score * t_rel.score


# ============================================================
# Hypergraph detection
# ============================================================

def infer_hyper_relation_label(
    df: pd.DataFrame,
    cols: Tuple[str, ...],
    support_threshold: int = 2,
    coverage_threshold: float = 0.5,
    uniqueness_threshold: float = 0.6,
) -> Optional[Tuple[str, float, int, float, float]]:
    subset = df[list(cols)].dropna()
    if len(subset) < support_threshold:
        return None

    support = len(subset)
    uniqueness = len(subset.drop_duplicates()) / max(1, support)
    coverage = support / max(1, len(df))

    if coverage < coverage_threshold:
        return None
    if uniqueness < uniqueness_threshold:
        return None

    colnames = [norm_text(c) for c in cols]
    joined = "_".join(colnames)

    if any("year" in c or "date" in c for c in colnames):
        return f"temporal_hyper_relation_{joined}", 0.85, support, uniqueness, coverage
    if any("team" in c or "player" in c or "club" in c for c in colnames):
        return f"sport_hyper_relation_{joined}", 0.88, support, uniqueness, coverage
    if any("film" in c or "director" in c for c in colnames):
        return f"film_hyper_relation_{joined}", 0.88, support, uniqueness, coverage
    if any("city" in c or "country" in c or "population" in c for c in colnames):
        return f"location_hyper_relation_{joined}", 0.80, support, uniqueness, coverage

    return f"composite_hyper_relation_{joined}", 0.75, support, uniqueness, coverage


def build_hyper_relation_info(
    df: pd.DataFrame,
    cols: Tuple[str, ...],
    encoder: TextEncoder,
) -> Optional[HyperRelation]:
    res = infer_hyper_relation_label(df, cols)
    if res is None:
        return None

    label, score, support, uniqueness, coverage = res
    emb = encoder.encode_text(label)

    return HyperRelation(
        columns=cols,
        label=label,
        score=score,
        emb=emb,
        support=support,
        uniqueness=uniqueness,
        coverage=coverage,
    )


def hyper_relation_match(q_h: HyperRelation, t_h: HyperRelation) -> float:
    sim = cosine(q_h.emb, t_h.emb)
    meta = (
        0.4 * min(q_h.uniqueness, t_h.uniqueness) +
        0.3 * min(q_h.coverage, t_h.coverage) +
        0.3 * min(1.0, min(q_h.support, t_h.support) / 10.0)
    )
    return max(0.0, sim) * q_h.score * t_h.score * meta


def hyper_column_set_match(
    q_graph: SemanticGraph,
    t_graph: SemanticGraph,
    q_cols: Tuple[str, ...],
    t_cols: Tuple[str, ...],
) -> float:
    if len(q_cols) != len(t_cols):
        return 0.0

    scores = []
    for qc, tc in zip(q_cols, t_cols):
        scores.append(column_match(q_graph.column_info(qc), t_graph.column_info(tc)))

    if not scores:
        return 0.0
    return float(np.mean(scores))


# ============================================================
# Build graph
# ============================================================

def build_semantic_graph(
    df: pd.DataFrame,
    table_name: str,
    text_encoder: TextEncoder,
    kglink_encoder: KGLinkEncoder,
    max_hyperedge_size: int = 3,
) -> SemanticGraph:
    sg = SemanticGraph(table_name)

    for col in df.columns:
        info = kglink_encoder.encode_column(df, col)
        sg.add_column(col, info)

    for c1, c2 in itertools.combinations(df.columns, 2):
        rel = build_relation_info(df, c1, c2, text_encoder)
        if rel.score > 0:
            sg.add_relation(c1, c2, rel)

    cols = list(df.columns)
    for size in range(3, min(max_hyperedge_size, len(cols)) + 1):
        for combo in itertools.combinations(cols, size):
            hrel = build_hyper_relation_info(df, combo, text_encoder)
            if hrel is not None:
                sg.add_hyper_relation(hrel)

    return sg


# ============================================================
# Matching
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


def hyper_match(
    q_graph: SemanticGraph,
    t_graph: SemanticGraph,
    q_hrel: HyperRelation,
    t_hrel: HyperRelation,
) -> float:
    base = hyper_relation_match(q_hrel, t_hrel)
    if base == 0:
        return 0.0

    best_col_alignment = 0.0
    for perm in itertools.permutations(t_hrel.columns):
        s = hyper_column_set_match(q_graph, t_graph, q_hrel.columns, perm)
        best_col_alignment = max(best_col_alignment, s)

    return base * best_col_alignment


def best_alignment_score(
    q_graph: SemanticGraph,
    t_graph: SemanticGraph,
    binary_threshold: float = 0.35,
    hyper_threshold: float = 0.20,
    alpha_binary: float = 1.0,
    alpha_hyper: float = 1.5,
) -> float:
    total_binary = 0.0
    total_hyper = 0.0

    q_cols = q_graph.columns()
    t_cols = t_graph.columns()

    for qc1, qc2 in itertools.combinations(q_cols, 2):
        best = 0.0
        for tc1, tc2 in itertools.combinations(t_cols, 2):
            s1 = pair_match(q_graph, t_graph, qc1, qc2, tc1, tc2)
            s2 = pair_match(q_graph, t_graph, qc1, qc2, tc2, tc1)
            best = max(best, s1, s2)
        if best >= binary_threshold:
            total_binary += best

    q_by_arity: Dict[int, List[HyperRelation]] = {}
    t_by_arity: Dict[int, List[HyperRelation]] = {}

    for h in q_graph.hyperedges:
        q_by_arity.setdefault(len(h.columns), []).append(h)
    for h in t_graph.hyperedges:
        t_by_arity.setdefault(len(h.columns), []).append(h)

    for arity, q_hrels in q_by_arity.items():
        t_hrels = t_by_arity.get(arity, [])
        if not t_hrels:
            continue

        for q_h in q_hrels:
            best = 0.0
            for t_h in t_hrels:
                best = max(best, hyper_match(q_graph, t_graph, q_h, t_h))
            if best >= hyper_threshold:
                total_hyper += best

    return alpha_binary * total_binary + alpha_hyper * total_hyper


# ============================================================
# Search engine
# ============================================================

class SantosHyperKGLink:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        kg_backend: Optional[KGBackend] = None,
        max_hyperedge_size: int = 3,
    ):
        self.text_encoder = TextEncoder(model_name=model_name)
        self.kg_backend = kg_backend if kg_backend is not None else DemoKGBackend()
        self.kglink_encoder = KGLinkEncoder(self.text_encoder, self.kg_backend)
        self.max_hyperedge_size = max_hyperedge_size

        self.tables: Dict[str, pd.DataFrame] = {}
        self.graphs: Dict[str, SemanticGraph] = {}

    def add_table(self, name: str, df: pd.DataFrame) -> None:
        self.tables[name] = df.copy()
        self.graphs[name] = build_semantic_graph(
            df=df,
            table_name=name,
            text_encoder=self.text_encoder,
            kglink_encoder=self.kglink_encoder,
            max_hyperedge_size=self.max_hyperedge_size,
        )

    def search(self, query_df: pd.DataFrame, k: int = 5) -> List[Tuple[str, float]]:
        q_graph = build_semantic_graph(
            df=query_df,
            table_name="query",
            text_encoder=self.text_encoder,
            kglink_encoder=self.kglink_encoder,
            max_hyperedge_size=self.max_hyperedge_size,
        )

        scores = []
        for name, g in self.graphs.items():
            score = best_alignment_score(q_graph, g)
            scores.append((name, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def explain_column(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        info = self.kglink_encoder.encode_column(df, col)
        return {
            "column": col,
            "candidate_types": info.candidate_types,
            "linked_entities": info.linked_entities,
            "coverage": info.coverage,
        }

    def explain_hyperedges(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        g = build_semantic_graph(
            df=df,
            table_name="tmp",
            text_encoder=self.text_encoder,
            kglink_encoder=self.kglink_encoder,
            max_hyperedge_size=self.max_hyperedge_size,
        )
        out = []
        for h in g.hyperedges:
            out.append({
                "columns": h.columns,
                "label": h.label,
                "score": h.score,
                "support": h.support,
                "uniqueness": round(h.uniqueness, 4),
                "coverage": round(h.coverage, 4),
            })
        return out


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    santos = SantosHyperKGLink(max_hyperedge_size=3)

    sports = pd.DataFrame({
        "Player": ["Messi", "Messi", "Akram Afif"],
        "Team": ["PSG", "PSG", "PSG"],
        "Year": [2022, 2023, 2022],
    })

    sports2 = pd.DataFrame({
        "Athlete": ["Messi", "Akram Afif"],
        "Club": ["PSG", "PSG"],
        "Season": [2022, 2022],
    })

    films = pd.DataFrame({
        "Film Title": ["Coco", "Moana"],
        "Film Director": ["Adrian Molina", "Ron Clements"],
        "Year": [2017, 2016],
    })

    cities = pd.DataFrame({
        "City": ["Chicago", "Fresno", "Doha"],
        "Country": ["USA", "USA", "Qatar"],
        "Population": [2700000, 540000, 956000],
    })

    santos.add_table("sports", sports)
    santos.add_table("sports2", sports2)
    santos.add_table("films", films)
    santos.add_table("cities", cities)

    query = pd.DataFrame({
        "Player": ["Messi", "Akram Afif"],
        "Team": ["PSG", "PSG"],
        "Year": [2022, 2022],
    })

    print("=== Query hyperedges ===")
    for h in santos.explain_hyperedges(query):
        print(h)

    print("\n=== Explain columns ===")
    for c in query.columns:
        print(santos.explain_column(query, c))

    print("\n=== Top unionable tables ===")
    for name, score in santos.search(query, k=5):
        print(f"{name:10s} -> {score:.4f}")