import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.preprocessing.candidate_types import extract_candidate_types


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class KGLinkEncoder:
    def __init__(self, linker, model_name="all-MiniLM-L6-v2"):
        self.linker = linker
        self.model = SentenceTransformer(model_name)

    def encode_text(self, text: str):
        return self.model.encode(text, normalize_embeddings=True)

    def encode_column(self, df: pd.DataFrame, col: str):
        values = df[col].dropna().astype(str).tolist()[:20]

        linked_entities = []
        for v in values:
            ents = self.linker.link_cell(v)
            if ents:
                linked_entities.append(ents[0])

        candidate_types = extract_candidate_types(linked_entities)
        type_labels = [t for t, _ in candidate_types]

        header_emb = self.encode_text(col)

        if values:
            value_embs = self.model.encode(values, normalize_embeddings=True)
            values_emb = np.mean(value_embs, axis=0)
        else:
            values_emb = np.zeros_like(header_emb)

        if type_labels:
            type_embs = self.model.encode(type_labels, normalize_embeddings=True)
            type_emb = np.mean(type_embs, axis=0)
        else:
            type_emb = np.zeros_like(header_emb)

        col_emb = 0.2 * header_emb + 0.3 * values_emb + 0.5 * type_emb
        denom = np.linalg.norm(col_emb)
        if denom > 0:
            col_emb = col_emb / denom

        return {
            "name": col,
            "embedding": col_emb,
            "candidate_types": candidate_types,
            "linked_entities": linked_entities
        }

    def column_similarity(self, q_col_info, t_col_info):
        return cosine(q_col_info["embedding"], t_col_info["embedding"])