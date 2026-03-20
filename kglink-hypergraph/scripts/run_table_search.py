import os
import pandas as pd

from src.utils.config_loader import load_config
from src.kg.elastic_backend import ElasticBackend
from src.kg.entity_linking import EntityLinker
from src.kg.kglink_encoder import KGLinkEncoder
from src.santos.kglink_table_matching import match_tables_with_kglink
from src.hypergraph.hyperedge_detection import detect_hyperedges
from src.hypergraph.hyper_match import hyper_match


TABLE_FOLDER = "data/tables"
QUERY_TABLE = "query.csv"

config = load_config()

backend = ElasticBackend(
    host=config["elasticsearch"]["host"],
    index=config["elasticsearch"]["index"]
)

linker = EntityLinker(backend)
kglink_encoder = KGLinkEncoder(linker)

query_path = os.path.join(TABLE_FOLDER, QUERY_TABLE)
query_df = pd.read_csv(query_path)

query_edges = detect_hyperedges(query_df)

print("=== Query hyperedges ===")
print(query_edges)
print()

scores = {}

for file in os.listdir(TABLE_FOLDER):
    if file == QUERY_TABLE or not file.endswith(".csv"):
        continue

    path = os.path.join(TABLE_FOLDER, file)
    df = pd.read_csv(path)

    # 1) score hypergraph
    table_edges = detect_hyperedges(df)
    hyper_score = 0.0
    for e1 in query_edges:
        for e2 in table_edges:
            hyper_score += hyper_match(e1, e2)

    # 2) score KGLink
    kglink_score = match_tables_with_kglink(query_df, df, kglink_encoder)

    # 3) score final
    final_score = 0.5 * hyper_score + 0.5 * kglink_score

    scores[file] = {
        "hyper_score": hyper_score,
        "kglink_score": kglink_score,
        "final_score": final_score
    }

print("=== Top unionable tables ===")
for table, vals in sorted(scores.items(), key=lambda x: x[1]["final_score"], reverse=True):
    print(
        f"{table} -> score_final={vals['final_score']:.4f} | "
        
    )