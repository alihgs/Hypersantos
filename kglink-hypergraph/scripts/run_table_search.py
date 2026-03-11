import pandas as pd
import os

from src.hypergraph.hyperedge_detection import detect_hyperedges
from src.hypergraph.hyper_match import hyper_match


TABLE_FOLDER = "data/tables"
QUERY_TABLE = "query.csv"


query_path = os.path.join(TABLE_FOLDER, QUERY_TABLE)

query_df = pd.read_csv(query_path)

query_edges = detect_hyperedges(query_df)

print("=== Query hyperedges ===")
print(query_edges)
print()

scores = {}


for file in os.listdir(TABLE_FOLDER):

    if file == QUERY_TABLE:
        continue

    path = os.path.join(TABLE_FOLDER, file)

    df = pd.read_csv(path)

    edges = detect_hyperedges(df)

    score = 0

    for e1 in query_edges:
        for e2 in edges:
            score += hyper_match(e1, e2)

    scores[file] = score


print("=== Top unionable tables ===")

for table, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):

    print(f"{table} -> {score:.4f}")