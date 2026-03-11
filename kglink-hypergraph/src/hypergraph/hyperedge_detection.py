import itertools

def detect_hyperedges(df):
    edges = []
    cols = list(df.columns)

    for combo in itertools.combinations(cols, 3):
        subset = df[list(combo)].dropna()
        unique = subset.drop_duplicates()

        if len(unique) >= 2:
            edges.append(combo)

    return edges