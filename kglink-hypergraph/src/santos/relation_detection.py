def detect_relation(df, c1, c2):

    pairs = df[[c1, c2]].dropna()

    if len(pairs) == 0:
        return None

    if pairs.groupby(c1)[c2].nunique().max() == 1:
        return f"{c1}_determines_{c2}"

    return "associated"