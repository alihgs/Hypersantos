def compute_row_score(row_entities):

    scores = []

    for col, ents in row_entities.items():

        if not ents:
            scores.append(0)

        else:
            scores.append(max(e["score"] for e in ents))

    return sum(scores)


def filter_rows(rows_entities, top_k=25):

    scores = [(i, compute_row_score(r)) for i, r in enumerate(rows_entities)]

    scores.sort(key=lambda x: x[1], reverse=True)

    return [i for i, _ in scores[:top_k]]