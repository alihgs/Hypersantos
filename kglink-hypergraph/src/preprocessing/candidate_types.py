def extract_candidate_types(entities):

    scores = {}

    for ent in entities:

        for t in ent["types"]:
            scores[t] = scores.get(t, 0) + ent["score"]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked[:5]