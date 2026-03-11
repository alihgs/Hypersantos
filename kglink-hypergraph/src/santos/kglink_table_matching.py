import itertools


def match_tables_with_kglink(query_df, table_df, encoder):
    q_infos = {c: encoder.encode_column(query_df, c) for c in query_df.columns}
    t_infos = {c: encoder.encode_column(table_df, c) for c in table_df.columns}

    total = 0.0

    for qc in query_df.columns:
        best = 0.0
        for tc in table_df.columns:
            s = encoder.column_similarity(q_infos[qc], t_infos[tc])
            best = max(best, s)
        total += best

    return total / max(1, len(query_df.columns))