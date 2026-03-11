def serialize_table(df, target_column):

    parts = []

    for col in df.columns:

        values = " | ".join(df[col].astype(str).tolist()[:5])

        if col == target_column:
            label = "[MASK]"
        else:
            label = "[CTX]"

        block = f"[CLS] {label} [COL] {col} [VAL] {values}"

        parts.append(block)

    return " ".join(parts)