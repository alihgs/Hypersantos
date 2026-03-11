import pandas as pd

def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

def unique_values(series):
    vals = series.dropna().astype(str).str.lower().unique()
    return list(vals)