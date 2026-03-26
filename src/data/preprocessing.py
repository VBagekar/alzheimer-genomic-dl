import GEOparse
import pandas as pd
import numpy as np
import os
from src.data.loader import load_gse, extract_gse48350, extract_gse5281
from src.data.loader import build_expression_matrix as build_expr_from_gse
from src.data.batch_correction import apply_combat, clean_and_filter, save_combined

RAW_DATA_PATH  = "data/raw"
PROCESSED_PATH = "data/processed"
os.makedirs(PROCESSED_PATH, exist_ok=True)


def build_single_dataset():
    gse = GEOparse.get_GEO(geo="GSE48350", destdir=RAW_DATA_PATH, silent=True)

    records = []
    for gsm_name, gsm in gse.gsms.items():
        disease_val = gsm.metadata.get("characteristics_ch1", [""])[0]
        source_val  = gsm.metadata.get("source_name_ch1",    [""])[0]

        if disease_val.startswith("gender:"):
            continue
        if disease_val.endswith(", C"):
            label = 0
        elif disease_val.endswith(", AA"):
            label = 1
        else:
            continue

        records.append({"sample_id": gsm_name, "label": label, "source": source_val})

    labels_df = pd.DataFrame(records).set_index("sample_id")
    counts    = labels_df["label"].value_counts()
    print(f"GSE48350 — Control: {counts.get(0,0)}, AD: {counts.get(1,0)}, Total: {len(labels_df)}")

    print("Building expression matrix...")
    expr_dict = {}
    valid     = labels_df.index.tolist()
    for i, gsm_name in enumerate(valid):
        if gsm_name not in gse.gsms:
            continue
        expr_dict[gsm_name] = gse.gsms[gsm_name].table.set_index("ID_REF")["VALUE"]
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(valid)}")

    expr_df = pd.DataFrame(expr_dict).T
    expr_df.index.name = "sample_id"

    expr_df   = expr_df.dropna(axis=1)
    variances = expr_df.var(axis=0)
    expr_df   = expr_df.loc[:, variances > variances.quantile(0.10)]
    print(f"After cleaning: {expr_df.shape}")

    common = expr_df.index.intersection(labels_df.index)
    X = expr_df.loc[common].values
    y = labels_df.loc[common, "label"].values

    np.save(f"{PROCESSED_PATH}/X_raw.npy",       X)
    np.save(f"{PROCESSED_PATH}/y_raw.npy",        y)
    np.save(f"{PROCESSED_PATH}/feature_names.npy", expr_df.columns.values)

    print(f"Saved X_raw: {X.shape}, y_raw: {y.shape}")


def build_combined_dataset():
    gse48350 = load_gse("GSE48350")
    gse5281  = load_gse("GSE5281")

    meta_48350 = extract_gse48350(gse48350)
    meta_5281  = extract_gse5281(gse5281)

    expr_48350 = build_expr_from_gse(gse48350, meta_48350.index.tolist())
    expr_5281  = build_expr_from_gse(gse5281,  meta_5281.index.tolist())

    corrected_df, combined_meta = apply_combat(expr_48350, expr_5281, meta_48350, meta_5281)
    corrected_df = clean_and_filter(corrected_df)

    X, y = save_combined(corrected_df, combined_meta)
    return X, y


if __name__ == "__main__":
    print("=== Single Dataset (GSE48350) ===")
    build_single_dataset()

    print("\n=== Combined Dataset (GSE48350 + GSE5281) ===")
    build_combined_dataset()