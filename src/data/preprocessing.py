import GEOparse
import pandas as pd
import numpy as np
import os

RAW_DATA_PATH  = "data/raw"
PROCESSED_PATH = "data/processed"
os.makedirs(PROCESSED_PATH, exist_ok=True)


def load_raw(geo_id="GSE48350"):
    print(f"Loading {geo_id}...")
    gse = GEOparse.get_GEO(geo=geo_id, destdir=RAW_DATA_PATH, silent=True)
    return gse


def extract_labels(gse):
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

        records.append({
            "sample_id": gsm_name,
            "label":     label,
            "source":    source_val,
        })

    labels_df = pd.DataFrame(records).set_index("sample_id")

    counts = labels_df["label"].value_counts()
    print(f"Control: {counts.get(0,0)}, Alzheimer's: {counts.get(1,0)}, Total: {len(labels_df)}")

    return labels_df


def build_expression_matrix(gse, labels_df):
    print("Building expression matrix...")
    valid_samples = labels_df.index.tolist()
    expr_dict = {}

    for i, gsm_name in enumerate(valid_samples):
        if gsm_name not in gse.gsms:
            continue
        gsm = gse.gsms[gsm_name]
        expr_dict[gsm_name] = gsm.table.set_index("ID_REF")["VALUE"]

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(valid_samples)}")

    expr_df = pd.DataFrame(expr_dict).T
    expr_df.index.name = "sample_id"
    return expr_df


def clean_matrix(expr_df):
    expr_df = expr_df.dropna(axis=1)
    variances = expr_df.var(axis=0)
    threshold = variances.quantile(0.10)
    expr_df   = expr_df.loc[:, variances > threshold]
    print(f"After cleaning: {expr_df.shape}")
    return expr_df


if __name__ == "__main__":
    gse       = load_raw()
    labels_df = extract_labels(gse)
    expr_df   = build_expression_matrix(gse, labels_df)
    expr_df   = clean_matrix(expr_df)

    common = expr_df.index.intersection(labels_df.index)
    X = expr_df.loc[common].values
    y = labels_df.loc[common, "label"].values

    np.save(f"{PROCESSED_PATH}/X_raw.npy", X)
    np.save(f"{PROCESSED_PATH}/y_raw.npy", y)
    np.save(f"{PROCESSED_PATH}/feature_names.npy", expr_df.columns.values)

    print(f"Saved X_raw: {X.shape}, y_raw: {y.shape}")
    print("No scaling or feature selection applied — will be done inside each fold")