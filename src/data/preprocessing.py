

import GEOparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle

RAW_DATA_PATH   = "data/raw"
PROCESSED_PATH  = "data/processed"
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
            print(f"  Skipping unknown label: {disease_val}")
            continue

        records.append({
            "sample_id": gsm_name,
            "label":     label,
            "source":    source_val,
            "raw_label": disease_val
        })

    labels_df = pd.DataFrame(records).set_index("sample_id")

    print("\n=== Label Distribution ===")
    counts = labels_df["label"].value_counts()
    print(f"  Control (0):     {counts.get(0, 0)} samples")
    print(f"  Alzheimer's (1): {counts.get(1, 0)} samples")
    print(f"  Total:           {len(labels_df)} samples")

    return labels_df



def build_expression_matrix(gse, labels_df):
    valid_samples = labels_df.index.tolist()
    expr_dict = {}

    for i, gsm_name in enumerate(valid_samples):
        if gsm_name not in gse.gsms:
            continue
        gsm = gse.gsms[gsm_name]
        
        expr_dict[gsm_name] = gsm.table.set_index("ID_REF")["VALUE"]

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(valid_samples)} samples")

    
    expr_df = pd.DataFrame(expr_dict).T
    expr_df.index.name = "sample_id"

    print(f"\nExpression matrix shape: {expr_df.shape}")
    print(f"  → {expr_df.shape[0]} samples × {expr_df.shape[1]} gene probes")

    return expr_df



def clean_matrix(expr_df):
    
    

    before = expr_df.shape[1]

    
    expr_df = expr_df.dropna(axis=1)
    print(f"  After dropping NaN probes:      {expr_df.shape[1]} (removed {before - expr_df.shape[1]})")
    variances  = expr_df.var(axis=0)
    threshold  = variances.quantile(0.10)   
    expr_df    = expr_df.loc[:, variances > threshold]
    print(f"  After dropping low-var probes:  {expr_df.shape[1]}")

    return expr_df



def select_top_features(expr_df, labels_df, top_k=5000):
    variances     = expr_df.var(axis=0)
    top_probes    = variances.nlargest(top_k).index
    expr_filtered = expr_df[top_probes]

    print(f"  Final matrix: {expr_filtered.shape[0]} samples × {expr_filtered.shape[1]} probes")
    return expr_filtered


def split_and_scale(expr_df, labels_df):
   
    common = expr_df.index.intersection(labels_df.index)
    X = expr_df.loc[common].values
    y = labels_df.loc[common, "label"].values

    
    print(f"  Total samples: {len(y)}, AD: {y.sum()}, Control: {(y==0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.20,
        stratify     = y,       
        random_state = 42
    )

    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def save_processed(X_train, X_test, y_train, y_test, scaler, feature_names):
    np.save(f"{PROCESSED_PATH}/X_train.npy",       X_train)
    np.save(f"{PROCESSED_PATH}/X_test.npy",        X_test)
    np.save(f"{PROCESSED_PATH}/y_train.npy",       y_train)
    np.save(f"{PROCESSED_PATH}/y_test.npy",        y_test)
    np.save(f"{PROCESSED_PATH}/feature_names.npy", feature_names)

    with open(f"{PROCESSED_PATH}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n Saved to {PROCESSED_PATH}/")
    print(f"   X_train.npy  → {X_train.shape}")
    print(f"   X_test.npy   → {X_test.shape}")
    print(f"   y_train.npy  → {y_train.shape}")
    print(f"   y_test.npy   → {y_test.shape}")


if __name__ == "__main__":
    gse        = load_raw()
    labels_df  = extract_labels(gse)
    expr_df    = build_expression_matrix(gse, labels_df)
    expr_df    = clean_matrix(expr_df)
    expr_df    = select_top_features(expr_df, labels_df, top_k=5000)

    X_train, X_test, y_train, y_test, scaler = split_and_scale(expr_df, labels_df)

    save_processed(
        X_train, X_test, y_train, y_test,
        scaler,
        feature_names = expr_df.columns.values
    )

    print("\n Preprocessing done.")