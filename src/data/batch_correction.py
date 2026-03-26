import numpy as np
import pandas as pd
from combat.pycombat import pycombat
import os

PROCESSED_PATH = "data/processed"
os.makedirs(PROCESSED_PATH, exist_ok=True)


def find_overlapping_probes(expr_48350, expr_5281):
    overlap = expr_48350.columns.intersection(expr_5281.columns)
    print(f"Overlapping probes: {len(overlap)}")
    return overlap


def apply_combat(expr_48350, expr_5281, meta_48350, meta_5281):
    overlap_probes = find_overlapping_probes(expr_48350, expr_5281)

    expr_a = expr_48350[overlap_probes]
    expr_b = expr_5281[overlap_probes]

    combined_expr = pd.concat([expr_a, expr_b], axis=0)
    combined_meta = pd.concat([meta_48350, meta_5281], axis=0)

    print(f"Combined matrix before ComBat: {combined_expr.shape}")
    print(f"Batch 0 (GSE48350): {(combined_meta.dataset == 'GSE48350').sum()} samples")
    print(f"Batch 1 (GSE5281):  {(combined_meta.dataset == 'GSE5281').sum()} samples")

    batch_labels = (combined_meta.loc[combined_expr.index, "dataset"] == "GSE5281").astype(int).values

    data_for_combat = combined_expr.T
    print("Running ComBat batch correction...")
    corrected = pycombat(data_for_combat, batch_labels)

    corrected_df = corrected.T
    corrected_df.index = combined_expr.index

    print(f"ComBat complete. Corrected matrix: {corrected_df.shape}")
    return corrected_df, combined_meta


def clean_and_filter(expr_df, variance_quantile=0.10):
    expr_df = expr_df.dropna(axis=1)
    variances = expr_df.var(axis=0)
    threshold = variances.quantile(variance_quantile)
    expr_df   = expr_df.loc[:, variances > threshold]
    print(f"After cleaning: {expr_df.shape}")
    return expr_df


def save_combined(expr_df, meta_df):
    common = expr_df.index.intersection(meta_df.index)
    X = expr_df.loc[common].values
    y = meta_df.loc[common, "label"].values

    np.save(f"{PROCESSED_PATH}/X_combined.npy",      X)
    np.save(f"{PROCESSED_PATH}/y_combined.npy",      y)
    np.save(f"{PROCESSED_PATH}/feature_names_combined.npy", expr_df.columns.values)

    meta_df.loc[common].to_csv(f"{PROCESSED_PATH}/meta_combined.csv")

    print(f"Saved X_combined: {X.shape}")
    print(f"Total AD: {y.sum()} | Control: {(y==0).sum()}")
    return X, y