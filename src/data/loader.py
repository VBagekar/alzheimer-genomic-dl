import GEOparse
import pandas as pd
import numpy as np
import os

RAW_DATA_PATH = "data/raw"

REGION_KEYWORDS = [
    "hippocampus",
    "entorhinal cortex",
    "superior frontal gyrus",
    "postcentral gyrus",
    "post-central gyrus",
]


def load_gse(geo_id):
    print(f"Loading {geo_id}...")
    return GEOparse.get_GEO(geo=geo_id, destdir=RAW_DATA_PATH, silent=True)


def region_matches(source_str):
    source_lower = source_str.lower()
    return any(kw in source_lower for kw in REGION_KEYWORDS)


def extract_gse48350(gse):
    records = []
    for gsm_name, gsm in gse.gsms.items():
        disease_val = gsm.metadata.get("characteristics_ch1", [""])[0]
        source_val  = gsm.metadata.get("source_name_ch1",    [""])[0]

        if disease_val.startswith("gender:"):
            continue
        if not region_matches(source_val):
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
            "dataset":   "GSE48350"
        })

    df = pd.DataFrame(records).set_index("sample_id")
    print(f"GSE48350 — {len(df)} samples after region filter | AD: {df.label.sum()} | Control: {(df.label==0).sum()}")
    return df


def extract_gse5281(gse):
    records = []
    for gsm_name, gsm in gse.gsms.items():
        source_val = gsm.metadata.get("source_name_ch1", [""])[0]
        chars      = gsm.metadata.get("characteristics_ch1", [])

        if not region_matches(source_val):
            continue

        label = None
        for c in chars:
            c_lower = c.lower()
            if "control" in c_lower or "normal" in c_lower or "healthy" in c_lower:
                label = 0
                break
            if "alzheimer" in c_lower or " ad" in c_lower:
                label = 1
                break

        if label is None:
            if "incipient" in source_val.lower() or "moderate" in source_val.lower() or "severe" in source_val.lower():
                label = 1
            elif "normal" in source_val.lower():
                label = 0

        if label is None:
            continue

        records.append({
            "sample_id": gsm_name,
            "label":     label,
            "source":    source_val,
            "dataset":   "GSE5281"
        })

    df = pd.DataFrame(records).set_index("sample_id")
    print(f"GSE5281  — {len(df)} samples after region filter | AD: {df.label.sum()} | Control: {(df.label==0).sum()}")
    return df


def build_expression_matrix(gse, sample_ids):
    print(f"Building expression matrix for {len(sample_ids)} samples...")
    expr_dict = {}

    for i, gsm_name in enumerate(sample_ids):
        if gsm_name not in gse.gsms:
            continue
        expr_dict[gsm_name] = gse.gsms[gsm_name].table.set_index("ID_REF")["VALUE"]

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sample_ids)}")

    expr_df = pd.DataFrame(expr_dict).T
    expr_df.index.name = "sample_id"
    return expr_df