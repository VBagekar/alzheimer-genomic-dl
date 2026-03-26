
import GEOparse
import pandas as pd
import numpy as np
import os

RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

def load_gse48350():
    
    print("Loading GSE48350 — this may take a few minutes first time...")
    
    gse = GEOparse.get_GEO(
        geo="GSE48350",
        destdir=RAW_DATA_PATH,
        silent=False
    )
    
    return gse


def extract_expression_and_labels(gse):
    
    print("\n=== Dataset Overview ===")
    print(f"Number of samples:  {len(gse.gsms)}")
    print(f"Number of GPLs:     {len(gse.gpls)}")
    metadata = []
    for gsm_name, gsm in gse.gsms.items():
        meta = gsm.metadata
        metadata.append({
            "sample_id":    gsm_name,
            "title":        meta.get("title", [""])[0],
            "source":       meta.get("source_name_ch1", [""])[0],
            "description":  meta.get("description", [""])[0],
            "disease":      meta.get("characteristics_ch1", [""])[0],
        })
    
    meta_df = pd.DataFrame(metadata)
    
    print("\n=== Sample Metadata Preview ===")
    print(meta_df.head(10))
    print(f"\nUnique sources: {meta_df['source'].unique()}")
    print(f"Unique disease values: {meta_df['disease'].unique()}")
    
    return meta_df


def quick_peek(gse):
    """
    Peek at the raw expression values of first sample.
    """
    first_sample = list(gse.gsms.values())[0]
    print("\n=== First Sample Expression Data (first 5 rows) ===")
    print(first_sample.table.head())
    print(f"\nShape: {first_sample.table.shape}")
    print("Columns:", first_sample.table.columns.tolist())


if __name__ == "__main__":
    gse = load_gse48350()
    meta_df = extract_expression_and_labels(gse)
    quick_peek(gse)
    
    print("\n✅ Dataset loaded successfully!")
    print("Next: preprocessing.py will clean and encode this data")