"""
run_normalization_test.py
=========================
Phase 5 – Normalize TEST data and write intermediate/ files.

Produces:
    intermediate/test_s1_normalized.tsv
    intermediate/test_s2_normalized.tsv
    intermediate/test_s3_normalized.tsv

Usage (from student_resource/ directory):
    python run_normalization_test.py
"""

import os
import sys
import pandas as pd

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

from normalization import (
    normalize_business_name,
    normalize_address,
    generate_phonetic_key,
    generate_sorted_token_key,
)

os.makedirs("intermediate", exist_ok=True)

OUT_COLS = [
    "entity_id", "business_name", "normalized_name",
    "business_address", "normalized_address",
    "phonetic_key", "sorted_token_key", "country",
]

for i in range(1, 4):
    src_path = f"dataset/test/test_source{i}.tsv"
    out_path = f"intermediate/test_s{i}_normalized.tsv"
    print(f"Normalizing test Source {i}: {src_path} ...")
    df = pd.read_csv(src_path, sep="\t", dtype=str).fillna("")
    df["normalized_name"] = df["business_name"].apply(normalize_business_name)
    df["normalized_address"] = df["business_address"].apply(normalize_address)
    df["phonetic_key"] = df["normalized_name"].apply(generate_phonetic_key)
    df["sorted_token_key"] = df["normalized_name"].apply(generate_sorted_token_key)
    df[OUT_COLS].to_csv(out_path, sep="\t", index=False)
    print(f"  Saved {len(df):,} rows → {out_path}")

print("\n✅ Test normalization complete.")
