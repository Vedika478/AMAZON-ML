import sys
import time
import pandas as pd

sys.path.append('.')
from src.normalization import (
    normalize_business_name,
    normalize_address,
    generate_phonetic_key,
    generate_sorted_token_key,
)
from src.candidate_generator import generate_candidates

def add_normalized_columns(df):
    df = df.copy()
    df['normalized_name'] = df['business_name'].apply(normalize_business_name)
    df['normalized_address'] = df['business_address'].apply(normalize_address)
    df['phonetic_key'] = df['normalized_name'].apply(generate_phonetic_key)
    df['sorted_token_key'] = df['normalized_name'].apply(generate_sorted_token_key)
    return df

print("Loading data...")
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')
s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t')
s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t')

print(f"S1: {len(s1):,} rows | S2: {len(s2):,} rows | S3: {len(s3):,} rows")

print("Normalizing S1...")
s1 = add_normalized_columns(s1)
print("Normalizing S2...")
s2 = add_normalized_columns(s2)
print("Normalizing S3...")
s3 = add_normalized_columns(s3)

print("S1 columns after normalization:", s1.columns.tolist())

sample_s1 = s1.sample(3000, random_state=42)

print("Running candidate generation on 3,000-entity sample...")
start = time.time()
result = generate_candidates(sample_s1, s2, s3)
elapsed = time.time() - start

print(f"\nDone in {elapsed:.1f} seconds")
total_candidates = sum(len(v) for v in result.values())
has_candidates = sum(1 for v in result.values() if v)
print(f"Total candidates: {total_candidates:,}")
print(f"S1 entities with at least 1 candidate: {has_candidates:,} / {len(result):,}")