import os
import pandas as pd
import sys
sys.path.append('src')
from normalization import normalize_business_name, normalize_address, generate_phonetic_key, generate_sorted_token_key

sys.stdout.reconfigure(encoding='utf-8')

os.makedirs('intermediate', exist_ok=True)

for i in range(1, 4):
    print(f"Processing Source {i}...")
    # Keep identifiers and address/name tokens as strings (leading zeroes
    # matter) and let the shared normalization functions handle missing data.
    df = pd.read_csv(f'dataset/train/train_source{i}.tsv', sep='\t', dtype=str).fillna('')
    
    df['normalized_name'] = df['business_name'].apply(normalize_business_name)
    df['normalized_address'] = df['business_address'].apply(normalize_address)
    df['phonetic_key'] = df['normalized_name'].apply(generate_phonetic_key)
    df['sorted_token_key'] = df['normalized_name'].apply(generate_sorted_token_key)
    
    out_cols = ['entity_id', 'business_name', 'normalized_name', 'business_address', 'normalized_address', 'phonetic_key', 'sorted_token_key', 'country']
    df[out_cols].to_csv(f'intermediate/s{i}_normalized.tsv', sep='\t', index=False)
    
    if i == 1:
        print("\nSanity Check (Source 1 Examples):")
        sample_df = df.sample(15, random_state=42)
        for _, row in sample_df.iterrows():
            print(f"ORIGINAL NAME: {row['business_name']}")
            print(f"NORMALIZED NAME: {row['normalized_name']}")
            print(f"ORIGINAL ADDRESS: {row['business_address']}")
            print(f"NORMALIZED ADDRESS: {row['normalized_address']}")
            print(f"PHONETIC KEY: {row['phonetic_key']}")
            print(f"SORTED TOKEN KEY: {row['sorted_token_key']}")
            print("-" * 50)
