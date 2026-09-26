import pandas as pd
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load files
print("Loading files...")
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')
s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t')
s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t')
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')

datasets = {'Source 1': s1, 'Source 2': s2, 'Source 3': s3}

for name, df in datasets.items():
    print(f"\\n--- {name} ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("Missing value %:")
    print((df.isnull().sum() / len(df) * 100).to_string())
    print("\\n10 sample business names:")
    print(df['business_name'].dropna().sample(10, random_state=42).tolist())
    print("\\n10 sample addresses:")
    print(df['business_address'].dropna().sample(10, random_state=42).tolist())
    print("\\nCountry distribution:")
    print(df['country'].value_counts(dropna=False).to_string())
    print(f"\\nNumber of unique entity IDs: {df['entity_id'].nunique()}")
    print(f"Duplicate rows count: {df.duplicated().sum()}")

print("\\n--- Ground Truth ---")
print(f"Number of Source 1 entities: {gt['source1_entity_id'].nunique()}")

gt['match_count'] = gt['matched_entity_ids'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) and x != '' else 0)

print("\\nDistribution of number of ground-truth matches per Source 1 entity:")
print(gt['match_count'].value_counts().sort_index().to_string())

singleton_pct = (gt['match_count'] == 0).sum() / len(gt) * 100
print(f"\\nPercentage of singleton Source 1 entities: {singleton_pct:.2f}%")
