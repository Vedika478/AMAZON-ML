"""
run_blocking_test.py
====================
Phase 5 – Run candidate generation on TEST data and produce final output.

Prerequisites:
    1. run_normalization_test.py  (creates intermediate/test_s*.tsv)
    2. run_blocking.py            (training recall validated >= 95%)

Produces:
    output/candidate_pairs.tsv   — final candidate set for test S1 entities
                                   (THIS is what goes into the submission zip
                                    IF Person 2 does NOT further filter candidates)

Usage (from student_resource/ directory):
    python run_blocking_test.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

from candidate_generator import generate_candidates

# ============================================================ Configuration
# Use the SAME config validated on training data
TFIDF_K = 300
CANDIDATE_CAP = 1000

LOAD_COLS = [
    "entity_id", "country",
    "normalized_name", "normalized_address",
    "phonetic_key", "sorted_token_key",
]

# ================================================================ Load data
print("Loading normalized test data ...", flush=True)
s1 = pd.read_csv("intermediate/test_s1_normalized.tsv", sep="\t", dtype=str, usecols=LOAD_COLS).fillna("")
s2 = pd.read_csv("intermediate/test_s2_normalized.tsv", sep="\t", dtype=str, usecols=LOAD_COLS).fillna("")
s3 = pd.read_csv("intermediate/test_s3_normalized.tsv", sep="\t", dtype=str, usecols=LOAD_COLS).fillna("")
print(f"  S1: {len(s1):,}  |  S2: {len(s2):,}  |  S3: {len(s3):,}", flush=True)

# ================================================= Generate candidates
print(f"\nGenerating candidates (TF-IDF K={TFIDF_K}, cap={CANDIDATE_CAP}) ...")
candidates_dict = generate_candidates(
    s1, s2, s3, tfidf_k=TFIDF_K, candidate_cap=CANDIDATE_CAP
)

# ========================================= Save candidate_pairs.tsv (test)
os.makedirs("output", exist_ok=True)
out_path = "output/candidate_pairs.tsv"
print(f"\nSaving → {out_path} ...")

all_s1_ids = s1["entity_id"].tolist()
with open(out_path, "w", encoding="utf-8", newline="") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for s1_id in all_s1_ids:
        cands = candidates_dict.get(s1_id, [])
        seen = set()
        deduped = []
        for c in cands:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        f.write(f"{s1_id}\t{','.join(deduped)}\n")

# ================================================= Summary stats
has_cands = sum(1 for v in candidates_dict.values() if v)
total_cands = sum(len(v) for v in candidates_dict.values())
zero_cands = len(all_s1_ids) - has_cands
max_cands = max((len(v) for v in candidates_dict.values()), default=0)
avg_cands = total_cands / len(all_s1_ids) if all_s1_ids else 0.0

print()
print("=" * 55)
print("  TEST CANDIDATE GENERATION SUMMARY")
print("=" * 55)
print(f"  Total S1 test entities        : {len(all_s1_ids):,}")
print(f"  S1 with candidates            : {has_cands:,}")
print(f"  S1 with zero candidates       : {zero_cands:,}")
print(f"  Total candidates generated    : {total_cands:,}")
print(f"  Avg candidates / S1           : {avg_cands:.2f}")
print(f"  Max candidates for any S1     : {max_cands}")
print("=" * 55)
print(f"\n  Output → {out_path}")
print()
print("  ⚠️  HANDOFF NOTE (Section 15):")
print("  This file is the BLOCKING-STAGE output.")
print("  If Person 2 filters candidates further before model inference,")
print("  the FINAL candidate_pairs.tsv for the submission zip must be")
print("  reconstructed from Person 2's pre-inference candidate set,")
print("  NOT from this file.  Confirm with Person 2 before submitting.")
print()
print("✅ Phase 5 (test candidates) complete.")

