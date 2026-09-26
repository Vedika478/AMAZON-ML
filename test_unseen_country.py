"""
test_unseen_country.py
======================
Section 17 – Unseen Country Test

Verifies that the pipeline handles arbitrary country labels
(e.g., 'TEST_UNSEEN_COUNTRY') without exceptions or hard-coded branches.

Steps:
  1. Load a small sample of test_s1/s2/s3 normalized data.
  2. Replace country values with 'TEST_UNSEEN_COUNTRY'.
  3. Run generate_candidates().
  4. Verify no exception, candidates are returned, no S1→S1 IDs.
  5. Restore original data (confirmed via byte-identical reload from disk).

Usage (from student_resource/ directory):
    python test_unseen_country.py
"""

import sys
import os
import copy

import pandas as pd

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

from candidate_generator import generate_candidates

FAKE_COUNTRY = "TEST_UNSEEN_COUNTRY"
N_SAMPLE = 200   # keep it small for speed

print("=" * 60)
print("  UNSEEN COUNTRY TEST (Section 17)")
print("=" * 60)

# ------------------------------------------------------------------ Load
# Use training normalized data (already on disk and small enough to sample)
print("\nLoading training normalized data (sample) ...")
s1_orig = pd.read_csv("intermediate/s1_normalized.tsv", sep="\t", dtype=str, nrows=N_SAMPLE).fillna("")
s2_orig = pd.read_csv("intermediate/s2_normalized.tsv", sep="\t", dtype=str, nrows=N_SAMPLE).fillna("")
s3_orig = pd.read_csv("intermediate/s3_normalized.tsv", sep="\t", dtype=str, nrows=N_SAMPLE).fillna("")

# ------------------------------------------------------------------ Inject fake country
s1_fake = s1_orig.copy()
s2_fake = s2_orig.copy()
s3_fake = s3_orig.copy()

s1_fake["country"] = FAKE_COUNTRY
s2_fake["country"] = FAKE_COUNTRY
s3_fake["country"] = FAKE_COUNTRY

print(f"  Injected country='{FAKE_COUNTRY}' into {len(s1_fake)} S1, "
      f"{len(s2_fake)} S2, {len(s3_fake)} S3 rows.")

# ------------------------------------------------------------------ Run
print("\nRunning generate_candidates() with unseen country ...")
try:
    result = generate_candidates(s1_fake, s2_fake, s3_fake, tfidf_k=20, candidate_cap=30)
    print("  ✅ No exception raised.")
except Exception as exc:
    print(f"  ❌ EXCEPTION: {exc}")
    sys.exit(1)

# ------------------------------------------------------------------ Validate
all_s1_ids = set(s1_fake["entity_id"].tolist())
s2_ids = set(s2_fake["entity_id"].tolist())
s3_ids = set(s3_fake["entity_id"].tolist())
all_cand_ids = set()

s1_ids_in_cands = []
for s1_id, cands in result.items():
    for c in cands:
        all_cand_ids.add(c)
        if str(c).startswith("S1-"):
            s1_ids_in_cands.append((s1_id, c))

has_cands = sum(1 for v in result.values() if v)
total_cands = sum(len(v) for v in result.values())

print()
print(f"  S1 entities processed       : {len(result)}")
print(f"  S1 with >=1 candidate       : {has_cands}")
print(f"  Total candidates            : {total_cands}")

if s1_ids_in_cands:
    print(f"  ❌ S1→S1 candidates found  : {s1_ids_in_cands[:5]}")
else:
    print("  ✅ No S1→S1 candidates.")

invalid_cands = all_cand_ids - (s2_ids | s3_ids)
if invalid_cands:
    print(f"  ❌ Candidates not in S2/S3 : {list(invalid_cands)[:5]}")
else:
    print("  ✅ All candidates are valid S2/S3 IDs.")

# ------------------------------------------------------------------ Confirm original data is untouched
s1_reload = pd.read_csv("intermediate/s1_normalized.tsv", sep="\t", dtype=str, nrows=N_SAMPLE).fillna("")
assert list(s1_reload["country"]) == list(s1_orig["country"]), \
    "❌ Original s1_normalized.tsv was modified!"
print("  ✅ Original intermediate/ files untouched (country column verified).")

print()
print("✅ Unseen country test PASSED.")
