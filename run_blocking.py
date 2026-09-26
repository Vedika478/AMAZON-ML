"""
run_blocking.py
===============
Phase 3 – Run candidate generation on TRAINING data, evaluate, log, and save.

Usage (from student_resource/ directory):
    python run_blocking.py                   # full training set
    python run_blocking.py --sample 2000     # sample 2000 S1 entities (quick test)

Outputs:
    output/candidate_pairs.tsv   — blocking candidates for training S1 entities
    logs/blocking_experiments.csv — experiment log
    logs/missed_pairs.json        — (if recall < 0.95) sample of missed pairs
"""

import argparse
import csv
import gc
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

from candidate_generator import generate_candidates
from evaluate_blocking import evaluate, print_missed_pairs_analysis

# ============================================================ Configuration
TFIDF_K = 100        # top-K TF-IDF neighbours per S1 record
CANDIDATE_CAP = 300  # max candidates kept per S1 entity after union
EXPERIMENT_ID = 2
STRATEGIES_DESC = "7-key country blocking (phonetic, token, words, address/name prefixes) + TF-IDF char 2-4 fallback"
NGRAM_RANGE = "(2,4)"
RECALL_TARGET = 0.95

# ============================================================ CLI args
parser = argparse.ArgumentParser(
    description="Run Phase 3 blocking + evaluation on training data."
)
parser.add_argument(
    "--sample", type=int, default=0,
    help="If > 0, randomly sample this many S1 entities (seed=42). 0 = use all."
)
parser.add_argument(
    "--pool-limit", type=int, default=0,
    help="Optional S2/S3 row limit for a smoke test. 0 keeps the full candidate pool."
)
args = parser.parse_args()

USE_SAMPLE = args.sample > 0
NOTES = (
    f"Sample pass (n={args.sample}, seed=42, "
    f"{'full S2/S3 pool' if args.pool_limit == 0 else f'pool limit={args.pool_limit}'})"
    if USE_SAMPLE else
    "Targeted pass after cap-75 recall miss; full training pool"
)

# ================================================================ Load data
print("Loading normalized training data ...", flush=True)

# Always load only the needed columns to save RAM
LOAD_COLS = [
    "entity_id", "country",
    "normalized_name", "normalized_address",
    "phonetic_key", "sorted_token_key",
]

if USE_SAMPLE:
    # Sample only S1 by default. Truncating S2/S3 makes recall invalid because
    # true matches may occur anywhere in the full candidate pool.
    pool_kwargs = {"nrows": args.pool_limit} if args.pool_limit > 0 else {}
    pool_desc = f"S2/S3 nrows={args.pool_limit:,}" if args.pool_limit > 0 else "full S2/S3 pool"
    print(f"  ⚡ SAMPLE MODE (seed=42, {pool_desc}):", flush=True)
    s1_full = pd.read_csv("intermediate/s1_normalized.tsv", sep="\t",
                          dtype=str, usecols=LOAD_COLS).fillna("")
    s2 = pd.read_csv("intermediate/s2_normalized.tsv", sep="\t",
                     dtype=str, usecols=LOAD_COLS, **pool_kwargs).fillna("")
    s3 = pd.read_csv("intermediate/s3_normalized.tsv", sep="\t",
                     dtype=str, usecols=LOAD_COLS, **pool_kwargs).fillna("")
    s1 = s1_full.sample(n=min(args.sample, len(s1_full)), random_state=42).reset_index(drop=True)
    del s1_full
    print(f"     S1 sample : {len(s1):,}", flush=True)
    print(f"     S2 rows   : {len(s2):,}", flush=True)
    print(f"     S3 rows   : {len(s3):,}", flush=True)
    if args.pool_limit > 0:
        print("  ⚠️  NOTE: capped S2/S3 means recall is a smoke-test metric only", flush=True)
else:
    s1 = pd.read_csv("intermediate/s1_normalized.tsv", sep="\t",
                     dtype=str, usecols=LOAD_COLS).fillna("")
    s2 = pd.read_csv("intermediate/s2_normalized.tsv", sep="\t",
                     dtype=str, usecols=LOAD_COLS).fillna("")
    s3 = pd.read_csv("intermediate/s3_normalized.tsv", sep="\t",
                     dtype=str, usecols=LOAD_COLS).fillna("")

gc.collect()
print(f"  S1: {len(s1):,} rows  |  S2: {len(s2):,} rows  |  S3: {len(s3):,} rows", flush=True)


# ================================================= Generate candidates
t0 = time.time()
print(f"\nGenerating candidates  (TF-IDF K={TFIDF_K}, cap={CANDIDATE_CAP}) ...", flush=True)
candidates_dict = generate_candidates(
    s1, s2, s3, tfidf_k=TFIDF_K, candidate_cap=CANDIDATE_CAP
)

t_gen = time.time() - t0
print(f"  Candidate generation time: {t_gen:.1f}s", flush=True)

# =================================================== Evaluate
print("\nEvaluating blocking quality ...", flush=True)
eval_res = evaluate(candidates_dict, "dataset/train/train_ground_truth.tsv")

s1_count = len(s1)
s23_count = len(s2) + len(s3)
full_cross = s1_count * s23_count
reduction_ratio = (
    eval_res["candidate_count"] / full_cross if full_cross > 0 else 0.0
)

# ======================================================= Print results
print()
t_total = time.time() - t0
print("=" * 55)
print("  BLOCKING EVALUATION RESULTS")
print("=" * 55)
print(f"  Candidate Recall          : {eval_res['candidate_recall']:.4f}")
print(f"  Total true pairs          : {eval_res['total_true_pairs']:,}")
print(f"  Captured true pairs       : {eval_res['captured_true_pairs']:,}")
print(f"  Missed true pairs         : {eval_res['missed_true_pairs']:,}")
print(f"  S1 with >=1 true match    : {eval_res['s1_with_matches']:,}")
print()
print(f"  Total candidates          : {eval_res['candidate_count']:,}")
print(f"  Full cross-product        : {full_cross:,}")
print(f"  Reduction ratio           : {reduction_ratio:.4e}")
print(f"  Avg candidates / S1       : {eval_res['avg_candidates']:.2f}")
print(f"  Max candidates for any S1 : {eval_res['max_candidates']}")
print(f"  S1 with zero candidates   : {eval_res['s1_zero_cands']:,}")
print(f"  Total wall time           : {t_total:.1f}s")
print("=" * 55)

# ============================================ Missed-pair error analysis
if eval_res["candidate_recall"] < RECALL_TARGET:
    print(f"\n⚠️  Recall {eval_res['candidate_recall']:.4f} < target {RECALL_TARGET}", flush=True)
    print("Running error analysis on missed pairs ...", flush=True)
    sample_pairs = eval_res["missed_pairs"][:30]
    s1_needed = {p[0] for p in sample_pairs}
    s23_needed = {p[1] for p in sample_pairs}

    def _load_targeted_lookup(path, needed_ids):
        if not needed_ids:
            return {}
        found = {}
        for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=100000):
            matches = chunk[chunk["entity_id"].isin(needed_ids)]
            for _, r in matches.iterrows():
                found[r["entity_id"]] = r.to_dict()
            if len(found) >= len(needed_ids):
                break
        return found

    print("  Loading targeted S1 records for diagnostic display ...", flush=True)
    s1_full_lookup = _load_targeted_lookup("intermediate/s1_normalized.tsv", s1_needed)
    print("  Loading targeted S2/S3 records for diagnostic display ...", flush=True)
    s2_lookup = _load_targeted_lookup("intermediate/s2_normalized.tsv", s23_needed)
    s23_remaining = s23_needed - set(s2_lookup.keys())
    s3_lookup = _load_targeted_lookup("intermediate/s3_normalized.tsv", s23_remaining)
    s23_full_lookup = {**s2_lookup, **s3_lookup}

    print_missed_pairs_analysis(
        eval_res["missed_pairs"], s1_full_lookup, s23_full_lookup, n_sample=30
    )
    del s1_full_lookup, s23_full_lookup
    gc.collect()
    os.makedirs("logs", exist_ok=True)
    missed_serializable = [
        {"s1_id": p[0], "true_id": p[1]} for p in eval_res["missed_pairs"][:100]
    ]
    with open("logs/missed_pairs.json", "w", encoding="utf-8") as f:
        json.dump(missed_serializable, f, indent=2)
    print("\nSaved up to 100 missed pairs → logs/missed_pairs.json", flush=True)
else:
    print(f"\n✅ Recall target met: {eval_res['candidate_recall']:.4f} >= {RECALL_TARGET}")

# ========================================================= Log experiment
os.makedirs("logs", exist_ok=True)
log_file = "logs/blocking_experiments.csv"
file_exists = os.path.isfile(log_file)

with open(log_file, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            "experiment_id", "strategies", "tfidf_ngram_range",
            "tfidf_k", "candidate_cap", "candidate_recall",
            "reduction_ratio", "avg_candidates", "max_candidates", "notes",
        ])
    writer.writerow([
        EXPERIMENT_ID,
        STRATEGIES_DESC,
        NGRAM_RANGE,
        TFIDF_K,
        CANDIDATE_CAP,
        f"{eval_res['candidate_recall']:.4f}",
        f"{reduction_ratio:.4e}",
        f"{eval_res['avg_candidates']:.2f}",
        eval_res["max_candidates"],
        NOTES,
    ])
print(f"\nLogged experiment → {log_file}")

# ========================================= Save candidate_pairs.tsv (training)
os.makedirs("output", exist_ok=True)
out_path = "output/candidate_pairs.tsv"
print(f"\nSaving candidate pairs → {out_path} ...")

all_s1_ids = s1["entity_id"].tolist()
with open(out_path, "w", encoding="utf-8", newline="") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for s1_id in all_s1_ids:
        cands = candidates_dict.get(s1_id, [])
        # Deduplicate preserving order
        seen = set()
        deduped = []
        for c in cands:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        f.write(f"{s1_id}\t{','.join(deduped)}\n")

print(f"  Written {len(all_s1_ids):,} rows.")
print("\n✅ Phase 3 (training) complete.")
