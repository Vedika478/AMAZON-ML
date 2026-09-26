"""
evaluate_blocking.py
====================
Phase 3 – Blocking Evaluation

Computes:
  1. Candidate recall (recall ceiling)
  2. Reduction ratio
  3. Average / max candidates per Source 1
  4. Number of S1 entities with zero candidates
  5. Missed pairs list (for error analysis)
"""

import pandas as pd
from collections import defaultdict


def evaluate(
    candidates_dict: dict,
    gt_file: str,
) -> dict:
    """
    Evaluate blocking quality against training ground truth.

    IMPORTANT: Only evaluates S1 entities that appear in candidates_dict.
    This means sample runs correctly report recall over the sampled subset,
    not 0.0 due to unmatched full-GT rows.

    Parameters
    ----------
    candidates_dict : {source1_entity_id: [candidate_ids]}
    gt_file         : path to train_ground_truth.tsv

    Returns
    -------
    dict with keys:
        candidate_recall, total_true_pairs, captured_true_pairs,
        missed_true_pairs, s1_with_matches, candidate_count,
        max_candidates, avg_candidates, s1_zero_cands, missed_pairs,
        n_s1_evaluated
    """
    gt = pd.read_csv(gt_file, sep="\t", dtype=str)

    # ── Only evaluate rows whose S1 ID is in our candidate set ──────────
    # This is critical for sample runs: if we only generated candidates
    # for 3K S1 entities, we should only score those 3K, not all 2.2M.
    s1_ids_in_scope = set(candidates_dict.keys())
    gt = gt[gt["source1_entity_id"].isin(s1_ids_in_scope)].copy()

    # Parse ground-truth matches — handle NaN / empty strings
    def _parse_matches(val):
        if pd.isna(val) or str(val).strip() in ("", "nan"):
            return []
        return [v.strip() for v in str(val).split(",") if v.strip()]

    gt["_true_matches"] = gt["matched_entity_ids"].apply(_parse_matches)

    total_true_pairs = 0
    captured_true_pairs = 0
    s1_with_matches = 0
    s1_zero_cands = 0
    total_candidates_generated = 0
    max_cands = 0
    missed_pairs = []

    for _, row in gt.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        true_matches = set(row["_true_matches"])

        cands = candidates_dict.get(s1_id, [])
        cand_set = set(cands)
        cand_len = len(cand_set)

        total_candidates_generated += cand_len
        if cand_len > max_cands:
            max_cands = cand_len
        if cand_len == 0:
            s1_zero_cands += 1

        if true_matches:
            s1_with_matches += 1
            total_true_pairs += len(true_matches)
            captured = true_matches & cand_set
            captured_true_pairs += len(captured)
            missed = true_matches - cand_set
            for m in missed:
                missed_pairs.append((s1_id, m))

    recall = (
        captured_true_pairs / total_true_pairs if total_true_pairs > 0 else 0.0
    )
    n_s1 = len(gt)
    avg_cands = total_candidates_generated / n_s1 if n_s1 > 0 else 0.0

    return {
        "candidate_recall": recall,
        "total_true_pairs": total_true_pairs,
        "captured_true_pairs": captured_true_pairs,
        "missed_true_pairs": total_true_pairs - captured_true_pairs,
        "s1_with_matches": s1_with_matches,
        "candidate_count": total_candidates_generated,
        "max_candidates": max_cands,
        "avg_candidates": avg_cands,
        "s1_zero_cands": s1_zero_cands,
        "missed_pairs": missed_pairs,
        "n_s1_evaluated": n_s1,
    }


def print_missed_pairs_analysis(
    missed_pairs: list,
    s1_lookup: dict,
    s23_lookup: dict,
    n_sample: int = 30,
) -> None:
    """
    Print diagnostic info for missed ground-truth pairs (Section 13).

    Parameters
    ----------
    missed_pairs : list of (s1_id, true_candidate_id)
    s1_lookup    : {entity_id: row_dict} for ALL Source 1 records
    s23_lookup   : {entity_id: row_dict} for ALL Source 2+3 records
    n_sample     : max pairs to print
    """
    sample = missed_pairs[:n_sample]
    print(f"\n{'='*60}")
    print(f"MISSED PAIR ANALYSIS — showing {len(sample)} of {len(missed_pairs)} missed pairs")
    print("="*60)

    for s1_id, true_id in sample:
        s1_row = s1_lookup.get(s1_id, {})
        cand_row = s23_lookup.get(true_id, {})

        print(f"\n  S1 ID      : {s1_id}")
        print(f"  S1 Name    : {s1_row.get('business_name', 'N/A')}")
        print(f"  S1 Norm    : {s1_row.get('normalized_name', 'N/A')}")
        print(f"  S1 Addr    : {s1_row.get('business_address', 'N/A')}")
        print(f"  S1 Country : {s1_row.get('country', 'N/A')}")
        print(f"  S1 Phonetic: {s1_row.get('phonetic_key', 'N/A')}")
        print(f"  S1 Token   : {s1_row.get('sorted_token_key', 'N/A')}")
        print()
        print(f"  TRUE ID      : {true_id}")
        print(f"  TRUE Name    : {cand_row.get('business_name', 'N/A')}")
        print(f"  TRUE Norm    : {cand_row.get('normalized_name', 'N/A')}")
        print(f"  TRUE Addr    : {cand_row.get('business_address', 'N/A')}")
        print(f"  TRUE Country : {cand_row.get('country', 'N/A')}")
        print(f"  TRUE Phonetic: {cand_row.get('phonetic_key', 'N/A')}")
        print(f"  TRUE Token   : {cand_row.get('sorted_token_key', 'N/A')}")
        print(f"  {'─'*50}")
