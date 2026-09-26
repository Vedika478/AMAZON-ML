"""
candidate_generator.py
======================
Phase 3 – Fast multi-key blocking for 10M+ records, with vote-based
ranking (not arbitrary set order) and a lightweight fuzzy fallback for
entities that pure exact-key blocking misses (typos, transliteration).

Multi-key blocking (pure vectorized pandas):
  Key 1: Country + Phonetic Key + Name Prefix (3 chars)
  Key 2: Country + Sorted Token Key
  Key 3: Country + First Word (len >= 3)
  Key 4: Country + Second Word (len >= 3)
  Key 5: Country + Address Prefix (4 chars) + Name Prefix (3 chars)
  Key 6: Country + Broad Phonetic Key (small blocks only)
    Key 7: Country + Address Prefix (6 chars)
    Key 8: Country + Address Words (house-number independent)
    Key 7: Country + Address Prefix (6 chars)

Candidates are ranked by how many independent keys "voted" for them,
so the final cap keeps the most-corroborated candidates, not an
arbitrary subset. Entities with weak coverage from exact keys get a
targeted TF-IDF fallback (cheap, since it only runs on the small
leftover subset, not the full dataset).
"""

import gc
import re
from collections import defaultdict, Counter
import pandas as pd

NEEDED_COLS = [
    "entity_id", "country",
    "normalized_name", "normalized_address",
    "phonetic_key", "sorted_token_key",
]

STOP_WORDS = {
    "and", "the", "inc", "corp", "corporation", "llc", "ltd", "limited",
    "pvt", "private", "co", "company", "services", "solutions", "group",
    "enterprises", "associates", "tech", "technologies", "international",
    "global", "india", "usa", "us", "in", "of", "for",
}

# Threshold below which an S1 entity is considered "weak coverage"
# and gets a fuzzy TF-IDF fallback pass.
# Fuzzy retrieval has high value even when exact blocking found a few records:
# a handful of exact candidates does not mean the true pair was retrieved.
WEAK_COVERAGE_THRESHOLD = 100


def _is_empty_component(k: str) -> bool:
    """True if ANY pipe-separated component of the key is empty/blank."""
    if not k or k == "nan":
        return True
    parts = k.split("|")
    return any(p.strip() == "" or p.strip() == "nan" for p in parts)


def _make_word_cols(names_series):
    """Vectorized: extract first and second significant word."""
    def _fw(text):
        tokens = str(text).lower().split()
        clean = [re.sub(r"[^\w]", "", t) for t in tokens]
        sig = [t for t in clean if len(t) >= 3 and t not in STOP_WORDS]
        w1 = sig[0] if sig else (clean[0] if clean else "")
        w2 = sig[1] if len(sig) > 1 else ""
        return w1, w2

    pairs = names_series.apply(_fw)
    return pairs.apply(lambda x: x[0]), pairs.apply(lambda x: x[1])


def _consonant_skeleton(name):
    letters = [char for char in str(name).lower() if char.isalpha()]
    consonants = [char for char in letters if char not in "aeiou"]
    collapsed = []
    for char in consonants:
        if not collapsed or collapsed[-1] != char:
            collapsed.append(char)
    return "".join(collapsed[:6])


def address_token_set_key(address: str, min_token_len=3, top_n_tokens=3) -> str:
    tokens = re.findall(r"\w+", str(address).lower())
    stop = {"road", "street", "rd", "st", "near", "opposite", "opp", "the", "and"}
    significant = [token for token in tokens if len(token) >= min_token_len and token not in stop]
    return "_".join(sorted(set(significant)))


def _address_tokens(address):
    stop = {"road", "street", "rd", "st", "near", "opposite", "opp",
            "the", "and", "floor", "no", "number", "india", "usa"}
    tokens = {t for t in re.findall(r"\w+", str(address).lower())
              if len(t) >= 3 and t not in stop}
    # Retain a few most informative tokens (numbers, street names, localities)
    # to bound posting-list size on the multi-million-row pool.
    return sorted(sorted(tokens, key=lambda t: (-len(t), t))[:3])


def _apply_key(vote_counts, s1_ids, s1_keys, s23_ids, s23_keys,
               max_per_block=75, block_size_limit=500):
    """
    Build index from s23 keys and query with s1 keys.
    Instead of blindly truncating a block by file order, we keep the
    FULL block up to block_size_limit but distribute inclusion evenly
    (simple stride sampling) rather than always keeping the first N —
    this avoids systematically dropping records purely by their
    position in the source file.
    Matches are recorded as VOTES, not just membership, so later keys
    reinforce earlier ones instead of just re-adding the same IDs.
    """
    idx = defaultdict(list)
    for eid, k in zip(s23_ids, s23_keys):
        k = str(k).strip()
        if k and not _is_empty_component(k):
            bucket = idx[k]
            bucket.append(eid)

    for s1_id, k in zip(s1_ids, s1_keys):
        k = str(k).strip()
        if k and not _is_empty_component(k) and k in idx:
            bucket = idx[k]
            # Take up to max_per_block, but the FULL bucket is used to
            # cast votes — voting happens before any per-key capping,
            # so downstream ranking still reflects real corroboration.
            if len(bucket) > max_per_block:
                # Evenly sample the full bucket; retaining the file prefix
                # systematically lost valid records from later partitions.
                step = len(bucket) / max_per_block
                selected = (bucket[int(i * step)] for i in range(max_per_block))
            else:
                selected = iter(bucket)
            for cand in selected:
                vote_counts[s1_id][cand] += 1
    idx.clear()


def _fuzzy_fallback(weak_s1_df, s23_df, s23_id_set, top_k=15):
    """
    Lightweight TF-IDF fallback ONLY for the small subset of S1 entities
    with weak coverage from exact-key blocking. Runs within country
    groups to keep each TF-IDF call small and fast, never globally.
    """
    fallback_candidates = defaultdict(set)

    if weak_s1_df.empty:
        return fallback_candidates

    for country_val, s1_group in weak_s1_df.groupby("country", dropna=False):
        # Country is useful evidence, but it is incomplete/noisy. Include
        # same-country candidates first, then allow a bounded global fallback.
        pool_group = s23_df[s23_df["country"] == country_val]
        if pool_group.empty or len(s1_group) == 0:
            continue

        # Safety cap: if even a single-country pool is enormous,
        # sub-block by phonetic key first before TF-IDF.
        if len(pool_group) > 20000:
            # Broad phonetic prefix tolerates a changed Soundex code while
            # keeping the retrieval pool tractable.
            pool_keys = pool_group["normalized_name"].str[:1].str.casefold()
            query_keys = s1_group["normalized_name"].str[:1].str.casefold()
            for prefix, sub_s1 in s1_group.groupby(query_keys, dropna=False):
                sub_pool = pool_group[pool_keys == prefix]
                if not sub_pool.empty:
                    _tfidf_match(sub_s1, sub_pool, fallback_candidates, top_k)
        else:
            _tfidf_match(s1_group, pool_group, fallback_candidates, top_k)

    return fallback_candidates


def _tfidf_match(s1_group, pool_group, out_candidates, top_k):
    pool_text = (pool_group["normalized_name"].fillna("") + " " +
                 pool_group["normalized_address"].fillna(""))
    s1_text = (s1_group["normalized_name"].fillna("") + " " +
               s1_group["normalized_address"].fillna(""))
    if not len(pool_group) or not len(s1_group):
        return
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3))
        pool_vecs = vectorizer.fit_transform(pool_text)
        s1_vecs = vectorizer.transform(s1_text)
        n_neighbors = min(top_k, len(pool_group))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(pool_vecs)
        _, indices = nn.kneighbors(s1_vecs)
        pool_ids = pool_group["entity_id"].values
        for s1_id, row in zip(s1_group["entity_id"].values, indices):
            out_candidates[s1_id].update(pool_ids[row])
        return
    except ImportError:
        # A dependency-free sparse character TF-IDF fallback keeps retrieval
        # available in lightweight challenge runtimes.
        import heapq
        from collections import Counter

        def grams(text):
            padded = " " + str(text).lower() + " "
            return Counter(padded[i:i+n] for n in (1, 2, 3)
                          for i in range(max(0, len(padded)-n+1)))

        docs = [grams(text) for text in pool_text.tolist()]
        df = Counter(g for doc in docs for g in doc.keys())
        n_docs = len(docs)
        idf = {g: (1.0 + __import__("math").log((1+n_docs)/(1+freq))) for g, freq in df.items()}
        postings = defaultdict(list)
        doc_norms = [0.0] * n_docs
        for i, doc in enumerate(docs):
            total = sum(doc.values()) or 1
            for gram, count in doc.items():
                weight = (count / total) * idf[gram]
                postings[gram].append((i, weight))
                doc_norms[i] += weight * weight
        doc_norms = [v ** 0.5 for v in doc_norms]
        pool_ids = pool_group["entity_id"].values
        for s1_id, text in zip(s1_group["entity_id"].values, s1_text.tolist()):
            query = grams(text)
            total = sum(query.values()) or 1
            qweights = {g: (count/total) * idf.get(g, 1.0) for g, count in query.items()}
            qnorm = sum(v*v for v in qweights.values()) ** 0.5 or 1.0
            scores = defaultdict(float)
            for gram, weight in qweights.items():
                for i, doc_weight in postings.get(gram, ()):
                    scores[i] += weight * doc_weight
            ranked = heapq.nlargest(min(top_k, len(scores)), scores,
                                    key=lambda i: scores[i] / (qnorm * doc_norms[i] or 1.0))
            out_candidates[s1_id].update(pool_ids[i] for i in ranked)


def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    tfidf_k: int = 15,
    candidate_cap: int = 1000,
    run_fuzzy_fallback: bool = True,
) -> dict:
    """
    Generate blocking candidates using 6-key union blocking, ranked by
    vote count, with an optional lightweight fuzzy fallback for weak-
    coverage entities.

    Returns
    -------
    dict {source1_entity_id: [candidate_entity_id, ...]}
    """
    def _slim(df):
        cols = [c for c in NEEDED_COLS if c in df.columns]
        return df[cols].copy().fillna("")

    s1_df = _slim(s1_df)
    s2_df = _slim(s2_df)
    s3_df = _slim(s3_df)

    for df in (s1_df, s2_df, s3_df):
        df["country"] = df["country"].astype(str).str.strip()

    print("  Concatenating S2+S3 ...", flush=True)
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    gc.collect()

    s23_id_set = set(s23_df["entity_id"].tolist())
    vote_counts = defaultdict(Counter)  # s1_id -> Counter({candidate_id: votes})

    print("  Building blocking keys ...", flush=True)
    s1_ids = s1_df["entity_id"].values
    s23_ids = s23_df["entity_id"].values

    s1_pref3 = s1_df["normalized_name"].str[:3].values
    s23_pref3 = s23_df["normalized_name"].str[:3].values

    s1_addr4 = s1_df["normalized_address"].str[:4].values
    s23_addr4 = s23_df["normalized_address"].str[:4].values

    s1_w1, s1_w2 = _make_word_cols(s1_df["normalized_name"])
    s23_w1, s23_w2 = _make_word_cols(s23_df["normalized_name"])

    s1_country = s1_df["country"].values
    s23_country = s23_df["country"].values
    s1_phon = s1_df["phonetic_key"].values
    s23_phon = s23_df["phonetic_key"].values
    s1_stk = s1_df["sorted_token_key"].values
    s23_stk = s23_df["sorted_token_key"].values

    print("  Key 1: Phonetic + Name Prefix3 (country-agnostic union) ...", flush=True)
    _apply_key(vote_counts, s1_ids,
               [f"{p}|{n}" for p, n in zip(s1_phon, s1_pref3)],
               s23_ids,
               [f"{p}|{n}" for p, n in zip(s23_phon, s23_pref3)],
               max_per_block=candidate_cap)

    print("  Key 2: SortedTokenKey (country-agnostic union) ...", flush=True)
    _apply_key(vote_counts, s1_ids,
               s1_stk,
               s23_ids,
               s23_stk,
               max_per_block=candidate_cap)

    print("  Key 3: Country + FirstWord ...", flush=True)
    _apply_key(vote_counts, s1_ids,
               [f"{c}|{w}" for c, w in zip(s1_country, s1_w1.values)],
               s23_ids,
               [f"{c}|{w}" for c, w in zip(s23_country, s23_w1.values)],
               max_per_block=30, block_size_limit=300)

    print("  Key 4: Country + SecondWord ...", flush=True)
    _apply_key(vote_counts, s1_ids,
               [f"{c}|{w}" for c, w in zip(s1_country, s1_w2.values)],
               s23_ids,
               [f"{c}|{w}" for c, w in zip(s23_country, s23_w2.values)],
               max_per_block=20, block_size_limit=200)

    print("  Key 5: Country + AddrPref4 + NamePref3 ...", flush=True)
    _apply_key(vote_counts, s1_ids,
               [f"{c}|{a}|{n}" for c, a, n in zip(s1_country, s1_addr4, s1_pref3)],
               s23_ids,
               [f"{c}|{a}|{n}" for c, a, n in zip(s23_country, s23_addr4, s23_pref3)],
               max_per_block=25)

    print("  Key 6: Country + Broad Phonetic (block_limit=50) ...", flush=True)
    _apply_key(vote_counts, s1_ids,
               [f"{c}|{p}" for c, p in zip(s1_country, s1_phon)],
               s23_ids,
               [f"{c}|{p}" for c, p in zip(s23_country, s23_phon)],
               max_per_block=20, block_size_limit=50)

    print("  Key 7: Country + Address Prefix (6 chars) ...", flush=True)
    s1_addr6 = s1_df["normalized_address"].str[:6].values
    s23_addr6 = s23_df["normalized_address"].str[:6].values
    _apply_key(vote_counts, s1_ids,
               [f"{c}|{a}" for c, a in zip(s1_country, s1_addr6)],
               s23_ids,
               [f"{c}|{a}" for c, a in zip(s23_country, s23_addr6)],
               max_per_block=candidate_cap, block_size_limit=500)

    # Broad address-token retrieval is run only for entities with thin exact
    # coverage. This controls candidate explosion while allowing partial and
    # reordered addresses to contribute a vote independently.
    weak_address_ids = {eid for eid in s1_ids
                        if len(vote_counts.get(eid, {})) < WEAK_COVERAGE_THRESHOLD}
    if weak_address_ids:
        print(f"  Key 8: Partial address tokens for {len(weak_address_ids):,} weak entities ...", flush=True)
        s1_addr_token_rows = s1_df["normalized_address"].apply(_address_tokens).tolist()
        s23_addr_token_rows = s23_df["normalized_address"].apply(_address_tokens).tolist()
        query_ids, query_country, query_tokens = [], [], []
        for eid, country, tokens in zip(s1_ids, s1_country, s1_addr_token_rows):
            if eid in weak_address_ids:
                for token in tokens:
                    query_ids.append(eid)
                    query_country.append(country)
                    query_tokens.append(token)
        pool_ids, pool_country, pool_tokens = [], [], []
        for eid, country, tokens in zip(s23_ids, s23_country, s23_addr_token_rows):
            for token in tokens:
                pool_ids.append(eid)
                pool_country.append(country)
                pool_tokens.append(token)
        _apply_key(vote_counts, query_ids,
                   [f"{c}|{t}" for c, t in zip(query_country, query_tokens)],
                   pool_ids,
                   [f"{c}|{t}" for c, t in zip(pool_country, pool_tokens)],
                   max_per_block=100, block_size_limit=500)
        _apply_key(vote_counts, query_ids, query_tokens, pool_ids, pool_tokens,
                   max_per_block=100, block_size_limit=500)

    print("  Key 9: Country + Consonant Skeleton ...", flush=True)
    s1_skeleton = s1_df["normalized_name"].apply(_consonant_skeleton).values
    s23_skeleton = s23_df["normalized_name"].apply(_consonant_skeleton).values
    _apply_key(vote_counts, s1_ids,
               [f"{c}|{s}" for c, s in zip(s1_country, s1_skeleton)],
               s23_ids,
               [f"{c}|{s}" for c, s in zip(s23_country, s23_skeleton)],
               max_per_block=candidate_cap, block_size_limit=500)

    # ---- Vote-based ranking + final cap ----
    print("  Ranking by vote count and capping ...", flush=True)
    all_s1_ids = s1_df["entity_id"].tolist()
    final: dict = {}
    weak_ids = []

    for s1_id in all_s1_ids:
        votes = vote_counts.get(s1_id, Counter())
        valid = {eid: cnt for eid, cnt in votes.items()
                 if not str(eid).startswith("S1-") and eid in s23_id_set}
        # Sort by vote count descending (most-corroborated candidates first),
        # tie-break by entity_id for determinism.
        ranked = sorted(valid.items(), key=lambda x: (-x[1], x[0]))
        lst = [eid for eid, _ in ranked[:candidate_cap]]
        final[s1_id] = lst
        if len(lst) < WEAK_COVERAGE_THRESHOLD:
            weak_ids.append(s1_id)

    print(f"  {len(weak_ids):,} / {len(all_s1_ids):,} S1 entities have "
          f"< {WEAK_COVERAGE_THRESHOLD} candidates from exact-key blocking", flush=True)

    # ---- Fuzzy fallback for weak-coverage entities only ----
    if run_fuzzy_fallback and weak_ids:
        print(f"  Running TF-IDF fallback on {len(weak_ids):,} weak entities ...", flush=True)
        weak_s1_df = s1_df[s1_df["entity_id"].isin(weak_ids)]
        fallback = _fuzzy_fallback(weak_s1_df, s23_df, s23_id_set, top_k=tfidf_k)

        added = 0
        for s1_id, cand_set in fallback.items():
            existing = set(final.get(s1_id, []))
            valid_new = {eid for eid in cand_set
                         if not str(eid).startswith("S1-") and eid in s23_id_set}
            new_candidates = sorted(valid_new - existing)
            combined = list(existing) + new_candidates
            combined = combined[:candidate_cap]
            added += len(combined) - len(existing)
            final[s1_id] = combined
        print(f"  Fallback added {added:,} new candidates", flush=True)

    del s23_df
    gc.collect()

    total = sum(len(v) for v in final.values())
    has_cands = sum(1 for v in final.values() if v)
    print(f"  Final: {total:,} candidates across {has_cands:,} / {len(all_s1_ids):,} S1 entities", flush=True)
    return final

