import unicodedata
import re
import math
import random
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Optional
import pandas as pd

def normalize_title(title: str) -> str:
    if title is None:
        return ""
    s = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"\[[^\]]*\]", " ", s)
    s = s.split(":")[0]
    s = s.split(" - ")[0]
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"^\s*(the|a|an)\s+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def fuzzy_match(a: str, b: str, threshold: float = 0.90) -> bool:
    a_norm = normalize_title(a)
    b_norm = normalize_title(b)
    if not a_norm or not b_norm:
        return False
    ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
    a_tok = " ".join(sorted(a_norm.split()))
    b_tok = " ".join(sorted(b_norm.split()))
    ratio_tok = SequenceMatcher(None, a_tok, b_tok).ratio()
    return max(ratio, ratio_tok) >= threshold

def find_positive_rank(ranked_titles: List[str], positive_title: str, fuzzy_threshold: float = 0.90) -> Tuple[Optional[int], Optional[str]]:
    pos_norm = normalize_title(positive_title)
    for i, t in enumerate(ranked_titles):
        if normalize_title(t) == pos_norm:
            return i, t
    for i, t in enumerate(ranked_titles):
        if fuzzy_match(t, positive_title, threshold=fuzzy_threshold):
            return i, t
    return None, None

def precision_at_k(rank_idx: Optional[int], k: int) -> float:
    if k <= 0 or rank_idx is None:
        return 0.0
    return 1.0 / k if rank_idx < k else 0.0

def recall_at_k(rank_idx: Optional[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return 1.0 if (rank_idx is not None and rank_idx < k) else 0.0

def accuracy_at_k(rank_idx: Optional[int], k: int) -> float:
    return recall_at_k(rank_idx, k)

def mrr(rank_idx: Optional[int]) -> float:
    return 1.0 / (rank_idx + 1) if rank_idx is not None else 0.0

def ndcg_at_k(rank_idx: Optional[int], k: int) -> float:
    if rank_idx is None or rank_idx >= k:
        return 0.0
    return 1.0 / math.log2((rank_idx + 1) + 1)

def evaluate_query_titles(ranked_titles: List[str], positive_title: str, k_list=(5,10,20), fuzzy_threshold=0.90) -> Dict[str, float]:
    idx, _ = find_positive_rank(ranked_titles, positive_title, fuzzy_threshold=fuzzy_threshold)
    out = {"MRR": mrr(idx), "Rank": (idx + 1) if idx is not None else None}
    for k in k_list:
        out[f"Precision@{k}"] = precision_at_k(idx, k)
        out[f"Recall@{k}"] = recall_at_k(idx, k)
        out[f"Accuracy@{k}"] = accuracy_at_k(idx, k)
        out[f"NDCG@{k}"] = ndcg_at_k(idx, k)
    return out

def evaluate_system(heldout: List[Dict[str,str]], predictions: Dict[str, List[str]], k_list=(5,10,20), fuzzy_threshold=0.90) -> pd.DataFrame:
    rows = []
    for ex in heldout:
        q = ex["query"]
        pos = ex["positive"]
        rank = predictions.get(q, [])
        metrics = evaluate_query_titles(rank, pos, k_list=k_list, fuzzy_threshold=fuzzy_threshold)
        metrics["query"] = q
        metrics["positive"] = pos
        rows.append(metrics)
    return pd.DataFrame(rows)

def summarize(df: pd.DataFrame) -> pd.Series:
    return df.drop(columns=["query","positive","Rank"]).mean(numeric_only=True)

def paired_bootstrap(dfA: pd.DataFrame, dfB: pd.DataFrame, metric: str, iters: int = 2000, seed: int = 13) -> Dict[str, float]:
    rng = random.Random(seed)
    merged = pd.merge(dfA[["query", metric]], dfB[["query", metric]], on="query", suffixes=("_A","_B"))
    qids = merged["query"].tolist()
    diffs = []
    for _ in range(iters):
        sample = [rng.choice(qids) for _ in qids]
        sm = merged.set_index("query").loc[sample]
        diffs.append((sm[f"{metric}_A"] - sm[f"{metric}_B"]).mean())
    diffs.sort()
    mean_diff = sum(diffs)/len(diffs)
    lo = diffs[int(0.025*len(diffs))]
    hi = diffs[int(0.975*len(diffs))]
    return {"mean_diff": mean_diff, "ci_lo": lo, "ci_hi": hi}