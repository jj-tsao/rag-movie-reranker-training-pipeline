from __future__ import annotations
from typing import Dict, List, Tuple
import torch

# RRF core
def rrf_fuse(rankings: List[List[int]], k: int = 60) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = {}
    for rank in rankings:
        for r, _id in enumerate(rank):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + r + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# Utilities for neural scoring
def _format_item_text(payload: Dict) -> str:
    """Build the item text for the reranker; prefer 'llm_context', fall back to title."""
    ctx = payload.get("embedding_text") or payload.get("llm_context") or ""
    return ctx

@torch.no_grad()
def score_neural_batch(model, tokenizer, query: str, items: List[Dict], device: str = "cuda", max_length: int = 512) -> List[float]:
    """Return neural scores aligned with items (list of payload dicts)."""
    if not items:
        return []
    texts_a = [query] * len(items)
    texts_b = [_format_item_text(p) for p in items]

    enc = tokenizer(texts_a, texts_b, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    model = model.to(device).eval()
    scores = model(**enc).detach().float().cpu().tolist()
    return scores


# Main: Hybrid RRF search
def hybrid_rrf_search(
    retriever,                   
    reranker_model,              
    tokenizer,                   
    query: str,
    media_type: str,
    top_k: int = 20,
    candidate_pool_dense: int = 300,  # dense depth
    candidate_pool_sparse: int = 20,  # sparse depth (BM25)
    rrf_k: int = 60,                  # RRF constant
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[List[Dict], List[Dict]]:
    """
    Pipeline:
    1) Base fusion (RRF) with Dense depth=300, Sparse depth=20, RRF k=60 -> take top-60.
    2) Metadata rerank base fusion top-60 -> keep the top-30 list (Hybrid top-30).
    3) Neural rerank the top-30 *dense* (from step dense list in step 1) -> then metadata-rerank that same 30 -> (Neural top-30).
    4) Final fusion: RRF(Hybrid top-30 ids, Neural top-30 ids) -> return top_k results with debug fields.
    """
    # --- Stage 0: embed query
    dense_vec = retriever.embed_dense(query)
    sparse_vec = retriever.embed_sparse(query, media_type)

    # --- Stage 1: retrieve raw dense/sparse with specific depths
    qf = retriever._build_filter()
    dense_results = retriever._query_dense(dense_vec, media_type, qf) # 300 dense results
    sparse_results = retriever._query_sparse(sparse_vec, media_type, qf) # 20 sparse results

    # Build pure rank lists (by engine scores, desc). Qdrant returns .points with .score and .id
    dense_rank_ids = [int(p.id) for p in dense_results.points]
    sparse_rank_ids = [int(p.id) for p in sparse_results.points]

    # Build a fused dict with normalized dense/sparse (appending, not ranked)
    fused_all: Dict[int, Dict] = retriever.fuse_dense_sparse(dense_results, sparse_results)  # Fused 300 dense + 20 sparse {id: {'point','dense_score','sparse_score'}}

    # --- Stage 2: Base RRF fusion over positions (not scores), then take top-60
    base_rrf = rrf_fuse([dense_rank_ids, sparse_rank_ids], k=rrf_k)
    base_ids_ranked = [i for i, _ in base_rrf]
    base_top_ids = base_ids_ranked[:60]

    # Limit 'fused_' to the base set
    fused_base = {i: fused_all[i] for i in base_top_ids if i in fused_all}

    if not fused_base:
        return [], []

    # --- Stage 3: Metadata rerank the base-60 and keep top-30 (List A)
    meta_ranked_points_A, scored_lookup_A = retriever.rerank_fused_results(fused_base)
    meta_ids_A = [int(p.id) for p in meta_ranked_points_A[:30]]

    # --- Stage 4: Neural path — take top-30 *dense*, neural rerank, then metadata sort that 30 (List B)
    dense_top30_points = dense_results.points[:30]
    dense_top30_ids = [int(p.id) for p in dense_top30_points]
    dense_top30_doc = [p.payload for p in dense_top30_points]

    neural_scores = score_neural_batch(reranker_model, tokenizer, query, dense_top30_doc, device=device)
    id2neural = {pid: float(s) for pid, s in zip(dense_top30_ids, neural_scores)}

    # Sort by neural descending, keep the same 30
    dense_top30_sorted_by_neural = sorted(dense_top30_points, key=lambda p: id2neural.get(int(p.id), -1e9), reverse=True)
    neural_ids_sorted = [int(p.id) for p in dense_top30_sorted_by_neural]

    # Build a fused dict for those 30 (so that metadata scorer can run consistently)
    fused_neu_subset = {i: fused_all[i] for i in neural_ids_sorted if i in fused_all}
    meta_ranked_points_B, scored_lookup_B = retriever.rerank_fused_results(fused_neu_subset)
    meta_ids_B = [int(p.id) for p in meta_ranked_points_B[:30]]

    # --- Stage 5: Final RRF fuse the two lists (A and B), return top_k
    final_rrf = rrf_fuse([meta_ids_A, meta_ids_B], k=rrf_k)
    final_ids = [i for i, _ in final_rrf][:top_k]

    # For pretty/debug output we want: metadata_score (from A’s lookup if available, else B),
    # neural_score (from id2neural if available, else None)
    # Use fused_all to access the PointStruct and payload.
    out: List[Dict] = []
    for rank, pid in enumerate(final_ids, start=1):
        point = fused_all[pid]["point"]
        payload = point.payload or {}
        meta_score = None
        if pid in scored_lookup_A:
            meta_score = scored_lookup_A[pid]["reranked_score"]
        elif pid in scored_lookup_B:
            meta_score = scored_lookup_B[pid]["reranked_score"]

        out.append({
            "rank": rank,
            "id": pid,
            "title": payload.get("title"),
            "vote_average": payload.get("vote_average"),
            "popularity": payload.get("popularity"),
            "metadata_score": float(meta_score) if meta_score is not None else None,
            "neural_score": id2neural.get(pid),
            "llm_context": payload.get("llm_context"),
        })

    # Also return a metadata-only list for inspection (top-20 from meta_ranked_points_A)
    meta_out: List[Dict] = []
    for idx, pt in enumerate(meta_ranked_points_A[:20], start=1):
        pid = int(pt.id)
        payload = pt.payload or {}
        scored = scored_lookup_A.get(pid, {})
        meta_out.append({
            "rank": idx,
            "id": pid,
            "title": payload.get("title"),
            "vote_average": payload.get("vote_average"),
            "popularity": payload.get("popularity"),
            # Match your previous print keys (keep the 'desne_score' typo for compatibility)
            "metadata_score": float(scored.get("reranked_score", 0.0)),
            "desne_score": float(scored.get("dense_score", 0.0)),
            "sparse_score": float(scored.get("sparse_score", 0.0)),
            "llm_context": payload.get("llm_context"),
        })

    return out, meta_out


# ----------------------------
# Simple helper for metadata-only benchmark
# ----------------------------
def metadata_rerank_only(retriever, media_type: str, query: str) -> List[Dict]:
    dense_vec = retriever.embed_dense(query)
    sparse_vec = retriever.embed_sparse(query, media_type)
    reranked_points, scored_lookup = retriever.retrieve_and_rerank(dense_vec, sparse_vec, media_type.lower())
    out: List[Dict] = []
    for idx, pt in enumerate(reranked_points[:20], start=1):
        pid = int(pt.id)
        payload = pt.payload or {}
        scored = scored_lookup.get(pid, {})
        out.append({
            "rank": idx,
            "id": pid,
            "title": payload.get("title"),
            "vote_average": payload.get("vote_average"),
            "popularity": payload.get("popularity"),
            "metadata_score": float(scored.get("reranked_score", 0.0)),
            "desne_score": float(scored.get("dense_score", 0.0)),  # keep key name for compatibility
            "sparse_score": float(scored.get("sparse_score", 0.0)),
            "llm_context": payload.get("llm_context"),
        })
    return out
