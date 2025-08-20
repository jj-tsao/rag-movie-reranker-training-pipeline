from core.eval_metrics import evaluate_system, summarize
from core.data_loader import load_held_out_sample
from core.bootstrap import setup_reranker, setup_retriever 
from core.rrf_hybrid import hybrid_rrf_search, metadata_rerank_only

media_type = "movie"
retriever = setup_retriever()
tokenizer, reranker = setup_reranker(model_name="JJTsao/movietv-reranker-cross-encoder-base-v1")

ds = load_held_out_sample(3600)

preds_ce_reranker = {}
preds_hybrid_search = {}
preds_metadata ={}

for d in ds:
    ce_reranker_top20, hybrid_search_top20  = hybrid_rrf_search(retriever, reranker, tokenizer, media_type, d['query'])
    metadata_top20  = metadata_rerank_only(retriever, media_type, d['query'])
    
    preds_ce_reranker[d['query']] = [r['title'] for r in ce_reranker_top20]
    preds_hybrid_search[d['query']] = [r['title'] for r in hybrid_search_top20]
    preds_metadata[d['query']] = [r['title'] for r in metadata_top20]

df_ce_reranker = evaluate_system(ds[:3600], preds_ce_reranker)
df_hybrid_search = evaluate_system(ds[:3600], preds_hybrid_search)
df_metadata  = evaluate_system(ds[:3600], preds_metadata)


print ("Neural Reranker RRF - Eval Metrics:")
print(summarize(df_ce_reranker))
print ("Hybrid Search - Eval Metrics:")
print(summarize(df_hybrid_search))
print ("Metadata Reranker Only - Eval Metrics:")
print(summarize(df_metadata))