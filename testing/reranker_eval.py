from src.bootstrap import setup_reranker, setup_retriever
from src.rrf_hybrid import hybrid_rrf_search, metadata_rerank_only
from transformers import AutoTokenizer

media_type = "movie"
retriever = setup_retriever(semantic_retrieval_limit=300, bm25_retrieval_limit=20)
reranker = setup_reranker("reranker_final_weighted-negative")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

query="Mind-bending sci-fi with philosophical undertones and existential stakes."

final_top20, metadata_top20 = hybrid_rrf_search(retriever, reranker, tokenizer, query, media_type)

print (f"Query: {query}\n")

print ("Final Reranked Top20")
for r in final_top20:
    print(f"#{r['rank']} {r['title']}  "
        #   f"[metadata_score={r['metadata_score']:.3f}  neural_score={r['neural_score']:.3f}  "
          f"rating={r['vote_average']}  pop={r['popularity']}]")

print ("\nMetadata Reranked Top20 (with RFF & nearul)")
for r in metadata_top20:
    print(f"#{r['rank']} {r['title']}  "
        #   f"[metadata_score={r['metadata_score']:.3f}  desne_score={r['desne_score']:.3f}  sparse_score={r['sparse_score']:.3f}. "
          f"rating={r['vote_average']}  pop={r['popularity']}]")

metadata_only = metadata_rerank_only(retriever, media_type, query)

print ("\n Metadata-only Reranked Top20 (without RFF & nearul)")
for r in metadata_only:
    print(f"#{r['rank']} {r['title']}  "
        #   f"[metadata_score={r['metadata_score']:.3f}  desne_score={r['desne_score']:.3f}  sparse_score={r['sparse_score']:.3f}. "
          f"rating={r['vote_average']}  pop={r['popularity']}]")
