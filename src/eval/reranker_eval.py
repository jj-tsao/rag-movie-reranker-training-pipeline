from core.bootstrap import setup_retriever, setup_reranker
from core.rrf_hybrid import hybrid_rrf_search, metadata_rerank_only

media_type = "movie"
retriever = setup_retriever(semantic_retrieval_limit=100)
tokenizer, reranker = setup_reranker(model_name="JJTsao/movietv-reranker-cross-encoder-base-v1")

QeuryExamples = [
    "Mind-bending sci-fi with philosophical undertones and existential stakes",
    "Psychological thrillers that are character-driven, satirical, and thought-provoking",
    "Heartwarming coming-of-age stories that explore friendship, growth, and family bonds",
    "Offbeat indie comedies with quirky charm and emotional depth",
    "Slow-burn crime dramas with a dark, gritty atmosphere and morally gray characters",
    "Playful rom-coms with quirky characters, heartfelt moments, and a touch of melancholic realism",
    "Visually lush musical dramas that blend artistic ambition with emotional resonance",
    "Atmospheric horror that relies on psychological tension, folklore, and slow-building dread rather than gore",
  ]

for query in QeuryExamples:

  final_top20, metadata_top20 = hybrid_rrf_search(retriever, reranker, tokenizer, media_type, query)

  print (f"Query: {query}\n")

  print ("Cross-Encoder RRF Reranked Top20")
  for r in final_top20:
      print(f"#{r['rank']} {r['title']}  "
          #   f"[metadata_score={r['metadata_score']:.3f}  neural_score={r['neural_score']:.3f}  "
            f"rating={r['vote_average']}  pop={r['popularity']}]")

  print ("\nDense/Sparce Hybrid Search Top20")
  for r in metadata_top20:
      print(f"#{r['rank']} {r['title']}  "
          #   f"[metadata_score={r['metadata_score']:.3f}  neural_score={r['neural_score']:.3f}  "
            f"rating={r['vote_average']}  pop={r['popularity']}]")

  metadata_only = metadata_rerank_only(retriever, media_type, query)

  print ("\nMetadata-only Reranked Top20")
  for r in metadata_only:
      print(f"#{r['rank']} {r['title']}  "
          #   f"[metadata_score={r['metadata_score']:.3f}  desne_score={r['desne_score']:.3f}  sparse_score={r['sparse_score']:.3f}. "
            f"rating={r['vote_average']}  pop={r['popularity']}]")

  print ("\n=====\n")
  
  

