import os
import time

import nltk
import torch
from src.config import (
    NLTK_PATH,
    QDRANT_API_KEY,
    QDRANT_ENDPOINT,
    QDRANT_MOVIE_COLLECTION_NAME,
    QDRANT_TV_COLLECTION_NAME,
    get_reranker_path,
)
from app.llm.custom_models import (
    load_bm25_files,
    load_sentence_model,
)
from src.media_retriever import MediaRetriever
from app.retrieval.vectorstore import connect_qdrant
from model.reranker_model import RerankerModel

start = time.time()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_retriever(semantic_retrieval_limit, bm25_retrieval_limit, top_k=20):
    embed_model = load_sentence_model()
    bm25_models, bm25_vocabs = load_bm25_files()
    nltk.data.path.append(str(NLTK_PATH))
    print("✅ NLTK resources loaded")

    qdrant_client = connect_qdrant(endpoint=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)

    return MediaRetriever(
        embed_model=embed_model,
        qdrant_client=qdrant_client,
        bm25_models=bm25_models,
        bm25_vocabs=bm25_vocabs,
        movie_collection_name=QDRANT_MOVIE_COLLECTION_NAME,
        tv_collection_name=QDRANT_TV_COLLECTION_NAME,
        semantic_retrieval_limit=semantic_retrieval_limit,
        bm25_retrieval_limit=bm25_retrieval_limit,
        top_k=top_k
    )

def setup_reranker(model_name:str):
    reranker_path = get_reranker_path(model_name)
    model = RerankerModel()
    model.load_state_dict(torch.load(reranker_path, map_location="cpu"))
    model.eval()
    print (f"Reranker model '{model_name}' loaded")
    return model    