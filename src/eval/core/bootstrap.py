import os
import time

import nltk
import torch
from model.reranker_model import RerankerModel
from core.config import (
    NLTK_PATH,
    QDRANT_API_KEY,
    QDRANT_ENDPOINT,
    QDRANT_MOVIE_COLLECTION_NAME,
    QDRANT_TV_COLLECTION_NAME,
    get_reranker_path,
)
from core.custom_models import (
    load_bm25_files,
    load_sentence_model,
)
from core.media_retriever import MediaRetriever
from core.vectorstore import connect_qdrant
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

start = time.time()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_retriever(semantic_retrieval_limit:int =300, bm25_retrieval_limit:int =20):
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
        bm25_retrieval_limit=bm25_retrieval_limit
    )

def setup_reranker(
    model_name: str,
    revision: str | None = None,
    dtype: torch.dtype | None = None,
):
    """
    Loads the reranker from the Hugging Face Hub.
    - trust_remote_code=True is required (custom class RerankerModelHF).
    - Optionally pin a specific `revision` (commit hash or tag) for reproducibility.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # (Optional) inspect config to fail fast if auto_map is missing
    cfg = AutoConfig.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    am = getattr(cfg, "auto_map", {})
    assert "AutoModelForSequenceClassification" in am, "auto_map missing for custom model"

    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    # sanity check: ensure we didn't fall back to stock BertForSequenceClassification
    assert type(model).__name__ == "RerankerModelHF", f"Loaded {type(model).__name__}, expected RerankerModelHF"

    print(f"✅ Reranker loaded from HF: {model_name} ({type(model).__name__}) on {device}")
    return tok, model


def setup_reranker_local(model_name:str):
    reranker_path = get_reranker_path(model_name)
    model = RerankerModel()
    model.load_state_dict(torch.load(reranker_path, map_location="cpu"))
    model.eval()
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    print (f"Reranker model '{model_name}' loaded")
    return tok, model    