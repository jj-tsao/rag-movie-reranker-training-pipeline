import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")
QDRANT_MOVIE_COLLECTION_NAME = "Movies_BGE_June"
QDRANT_TV_COLLECTION_NAME = "TV_Shows_BGE_June"

NLTK_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "nltk_data"
BM25_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "bm25_files"

EMBEDDING_MODEL = "JJTsao/fine-tuned_movie_retriever-bge-base-en-v1.5"

SAMPLE_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "reranker_train_36k_weighted-negative_0806.jsonl"
VAL_IDX_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "val_indices.pt"

def get_reranker_path(model_name: str) -> str:
    reranker_dir = (
        Path(__file__).resolve().parent.parent.parent / "trained_models"
    )
    return os.path.join(reranker_dir, f"{model_name}.pt")