import random
from collections import defaultdict
from typing import List

from core.config import QDRANT_MOVIE_COLLECTION_NAME, QDRANT_TV_COLLECTION_NAME
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer


class NegativeSampler:
    def __init__(
        self, 
        embd_model: SentenceTransformer,
        qdrant_client: QdrantClient,
        retrieval_limit: int = 50,
    ):
        self.usage_counter = defaultdict(int)
        self.embd_model = embd_model
        self.qdrant_client = qdrant_client
        self.retrieval_limit = retrieval_limit
        
    def sample_negative(self, media_type: str, query: str, positive_media: dict):
        query_embd = self._embed_dense(query)
        pos_genres = self._extract_genres(positive_media)
        pos_keywords = self._extract_keywords(positive_media)
        retrieved_media = self._query_dense(query_embd, media_type)

        if not retrieved_media.points:
            self.usage_counter["no_retrieval"] += 1
            return None, "no_retrieval"

        # Filter out the positive from candidates
        candidates = [
            s for s in retrieved_media.points
            if s.payload['media_id'] != positive_media.get('id')
        ]

        # Precompute tiers
        tiered = {
            "hard_neg": [],
            "keyword_only_neg": [],
            "genre_only_neg": [],
            "semantic_only_neg": []
        }

        for s in candidates:
            kw_overlap = self._has_overlap(pos_keywords, s.payload.get("keywords", []))
            genre_overlap = self._has_overlap(pos_genres, s.payload.get("genres", []))

            if kw_overlap and genre_overlap:
                tiered["hard_neg"].append(s)
            elif kw_overlap:
                tiered["keyword_only_neg"].append(s)
            elif genre_overlap:
                tiered["genre_only_neg"].append(s)
            else:
                tiered["semantic_only_neg"].append(s)

        # Weighted tier selection
        tier_order = random.choices(
            population=["genre_only_neg", "keyword_only_neg", "semantic_only_neg", "hard_neg"],
            weights=[0.10, 0.30, 0.05, 0.55],  # Adjust this as needed
            k=1
        )[0]

        # Try selected tier, then fall back in order
        tier_priority = [tier_order, "hard_neg", "keyword_only_neg", "genre_only_neg", "semantic_only_neg"]
        for tier in tier_priority:
            if tiered[tier]:
                self.usage_counter[tier] += 1
                return random.choice(tiered[tier]).payload['embedding_text'], tier

        # Last-resort fallback
        self.usage_counter["no_valid_neg"] += 1
        return None, "no_valid_neg"
    
    def _embed_dense(self, query: str) -> List[float]:
        return self.embd_model.encode(query).tolist()
    
    def _query_dense(self, vector, media_type):
        collection = (
            QDRANT_MOVIE_COLLECTION_NAME
            if media_type == "movie"
            else QDRANT_TV_COLLECTION_NAME
        )
        return self.qdrant_client.query_points(
            collection_name=collection,
            query=vector,
            using="dense_vector",
            limit=self.retrieval_limit,
            with_payload=["media_id", "embedding_text", "title", "genres", "keywords"],
            with_vectors=False,
        )
        
    def _extract_genres(self, media_detail):
        return [m['name'] for m in media_detail.get('genres', [])]
    
    def _extract_keywords(self, media_detail):
        return [k for k in media_detail.get('keywords', [])]   

    def _has_overlap(self, a, b):
        return bool(a and b and set(a) & set(b))


    def _build_filter(self, genres=None) -> Filter | None:
            must_clauses = []

            if genres:
                genre_conditions = [
                    FieldCondition(key="genres", match=MatchValue(value=genre))
                    for genre in genres
                ]
                must_clauses.append({"should": genre_conditions})

            return Filter(must=must_clauses) if must_clauses else None

    def print_usage_summary(self):
        print("Negative Sampling Distribution:")
        for tier, count in sorted(self.usage_counter.items()):
            print(f"  - {tier}: {count}")

    
    