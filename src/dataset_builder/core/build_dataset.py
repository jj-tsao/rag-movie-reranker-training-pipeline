import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List
import torch

from core.config import get_cache_path, RETRIEVER_MODEL, QDRANT_API_KEY, QDRANT_ENDPOINT
from core.format_utils import format_training_text
from core.generate_queries import generate_metadata_query, generate_vibe_query_gpt
from core.negative_sampler_weighted import NegativeSampler
from sentence_transformers import SentenceTransformer
from core.vectorstore import connect_qdrant
from tqdm import tqdm
from core.vibe_cache import VibeQueryCache

embd_model = SentenceTransformer(RETRIEVER_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
qdrant_client = connect_qdrant(endpoint=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)

async def generate_dataset_jsonl(
    media_raw: List[Dict],
    media_type: str,
    output_path: str,
    num_metadata_queries: int = 3,
    num_vibe_queries: int = 3,
    max_workers:int =10,
):
    print(f"🧠 Building dataset with {len(media_raw)*6} samples...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Initiate cache for vibe queries
    cache_path = get_cache_path(media_type)
    cache = VibeQueryCache(cache_path)

    # Generate vibe-based queries by calling LLM api (or using cached queries) asynchronously 
    vibe_query_results = {}
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            media_data.get("id"): loop.run_in_executor(executor, generate_vibe_query_gpt, cache, media_type, dict(media_data), num_vibe_queries)
            for media_data in media_raw
        }

        for media_id, coro in tqdm(futures.items(), desc="🤖 Generating vibe queries", ncols=80):
            result = await coro
            vibe_query_results[media_id] = result

    with open(output_path, "w", encoding="utf-8") as f_out:
        sampler = NegativeSampler(embd_model, qdrant_client)
        for idx, media_data in enumerate(tqdm(media_raw, desc="💾 Writing dataset", ncols=80)):

            # Generate positive docuemnt
            document = format_training_text(media_type, media_data)
            
            # Generate metadata-based queries
            metadata_queries = generate_metadata_query(media_type, media_data, num_queries=num_metadata_queries)

            # Get vibe queries using media_id
            media_id = media_data.get("id")
            vibe_queries = vibe_query_results.get(media_id, [])

            # Write to JSONL with a query type tag
            for query in metadata_queries:
                negative_doc, neg_tier = sampler.sample_negative(media_type, query, media_data)
                f_out.write(json.dumps({
                    "query": query,
                    "positive": document,
                    "negative": negative_doc,
                    "negative_tier": neg_tier,
                    "source": "metadata"
                }) + "\n")
            for query in vibe_queries:
                negative_doc, neg_tier = sampler.sample_negative(media_type, query, media_data)
                f_out.write(json.dumps({
                    "query": query,
                    "positive": document,
                    "negative": negative_doc,
                    "negative_tier": neg_tier,
                    "source": "semantic"
                }) + "\n")                
        sampler.print_usage_summary()

    print(f"✅ Dataset with {len(media_raw)*6} samples written to {output_path}")