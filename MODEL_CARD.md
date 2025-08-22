---
language: en
tags:
  - reranker
  - cross-encoder
  - information-retrieval
  - recommendations
  - movies
  - sentence-transformers
license: other
datasets:
  - reelix/reranker-synthetic-vibes  # placeholder if you publish
library_name: transformers
pipeline_tag: text-classification
---

# 🎬 Reelix Cross-Encoder Reranker (Movies & TV)

A BERT-based **cross-encoder** that scores `(query, title_context)` pairs to re-rank candidates for **vibe-driven** movie/TV recommendations.


## 🧠 Model Architecture

- **Backbone:** `bert-base-uncased`
- **Input packing:** `[CLS] {query} {title_context} `
  - `title_context` is a concatenation of: **Title | Genres | Overview | Tagline | Director | Cast | Keywords | Year**
- **Scoring head (2-layer MLP):**
  - `Linear(hidden → inter)`
  - `GELU`
  - **Residual** connection to the CLS-pooled representation
  - `LayerNorm`
  - `Dropout(p=0.1)`
  - `Linear(inter → 1)` → scalar relevance logit
- **Output:** Higher score ⇒ stronger match

**Intended use:** Re-rank the **top-N** items surfaced by a separate hybrid retrieval system (dense + BM25).  
**Out of scope:** Standalone retrieval over large corpora (use a bi-encoder); general classification tasks without adaptation.

---

## 📚 Training Data

The model was trained on balanced **triplets** `(query, positive, negative)` that mirror real retrieval noise patterns.

- **Queries**
  - LLM-generated **vibe** prompts (e.g., “Emotionally powerful space exploration film with themes of love and sacrifice.”)
  - Template-driven metadata prompts (e.g., “Any crime movies from the 1990s directed by Quentin Tarantino about heists?”)

- **Positives**
  - The source title for the query.
  - Fields provided to the model: **title, genres, overview, tagline, director, cast, keywords, year**.

- **Negatives** (weighted hard negatives from dense neighbors; positive excluded)
  - **Hard:** same genre **and** keyword overlap (forces fine-grained discrimination)
  - **Mid (A):** same genre, no keyword overlap (prevents overfitting to genre)
  - **Mid (B):** keyword overlap, different genre (prevents keyword bias)
  - **Easy:** semantically nearer but clearly off (stabilizes margin learning)

---

## 🏋️ Training Procedure

- **Objective:** Pairwise **margin ranking loss**

  $$
  L = \max\bigl(0,\, m - (s_\text{pos} - s_\text{neg})\bigr),\quad m=1.0
  $$
  
- **Batch:** 16 triplets (Q, Pos, Neg)
- **Max length:** 512
- **Epochs:** 3 (early stop on dev loss / ranking metrics)
- **Optimizer:** `AdamW`
  - `lr=2e-5`, weight decay `0.01`
  - Exempt bias/LayerNorm from weight decay
- **Scheduler:** Linear decay with **10% warmup**
- **Gradient clipping:** `max_norm=1.0`
- **Seed:** Fixed (for `torch` and `random`)

---

## 🧪 Evaluation

Evaluation was perfomed on held-out `(query, positive_title)` pairs using **normalized title matching**. Metrics:

- **MRR** — Mean Reciprocal Rank of the first relevant item
- **Precision@k** — with a single positive, `1/k` if positive appears in top-k; else `0`
- **Recall@k / Accuracy@k** — identical for single-positive; `1` if positive appears in top-k; else `0`
- **NDCG@k** — discounts gains by rank; rewards early hits

### Pipelines Compared

- **Reranker**: Cross-Encoder reranker **+** metadata features with **RRF** fusion  
- **Baseline**: Metadata-only reranking (no cross-encoder)

### Results

The cross-encoder lifts early ranking quality (MRR, NDCG@k) and improves inclusion at k=5/10/20, which translates to cleaner top-20 lists for downstream LLM write-ups.

| Metric       | Reranker | Baseline  |   Δ (Abs) | Δ (Rel) |
| ------------ | -------: | --------: | --------: | ------: |
| MRR          | 0.554752 |  0.365887 | +0.188865 | **+51.6%** |
| Precision@5  | 0.129222 |  0.111722 | +0.017500 |  +15.7% |
| Recall@5     | 0.646111 |  0.558611 | +0.087500 |  +15.7% |
| NDCG@5       | 0.570416 |  0.403535 | +0.166881 | **+41.3%** |
| Precision@10 | 0.069250 |  0.063222 | +0.006028 |   +9.5% |
| Recall@10    | 0.692500 |  0.632222 | +0.060278 |   +9.5% |
| NDCG@10      | 0.585627 |  0.427452 | +0.158175 | **+37.0%** |
| Precision@20 | 0.037111 |  0.034944 | +0.002167 |   +6.2% |
| Recall@20    | 0.742222 |  0.698889 | +0.043333 |   +6.2% |
| NDCG@20      | 0.598061 |  0.444327 | +0.153734 | **+34.6%** |


---

### Thematic Noise Ratio (TNR) — Human-in-the-loop Quality Check

#### What:
We rate the **on-briefness** of the top-k results using a simple rubric:  
`1 = highly relevant`, `0.5 = borderline`, `0 = not relevant`.  
**RS (Relevance Score)** is the mean label; **TNR = 1 − RS** (lower is better).

#### How:
For each query, a human labels top-k (k∈{10,20}) items for:
- **Reranker** (cross-encoder + metadata RRF)
- **Baseline** (metadata-only)

#### Results:

Reranker reduces thematic noise, especially in **Top-10**, producing a stronger prompt substrate for the LLM.

| Metric     | Reranker | Baseline | Δ (Abs)  | Δ (Rel)     | ↑/↓ Better |
|------------|---------:|---------:|---------:|------------:|:----------:|
| RS@10      | 0.806    | 0.612    | +0.194   | **+31.7%**  | ↑          |
| TNR@10     | 0.194    | 0.388    | -0.194   | **−50.0%**  | ↓          |
| RS@20      | 0.731    | 0.669    | +0.062   |   +9.3%     | ↑          |
| TNR@20     | 0.269    | 0.331    | -0.062   |  −18.7%     | ↓          |
| Count_1    | 11.625   | 9.375    | +2.250   |  +24.0%     | ↑          |
| Count_0.5  | 6.000    | 8.000    | -2.000   |  −25.0%     | ↓          |
| Count_0    | 2.375    | 2.625    | -0.250   |   −9.5%     | ↓          |

#### Per-intent Highlights (RS ↑)

- **Mind-bending sci-fi:** 0.95 @10 vs 0.75; 0.90 @20 vs 0.80  
- **Atmospheric folk/psych horror:** 0.80 @10 vs 0.30; 0.725 @20 vs 0.475  
- **Musical dramas (visually lush):** 0.90 @10 vs 0.70; 0.875 @20 vs 0.775  
- **Slow-burn crime (gritty):** 0.85 @10 vs 0.70; parity 0.65 @20  
- **Psych thrillers (satirical):** 0.70 @10 vs 0.65; 0.70 @20 vs 0.625  
- **Coming-of-age (heartwarming):** 0.90 @10 vs 0.75; 0.90 @20 vs 0.825  
- **Offbeat indie comedies:** 0.70 @10 vs 0.60; slight drop 0.575 @20 vs 0.60 → add indie/major-studio gates  
- **Playful rom-coms:** 0.65 @10 vs 0.45; 0.525 @20 vs 0.60 → enforce Romance|Comedy and down-weight heavy drama

**Guardrails informed by TNR**
- **Intent-aware gates** (genre/format constraints; exclude TV specials/game tie-ins)
- **Franchise dedupe** (limit sequels)
- Final fusion tuning (RRF `k≈20–30` or **Weighted-RRF** after score calibration)

---

## 💻 Usage

If exported as `AutoModelForSequenceClassification` (`num_labels=1`):

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

mname = "JJTsao/movietv-reranker-cross-encoder-base-v1"
tok = AutoTokenizer.from_pretrained(mname)
model = AutoModelForSequenceClassification.from_pretrained(mname, trust_remote_code=True)
model.eval()

def score(query: str, context: str, max_len=320):
    inputs = tok(query, context, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    return float(out.logits.squeeze(-1))
```

> If you keep a custom head, include `auto_map` in the config or provide a wrapper class in the Hub repo.

---

## 📄 License

MIT

---

## 📚 Citation

```
@software{reelix_reranker_2025,
  title  = {Reelix Cross-Encoder Reranker},
  author = {JJ Tsao},
  year   = {2025},
  url    = {https://huggingface.co/JJTsao/movietv-reranker-cross-encoder-base-v1}
}
```