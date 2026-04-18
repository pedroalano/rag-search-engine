# RAG Search Engine

A Retrieval-Augmented Generation search engine for a movie dataset. Built for Hoopla, a movie streaming service. Combines keyword (BM25), semantic, hybrid, and multimodal search with LLM-powered answer generation via the Gemini API.

## Setup

**Requirements:** Python 3.14+, [uv](https://docs.astral.sh/uv/)

```bash
uv sync
```

Create a `.env` file with your Gemini API key:

```
GEMINI_API_KEY=your_key_here
```

## CLI Scripts

All scripts are in the `cli/` directory. Run with `uv run cli/<script>.py <command>`.

### Keyword Search (`keyword_search_cli.py`)

BM25-based keyword search over the movie dataset.

```bash
uv run cli/keyword_search_cli.py build                    # Build inverted index
uv run cli/keyword_search_cli.py search "dinosaur"        # Search movies
uv run cli/keyword_search_cli.py bm25search "dinosaur" --limit 10
uv run cli/keyword_search_cli.py tf 1 "dinosaur"          # Term frequency
uv run cli/keyword_search_cli.py idf "dinosaur"           # Inverse document frequency
uv run cli/keyword_search_cli.py tfidf 1 "dinosaur"       # TF-IDF score
uv run cli/keyword_search_cli.py bm25idf "dinosaur"       # BM25 IDF
uv run cli/keyword_search_cli.py bm25tf 1 "dinosaur"      # BM25 TF
```

### Semantic Search (`semantic_search_cli.py`)

Embedding-based search using `all-MiniLM-L6-v2` with sentence chunking.

```bash
uv run cli/semantic_search_cli.py verify                  # Verify model loads
uv run cli/semantic_search_cli.py verify_embeddings       # Build/verify embeddings cache
uv run cli/semantic_search_cli.py embed_text "some text"  # Embed text
uv run cli/semantic_search_cli.py search "romantic comedy" --limit 5
uv run cli/semantic_search_cli.py search_chunked "romantic comedy" --limit 5
uv run cli/semantic_search_cli.py semantic_chunk "Long text here..." --max-chunk-size 4
```

### Hybrid Search (`hybrid_search_cli.py`)

Combines BM25 and semantic search with weighted or Reciprocal Rank Fusion (RRF).

```bash
uv run cli/hybrid_search_cli.py weighted-search "dinosaur movies" --alpha 0.5 --limit 5
uv run cli/hybrid_search_cli.py rrf-search "dinosaur movies" -k 60 --limit 5
uv run cli/hybrid_search_cli.py rrf-search "dinsoar" --enhance spell         # Spell correction
uv run cli/hybrid_search_cli.py rrf-search "action" --enhance rewrite        # Query rewriting
uv run cli/hybrid_search_cli.py rrf-search "action" --enhance expand         # Query expansion
uv run cli/hybrid_search_cli.py rrf-search "action" --rerank-method cross_encoder  # Re-ranking
uv run cli/hybrid_search_cli.py rrf-search "action" --debug                  # Debug logging
uv run cli/hybrid_search_cli.py rrf-search "action" --evaluate               # Evaluation metrics
```

### Evaluation (`evaluation_cli.py`)

Evaluate search quality with precision@K, recall@K, and F1 score.

```bash
uv run cli/evaluation_cli.py --limit 5
```

### Augmented Generation (`augmented_generation_cli.py`)

LLM-powered answer generation using search results. Requires `GEMINI_API_KEY`.

```bash
uv run cli/augmented_generation_cli.py rag "dinosaur movies"                  # RAG answer
uv run cli/augmented_generation_cli.py summarize "dinosaur movies" --limit 5  # Multi-doc summary
uv run cli/augmented_generation_cli.py citations "dinosaur movies" --limit 5  # Answer with citations
uv run cli/augmented_generation_cli.py question "What dinosaur movies do you have?"  # Conversational Q&A
```

### Image Query Rewriting (`describe_image_cli.py`)

Multimodal query rewriting — combines an image with a text query via Gemini to produce a better search query.

```bash
uv run cli/describe_image_cli.py --image data/paddington.jpeg --query "bear movie for kids"
```

### Multimodal Search (`multimodal_search_cli.py`)

CLIP-based image search over the movie dataset.

```bash
uv run cli/multimodal_search_cli.py verify_image_embedding data/paddington.jpeg  # Verify CLIP embeddings
uv run cli/multimodal_search_cli.py image_search data/paddington.jpeg            # Search movies by image
```

## Architecture

```
cli/
  lib/
    semantic_search.py      # SemanticSearch, ChunkedSemanticSearch, cosine similarity
    hybrid_search.py        # HybridSearch (BM25 + semantic, weighted & RRF)
    multimodal_search.py    # MultimodalSearch (CLIP image embeddings)
  inverted_index.py         # InvertedIndex with BM25 scoring
data/
  movies.json               # 5000 movie dataset (id, title, description)
  paddington.jpeg           # Sample image for multimodal search
```
