import os

# libpq conninfo or URL. Default points at a local Postgres 16 + pgvector on port 5433.
DATABASE_URL = os.getenv("DATABASE_URL", "host=/tmp port=5433 dbname=ragdb user=postgres")

EMBEDDING_DIM = 384
RRF_K = 60            # reciprocal-rank-fusion constant
CANDIDATE_POOL = 20   # candidates pulled from each retriever before fusion
# Guardrail: refuse to answer when the best chunk covers less than this share of the question terms (chosen on the dev split of the eval set).
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.34"))

EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "hashing")  # hashing | azure_openai
ANSWER_PROVIDER = os.getenv("ANSWER_PROVIDER", "extractive")       # extractive | anthropic
