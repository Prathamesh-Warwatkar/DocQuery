"""Embedding providers.

* HashingEmbedder (default): offline, deterministic word + character n-gram feature hashing into
  384 dims. It is a lexical-fuzzy embedding (no pretrained semantics), so it runs with no model download.
* AzureOpenAIEmbedder: real semantic embeddings. Selected with EMBEDDINGS_PROVIDER=azure_openai and the
  AZURE_OPENAI_* variables. NOTE: written against the SDK but not exercised live in the build environment.
"""
import os
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from .config import EMBEDDING_DIM, EMBEDDINGS_PROVIDER


class HashingEmbedder:
    name = "hashing-384"

    def __init__(self) -> None:
        self._word = HashingVectorizer(
            n_features=EMBEDDING_DIM, analyzer="word", ngram_range=(1, 2),
            stop_words="english", norm="l2", alternate_sign=True,
        )
        self._char = HashingVectorizer(
            n_features=EMBEDDING_DIM, analyzer="char_wb", ngram_range=(3, 5),
            norm="l2", alternate_sign=True,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        m = (self._word.transform(texts) + 0.6 * self._char.transform(texts)).toarray()
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (m / norms).astype(np.float32)


class AzureOpenAIEmbedder:
    name = "azure-openai"

    def __init__(self) -> None:
        from openai import AzureOpenAI

        self._client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )
        self._deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._deployment, input=texts, dimensions=EMBEDDING_DIM)
        arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


@lru_cache(maxsize=1)
def get_embedder():
    return AzureOpenAIEmbedder() if EMBEDDINGS_PROVIDER == "azure_openai" else HashingEmbedder()


def to_pgvector(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
