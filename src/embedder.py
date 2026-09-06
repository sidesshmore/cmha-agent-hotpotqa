"""Local embeddings via Ollama's native /api/embeddings route.

No HuggingFace download, no PyTorch — the embedding model (nomic-embed-text,
~274MB) is served by the same local Ollama process as the chat models, so
this whole project has exactly one runtime dependency: Ollama.
"""

import hashlib
import os

import numpy as np
import requests


class Embedder:
    def __init__(self, model: str = "nomic-embed-text", host: str | None = None, mock: bool = False):
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.mock = mock

    def embed(self, texts: list[str]) -> np.ndarray:
        """Returns an (n_texts, dim) L2-normalized float32 array."""
        if self.mock:
            vecs = [self._mock_vector(t) for t in texts]
        else:
            vecs = [self._embed_one(t) for t in texts]
        arr = np.asarray(vecs, dtype=np.float32)
        return self._normalize(arr)

    def _embed_one(self, text: str) -> list[float]:
        resp = requests.post(
            f"{self.host}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    @staticmethod
    def _mock_vector(text: str, dim: int = 32) -> np.ndarray:
        """Deterministic pseudo-embedding — needs no Ollama server at all.

        Seeded from an MD5 hash (not Python's built-in hash(), which is
        randomized per-process) so the same text always maps to the same
        vector across runs.
        """
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        return rng.normal(size=dim).astype(np.float32)

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms
