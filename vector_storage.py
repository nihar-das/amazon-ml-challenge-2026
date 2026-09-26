"""Persistent storage for embeddings already computed by the caller.

Example:
    store = VectorStorage("./chroma_data")
    store.store_embeddings("business_names", ["1", "2"], [[0.1, 0.2], [0.3, 0.4]])
    collection = store.get_collection("business_names")
"""

import logging
from pathlib import Path
from typing import Sequence

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError


log = logging.getLogger(__name__)


class VectorStorage:
    def __init__(self, directory: str | Path):
        self._client = chromadb.PersistentClient(path=str(directory))
        log.info("Chroma storage ready at %s", directory)

    def get_collection(self, name: str) -> Collection:
        """Fetch an existing collection, requiring an HNSW cosine index."""
        collection = self._client.get_collection(name=name, embedding_function=None)
        self._require_cosine_hnsw(collection)
        log.info("Fetched collection %s", name)
        return collection

    def missing_ids(self, name: str, ids: Sequence[str]) -> list[str]:
        """Return IDs that are not yet saved in a collection."""
        if len(ids) == 0:
            return []

        try:
            collection = self._client.get_collection(name=name, embedding_function=None)
        except NotFoundError:
            return list(ids)

        self._require_cosine_hnsw(collection)
        saved_ids = set(collection.get(ids=list(ids), include=[])["ids"])
        return [item_id for item_id in ids if item_id not in saved_ids]

    def store_embeddings(
        self, name: str, ids: Sequence[str], embeddings: Sequence[Sequence[float]]
    ) -> Collection:
        """Insert or replace embeddings by ID in a persistent cosine collection."""
        if len(ids) != len(embeddings):
            raise ValueError("ids and embeddings must have the same length")

        collection = self._client.get_or_create_collection(
            name=name,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )
        self._require_cosine_hnsw(collection)

        batch_size = self._client.get_max_batch_size()
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            collection.upsert(ids=list(ids[start:end]), embeddings=embeddings[start:end])
        log.info("Saved %d embeddings to collection %s", len(ids), name)
        return collection

    @staticmethod
    def _require_cosine_hnsw(collection: Collection) -> None:
        hnsw = collection.configuration.get("hnsw")
        if not hnsw or hnsw.get("space") != "cosine":
            raise ValueError(
                f"Collection {collection.name!r} does not use HNSW with cosine distance"
            )
