from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod
from typing import ClassVar, Any


@dataclass
class Source:
    id: str
    path: str
    url: str
    branch: str
    indexed_at: datetime
    file_count: int
    chunk_count: int


@dataclass
class Chunk:
    text: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    source_id: str


class EmbeddingPort(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        pass


class RerankPort(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        pass


class VectorStorePort(ABC):
    @abstractmethod
    def init_collection(self):
        pass

    @abstractmethod
    def insert(self, chunks: list[Chunk], embeddings: list[list[float]]):
        pass

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[Any]:
        pass

    @abstractmethod
    def hybrid_search(self, query: str, query_embedding: list[float], top_k: int) -> list[Any]:
        pass

    @abstractmethod
    def delete_by_filter(self, field: str, value: str):
        pass


class IndexingService(ABC):
    @abstractmethod
    def index_source(self, source: Source):
        pass

    @abstractmethod
    def delete_source(self, source_id: str):
        pass
