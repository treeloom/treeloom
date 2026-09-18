from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any


@dataclass
class Query:
    text: str
    embedding: list[float]
    language: str | None = None
    path_prefix: str | None = None
    source_id: str | None = None
    top_k: int = 10
    use_hybrid: bool = False
    use_hyde: bool = False
    use_summary_vector: bool = False
    use_graph_scoring: bool = False


@dataclass
class SearchResult:
    chunk_text: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    source_id: str
    relevance_score: float
    final_score: float
    graph_signals: dict[str, Any]


class CommunityPort(ABC):
    @abstractmethod
    def detect_communities(self, project_id: str):
        pass

    @abstractmethod
    def summarize_community(self, community_id: int) -> str:
        pass

    @abstractmethod
    def get_community_summaries(self, community_ids: list[str]) -> dict[str, str]:
        pass

    @abstractmethod
    def load_community_embeddings(self, community_ids: list[int]) -> dict[int, list[float]]:
        pass


class GraphSearchPort(ABC):
    @abstractmethod
    def get_entities_by_file(self, file_path: str, source_id: str | None = None) -> list[Any]:
        pass

    @abstractmethod
    def traverse(self, entity_id: str, depth: int = 1, max_nodes: int = 50) -> list[Any]:
        pass

    @abstractmethod
    def get_community_ids(self, entity_ids: list[str]) -> set[int]:
        pass

    @abstractmethod
    def compute_centrality(self, entity_id: str) -> float:
        pass


class SearchService(ABC):
    @abstractmethod
    def search(self, query: Query) -> list[SearchResult]:
        pass

    @abstractmethod
    def enhanced_search(self, query: Query) -> list[SearchResult]:
        pass
