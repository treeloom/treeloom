"""Vector store factory — creates the appropriate VectorStorePort adapter from config."""

from treeloom.adapters.chromadb.vector_store import ChromaAdapter
from treeloom.adapters.milvus.vector_store import MilvusAdapter
from treeloom.domain.indexing import VectorStorePort
from treeloom.infrastructure.config import ConfigError


def create_vector_store(config: dict) -> VectorStorePort:
    """Create a VectorStorePort adapter from a configuration dictionary.

    Args:
        config: Dictionary with at minimum a ``backend`` key and a sub-section
                with backend-specific settings (e.g. ``milvus``, ``chromadb``).

    Returns:
        An initialized VectorStorePort adapter instance.

    Raises:
        ConfigError: If ``backend`` is missing from the configuration.
        ValueError: If the ``backend`` value is not one of the supported backends:
                    ``milvus``, ``chromadb``.
    """
    backend = config.get("backend")
    if backend is None:
        raise ConfigError(
            "Missing 'backend' key in vector store config. "
            "Set backend to one of: milvus, chromadb."
        )

    match backend:
        case "milvus":
            return MilvusAdapter(**config.get("milvus", {}))
        case "chromadb":
            return ChromaAdapter(**config.get("chromadb", {}))
        case _:
            raise ValueError(
                f"Unknown vector store backend: {backend!r}. "
                f"Supported backends: milvus, chromadb."
            )
