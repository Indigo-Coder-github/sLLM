# vectordb_factory.py
import os
from pathlib import Path

from autogen_ext.memory.chromadb import PersistentChromaDBVectorMemoryConfig

from rag.vector_db import VectorDB


def create_preferences_memory() -> VectorDB:
    config = PersistentChromaDBVectorMemoryConfig(
        collection_name="pubmed",
        persistence_path=os.path.join(os.path.expanduser("~"), "chroma_db"),
        k=5,
        score_threshold=0.4,
    )
    return VectorDB(config=config)
