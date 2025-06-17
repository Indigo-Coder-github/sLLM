import json
import os
from typing import List

from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core.memory._base_memory import ContentType
from autogen_ext.memory.chromadb import (ChromaDBVectorMemory,
                                         PersistentChromaDBVectorMemoryConfig)
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

class VectorDB(ChromaDBVectorMemory):
    def __init__(self, config: PersistentChromaDBVectorMemoryConfig):
        """Initialize the VectorDB with the given configuration.

        Args:
            config (PersistentChromaDBVectorMemoryConfig): The configuration for the vector database.

        Example:
            PersistentChromaDBVectorMemoryConfig(
                collection_name="preferences",
                persistence_path=os.path.join(str(Path.home()), "chroma_db"),
                k=2,
                score_threshold=0.4,
            )
        """
        super().__init__(config)
        self.config = config

    async def add_content(self, content: MemoryContent) -> None:
        """
        content 삽입 함수. 예시의 형태를 따라야 함.
        metadata는 dict 형태로, 알아서 넣으면 됨.

        Args:
            content (MemoryContent): The content to be added to the vector database.

        Example:
            MemoryContent(
                content="The weather should be in metric units",
                mime_type=MemoryMimeType.TEXT,
                metadata={"category": "preferences", "type": "units"},
            )
        """
        await super().add(content)

    async def _process_file(self, path: str):
        """단일 JSONL 파일을 읽어 MemoryContent 삽입."""
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                memory_content = MemoryContent(
                    content=data.get("contents", ""),
                    mime_type=MemoryMimeType.TEXT,
                    metadata={"category": "pubmed", "type": "text"},
                )
                await self.add_content(memory_content)
        count += 1
        return os.path.basename(path), count

    async def add_contents_from_path(self, path: str) -> None:
        """
        path에 있는 파일들을 읽어와서 content로 변환 후 삽입.

        Args:
            path (str): The path to the file or directory containing content to be added.

        1. jsonl 파일을 읽어온다
        2. 각 줄을 MemoryContent로 변환한다. 이때, 각 key-value쌍을 하나의 문자열로 합친다.
        3. 변환된 MemoryContent를 add_content 함수를 통해 삽입한다.
        """

        # 파일 목록 수집
        if os.path.isfile(path) and path.endswith(".jsonl"):
            jsonl_files = [path]
        elif os.path.isdir(path):
            jsonl_files = [
                os.path.join(path, fn)
                for fn in os.listdir(path)
                if fn.endswith(".jsonl")
            ]
        else:
            raise FileNotFoundError(f"No valid file or folder: {path}")

        # JSONL 파일마다 create_task
        tasks = [asyncio.create_task(self._process_file(fp)) for fp in jsonl_files]

        # 파일 단위 비동기 진행바
        for result in async_tqdm.as_completed(tasks, total=len(tasks), desc="파일 처리 진행"):
            filename, count = await result
            tqdm.write(f"[완료] {filename}: {count} 줄 처리")

    async def retrieve_relevant(self, query: str) -> List[ContentType]:
        """
        query와 연관있는 결과 반환.
        이때, 임계값은 config에서 설정된 score_threshold 사용.

        Args:
            query (str): The content to query the vector database with.
        Example:
            query_content = MemoryContent(
                content="What is the weather in metric units?",
                mime_type=MemoryMimeType.TEXT,
                metadata={"category": "preferences", "type": "units"},
            )
        """
        results = (await super().query(query=query)).results
        return [r.content for r in results if isinstance(r, MemoryContent)]

    async def clear_memory(self) -> None:
        """DB의 구조는 유지한채, 데이터만 정리"""
        await super().clear()

    async def reset_memory(self) -> None:
        """DB의 구조와 데이터 모두 정리"""
        # allow_reset이 True이어야 함
        self.config.allow_reset = True
        await super().reset()


if __name__ == "__main__":
    import asyncio

    # Example usage
    config = PersistentChromaDBVectorMemoryConfig(
        collection_name="preferences",
        persistence_path=os.path.join(os.path.expanduser("~"), "chroma_db"),
        k=2,
        score_threshold=0.4,
    )
    vector_db = VectorDB(config)

    async def main():
        await vector_db.add_contents_from_path(
            "/home/tako/Documents/yonghan/sLLM/raw_data/MedQA/5_options/train.jsonl"
        )
        results = await vector_db.retrieve_relevant(
            "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. "
        )
        for result in results:
            print(result)

    asyncio.run(main())
