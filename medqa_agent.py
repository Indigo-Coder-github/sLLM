from typing import Literal

from autogen_core import CancellationToken
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel

from tqdm import tqdm
from rag.stateful_assistant_agent import StatefulAssistantAgent

model_client = OpenAIChatCompletionClient(
    model="medqa_agent/",  # 서빙 시 사용되는 모델 이름
    base_url="http://localhost:8000/v1",
    model_info=ModelInfo(
        vision=False,
        function_calling=False,
        json_output=True,
        family="unknown",
        structured_output=True,
        multiple_system_messages=True
    ),
    api_key="anything",
)


class MedQAResponse4Options(BaseModel):
    """Response model for medical questions."""

    thought: str
    response: Literal["A", "B", "C", "D"]


class MedQAResponse5Options(MedQAResponse4Options):
    """Response model for 5-option medical questions."""

    response: Literal["A", "B", "C", "D", "E"]


class MedQAAgent(StatefulAssistantAgent):
    """Medical question answering agent."""

    def __init__(self, len_options: int = 5):
        """Initialize MedQA agent.

        Args:
            len_options: Number of answer options (4 or 5)
        """
        output_content_type = (
            MedQAResponse4Options if len_options == 4 else MedQAResponse5Options
        )

        super().__init__(
            name="MedQAAgent",
            model_client=model_client,
            system_message="You are a medical question answering agent.",
            output_content_type=output_content_type,
        )

    async def completion(self, query: str | list[str]) -> str:
        if isinstance(query, str):
            query = [query]

        results = []
        for q in tqdm(query):
            resp = await self.run(task=q)
            # print(f"질의: {q}")
            # print(f"응답: {resp.messages[-1].content.response}")
            # print(f"전체 응답: {resp}")
            results.append(resp.messages[-1].content.response)
            await self.reset_context()
            

        # query와 key-value 쌍으로 합쳐서 반환
        return dict(zip(query, results))

    async def reset_context(self) -> None:
        """Close the agent."""
        await super().on_reset(cancellation_token=CancellationToken())

if __name__ == "__main__":
    import asyncio

    agent = MedQAAgent()
    query = """
A 21-year-old sexually active male complains of fever.
pain during urination, and inflammation and pain in the right knee.
A culture of the joint fluid shows a bacteria that does not ferment maltose and has no polysaccharide capsule.
The physician orders antibiotic therapy for the patient.
The mechanism of action of the medication given blocks cell wall synthesis, which of the following was given?

"options"
    "A": "Chloramphenicol"
    "B": "Gentamicin"
    "C": "Ciprofloxacin"
    "D": "Ceftriaxone"
    "E": "Trimethoprim"
"""

    result = asyncio.run(agent.completion(query=query))
    print("Final Result:", result)
