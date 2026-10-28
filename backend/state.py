from typing import TypedDict, List, Dict, Annotated, Optional
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def reduce_steps(current: Optional[List[str]], update: Optional[List[str]]) -> List[str]:
    if update == []:
        return []
    return (current or []) + (update or [])


class State(TypedDict, total=False):
    thread_id: str         
    raw_question: str
    question: str
    documents: List[Document]
    generation: str
    search: bool
    web_searched: bool
    steps: Annotated[List[str], reduce_steps]
    pii_map: Dict[str, str]
    pii_redacted: bool
    detail_requested: bool
    is_detailed: bool
    messages: Annotated[List[BaseMessage], add_messages]
    # Output guardrails fields
    safety_passed: bool
    safety_violation: Optional[str]
    faithfulness_passed: bool
    faithfulness_score: Optional[float]
    hallucination_retry_count: int
    output_pii_detected: bool

