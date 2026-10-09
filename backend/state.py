import operator
from typing import TypedDict, List, Dict, Annotated
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class State(TypedDict, total=False):
    raw_question: str
    question: str
    documents: List[Document]
    generation: str
    search: bool
    web_searched: bool
    steps: Annotated[List[str], operator.add]
    pii_map: Dict[str, str]
    pii_redacted: bool
    detail_requested: bool
    is_detailed: bool
    messages: Annotated[List[BaseMessage], add_messages]

