from typing import TypedDict, List, Dict
from langchain_core.documents import Document


class State(TypedDict, total=False):
    raw_question: str
    question: str
    documents: List[Document]
    generation: str
    search: bool
    web_searched: bool
    steps: List[str]
    pii_map: Dict[str, str]
    pii_redacted: bool
    detail_requested: bool
    is_detailed: bool
