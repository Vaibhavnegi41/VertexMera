from typing import TypedDict, List
from langchain_core.documents import Document


class State(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    search: bool
    web_searched: bool
    steps: List[str]
