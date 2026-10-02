from typing import Literal
from pydantic import BaseModel
from langchain_core.documents import Document

from backend.state import State
from backend.config import model, grading_model, vector_store, web_search_tool
from backend.prompts import scoring_prompt, rag_prompt, web_prompt, web_detailed_prompt
from backend.pii_guardrails import get_guardrails, mask_documents

CONFIDENCE_THRESHOLD = 0.85


class ScoreFormat(BaseModel):
    score: bool


score_model = grading_model.with_structured_output(ScoreFormat)


def pii_guardrail_node(state: State):
    question = state.get("question", "")
    steps = state.get("steps", [])
    steps.append("pii_guardrail_node")

    guard = get_guardrails()
    res = guard.mask(question)

    return {
        "raw_question": question,
        "question": res.masked_text,
        "pii_map": guard.pii_map,
        "pii_redacted": True,
        "steps": steps,
    }


def retrieve(state: State):
    question = state["question"]
    docs_with_scores = vector_store.similarity_search_with_score(question, k=4)

    documents = []
    for doc, score in docs_with_scores:
        if not hasattr(doc, "metadata") or doc.metadata is None:
            doc.metadata = {}
        doc.metadata["score"] = float(score)
        documents.append(doc)

    steps = state["steps"]
    steps.append("retrieval_node")

    return {
        "documents": documents,
        "steps": steps
    }


def grade_documents(state: State):
    question = state["question"]
    documents = state["documents"]
    steps = state["steps"]
    steps.append("grade_documents_node")

    if not documents:
        return {"documents": [], "search": True, "steps": steps}

    top_score = max([doc.metadata.get("score", 0.0) for doc in documents], default=0.0)
    if top_score >= CONFIDENCE_THRESHOLD:
        return {
            "documents": documents,
            "search": False,
            "steps": steps
        }

    prompts = [
        scoring_prompt.format_messages(
            question=question,
            documents=doc.page_content
        )
        for doc in documents
    ]

    results = score_model.batch(prompts)
    filtered_docs = [doc for doc, result in zip(documents, results) if result.score]
    search = len(filtered_docs) == 0

    return {
        "documents": filtered_docs,
        "search": search,
        "steps": steps
    }


def generate_node(state: State):
    steps = state["steps"]
    steps.append("Generation_node")

    documents = state["documents"]
    context_str = "\n\n".join([doc.page_content for doc in documents])
    
    if state.get("web_searched"):
        # For public web search results, preserve authentic source entities
        question = state.get("raw_question") or state["question"]
        prompt = web_detailed_prompt if state.get("detail_requested") else web_prompt
    else:
        question = state["question"]
        prompt = rag_prompt

    generation = model.invoke(
        prompt.format_messages(
            question=question,
            documents=context_str
        )
    )

    return {
        "generation": generation.content,
        "steps": steps,
        "is_detailed": state.get("detail_requested", False)
    }


def generate_detailed_web_node(state: State):
    """Generates an in-depth, structured breakdown using already cached web search documents (zero extra search latency)."""
    question = state.get("raw_question") or state.get("question", "")
    documents = state.get("documents", [])
    steps = list(state.get("steps", []))
    steps.append("detailed_web_generation_node")

    context_str = "\n\n".join([doc.page_content for doc in documents])

    generation = model.invoke(
        web_detailed_prompt.format_messages(
            question=question,
            documents=context_str
        )
    )

    return {
        "generation": generation.content,
        "steps": steps,
        "web_searched": True,
        "is_detailed": True,
        "detail_requested": True,
        "documents": documents
    }


def web_search(state: State):
    # Search external web using original question without PII masking to find accurate public web information
    search_query = state.get("raw_question") or state["question"]
    steps = state["steps"]
    steps.append("web_search_node")

    web_response = web_search_tool.invoke(search_query)
    web_docs = []

    if isinstance(web_response, list):
        for result in web_response:
            content = result.get("content", "")
            if content:
                web_docs.append(Document(page_content=content))

    elif isinstance(web_response, dict):
        for result in web_response.get("results", []):
            content = result.get("content", "")
            if content:
                web_docs.append(Document(page_content=content))

    # Public web search sources do not contain internal sensitive PII,
    # keep raw source information for high accuracy.
    return {
        "documents": web_docs,
        "web_searched": True,
        "steps": steps
    }


def route_decision(state: State) -> Literal["web_search", "generate"]:
    if state["search"]:
        return "web_search"
    return "generate"
