import asyncio
from typing import Literal
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from backend.state import State
from backend.config import model, grading_model, vector_store, web_search_tool
from backend.prompts import scoring_prompt, rag_prompt, web_prompt, web_detailed_prompt
from backend.pii_guardrails import get_guardrails, get_session_guardrails, mask_documents
from backend.intent_router import should_web_search_directly

CONFIDENCE_THRESHOLD = 0.85


class ScoreFormat(BaseModel):
    score: bool


score_model = grading_model.with_structured_output(ScoreFormat)


async def pii_guardrail_node(state: State):
    question = state.get("question", "")

    # Use per-session guard keyed by thread_id to prevent cross-user PII bleed.
    thread_id: str = state.get("thread_id") or "__global__"
    guard = get_session_guardrails(thread_id)
    res = await asyncio.to_thread(guard.mask, question)

    messages = state.get("messages", [])
    new_messages = []
    if question and (not messages or getattr(messages[-1], "content", None) != question):
        new_messages.append(HumanMessage(content=question))

    return {
        "raw_question": question,
        "question": res.masked_text,
        "pii_map": guard.pii_map,
        "pii_redacted": True,
        "steps": ["pii_guardrail_node"],
        "messages": new_messages
    }


async def retrieve(state: State):
    question = state.get("raw_question") or state.get("question", "")

    docs_with_scores = await vector_store.asimilarity_search_with_score(question, k=4)

    documents = []
    for doc, score in docs_with_scores:
        if not hasattr(doc, "metadata") or doc.metadata is None:
            doc.metadata = {}
        doc.metadata["score"] = float(score)
        documents.append(doc)

    return {
        "documents": documents,
        "steps": ["retrieval_node"]
    }


async def grade_documents(state: State):
    question = state.get("question", "")
    documents = state.get("documents", [])

    if not documents:
        return {"documents": [], "search": True, "steps": ["grade_documents_node"]}

    top_score = max([doc.metadata.get("score", 0.0) for doc in documents], default=0.0)
    if top_score >= CONFIDENCE_THRESHOLD:
        return {
            "documents": documents,
            "search": False,
            "steps": ["grade_documents_node"]
        }

    prompts = [
        scoring_prompt.format_messages(
            question=question,
            documents=doc.page_content
        )
        for doc in documents
    ]

    results = await score_model.abatch(prompts)
    filtered_docs = [doc for doc, result in zip(documents, results) if result.score]
    search = len(filtered_docs) == 0

    return {
        "documents": filtered_docs,
        "search": search,
        "steps": ["grade_documents_node"]
    }


async def generate_node(state: State):
    documents = state.get("documents", [])
    context_str = "\n\n".join([doc.page_content for doc in documents])
    
    if state.get("web_searched"):
        question = state.get("raw_question") or state.get("question", "")
        prompt = web_detailed_prompt if state.get("detail_requested") else web_prompt
    else:
        question = state.get("question", "")
        prompt = rag_prompt

    chat_history = state.get("messages", [])
    current_q = state.get("raw_question") or state.get("question", "")
    prior_messages = [
        m for m in chat_history
        if getattr(m, "content", "") != current_q
    ]

    generation = await model.ainvoke(
        prompt.format_messages(
            question=question,
            documents=context_str,
            chat_history=prior_messages
        )
    )

    ai_msg = AIMessage(content=generation.content)

    return {
        "generation": generation.content,
        "steps": ["generation_node"],
        "is_detailed": state.get("detail_requested", False),
        "messages": [ai_msg]
    }


async def generate_detailed_web_node(state: State):
    question = state.get("raw_question") or state.get("question", "")
    documents = state.get("documents", [])

    context_str = "\n\n".join([doc.page_content for doc in documents])

    chat_history = state.get("messages", [])
    prior_messages = [
        m for m in chat_history
        if getattr(m, "content", "") != question
    ]

    generation = await model.ainvoke(
        web_detailed_prompt.format_messages(
            question=question,
            documents=context_str,
            chat_history=prior_messages
        )
    )

    ai_msg = AIMessage(content=generation.content)

    return {
        "generation": generation.content,
        "steps": ["detailed_web_generation_node"],
        "web_searched": True,
        "is_detailed": True,
        "detail_requested": True,
        "documents": documents,
        "messages": [ai_msg]
    }


async def web_search(state: State):
    search_query = state.get("raw_question") or state.get("question", "")
    web_response = await web_search_tool.ainvoke(search_query)
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

    return {
        "documents": web_docs,
        "web_searched": True,
        "steps": ["web_search_node"]
    }


def route_after_pii(state: State) -> Literal["web_search", "retrieve"]:
    query = state.get("raw_question") or state.get("question", "")
    if should_web_search_directly(query):
        return "web_search"
    return "retrieve"


def route_decision(state: State) -> Literal["web_search", "generate"]:
    if state.get("search", False):
        return "web_search"
    return "generate"
