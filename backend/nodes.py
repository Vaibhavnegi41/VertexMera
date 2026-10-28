import asyncio
from typing import Literal
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from backend.state import State
from backend.config import model, grading_model, vector_store, web_search_tool
from backend.prompts import scoring_prompt, multi_doc_scoring_prompt, rag_prompt, web_prompt, web_detailed_prompt
from backend.pii_guardrails import get_guardrails, get_session_guardrails, mask_documents
from backend.intent_router import should_web_search_directly
from backend.output_guardrails import apply_output_guardrails

CONFIDENCE_THRESHOLD = 0.85
RETRIEVAL_SCORE_THRESHOLD = 0.70


class ScoreFormat(BaseModel):
    score: bool


class MultiDocRelevance(BaseModel):
    relevant_indices: list[int]


score_model = grading_model.with_structured_output(ScoreFormat)
multi_score_model = grading_model.with_structured_output(MultiDocRelevance)


async def pii_guardrail_node(state: State):
    question = state.get("question", "")
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
        score_val = float(score)
        if score_val >= RETRIEVAL_SCORE_THRESHOLD:
            if not hasattr(doc, "metadata") or doc.metadata is None:
                doc.metadata = {}
            doc.metadata["score"] = score_val
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

    formatted_docs = [
        f"[Doc {idx}]\n{doc.page_content}"
        for idx, doc in enumerate(documents)
    ]
    docs_block = "\n\n".join(formatted_docs)

    prompt = multi_doc_scoring_prompt.format_messages(
        question=question,
        documents=docs_block
    )

    try:
        res = await multi_score_model.ainvoke(prompt)
        relevant_indices = set(res.relevant_indices or [])
    except Exception as e:
        print(f"[Warning] Single-prompt grading parsing fallback: {e}")
        relevant_indices = set(range(len(documents)))

    filtered_docs = [doc for idx, doc in enumerate(documents) if idx in relevant_indices]
    search = len(filtered_docs) == 0

    return {
        "documents": filtered_docs,
        "search": search,
        "steps": ["grade_documents_node"]
    }


async def generate_node(state: State):
    documents = state.get("documents", [])
    context_str = "\n\n".join([doc.page_content for doc in documents])
    
    is_web = state.get("web_searched", False)
    is_detailed = state.get("detail_requested", False)

    if is_web:
        question = state.get("raw_question") or state.get("question", "")
        prompt = web_detailed_prompt if is_detailed else web_prompt
        llm = model if is_detailed else model.bind(max_tokens=200)
    else:
        question = state.get("question", "")
        prompt = rag_prompt
        llm = model

    chat_history = state.get("messages", [])
    current_q = state.get("raw_question") or state.get("question", "")
    prior_messages = [
        m for m in chat_history
        if getattr(m, "content", "") != current_q
    ]

    generation = await llm.ainvoke(
        prompt.format_messages(
            question=question,
            documents=context_str,
            chat_history=prior_messages
        )
    )

    return {
        "generation": generation.content,
        "steps": ["generation_node"],
        "is_detailed": state.get("detail_requested", False),
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

    guard_res = await apply_output_guardrails({
        **state,
        "generation": generation.content,
    })
    final_content = guard_res["generation"]
    ai_msg = AIMessage(content=final_content)

    return {
        "generation": final_content,
        "steps": ["detailed_web_generation_node", "output_guardrail_node"],
        "web_searched": True,
        "is_detailed": True,
        "detail_requested": True,
        "documents": documents,
        "messages": [ai_msg],
        "safety_passed": guard_res.get("safety_passed", True),
        "safety_violation": guard_res.get("safety_violation"),
        "faithfulness_passed": guard_res.get("faithfulness_passed", True),
        "faithfulness_score": guard_res.get("faithfulness_score"),
        "output_pii_detected": guard_res.get("output_pii_detected", False),
    }


async def output_guardrail_node(state: State):
    res = await apply_output_guardrails(state)
    updates = dict(res)
    is_routing_web = (
        not updates.get("faithfulness_passed", True)
        and not state.get("web_searched", False)
        and updates.get("hallucination_retry_count", 0) <= 1
        and updates.get("safety_passed", True)
    )
    if not is_routing_web:
        updates["messages"] = [AIMessage(content=updates["generation"])]
    return updates


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


def route_after_retrieve(state: State) -> Literal["web_search", "grading"]:
    documents = state.get("documents", [])
    if not documents:
        return "web_search"
    return "grading"


def route_decision(state: State) -> Literal["web_search", "generate"]:
    if state.get("search", False):
        return "web_search"
    return "generate"


def route_after_output_guardrail(state: State) -> Literal["web_search", "end"]:
    if (
        not state.get("faithfulness_passed", True)
        and not state.get("web_searched", False)
        and state.get("hallucination_retry_count", 0) <= 1
        and state.get("safety_passed", True)
    ):
        return "web_search"
    return "end"
