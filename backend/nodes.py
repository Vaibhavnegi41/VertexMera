from typing import Literal
from pydantic import BaseModel
from langchain_core.documents import Document

from backend.state import State
from backend.config import model, grading_model, retriever, web_search_tool
from backend.prompts import scoring_prompt, rag_prompt, web_prompt


class ScoreFormat(BaseModel):
    score: bool


score_model = grading_model.with_structured_output(ScoreFormat)


def retrieve(state: State):
    """Query the Pinecone vector store for relevant documents."""
    question = state["question"]
    response = retriever.invoke(question)

    steps = state["steps"]
    steps.append("retrieval_node")

    return {
        "documents": response,
        "steps": steps
    }


def grade_documents(state: State):
    question = state["question"]
    documents = state["documents"]

    steps = state["steps"]
    steps.append("grade_documents_node")

    prompts = [
        scoring_prompt.format_messages(
            question=question,
            documents=doc.page_content
        )
        for doc in documents
    ]

    results = score_model.batch(prompts)

    filtered_docs = [
        doc for doc, result in zip(documents, results)
        if result.score
    ]

    search = len(filtered_docs) == 0

    print("SEARCH :", search)

    return {
        "documents": filtered_docs,
        "search": search,
        "steps": steps
    }


def generate_node(state: State):
    question = state["question"]
    documents = state["documents"]
    steps = state["steps"]
    steps.append("Generation_node")

    context_str = "\n\n".join([doc.page_content for doc in documents])

    prompt = web_prompt if state.get("web_searched") else rag_prompt

    generation = model.invoke(
        prompt.format_messages(
            question=question,
            documents=context_str
        )
    )

    return {
        "generation": generation.content,
        "steps": steps
    }


def web_search(state: State):
    question = state["question"]
    steps = state["steps"]
    steps.append("web_search_node")

    web_response = web_search_tool.invoke(question)

    web_docs = []

    # TavilySearch returns a list directly
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

    print("WEB DOCS FOUND:", len(web_docs))

    return {
        "documents": web_docs,
        "web_searched": True,
        "steps": steps
    }


def route_decision(state: State) -> Literal["web_search", "generate"]:
    """Decide whether to fall back to web search or go straight to generation."""
    if state["search"]:
        return "web_search"
    return "generate"
