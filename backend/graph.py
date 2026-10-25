from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import MemorySaver
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

from backend.config import POSTGRES_URI
from backend.state import State
from backend.nodes import (
    pii_guardrail_node,
    retrieve,
    grade_documents,
    generate_node,
    web_search,
    route_after_pii,
    route_after_retrieve,
    route_decision,
    generate_detailed_web_node,
)

_checkpointer = None
_pool = None


async def get_async_checkpointer():
    global _checkpointer, _pool
    if _checkpointer is not None:
        return _checkpointer

    try:
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        }
        _pool = AsyncConnectionPool(
            conninfo=POSTGRES_URI,
            max_size=10,
            kwargs=connection_kwargs,
        )
        _checkpointer = AsyncPostgresSaver(_pool)
        await _checkpointer.setup()
        return _checkpointer
    except Exception as e:
        print(f"[Warning] Could not initialize AsyncPostgresSaver at {POSTGRES_URI}: {e}")
        return None


def build_graph(checkpointer=None):
    graph = StateGraph(State)

    graph.add_node("pii_guardrail", pii_guardrail_node)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grading", grade_documents)
    graph.add_node("web_search", web_search)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "pii_guardrail")
    graph.add_conditional_edges("pii_guardrail", route_after_pii, {
        "web_search": "web_search",
        "retrieve": "retrieve"
    })
    graph.add_conditional_edges("retrieve", route_after_retrieve, {
        "web_search": "web_search",
        "grading": "grading"
    })
    graph.add_conditional_edges("grading", route_decision, {
        "web_search": "web_search",
        "generate": "generate"
    })
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


async def build_graph_async():
    checkpointer = await get_async_checkpointer()
    if checkpointer is None:
        checkpointer = MemorySaver()
    return build_graph(checkpointer=checkpointer)


chatbot = build_graph()
__all__ = ["chatbot", "build_graph", "build_graph_async", "get_async_checkpointer", "generate_detailed_web_node"]
