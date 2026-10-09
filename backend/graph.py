from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import MemorySaver
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

from backend.config import POSTGRES_URI
from backend.state import State
from backend.nodes import (
    pii_guardrail_node,
    retrieve,
    grade_documents,
    generate_node,
    web_search,
    route_decision,
    generate_detailed_web_node,
)

_checkpointer = None
_pool = None


def get_checkpointer():
    """Initializes and returns a singleton PostgresSaver checkpointer instance.
    Uses psycopg ConnectionPool and automatically executes PostgresSaver.setup()
    to create checkpoint tables if they don't already exist.
    """
    global _checkpointer, _pool
    if _checkpointer is not None:
        return _checkpointer

    try:
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        }
        _pool = ConnectionPool(
            conninfo=POSTGRES_URI,
            max_size=10,
            kwargs=connection_kwargs,
        )
        _checkpointer = PostgresSaver(_pool)
        _checkpointer.setup()
        return _checkpointer
    except Exception as e:
        print(f"[Warning] Could not initialize PostgresSaver at {POSTGRES_URI}: {e}")
        return None


def build_graph(checkpointer=None):
    graph = StateGraph(State)

    graph.add_node("pii_guardrail", pii_guardrail_node)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grading", grade_documents)
    graph.add_node("web_search", web_search)
    graph.add_node("generate", generate_node)

    # Parallel fork (Fan-Out) from START to both PII Guardrail and Retrieval
    graph.add_edge(START, "pii_guardrail")
    graph.add_edge(START, "retrieve")

    # Fan-In Join: grading waits for both pii_guardrail and retrieve to complete
    graph.add_edge("pii_guardrail", "grading")
    graph.add_edge("retrieve", "grading")

    graph.add_conditional_edges("grading", route_decision, {
        "web_search": "web_search",
        "generate": "generate"
    })
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)

    if checkpointer is None:
        checkpointer = get_checkpointer()

    if checkpointer is None:
        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


chatbot = build_graph()
__all__ = ["chatbot", "build_graph", "get_checkpointer", "generate_detailed_web_node"]

