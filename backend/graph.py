from langgraph.graph import StateGraph, START, END

from backend.state import State
from backend.nodes import (
    retrieve,
    grade_documents,
    generate_node,
    web_search,
    route_decision,
)


def build_graph():
    """Build and compile the Corrective-RAG state graph."""
    graph = StateGraph(State)

    # ── Nodes ─
    graph.add_node("retrieve", retrieve)
    graph.add_node("grading", grade_documents)
    graph.add_node("web_search", web_search)
    graph.add_node("generate", generate_node)

    # ── Edges 
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grading")
    graph.add_conditional_edges("grading", route_decision, {
        "web_search": "web_search",
        "generate": "generate"
    })
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

chatbot = build_graph()
