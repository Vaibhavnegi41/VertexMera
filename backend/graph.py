from langgraph.graph import StateGraph, START, END

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


def build_graph():
    graph = StateGraph(State)

    graph.add_node("pii_guardrail", pii_guardrail_node)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grading", grade_documents)
    graph.add_node("web_search", web_search)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "pii_guardrail")
    graph.add_edge("pii_guardrail", "retrieve")
    graph.add_edge("retrieve", "grading")
    graph.add_conditional_edges("grading", route_decision, {
        "web_search": "web_search",
        "generate": "generate"
    })
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


chatbot = build_graph()
__all__ = ["chatbot", "build_graph", "generate_detailed_web_node"]
