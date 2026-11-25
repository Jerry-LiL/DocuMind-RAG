"""Assemble the router, the two tools, and the grounding grade."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from documind.config import Settings
from documind.graph.nodes import make_nodes
from documind.graph.state import GraphState, initial_state
from documind.llm.providers import ModelProvider
from documind.retrieval.store import CorpusStore


def build_graph(settings: Settings, store: CorpusStore, model: ModelProvider):
    nodes = make_nodes(settings, store, model)
    graph = StateGraph(GraphState)
    graph.add_node("router", nodes["router"])
    graph.add_node("retrieve", nodes["retrieve"])
    graph.add_node("sql", nodes["sql"])
    graph.add_node("clarify", nodes["clarify"])
    graph.add_node("generate", nodes["generate"])
    graph.add_node("grade", nodes["grade"])

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        _choose_route,
        {"retrieve": "retrieve", "sql": "sql", "clarify": "clarify"},
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("sql", "generate")
    graph.add_edge("clarify", END)
    graph.add_edge("generate", "grade")
    graph.add_conditional_edges("grade", _after_grade, {"generate": "generate", "end": END})
    return graph.compile()


def run_question(graph, question: str) -> GraphState:
    return graph.invoke(initial_state(question))


def _choose_route(state: GraphState) -> str:
    route = state.get("route")
    if route in {"retrieve", "sql", "clarify"}:
        return route
    return "clarify"


def _after_grade(state: GraphState) -> str:
    if state.get("needs_rewrite"):
        return "generate"
    return "end"
