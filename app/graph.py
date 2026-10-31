"""Graph wiring + all routing (retry logic) in one place.

    START -> analyze_query -> retrieve -> grade_documents --(relevant found)-----------> generate
                                ^                |
                                |                +--(none, retries left)--> rewrite_query -+
                                +-----------------------------------------------------------+
                                                 +--(none, retries used up, web on)--> web_search --> generate
                                                 +--(none, retries used up, web off)--> fallback --> END
    generate -> check_grounding --(grounded)--> END
                      +--(not grounded, attempts left)--> generate (with feedback)
                      +--(not grounded, attempts used up)--> END (with grounded=False flag)
"""
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app import nodes
from app.config import settings
from app.state import GraphState

# --------------------------------------------------------------------------
# Routing functions (conditional edges)
# --------------------------------------------------------------------------


def route_after_grading(state: GraphState) -> str:
    if state.get("relevant_documents"):
        return "generate"
    if state.get("retry_count", 0) < state.get("max_retries", settings.max_retries):
        return "rewrite_query"
    if nodes.web_search_available():
        return "web_search"
    return "fallback"


def route_after_web_search(state: GraphState) -> str:
    return "generate" if state.get("relevant_documents") else "fallback"


def route_after_generation(state: GraphState) -> str:
    return "check_grounding" if settings.enable_hallucination_check else END


def route_after_grounding(state: GraphState) -> str:
    if state.get("grounded", True):
        return END
    if state.get("generation_attempts", 0) < settings.max_generation_attempts:
        return "generate"
    return END  # limit reached: the answer is returned, but with the grounded=False flag


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("analyze_query", nodes.analyze_query)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade_documents", nodes.grade_documents)
    g.add_node("rewrite_query", nodes.rewrite_query)
    g.add_node("web_search", nodes.web_search)
    g.add_node("generate", nodes.generate)
    g.add_node("check_grounding", nodes.check_grounding)
    g.add_node("fallback", nodes.fallback)

    g.add_edge(START, "analyze_query")
    g.add_edge("analyze_query", "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"generate": "generate", "rewrite_query": "rewrite_query", "web_search": "web_search", "fallback": "fallback"},
    )
    g.add_edge("rewrite_query", "retrieve")
    g.add_conditional_edges("web_search", route_after_web_search, {"generate": "generate", "fallback": "fallback"})
    g.add_conditional_edges("generate", route_after_generation, {"check_grounding": "check_grounding", END: END})
    g.add_conditional_edges("check_grounding", route_after_grounding, {"generate": "generate", END: END})
    g.add_edge("fallback", END)
    return g.compile()


@lru_cache
def get_graph():
    return build_graph()


def run_query(question: str, chat_history: list[dict] | None = None) -> dict:
    """Run one question through the graph and return the final state."""
    history = (chat_history or [])[-2 * settings.max_history_turns :]
    initial: GraphState = {
        "question": question,
        "chat_history": history,
        "retry_count": 0,
        "max_retries": settings.max_retries,
        "generation_attempts": 0,
        "used_web_search": False,
        "grounded": None,
        "trace": [],
    }
    final = get_graph().invoke(initial)
    final["status"] = _status(final)
    return final


def _status(state: dict) -> str:
    if not state.get("relevant_documents"):  # came from the fallback node: nothing was found
        return "no_answer"
    if state.get("grounded") is False:
        return "unverified"
    return "answered_from_web" if state.get("used_web_search") else "answered"