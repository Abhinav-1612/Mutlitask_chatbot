"""
app/graph.py — LangGraph Workflow Compilation
=============================================
Assembles the full Supervisor architecture into a compiled StateGraph.

Flow:
  START
    └─▶ gateway_router
          ├─▶ general_node  ─────────────▶ END
          └─▶ supervisor_node
                ├─▶ rag_node     ──────▶ END
                ├─▶ web_node     ──────▶ END
                └─▶ finance_node ──────▶ END
"""
from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import StateGraph, START, END

from app.agents.state import UniversalAgentState
from app.agents.nodes import (
    gateway_router,
    supervisor_node,
    general_node,
    rag_node,
    web_node,
    finance_node,
    farmer_node,
)

logger = logging.getLogger(__name__)

# ── Conditional edge functions ────────────────────────────────────────────────

def route_start(
    state: UniversalAgentState,
) -> Literal["farmer_node", "gateway_router"]:
    """
    If farmer_mode is enabled, bypass gateway and go straight to farmer_node.
    Otherwise, standard gateway routing.
    """
    if state.get("farmer_mode"):
        logger.debug("[graph] START → farmer_node (Farmer Mode active)")
        return "farmer_node"
    return "gateway_router"


def route_after_gateway(
    state: UniversalAgentState,
) -> Literal["general_node", "supervisor_node"]:
    """
    After gateway_router:
    - "general" → skip supervisor, go straight to general_node
    - anything else → validate through supervisor_node
    """
    if state["next_node"] == "general":
        logger.debug("[graph] Gateway → general_node (bypassing supervisor)")
        return "general_node"
    logger.debug("[graph] Gateway → supervisor_node (route=%s)", state["next_node"])
    return "supervisor_node"


def route_after_supervisor(
    state: UniversalAgentState,
) -> Literal["rag_node", "web_node", "finance_node", "general_node"]:
    """
    After supervisor_node, dispatch to the correct specialist agent.
    """
    route = state["next_node"]
    mapping = {
        "rag":     "rag_node",
        "web":     "web_node",
        "finance": "finance_node",
    }
    dest = mapping.get(route, "general_node")
    logger.debug("[graph] Supervisor → %s", dest)
    return dest


# ── Graph compilation ─────────────────────────────────────────────────────────

_compiled_graph = None


def compile_graph():
    """
    Build and compile the LangGraph StateGraph (cached singleton).
    Returns the compiled graph ready for .ainvoke() or .astream().
    """
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    builder = StateGraph(UniversalAgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("gateway_router",  gateway_router)
    builder.add_node("supervisor_node", supervisor_node)
    builder.add_node("general_node",    general_node)
    builder.add_node("rag_node",        rag_node)
    builder.add_node("web_node",        web_node)
    builder.add_node("finance_node",    finance_node)
    builder.add_node("farmer_node",     farmer_node)

    # ── Wire edges ────────────────────────────────────────────────────────────
    # Route from start based on farmer_mode flag
    builder.add_conditional_edges(
        START,
        route_start,
        {"farmer_node": "farmer_node", "gateway_router": "gateway_router"},
    )

    # Gateway → general (fast path) or supervisor (tool path)
    builder.add_conditional_edges(
        "gateway_router",
        route_after_gateway,
        {"general_node": "general_node", "supervisor_node": "supervisor_node"},
    )

    # Supervisor → specialist agents
    builder.add_conditional_edges(
        "supervisor_node",
        route_after_supervisor,
        {
            "rag_node":     "rag_node",
            "web_node":     "web_node",
            "finance_node": "finance_node",
            "general_node": "general_node",
        },
    )

    # All specialists → END
    builder.add_edge("general_node",  END)
    builder.add_edge("rag_node",      END)
    builder.add_edge("web_node",      END)
    builder.add_edge("finance_node",  END)
    builder.add_edge("farmer_node",   END)

    _compiled_graph = builder.compile()
    logger.info("[graph] ✅ LangGraph pipeline compiled successfully.")
    return _compiled_graph


def get_graph():
    """Return the cached compiled graph (compiles on first call)."""
    return compile_graph()
