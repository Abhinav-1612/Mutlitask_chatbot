"""
app/graph.py — LangGraph Workflow Compilation
=============================================
Assembles the full Supervisor architecture into a compiled StateGraph.

Flow:
START
   │
   ▼
Farmer Mode?
   │
Yes────────►Farmer Agent
   │
No
   ▼
Agent Node (Omni-Agent)
   │
   ▼
  END
"""
from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import StateGraph, START, END

from app.agents.state import UniversalAgentState
from app.agents.nodes import (
    agent_node,
    farmer_node,
)

logger = logging.getLogger(__name__)

# ── Conditional edge functions ────────────────────────────────────────────────

def route_start(
    state: UniversalAgentState,
) -> Literal["farmer_node", "agent_node"]:
    """
    If farmer_mode is enabled, bypass gateway and go straight to farmer_node.
    Otherwise, route to the omni agent_node.
    """
    if state.get("farmer_mode"):
        logger.debug("[graph] START → farmer_node (Farmer Mode active)")
        return "farmer_node"
    return "agent_node"


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
    builder.add_node("agent_node", agent_node)
    builder.add_node("farmer_node", farmer_node)

    # ── Wire edges ────────────────────────────────────────────────────────────
    # Route from start based on farmer_mode flag
    builder.add_conditional_edges(
        START,
        route_start,
        {"farmer_node": "farmer_node", "agent_node": "agent_node"},
    )

    builder.add_edge("agent_node", END)
    builder.add_edge("farmer_node", END)

    _compiled_graph = builder.compile()
    logger.info("[graph] ✅ LangGraph pipeline compiled successfully.")
    return _compiled_graph

def get_graph():
    """Return the cached compiled graph (compiles on first call)."""
    return compile_graph()