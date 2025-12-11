"""Influence analysis subagent."""

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph

from script.context import Context
from script.model import get_response
from script.state import State

def influence_analysis_agent() -> StateGraph:
    """Influence analysis subagent."""
    builder = StateGraph(
        State,
        context_schema=Context,
    )
    # Nodes

    # Edges
    builder.add_edge(START, END)

    return builder