"""Graph."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from context import Context
from state import InputState, OutputState, State
from subagents.event_summary import event_summary_agent
from subagents.influence_analysis import influence_analysis_agent
from subagents.answer_merge import answer_merge_agent


def graph() -> StateGraph:
    """Agent graph."""
    builder = StateGraph(
        State,
        input_schema=InputState,
        output_schema=OutputState,
        context_schema=Context,
    )

    # Nodes
    builder.add_node("event_summary", event_summary_agent().compile())
    builder.add_node("influence_analysis", influence_analysis_agent().compile())
    builder.add_node("answer_merge", answer_merge_agent().compile())
    builder.add_node("compose_report", compose_report)

    # Edges
    builder.add_edge(START, "event_summary")
    builder.add_edge("event_summary", "influence_analysis")
    builder.add_edge("influence_analysis", "answer_merge")
    builder.add_edge("answer_merge", "compose_report")
    builder.add_edge("compose_report", END)

    return builder


def compose_report(
    state: State,
) -> dict[str, Any]:
    """Compose report."""
    return {
        "report": state.get("answer_merge")
    }
