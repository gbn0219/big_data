"""State."""

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class InputState(TypedDict):
    """Input State."""

    task_id: int
    topic: str


class OutputState(TypedDict):
    """Output State."""

    report: dict[str, Any]


class State(InputState, OutputState):
    """State."""

    history: Annotated[list, add_messages]
    event_summary: dict[str, Any]
    influence_analysis: dict[str, Any]
    answer_merge: dict[str, Any]