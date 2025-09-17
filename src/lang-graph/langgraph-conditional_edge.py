from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph


class State(TypedDict):
    tool: Literal['search', 'web_search']


def route_tools(
    state: State
) -> Literal["tools", "__end__"]:
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """

    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get('messages', []):
        ai_message = messages[-1]
    else:
        raise ValueError(
            f"No messages found in input state to tool_edge:{state}"
        )
    if hasattr(ai_message, 'tool_calls') and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"


graph_builder = StateGraph(State)
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", "__end__": "__end__"}
)
