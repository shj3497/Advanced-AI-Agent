from typing import Annotated
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import asyncio

load_dotenv()


web_search = TavilySearchResults(k=3)
repl = PythonREPL()


class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def python_repl(code: Annotated[str, "The python code to generate your chart."]):
    """
    Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. chart labels should be written in English.
    This is visible to the user.
    """

    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )


tools = [web_search, python_repl]
tool_node = ToolNode(tools)

# 에이전트에게 도구 인지시키기
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def agent(state: State):
    result = llm_with_tools.invoke(state["messages"])
    return {"messages": [result]}


def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


workflow = StateGraph(State)

workflow.add_node("agent", agent)
workflow.add_node('tool', tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent", should_continue,
    {
        "continue": "tool",
        "end": END
    }
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tool", "agent")

# Set up memory
memory = MemorySaver()
#! interrupt_before=["tool"] 도구 호출 전에 인터럽트 발생 > 사용자가 확인해야함
graph = workflow.compile(checkpointer=memory, interrupt_before=["tool"])


# 이미지 생성
# try:
#     # Get the Mermaid PNG bytes
#     png_bytes = graph.get_graph().draw_mermaid_png()
#     # Save the bytes to a file
#     with open("content/langgraph-prompt-generator-2.png", "wb") as f:
#         f.write(png_bytes)
#     print("Graph image saved to langgraph-prompt-generator-2.png")
# except Exception as e:
#     # This requires some extra dependencies and is optional
#     print(f"Could not generate graph image: {e}")
#     pass


async def main():
    initial_input = {"messages": [HumanMessage(
        content="미국의 최근 5개년(~2023) GDP 차트를 그려줄래?")]}
    thread = {"configurable": {"thread_id": "13"}}

    async for chunk in graph.astream(initial_input, thread, stream_mode="updates"):
        for node, values in chunk.items():
            print(f"Receiving update from node: '{node}'")
            print(values)
            print("\n\n")


if __name__ == "__main__":
    asyncio.run(main())
