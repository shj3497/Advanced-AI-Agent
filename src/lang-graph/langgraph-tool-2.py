from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph import graph
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


@tool
def get_weather(location: str):
    """Call to get the weather"""

    if location in ["서울", "인천"]:
        return "It's 60 degrees and foggy"
    else:
        return "It's 90 degrees and sunny."


@tool
def get_coolest_cities():
    """Get a list of coolest cities"""

    return "서울, 고성"


tools = [get_weather, get_coolest_cities]
tool_node = ToolNode(tools)
model_with_tools = ChatOpenAI(
    model="gpt-4o-mini", temperature=0
).bind_tools(tools)


def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    latest_message = messages[-1]
    if (latest_message.tool_calls):
        return "tools"
    return END


def call_model(state: MessagesState):
    messages = state['messages']
    response = model_with_tools.invoke(messages)

    return {"messages": [response]}


workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node('tools', tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges('agent', should_continue)
workflow.add_edge("tools", 'agent')

app = workflow.compile()


final_state = app.invoke(
    {"messages": [HumanMessage(content="대한민국 서울 날씨는 어때?")]}
)

res = final_state['messages'][-1].content
# print(res)

# example with a multiple tool calls in succession
for chunk in app.stream(
    {"messages": [("human", "가장 추운 도시의 날씨는 어때?")]},
    stream_mode="values"
):
    res = chunk['messages'][-1].pretty_print()
    print(res)


# 이미지 생성
# try:
#     # Get the Mermaid PNG bytes
#     png_bytes = app.get_graph().draw_mermaid_png()

#     # Save the bytes to a file
#     with open("content/langgraph-tool-2.png", "wb") as f:
#         f.write(png_bytes)
#     print("Graph image saved to langgraph--tool-2.png")
# except Exception as e:
#     # This requires some extra dependencies and is optional
#     print(f"Could not generate graph image: {e}")
#     pass
