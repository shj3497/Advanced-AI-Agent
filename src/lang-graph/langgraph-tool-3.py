from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph


load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")


class State(TypedDict):
    messages: Annotated[list, add_messages]


tool = TavilySearchResults(max_results=2)
tools = [tool]
tool_node = ToolNode(tools)

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    result = llm_with_tools.invoke(state['messages'])
    return {"messages": [result]}


graph_builder = StateGraph(State)

graph_builder.add_node('chatbot', chatbot)
graph_builder.add_node('tools', tool_node)

graph_builder.add_edge('tools', 'chatbot')
graph_builder.add_conditional_edges('chatbot', tools_condition)

graph_builder.set_entry_point('chatbot')
graph = graph_builder.compile()

# response = graph.invoke(
#     {"messages": [{"role": "user", "content": "지금 한국 대통령은 누구야?"}]})

response = graph.invoke(
    {"messages": [{"role": "user", "content": "마이크로소프트가 어떤 회사야?"}]})

print(response)
