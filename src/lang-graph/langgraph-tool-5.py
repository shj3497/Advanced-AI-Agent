from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

from langgraph.checkpoint.memory import MemorySaver


load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
memory = MemorySaver()

#! 기억할 메세지 개수 제한하기


class State(TypedDict):
    messages: Annotated[list, add_messages]


tool = TavilySearchResults(max_results=2)
tools = [tool]
tool_node = ToolNode(tools)

llm_with_tools = llm.bind_tools(tools)


def filter_messages(messages: list):
    return messages[-2:]


def chatbot(state: State):
    messages = filter_messages(state['messages'])
    result = llm_with_tools.invoke(messages)
    return {"messages": [result]}


graph_builder = StateGraph(State)

graph_builder.add_node('chatbot', chatbot)
graph_builder.add_node('tools', tool_node)

graph_builder.add_edge('tools', 'chatbot')
graph_builder.add_conditional_edges('chatbot', tools_condition)

graph_builder.set_entry_point('chatbot')
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("User : ")
    if (user_input.lower() in ['quit', 'exit', 'q']):
        print("Goodbye!")
        break

    for event in graph.stream({"messages": ("user", user_input)}, config):
        for value in event.values():
            print('Assistant : ', value['messages'][-1].content)
