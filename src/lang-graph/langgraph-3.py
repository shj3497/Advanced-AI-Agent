from typing import Annotated
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

# ? Message를 담는 StateGraph 만들기

model = ChatOpenAI(model="gpt-4o-mini")


# class State(TypedDict):
#     messages: Annotated[list, add_messages]
#! == MessagesState 사용

class State(MessagesState):
    counter: int


graph_builder = StateGraph(State)

llm = ChatOpenAI(model='gpt-4o-mini')


def chatbot(state: MessagesState):
    state['counter'] = state.get('counter', 0)+1
    return {"messages": [llm.invoke(state['messages'])], "counter": state['counter']}


graph_builder.add_node('chatbot', chatbot)
graph_builder.set_entry_point('chatbot')
graph_builder.set_finish_point('chatbot')
graph = graph_builder.compile()


while True:
    user_input = input('User :')
    if user_input.lower() in ['quit', 'exit', 'q']:
        print('Goodbye!')
        break

    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            print('Asisstant:', value['messages'][-1].content)
