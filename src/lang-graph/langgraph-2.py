from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
import operator

load_dotenv()

# ? Annotated 사용하여 상태 변경 시 연산 적용


class State(TypedDict):
    counter: int
    alphabet: Annotated[list[str], operator.add]


def node_a(state: State):
    state['counter'] += 1
    state['alphabet'] = ["Hello"]
    return state


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", node_a)

graph_builder.add_edge(START, 'chatbot')
graph_builder.add_edge('chatbot', END)

graph = graph_builder.compile()


initial_state = {
    "counter": 0,
    "alphabet": []
}

state = initial_state

for _ in range(3):
    state = graph.invoke(state)
    print(state)
