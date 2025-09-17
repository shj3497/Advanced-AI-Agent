from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
import operator

load_dotenv()


class State(TypedDict):
    counter: int
    alphabet: list[str]


def node_a(state: State):
    state['counter'] += 1
    state['alphabet'] = ["Hello"]
    return state


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", node_a)

graph_builder.add_edge(START, 'chatbot')
graph_builder.add_edge('chatbot', END)

graph = graph_builder.compile()

# 이미지 생성
# try:
#     # Get the Mermaid PNG bytes
#     png_bytes = graph.get_graph().draw_mermaid_png()

#     # Save the bytes to a file
#     with open("content/langgraph-1.png", "wb") as f:
#         f.write(png_bytes)
#     print("Graph image saved to langgraph-1.png")
# except Exception as e:
#     # This requires some extra dependencies and is optional
#     print(f"Could not generate graph image: {e}")
#     pass

initial_state = {
    "counter": 0,
    "alphabet": []
}

state = initial_state

for _ in range(3):
    state = graph.invoke(state)
    print(state)
