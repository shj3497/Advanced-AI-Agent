from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

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

test_res = model_with_tools.invoke("대한민국 대통령은 누구야?").tool_calls
print(test_res)  # 빈배열

res1 = model_with_tools.invoke("대한민국 서울 날씨는 어때?").tool_calls
# [{'name': 'get_weather', 'args': {'location': '서울, 대한민국'}, 'id': 'call_R5ySzWT6mKyBNTJRCMkOLkLb', 'type': 'tool_call'}]
print('### start res1 ###')
print(res1)
print("### end res1 ###")

res2 = model_with_tools.invoke('한국에서 가장 추운 도시는?').tool_calls
print('### start res2 ###')
print(res2)
print("### end res2 ###")
