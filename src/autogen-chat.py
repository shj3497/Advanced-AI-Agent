import os
from dotenv import load_dotenv
from autogen import ConversableAgent

load_dotenv()

corder_agent = ConversableAgent(
    name='Junior_Corder_Agent',
    system_message="당신은 4년차 Next.js, React, Typescript 전문 개발자입니다. 이해가 안되는 부분은 Senior_Corder_Agent에게 물어보세요.",
    llm_config={"config_list": [
        {"model": "gpt-4o", "api_key": os.getenv("OPENAI_API_KEY")}]}

)

senior_corder_agent = ConversableAgent(
    name='Senior_Corder_Agent',
    system_message="당신은 10년차 Next.js, React, Typescript 전문 개발자입니다. 질문이 주어지면, 해당 질문에 전문성을 가지고 대답해주세요. 만약 코드가 주어졌다면 주어진 코드를 검토하고 효율성을 높일 방안을 탐구하세요.",
    llm_config={"config_list": [
        {"model": "gpt-4o", "api_key": os.getenv("OPENAI_API_KEY")}]}
)


def get_user_input():
    message = input('코드 작성을 요청하고 싶은 내용을 입력해주세요:')
    return message


message = get_user_input()

chat_result = corder_agent.initiate_chat(
    senior_corder_agent,
    message=message,
    summary_method="reflection_with_llm",
    max_turns=5
)

print(chat_result.summary)
