import os
from dotenv import load_dotenv
from autogen import ConversableAgent, GroupChat, GroupChatManager

load_dotenv()

topic_agent = ConversableAgent(
    name="topic_agent",
    system_message="당신은 토론할 주제를 제시하는 역할입니다.",
    llm_config={"config_list": [
        {"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}]
    },
    human_input_mode="NEVER"
)

긍적적_생각_agent = ConversableAgent(
    name="긍적적_생각_agent",
    system_message="당신은 연애 박사입니다. 주어진 주제와 이전 의견들에 대해 긍적적 생각을 제시하는 역할입니다.",
    llm_config={"config_list": [
        {"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}]
    }
)

부정적_생각_agent = ConversableAgent(
    name="부정적_생각_agent",
    system_message="당신은 연애 박사입니다. 주어진 주제와 이전 의견들에 대해 부정적 생각을 제시하는 역할입니다.",
    llm_config={"config_list": [
        {"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}]
    }
)

group_chat = GroupChat(
    agents=[긍적적_생각_agent, 부정적_생각_agent],
    messages=[],
    max_round=9
)

group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config={"config_list": [
        {"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}]
    }
)

chat_result = topic_agent.initiate_chat(
    group_chat_manager,
    message="주어진 상황에 대해 토론해주세요. 주어진 상황은 다음과 같습니다. ",
    summary_method="reflection_with_llm",

)

print(chat_result.summary)
