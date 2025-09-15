import os
from dotenv import load_dotenv
from autogen import ConversableAgent, GroupChat, GroupChatManager

load_dotenv()

topic_agent = ConversableAgent(
    name="topic_agent",
    system_message="당신은 토론할 주제를 제시하는 역할입니다.",
    llm_config={"config_list": [
        {"model": "gpt-4o", "api_key": os.getenv("OPENAI_API_KEY")}]
    },
    human_input_mode="NEVER"
)

긍적적_생각_agent = ConversableAgent(
    name="긍적적_생각_agent",
    system_message="당신은 프론트엔드 프레임워크인 Next.js에 대해 긍적적 생각을 제시하는 역할입니다.",
    llm_config={"config_list": [
        {"model": "gpt-4o", "api_key": os.getenv("OPENAI_API_KEY")}]
    }
)

부정적_생각_agent = ConversableAgent(
    name="부정적_생각_agent",
    system_message="당신은 프론트엔드 프레임워크인 Next.js가 너무 복잡하다고 생각하여 부정적 생각을 제시하는 역할입니다.",
    llm_config={"config_list": [
        {"model": "gpt-4o", "api_key": os.getenv("OPENAI_API_KEY")}]
    }
)

group_chat = GroupChat(
    agents=[긍적적_생각_agent, 부정적_생각_agent],
    messages=[],
    max_round=6
)

group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config={"config_list": [
        {"model": "gpt-4o", "api_key": os.getenv("OPENAI_API_KEY")}]
    }
)

chat_result = 긍적적_생각_agent.initiate_chat(
    group_chat_manager,
    message="Next.js에 대해 토론해주세요.",
    summary_method="reflection_with_llm",

)
