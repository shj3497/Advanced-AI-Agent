from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    # SystemMessage : 유용한 챗봇이라는 역할과 이름을 부여
    ("system", "You are a helpful AI bot. Your name is {name}."),
    # HumanMessage와 AIMessage: 서로 안부를 묻고 대답하는 대화 히스토리 주입.
    ('human', 'Hello, how are you doing?'),
    ('ai', "I'm doing well, thanks!"),
    # HumanMessage로 사용자가 입력한 프롬프트를 전달
    ('human', "{user_input}")
])

messages = chat_template.format_messages(
    name="Bob", user_input="What is your name?")

print('########### start messages #############')
print(messages)
print('########### end messages #############')
