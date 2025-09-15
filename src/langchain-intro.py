import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI

load_dotenv()

# langchain을 사용하지 않고 openai를 사용하는 방법
client = OpenAI()

# `client.chat.completions.create` 메서드를 사용하여 Chat Completion API를 호출합니다.
response = client.chat.completions.create(
    model="gpt-4o-mini",  # 사용할 모델을 지정합니다.
    messages=[
        {"role": "user", "content": "2002년 월드컵 4강 국가 알려줘"}
    ]
)

# API 응답에서 메시지 내용(content)을 추출하여 출력합니다.
print('########### start openai #############')
print(response.choices[0].message.content)
print('########### end openai #############')

# langchain을 사용하는 방법
llm = ChatOpenAI(model="gpt-4o-mini", )
response = llm.invoke("2002년 월드컵 4강 국가 알려줘")
print('########### start langchain #############')
print(response.content)
print('########### end langchain #############')
