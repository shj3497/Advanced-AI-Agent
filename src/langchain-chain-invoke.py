import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# 프롬프트 템플릿 설정
prompt = ChatPromptTemplate.from_template("{topic}을 주제로 넌센스 퀴즈를 만들어줘")

# LLM 호출
model = ChatOpenAI(model="gpt-4o-mini")

# LCEL로 프롬프트템플릿 - LLM - 출력파서 연결하기
chain = prompt | model | StrOutputParser()

# ? invoke 함수로 chain 실행
response = chain.invoke({"topic": "고구마"})
print('########### start chain #############')
print(response)
print('########### end chain #############')
