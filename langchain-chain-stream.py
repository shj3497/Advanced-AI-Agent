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

# ? Chain의 stream() 함수를 통해 스트리밍 기능 추가
for s in chain.stream({"topic": "고구마"}):
    print(s, end="", flush=True)
