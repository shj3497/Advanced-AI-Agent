import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ChatPromptTemplate에 SystemMessage로 LLM 역할과 출력 형식 지정
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(
        content=(
            "너는 영화 전문가 AI야. 사용자가 원하는 장르의 영화를 리스트 형태로 추천해줘"
            "ex) Query: SF영화 3개 추천해줘 / 답변 : ['인터스텔라', '스페이스오디세이', '혹성탈출']"
        )
    ),
    HumanMessagePromptTemplate.from_template("{text}")
])

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = chat_template | model | StrOutputParser()
response = chain.invoke("멜로")

print(response)
