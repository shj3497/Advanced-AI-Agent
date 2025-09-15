import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate


load_dotenv()


# CSV 파서 선언
output_parser = CommaSeparatedListOutputParser()

# CSV 파서 작동을 위한 형식 지정 프롬프트 로드
format_instructions = output_parser.get_format_instructions()

# 프롬프트 템플릿의 partial_variables에 CSV 형식 지정 프롬프트 주입
prompt = PromptTemplate(
    template="List {subject}. answer in Korean \n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions}
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 프롬프트템플릿-모델-Output Parser를 체인으로 연결
chain = prompt | model | output_parser

response = chain.invoke({"subject": "SF영화 3개 추천해줘"})

print(response)
