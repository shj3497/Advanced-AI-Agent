from typing import List
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


# 데이터 구조를 정의합니다.
class Country(BaseModel):
    continent: str = Field(description="사용자가 물어본 나라가 속한 대륙")
    population: str = Field(description="사용자가 물어본 나라의 인구 (int 형식)")
    culture: str = Field(description="사용자가 물어본 나라의 문화")


# JsonOutputParser를 설정하고 프롬프트 템플릿에 format_instructions를 삽입합니다.
parser = JsonOutputParser(pydantic_object=Country)

prompt = PromptTemplate(
    template="Answer the user query.\n {format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()},
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = prompt | model | parser

response = chain.invoke({"query": "한국은 어떤 나라야?"})

print(response)
