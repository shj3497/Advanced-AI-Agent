from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


model = ChatOpenAI(model="gpt-4o-mini")


history_prompt = ChatPromptTemplate.from_template("{topic}가 무엇의 약자인지 알려주세요.")
celeb_prompt = ChatPromptTemplate.from_template(
    "{topic} 분야의 유명인사 3명의 이름만 알려주세요.")

output_parser = StrOutputParser()

history_chain = history_prompt | model | output_parser
celeb_chain = celeb_prompt | model | output_parser

# ? RunnableParallel
# ? 여러 개의 Runnable(실행 가능한 LangChain 컴포넌트)을 동시에 병렬로 실행시키기 위한 LangChain의 도구입니다.

map_chain = RunnableParallel(
    history=history_chain,
    celeb=celeb_chain
)

response = map_chain.invoke({"topic": "AI"})
print(response)
