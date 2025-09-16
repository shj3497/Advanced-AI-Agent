from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_template(
    """
    다음 한글 문장을 일본어 번역해줘 {sentence}
    sentence: (print from here)
    """
)

output_parser = StrOutputParser()


model = ChatOpenAI(model="gpt-4o-mini")

chain = {"sentence": RunnablePassthrough()} | prompt | model | output_parser

response = chain.invoke({"sentence": "안녕하세요. 아침식사 하셨습니까?"})

print(response)
