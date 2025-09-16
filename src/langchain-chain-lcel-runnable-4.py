from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

prompt_str = "{topic}의 역사에 대해 세문장으로 설명해주세요."
prompt = ChatPromptTemplate.from_template(prompt_str)

model = ChatOpenAI(model="gpt-4o-mini")


def add_thank(x):
    return x + "\n감사합니다. :)"


add_thank = RunnableLambda(add_thank)

chain = prompt | model | StrOutputParser() | add_thank

response = chain.invoke({"topic": "한국"})

print(response)
