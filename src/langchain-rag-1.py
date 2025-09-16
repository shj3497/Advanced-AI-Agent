from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

loader = PyPDFLoader("content/autogen_paper.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

texts = text_splitter.split_documents(pages)

# FAISS 벡터 데이터베이스는 메모리에 존재, 영구적으로 보관하려면 db 객체를 로컬 파일 시스템에 저장해야함
db = FAISS.from_documents(texts, embeddings_model)

retriever = db.as_retriever()

model = ChatOpenAI(model="gpt-4o-mini")
prompt = hub.pull('rlm/rag-prompt')


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    } | prompt | model | StrOutputParser()
)

response = rag_chain.invoke("autogen이 뭐야?")

print(response)
