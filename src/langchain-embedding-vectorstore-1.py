from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

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

query = 'autogen 이 뭐야?'
res = retriever.invoke(query)

print(res)
