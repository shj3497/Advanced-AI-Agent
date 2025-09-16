from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

embeddings = embeddings_model.embed_documents([i.page_content for i in texts])

print(len(embeddings))
print(len(embeddings[0]))
