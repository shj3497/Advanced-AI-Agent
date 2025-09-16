from langchain_community.document_loaders import PyPDFLoader
# ? CharacterTextSplitter 모듈 로드
from langchain_text_splitters import CharacterTextSplitter

loader = PyPDFLoader("content/autogen_paper.pdf")

pages = loader.load()

# 구분자: 줄넘김, 청크 길이: 500, 청크 오버랩: 100, length_function: 글자수
text_splitter = CharacterTextSplitter(
    separator="\n",   # * 특정한 구분자를 기준으로 분할
    chunk_size=500,  # * 텍스트의 길이
    chunk_overlap=100,
    length_function=len,  # * 글자수
)

# ? 텍스트 분할
texts = text_splitter.split_documents(pages)

print([len(i.page_content) for i in texts])

print(texts[0])
