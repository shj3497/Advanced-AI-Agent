from langchain_community.document_loaders import PyPDFLoader
# ? CharacterTextSplitter 모듈 로드
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("content/autogen_paper.pdf")

pages = loader.load()

# 청크 길이: 500, 청크 오버랩: 100, length_function: 글자수
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # * 텍스트의 길이
    chunk_overlap=100,
    length_function=len,  # * 글자수
    # is_separator_regex=False,
)

# ? 텍스트 분할
texts = text_splitter.split_documents(pages)

print([len(i.page_content) for i in texts])

print(texts[0])
