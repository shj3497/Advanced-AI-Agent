from dotenv import load_dotenv

# PyPDFLoader 불러오기
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# PDF 파일 불러올 객체 PyPDFLoader 선언
loader = PyPDFLoader(
    "content/autogen_paper.pdf")

# PDF 파일 로드 및 페이지별로 자르기
pages = loader.load_and_split()
