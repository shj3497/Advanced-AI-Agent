import bs4
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

# 텍스트 추출할 URL 입력
loader = WebBaseLoader(
    "https://megazone.com/resources/newsroom/51",
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("mui-1d89a32")
        )
    )
)

# SSL Verification 비활성화
loader.request_kwargs = {"verify": False}
data = loader.load()
print(data)
