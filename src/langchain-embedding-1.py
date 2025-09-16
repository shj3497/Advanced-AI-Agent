from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = embeddings_model.embed_documents(
    [
        "Hi, there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me world",
        "Hellow world"
    ]
)

print(len(embeddings))

print(len(embeddings[0]))
