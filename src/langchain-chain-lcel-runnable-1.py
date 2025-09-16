from langchain_core.runnables import RunnablePassthrough

# invoke 로 들어온 인자를 그대로 반환
response = RunnablePassthrough().invoke("Hello, world!")

print(response)
