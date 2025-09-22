# Self Rag의 개념
# 3개의 분기(관련성 검토, 환각 검토, 답변 적절성 검토)로 사용자의 질문에 더 제대로된 답변을 할 수 있음.
# 관련된 문서를 찾을 수 없거나 답변이 적절치 않은 경우 쿼리를 재작성, 환각 발생한 경우 LLM 답변 재작성


from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langgraph.graph import END, START, StateGraph

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

docs_splits = text_splitter.split_documents(docs_list)

# Add to VectorDB
vectorstore = Chroma.from_documents(
    documents=docs_splits,
    collection_name='rag-chroma',
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
)

retriever = vectorstore.as_retriever()


# 문서 관련성 검토 함수 정의
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents"""
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")


# LLM with function call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt
system = """
You are a grader assessing relevance of a retrieved document to ad user question.
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
"""
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("user", "Retrieved document: {document} \n\n User question: {question}"),
])

retrieval_grader = grade_prompt | structured_llm_grader
question = 'agent memory'
docs = retriever.get_relevant_documents(question)
docs_txt = docs[1].page_content
print(retrieval_grader.invoke({"question": question, "document": docs_txt}))

# 답변 작성 Chain 정의하기

prompt = hub.pull('rlm/rag-prompt')

# Post-processing


def format_docs(docs):
    return "\n\n".join(docs.page_content for docs in docs)


# Chain
rag_chain = prompt | llm | StrOutputParser()

# Question Re-writer
# Prompt
system = """
You a question re-writer. that converts an input question to a better version that is optimized \n
for web search. Look at the input and try to reason about the underlying semantic intent / meaning.
"""

re_write_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    (
        "human",
        "Here is the initial question: \n\n{question}\n Formulate an improved question."
    ),
])

question_rewriter = re_write_prompt | llm | StrOutputParser()
question_rewriter.invoke({"question": question})

# 웹 검색툴 설정
web_search_tool = TavilySearchResults(max_results=3)

# GraphState 정의하기


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
      question: question
      generation: LLM generation
      web_search: wheter to add search
      documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]


def retrieve(state):
    """
    Retrieve documents

    Args:
      state (dict): The current graph state

    Returns:
      state (dict): New key added to state, documents, that contains retrieved documents.
    """

    print("--- RETRIEVE ---")
    question = state['question']

    # Retrieval
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer

    Args:
      state (dict): The current graph state

    Returns:
      state (dict): New key added to state, generation, that contains LLM generation.
    """

    print("--- GENERATE ---")
    question = state['question']
    documents = state['documents']

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    """

    print("--- CHECK RELEVANCE ---")
    documents = state['documents']
    question = state['question']

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({
            "question": question,
            "document": d.page_content
        })
        grade = score.binary_score
        if grade == 'yes':
            print("--- GRADE: DOCUMENT RELEVANT ---")
            filtered_docs.append(d)
        else:
            print("--- GRADE: DOCUMENT NOT RELEVANT ---")
            web_search = "Yes"
            continue

    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
      state (dict): The current graph state

    Returns:
      state (dict): Updates question key with a re-phrased question.
    """
    print("--- TRANSFORM QUERY ---")
    question = state['question']
    documents = state['documents']

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"question": better_question, "documents": documents}


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
      state (dict): The current graph state

    Returns:
      state (dict): Updates documents key with appended web results.
    """

    print("--- WEB SEARCH ---")
    question = state['question']
    documents = state['documents']

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents, "question": question}

# Edges


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
      state (dict): The current graph state

    Returns:
      str: Binary decision or next node to call
    """

    print("--- ACCESS GRADED DOCUMENTS ---")
    web_search = state['web_search']
    if web_search == 'Yes':
        print(
            "--- DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY ---")
        return 'transform_query'
    else:
        print("--- DECISION: GENERATE ---")
        return 'generate'


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node('retrieve', retrieve)
workflow.add_node('grade_documents', grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search)

# Build graph
workflow.add_edge(START, 'retrieve')
workflow.add_edge('retrieve', 'grade_documents')
workflow.add_conditional_edges('grade_documents', decide_to_generate, {
    "transform_query": "transform_query",
    "generate": "generate"
})
workflow.add_edge('transform_query', 'web_search_node')
workflow.add_edge('web_search_node', 'generate')
workflow.add_edge('generate', END)

# Compile
graph = workflow.compile()


# 이미지 생성
# try:
#     # Get the Mermaid PNG bytes
#     png_bytes = graph.get_graph().draw_mermaid_png()

#     # Save the bytes to a file
#     with open("content/langgraph-self-rag.png", "wb") as f:
#         f.write(png_bytes)
#     print("Graph image saved to langgraph-self-rag.png")
# except Exception as e:
#     # This requires some extra dependencies and is optional
#     print(f"Could not generate graph image: {e}")
#     pass

inputs = {
    "question": "What player at the Bears expected to draft first in the 2024 NFL draft?"
}
for output in graph.stream(inputs):
    for key, value in output.items():
        # Node
        print(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    print("\n---\n")

# Final generation
print(value["generation"])
