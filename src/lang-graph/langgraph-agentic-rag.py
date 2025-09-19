from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

import pprint

from dotenv import load_dotenv

load_dotenv()


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100,
    chunk_overlap=50,
)

doc_splits = text_splitter.split_documents(docs_list)
chroma = Chroma()

# Add to vectorDB

vectorstore = chroma.from_documents(
    documents=doc_splits,
    collection_name='rag-chroma',
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
)

retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retriever_blog_posts",
    "Search and return information about Lilian Weng blog posts on LLM agents prompt engineering, and adversarial attacks on LLMs"
)
tools = [retriever_tool]


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
      Determines whether the retrieved documents are relevant to the question.
      Args : state (messages): The current state
      Returns: str: A decision for whether the documents are relevant or not
    """

    print("--- CHECK RELEVANCE ---")
    # Data model

    class grade(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""
        You are a grader assessing relevance of a retrieved document to ad user question.\n
        Here is the retrieved document : \n\n {context} \n\n
        Here is the use question : {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.\n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"]
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state['messages']
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if (score == "yes"):
        print("--- DECISION: DOCS RELEVANT ---")
        return "generate"
    else:
        print("--- DECISION: DOCS NOT RELEVANT ---")
        return "rewrite"


def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. 
    Given the question, it will decide to retrieve using the retriever tool, or simply end.

    Args: state (messages): The current state

    Returns: dict: The updated state with the agent response appended to messages
    """

    print("--- CALL AGENT ---")
    messages = state['messages']
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    model = model.bind_tools(tools)
    response = model.invoke(messages)

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args: state (messages): The current state

    Returns: dict: The updated state with re-phrased question
    """

    print("--- TRANSFORM QUERY ---")
    messages = state['messages']
    question = messages[0].content

    msg = [
        HumanMessage(content=f"""\n
      Look at the input and try to reason about the underlying semantic intent / meaning. \n
      Here is the initial question:
      \n ------ \n
      {question}
      \n ------ \n
      Formulate an improved question:""")
    ]

    # Grader
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    response = model.invoke(msg)

    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args: state (messages): The current state

    Returns: dict: The updated state with re-phrased question
    """
    print("--- GENERATE ---")
    messages = state['messages']
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = hub.pull('rlm/rag-prompt')

    # LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(docs.page_content for docs in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"question": docs, "question": question})
    return {"messages": [response]}


print("*"*20 + "Prompt[rlm/rag-prompt]" + "*"*20)
# Show what the prompt looks like
prompt = hub.pull('rlm/rag-prompt').pretty_print()

# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)
retrieve = ToolNode([retriever_tool])
workflow.add_node('retrieve', retrieve)  # retrieval
workflow.add_node('rewrite', rewrite)  # Re-writing the question
# Generating a response after we know the documents are relevant
workflow.add_node('generate', generate)

# Call agent node to decide to retrieve or not

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    'agent',
    tools_condition,
    {
        "tools": "retrieve",
        END: END
    }
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
)

workflow.add_edge('generate', END)
workflow.add_edge('rewrite', 'agent')

graph = workflow.compile()

# 이미지 생성
# try:
#     # Get the Mermaid PNG bytes
#     png_bytes = graph.get_graph().draw_mermaid_png()

#     # Save the bytes to a file
#     with open("content/langgraph-agentic-rag.png", "wb") as f:
#         f.write(png_bytes)
#     print("Graph image saved to langgraph-agentic-rag.png")
# except Exception as e:
#     # This requires some extra dependencies and is optional
#     print(f"Could not generate graph image: {e}")
#     pass

query_1 = 'agent memory가 무엇인가요?'  # ?블로그 포스트에 있는것
query_2 = 'lilian weng은 agent memory를 어떤 것에 비유했나요?'  # ?블로그 포스트에 없는것

inputs = {"messages": [("user", query_2)]}

for output in graph.stream(inputs, {"recursion_limit": 10}):
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
    print("\n---\n")
