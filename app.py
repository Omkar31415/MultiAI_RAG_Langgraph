import cassio
import streamlit as st
import os
from typing import Literal, List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Cassandra
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langchain.schema import Document
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
import dotenv
dotenv.load_dotenv()

# Environment variables (keep your existing configuration)
# os.environ['HF_TOKEN']= os.getenv("HF_TOKEN")
# ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
# ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit Secrets Configuration
# You'll need to add these in Streamlit's secrets management
ASTRA_DB_APPLICATION_TOKEN = st.secrets.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = st.secrets.get("ASTRA_DB_ID")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
os.environ['HF_TOKEN'] = st.secrets.get("HF_TOKEN")

# Initialize Cassio for Astra DB connection
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Cache vector store initialization
@st.cache_resource
def load_vector_store():
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    astra_vector_store = Cassandra(
        embedding=embeddings,
        table_name="multiAI_webdata",
        session=None,
        keyspace=None
    )
    astra_vector_store.add_documents(doc_splits)
    return astra_vector_store

# Load vector store and retriever
astra_vector_store = load_vector_store()
retriever = astra_vector_store.as_retriever(search_kwargs={'k': 1})

# Initialize LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Gemma2-9b-It")

# Define routing model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "wiki_search"] = Field(
        description="Given a user question choose to route it to Wikipedia or a vectorstore."
    )

structured_llm_router = llm.with_structured_output(RouteQuery)

# Router prompt
system = """You are an expert at routing a user question to a vectorstore or Wikipedia.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use Wikipedia."""
route_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}"),
])
question_router = route_prompt | structured_llm_router

# Initialize Wikipedia tool
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# Define graph state
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

# Define nodes
def retrieve(state):
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def wiki_search(state):
    question = state["question"]
    docs = wiki.invoke({"query": question})
    wiki_results = Document(page_content=docs)
    return {"documents": [wiki_results], "question": question}

# Define routing logic
def route_question(state):
    question = state["question"]
    source = question_router.invoke({"question": question})
    return source.datasource

# Build workflow
workflow = StateGraph(GraphState)
workflow.add_node("wiki_search", wiki_search)
workflow.add_node("retrieve", retrieve)
workflow.add_conditional_edges(
    START,
    route_question,
    {"wiki_search": "wiki_search", "vectorstore": "retrieve"}
)
workflow.add_edge("retrieve", END)
workflow.add_edge("wiki_search", END)
app = workflow.compile()

# Streamlit interface
st.title("Multi-Source AI Agent RAG ChatBot")
st.write("Ask a question, and I'll fetch the answer from my knowledge base or Wikipedia!")

question = st.text_input("Enter your question here:", placeholder="e.g., What is an agent?")
if st.button("Get Answer"):
    if question:
        with st.spinner("Fetching your answer..."):
            inputs = {"question": question}
            result = None
            for output in app.stream(inputs):
                result = output
            # Extract and display documents
            for node, value in result.items():
                documents = value["documents"]
                st.subheader(f"Answer from {'Vector Store' if node == 'retrieve' else 'Wikipedia'}:")
                if isinstance(documents, list):
                    for doc in documents:
                        st.write(doc.page_content)
                else:
                    st.write(documents.page_content)
    else:
        st.warning("Please enter a question!")
