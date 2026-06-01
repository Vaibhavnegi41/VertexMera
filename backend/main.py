from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Literal
import os
from langchain_pinecone.vectorstores import Pinecone as LangchainPinecone
from pydantic import BaseModel
from langchain_core.documents import Document
from pinecone import Pinecone
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY
)

grading_model = ChatGroq(model="llama-3.1-8b-instant",api_key=GROQ_API_KEY)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_store = LangchainPinecone(
    index_name=INDEX_NAME,
    embedding=embedding_model
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

web_search_tool = TavilySearch(max_results=2)



class State(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    search: bool
    web_searched: bool   
    steps: List[str]


def retrieve(state: State):
    question = state["question"]
    response = retriever.invoke(question)

    steps = state["steps"]
    steps.append("retrieval_node")

    return {
        "documents": response,
        "steps": steps
    }


class ScoreFormat(BaseModel):
    score: bool

score_model = model.with_structured_output(ScoreFormat)

scoring_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a relevance grader.

        Determine whether the document contains information relevant to the question.

        Return ONLY:

        {{"score": true}}

        or

        {{"score": false}}
        """
    ),
    (
        "human",
        """
        QUESTION:
        {question}

        DOCUMENT:
        {documents}
        """
    )
])

def grade_documents(state: State):
    question = state["question"]
    documents = state["documents"]

    steps = state["steps"]
    steps.append("grade_documents_node")

    
    prompts = [
        scoring_prompt.format_messages(
            question=question,
            documents=doc.page_content
        )
        for doc in documents
    ]

    results = score_model.batch(prompts)

    filtered_docs = [
        doc for doc, result in zip(documents, results)
        if result.score
    ]

    search = len(filtered_docs) == 0

    print("SEARCH :", search)

    return {
        "documents": filtered_docs,
        "search": search,
        "steps": steps
    }


rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a factual assistant.

        Answer ONLY from the provided context.

        RULES:
        1. Do not use outside knowledge.
        2. If the answer is not present in the context, say:
           'I cannot find the answer in the provided documents.'
        3. Keep your answer concise and factual.
        """
    ),
    (
        "human",
        """
        QUESTION:
        {question}

        CONTEXT:
        {documents}
        """
    )
])

web_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful assistant.

        Answer the question using the provided web search results as your primary source.

        RULES:
        1. Synthesise information from the results naturally and concisely.
        2. If results are partially relevant, use what is useful.
        3. Do not fabricate facts beyond what the results support.
        4. Keep your answer concise and factual.
        """
    ),
    (
        "human",
        """
        QUESTION:
        {question}

        WEB SEARCH RESULTS:
        {documents}
        """
    )
])



def generate_node(state: State):
    question = state["question"]
    documents = state["documents"]
    steps = state["steps"]
    steps.append("Generation_node")

    context_str = "\n\n".join([doc.page_content for doc in documents])

    # Use a different prompt depending on where the documents came from
    prompt = web_prompt if state.get("web_searched") else rag_prompt

    generation = model.invoke(
        prompt.format_messages(
            question=question,
            documents=context_str
        )
    )

    return {
        "generation": generation.content,
        "steps": steps
    }


def web_search(state: State):
    question = state["question"]
    steps = state["steps"]
    steps.append("web_search_node")

    web_response = web_search_tool.invoke(question)

    web_docs = []

    # TavilySearch returns a list directly
    if isinstance(web_response, list):
        for result in web_response:
            content = result.get("content", "")
            if content:
                web_docs.append(Document(page_content=content))

    # Fallback: older Tavily dict format
    elif isinstance(web_response, dict):
        for result in web_response.get("results", []):
            content = result.get("content", "")
            if content:
                web_docs.append(Document(page_content=content))

    print("WEB DOCS FOUND:", len(web_docs))

    return {
        "documents": web_docs,
        "web_searched": True,   
        "steps": steps
    }


def route_decision(state: State) -> Literal["web_search", "generate"]:
    if state["search"]:
        return "web_search"
    return "generate"



graph = StateGraph(State)

graph.add_node("retrieve", retrieve)
graph.add_node("grading", grade_documents)
graph.add_node("web_search", web_search)
graph.add_node("generate", generate_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grading")
graph.add_conditional_edges("grading", route_decision, {
    "web_search": "web_search",
    "generate": "generate"
})
graph.add_edge("web_search", "generate")
graph.add_edge("generate", END)

chatbot = graph.compile()


if __name__ == "__main__":
    results = chatbot.invoke({
        "question": "what is linear regression?",
        "documents": [],
        "steps": [],
        "generation": "",
        "web_searched": False   
    })

    print(results["generation"])
    print(results["steps"])