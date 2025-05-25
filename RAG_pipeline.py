from typing import List
from langchain_core.documents.base import Document
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from typing import List, Tuple
from utils import RAG_CHAIN_PROMPT, VECTOR_DB, RAG_CHAIN_PROMPT_DEBUG

LANGUAGE_MODEL = ChatOllama(
    model = "llama3.1:8b",
    num_predict = -1,
)

def process_query(query: str, top_k=20) -> List[Tuple[Document, float]]:
    """Processes the query and retrieves the most relevant documents."""
    results = VECTOR_DB.similarity_search_with_score(query, top_k)
    return results

def generate_response(query: str, retrieved_chunks: List[str], chat_model = LANGUAGE_MODEL):
    """Generates a response based on retrieved document chunks."""  
    label_prompt = ChatPromptTemplate.from_template(RAG_CHAIN_PROMPT_DEBUG, stream = False)
    label_chain = label_prompt | chat_model
    response = label_chain.invoke({"question": query,
                                   "documents": retrieved_chunks})
    return response


def deploy_pipeline():
    """Deploys the query processing and retrieval pipeline."""
    print("Pipeline deployed successfully. Ready to handle queries.")


