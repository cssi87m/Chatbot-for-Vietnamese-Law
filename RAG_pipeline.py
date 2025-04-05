from typing import List
from langchain_core.documents.base import Document
from langchain.prompts import ChatPromptTemplate
import numpy as np
from langchain_ollama import ChatOllama

from typing import List, Tuple
from utils import * 




PROMPT_TEMPLATE = """
Bạn là một chuyên gia về pháp luật, đặc biệt trong lĩnh vực luật an toàn thông tin và an ninh mạng. Dựa trên các tài liệu pháp lý được cung cấp dưới đây, 
trả lời câu hỏi sau.

Tài liệu: {documents}

Câu hỏi: {question}

Câu trả lời của bạn phải nêu rõ các quy định pháp luật liên quan, bao gồm tên luật hoặc nghị định, số điều luật, 
và nội dung trích dẫn phù hợp để làm cơ sở pháp lý cho lập luận của bạn. 
Hãy trình bày một cách logic, mạch lạc và dễ hiểu, đảm bảo thông tin có tính chính xác và phù hợp với ngữ cảnh câu hỏi.
"""

LANGUAGE_MODEL = ChatOllama(
    model = "llama3.2",
    num_predict = -1,
)

def process_query(query: str, top_k=5) -> List[Tuple[Document, float]]:
    """Processes the query and retrieves the most relevant documents."""
    results = VECTOR_DB.similarity_search_with_score(query, top_k)
    return results

def generate_response(query: str, retrieved_chunks: List[str], chat_model = LANGUAGE_MODEL):
    """Generates a response based on retrieved document chunks."""  
    label_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE, stream = False)
    label_chain = label_prompt | chat_model
    response = label_chain.invoke({"question": query,
                                   "documents": retrieved_chunks})
    return response


def deploy_pipeline():
    """Deploys the query processing and retrieval pipeline."""
    print("Pipeline deployed successfully. Ready to handle queries.")


# def refine_retrieval_quality(user_feedback, index):
#     """Refines the retrieval process using user feedback."""
#     for feedback in user_feedback:
#         if feedback['relevant']:
#             index.update(feedback['doc_id'])
#     print("Retrieval quality refined based on user feedback.")

