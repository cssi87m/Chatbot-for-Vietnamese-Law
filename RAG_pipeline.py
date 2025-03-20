from typing import List
from langchain_core.documents.base import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.utils.math import cosine_similarity
from langchain.prompts import ChatPromptTemplate
import numpy as np
from langchain_ollama import ChatOllama
from config import * 

VECTOR_DB = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=EMBEDDING
)

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
    additional_kwargs = {'gpu': True}
)

def process_query(query: str, top_k=5) -> List[Document]:
    """Processes the query and retrieves the most relevant documents."""
    # query_embedding = embedding_model.embed_query(query)
    # print(query_embedding)
    results = VECTOR_DB.similarity_search_with_score(query, top_k)
    return results

def generate_response(query: str, retrieved_chunks: List[str], chat_model = LANGUAGE_MODEL):
    """Generates a response based on retrieved document chunks."""  
    label_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE, stream = False)
    label_chain = label_prompt | chat_model
    response = label_chain.invoke({"question": query,
                                   "documents": retrieved_chunks})
    return response

def evaluate_retrieval(retrieved_chunks, ground_truth):
    """Evaluates the retrieval quality using similarity metrics."""
    ground_truth_embedding = EMBEDDING.embed_documents([ground_truth])
    retrieved_embeddings = EMBEDDING.embed_documents(retrieved_chunks)
    scores = cosine_similarity(retrieved_embeddings, ground_truth_embedding)
    return np.mean(scores)

def deploy_pipeline():
    """Deploys the query processing and retrieval pipeline."""
    print("Pipeline deployed successfully. Ready to handle queries.")

def main(): 
    deploy_pipeline()
    print("Nhập câu hỏi của bạn: ")
    query = input()
    retrieved_documents = process_query(query)
    retrieved_chunks = [document[0].page_content for document in retrieved_documents]
    response = generate_response(query = query, retrieved_chunks=retrieved_chunks)
    print(response.content)

if __name__ == "__main__":
    main()

# def fine_tune_model(model, dataset):
#     """Fine-tunes the language model using a custom dataset."""
#     model.train()
#     for data in dataset:
#         model.update(data)
#     print("Model fine-tuning complete.")

# def refine_retrieval_quality(user_feedback, index):
#     """Refines the retrieval process using user feedback."""
#     for feedback in user_feedback:
#         if feedback['relevant']:
#             index.update(feedback['doc_id'])
#     print("Retrieval quality refined based on user feedback.")
