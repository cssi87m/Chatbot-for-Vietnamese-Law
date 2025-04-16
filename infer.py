from RAG_pipeline import *
from langchain_core.messages import BaseMessage

def inference(query: str) -> BaseMessage:
    """Main function to handle inference."""
    if not isinstance(query, str):
        raise ValueError("Input must be a string. Do not accept any other type")
    
    retrieved_documents = process_query(query)
    retrieved_chunks = [document[0].page_content for document in retrieved_documents]
    response = generate_response(query=query, retrieved_chunks=retrieved_chunks)
    return response

def main(): 
    deploy_pipeline()
    print("Nhập câu hỏi của bạn: ")
    query = input()
    response = inference(query)
    print(response)

if __name__ == "__main__":
    main()