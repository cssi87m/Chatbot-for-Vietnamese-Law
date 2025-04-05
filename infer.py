from RAG_pipeline import *


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