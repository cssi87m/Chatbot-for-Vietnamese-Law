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
    # from tqdm.auto import tqdm
    # import pandas as pd

    # result = {
    #     "question": [],
    #     "answer": []
    # }
    # with open("data/testing_data/test_data_usecase3.txt", "r", encoding="utf-8") as f:
    #     questions = f.readlines()
    #     for i in tqdm(range(len(questions))):
    #         question = questions[i]
    #         print(f"Evaluating question {i+1}: {question}")
    #         response = inference(question).content
    #         print(response)
    #         result["question"].append(question)
    #         result["answer"].append(response)
    #     result_df = pd.DataFrame(result)
    #     result_df.to_csv("data/testing_data/usecase3.csv", index=False)
    #     print("Evaluation results saved to data/testing_data/temp.csv")

if __name__ == "__main__":
    main()