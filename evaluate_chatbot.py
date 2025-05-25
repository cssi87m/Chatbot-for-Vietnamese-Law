import pandas as pd
import torch
from safetensors import safe_open
from tqdm.auto import tqdm
import json
# from RAG_pipeline import * 
from expert_LLM_evaluate import expert_evaluate
import argparse
# from GraphRAG import *

def evaluate_chatbot_normal_RAG(test_dataset_dir: str):
    """Evaluate the chatbot using normal RAG."""
    # Load the test dataset
    test_data = pd.read_csv(test_dataset_dir)
    questions = test_data["question"].tolist()
    answers = test_data["answer"].tolist()
    print("Load checkpoint successfully.")
    loop = tqdm(range(10))
    loop.set_description("Evaluating response quality")
    result = []
    max_try = 3
    for i in loop: 
        question, answer = questions[i], answers[i]
        # Get the response from the chatbot
        for _ in range(max_try):
            try:
                # Evaluate the response using the expert model
                judgement = expert_evaluate(answer, question)
                # Dump judgement in a json file
                print(judgement)
                json.dump(judgement, open(f"eval_test/{i+1}.json", "w"), ensure_ascii=False, indent=4)
                loop.set_postfix_str(f"Câu hỏi {i+1}: {judgement['Câu hỏi']} Loại câu hỏi: {judgement['Loại câu hỏi']}, Điểm: {judgement['Điểm tổng thể']}")
                break 
            except Exception as e:
                print(f"Error evaluating question {i+1}: {e}")
        else: 
            loop.set_postfix_str(f"Failed to evaluate question {i+1} after {max_try} attempts.")
    
    # # Save the results to a CSV file
    # result_df = pd.DataFrame(result)
    # result_df.to_csv("evaluation_results.csv", index=False)
    # print("Evaluation results saved to evaluation_results.csv")
    
def main(): 
    parser = argparse.ArgumentParser(description="Evaluate the chatbot using normal RAG.")
    parser.add_argument("--test_dataset_dir", type=str, default="data/testing_data/temp.csv", help="Path to the test dataset.")
    args = parser.parse_args()
    evaluate_chatbot_normal_RAG(args.test_dataset_dir)

if __name__ == "__main__":
    main()