import pandas as pd
import torch
from safetensors import safe_open
from tqdm.auto import tqdm
import json
from RAG_pipeline import * 
from infer import inference
from utils import EMBEDDING
from expert_LLM_evaluate import expert_evaluate
import argparse
# from GraphRAG import *

def evaluate_chatbot_normal_RAG(test_dataset_dir: str = "data/testing_data/chatbot_test_data.csv", model_path: str = "checkpoint/embedding_model.pt"):
    """Evaluate the chatbot using normal RAG."""
    # Load the test dataset
    test_data = pd.read_csv(test_dataset_dir)
    questions, answers = test_data['question'], test_data['answer']
    if model_path.endswith(".safetensors"):
        # Load the model
        tensors = {}
        with safe_open(model_path, framework="pt", device="cuda") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        
        EMBEDDING._client.load_state_dict(tensors, strcict=False)
    else: 
        # Load the model
        tensors = torch.load(model_path, map_location="cuda", weights_only=False)
        EMBEDDING._client = tensors
    print("Load checkpoint successfully.")
    loop = tqdm(range(len(test_data)))
    loop.set_description("Evaluating response quality")
    result = []
    max_try = 3
    for i in loop: 
        question, ground_answer = questions[i], answers[i]
        # Get the response from the chatbot
        response = inference(question)
        for _ in range(max_try):
            try:
                # Evaluate the response using the expert model
                judgement = expert_evaluate(response.content, question)
                # Dump judgement in a json file
                print(judgement)
                json.dump(judgement, open(f"eval/{i+1}.json", "w"), ensure_ascii=False, indent=4)
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
    parser.add_argument("--test_dataset_dir", type=str, default="data/testing_data/chatbot_test_data.csv", help="Path to the test dataset.")
    parser.add_argument("--model_path", type=str, default="checkpoint/embedding_model.pt", help="Path to the model checkpoint.")
    args = parser.parse_args()
    evaluate_chatbot_normal_RAG(args.test_dataset_dir, args.model_path)

if __name__ == "__main__":
    main()