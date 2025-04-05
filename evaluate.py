from RAG_pipeline import *
import pandas as pd
from tqdm import tqdm

def evaluate_retrieval(retrieved_chunks, ground_truth):
    """Evaluates the retrieval quality."""
    for chunk in retrieved_chunks:
        if chunk == ground_truth:
            return 1
    return 0

def main(): 
    deploy_pipeline()
    test_data = pd.read_csv("test_data.csv")
    loop = tqdm(range(len(test_data)))
    total_score = 0
    loop.set_description("Evaluating retrieval quality")
    for i in tqdm(range(len(test_data))): 
        print(f"Evaluating question {i+1}/{len(test_data)}")
        query = test_data.iloc[i]['question']
        ground_truth = test_data.iloc[i]['corpus']
        retrieved_documents = process_query(query)
        retrieved_chunks = [document[0].page_content for document in retrieved_documents]
        score = evaluate_retrieval(retrieved_chunks, ground_truth)
        total_score += score
        loop.set_postfix({"Total documents retrieved": total_score})

if __name__ == "__main__":
    main()
