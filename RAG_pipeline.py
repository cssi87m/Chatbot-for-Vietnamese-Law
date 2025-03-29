from typing import List
from langchain_core.documents.base import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.utils.math import cosine_similarity
from langchain.prompts import ChatPromptTemplate
import numpy as np
from langchain_ollama import ChatOllama
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers 
import pandas as pd
from datasets import Dataset
import argparse
from typing import List, Tuple
from utils import * 

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

def evaluate_retrieval(retrieved_chunks, ground_truth):
    """Evaluates the retrieval quality using similarity metrics."""
    ground_truth_embedding = EMBEDDING.embed_documents([ground_truth])
    retrieved_embeddings = EMBEDDING.embed_documents(retrieved_chunks)
    scores = cosine_similarity(retrieved_embeddings, ground_truth_embedding)
    return np.mean(scores)

def deploy_pipeline():
    """Deploys the query processing and retrieval pipeline."""
    print("Pipeline deployed successfully. Ready to handle queries.")

def finetune_embedding(dataset_dir: str, output: str):
    """
        dataset: {"question": "<question>", "context": "<relevant context to answer>"}
    """ 
    dataset = pd.read_csv(dataset_dir)
    dataset.rename(columns={"question": "anchor", "context": "positive"}, inplace=True)

    dataset.drop(columns = ["answer", "cid"], inplace=True)
    dataset = Dataset.from_pandas(dataset)

    # Initialeize the loss function
    model = EMBEDDING._client
    loss = MultipleNegativesRankingLoss(model=model)

    # Train the model
    training_args =  SentenceTransformerTrainingArguments(
        output_dir = output, 
        num_train_epochs=4,                         
        per_device_train_batch_size=1,             
        gradient_accumulation_steps=16,             
        per_device_eval_batch_size=2,             
        warmup_ratio=0.1,                           
        learning_rate=2e-5,                         
        lr_scheduler_type="cosine",                 
        optim="adamw_torch_fused",                  
        tf32=True,                                  
        bf16=True,                                  
        batch_sampler=BatchSamplers.NO_DUPLICATES,  
        eval_strategy="no",                     
        save_strategy="epoch",                      
        logging_steps=1,                           
        save_total_limit=3,                         
        # load_best_model_at_end=True,                
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        loss=loss,
        train_dataset=dataset,
        args=training_args,
    )
    print("Training the model...")
    trainer.train()
    print("Model training complete.")
    trainer.save_model(output)
    print("Model saved successfully.")

def main(): 
    deploy_pipeline()
    # print("Nhập câu hỏi của bạn: ")
    # query = input()
    # retrieved_documents = process_query(query)
    # retrieved_chunks = [document[0].page_content for document in retrieved_documents]
    # response = generate_response(query = query, retrieved_chunks=retrieved_chunks)
    # print(response.content)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="train_data.csv")
    parser.add_argument("--output", type=str, default="output")

    args = parser.parse_args()
    finetune_embedding(args.dataset_dir, args.output)

if __name__ == "__main__":
    main()


# def refine_retrieval_quality(user_feedback, index):
#     """Refines the retrieval process using user feedback."""
#     for feedback in user_feedback:
#         if feedback['relevant']:
#             index.update(feedback['doc_id'])
#     print("Retrieval quality refined based on user feedback.")
