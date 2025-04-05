import pandas as pd 
from utils import *
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers 
import pandas as pd
from datasets import Dataset
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="train_data.csv")
    parser.add_argument("--output", type=str, default="output")

    args = parser.parse_args()
    finetune_embedding(args.dataset_dir, args.output)

if __name__ == "__main__":
    main()