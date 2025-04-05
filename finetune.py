import pandas as pd 
from utils import *
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers 
import pandas as pd
from datasets import Dataset
import argparse
def finetune_embedding(train_dir: str, eval_dir: str, output: str, epochs: int, batch_size: int, learning_rate: float, weight_decay: float = 0.01):
    """
        dataset: {"question": "<question>", "context": "<relevant context to answer>"}
    """ 
    train_dataset = pd.read_csv(train_dir)
    train_dataset.rename(columns={"question": "anchor", "context": "positive"}, inplace=True)
    train_dataset.drop(columns = ["answer", "cid"], inplace=True)
    train_dataset = Dataset.from_pandas(train_dataset)


    # Create a dataset for evaluation
    eval_dataset = pd.read_csv(eval_dir)
    eval_dataset.rename(columns={"question": "anchor", "context": "positive"}, inplace=True)
    eval_dataset.drop(columns = ["answer", "cid"], inplace=True)
    eval_dataset = Dataset.from_pandas(eval_dataset)
    # Initialeize the loss function
    model = EMBEDDING._client
    loss = MultipleNegativesRankingLoss(model=model)

    # Train the model
    training_args =  SentenceTransformerTrainingArguments(
        output_dir = output, 
        num_train_epochs=epochs,                         
        per_device_train_batch_size=batch_size,   
        learning_rate=learning_rate,                         
        weight_decay=weight_decay,
        gradient_accumulation_steps=16,             
        per_device_eval_batch_size=2,             
        warmup_ratio=0.1,                           
        lr_scheduler_type="cosine",                 
        optim="adamw_torch_fused",                                                   
        batch_sampler=BatchSamplers.NO_DUPLICATES,  
        eval_strategy="steps",                     
        save_strategy="epoch",                      
        logging_steps=1,                           
        save_total_limit=3,                         
        # load_best_model_at_end=True,                
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        loss=loss,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )
    print("Training the model...")
    trainer.train()
    print("Model training complete.")
    trainer.save_model(output)
    print("Model saved successfully.")

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="train_data.csv")
    parser.add_argument("--eval_dir", type=str, default="test_data.csv")
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    args = parser.parse_args()
    finetune_embedding(
        train_dir=args.train_dir,
        eval_dir=args.eval_dir,
        output=args.output,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )

if __name__ == "__main__":
    main()