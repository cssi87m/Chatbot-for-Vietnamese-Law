import pandas as pd 
import torch
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformerTrainer, SentenceTransformer
from sentence_transformers.training_args import BatchSamplers 
import pandas as pd
from datasets import Dataset
import argparse
def finetune_embedding(train_dir: str, eval_dir: str, model_path: str, output: str, epochs: int, batch_size: int, learning_rate: float, weight_decay: float, device: str):
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

    # Load the model
    if ('.pt' in model_path) or ('.bin' in model_path):
        model = torch.load(model_path, map_location=torch.device(device))
    
    if ('.safetensors' in model_path):
        from safetensors import safe_open
        tensors = {}
        with safe_open(model_path, framework="pt", device=device) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        model = SentenceTransformer(model_path)
        model.load_state_dict(tensors, strict = False)

    loss = MultipleNegativesRankingLoss(model=model)

    # Train the model
    training_args =  SentenceTransformerTrainingArguments(
        output_dir = output, 
        num_train_epochs=epochs,                         
        per_device_train_batch_size=batch_size,   
        learning_rate=learning_rate,                         
        weight_decay=weight_decay,
        torch_empty_cache_steps = 2,
        gradient_accumulation_steps=2,             
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

    # Log out all the param
    print(f"Training params: Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}, Weight decay: {weight_decay}")

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
    parser.add_argument("--model_path", type=str, default="checkpoint/embedding_model.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    finetune_embedding(
        train_dir=args.train_dir,
        eval_dir=args.eval_dir,
        output=args.output,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        model_path=args.model_path,
        device=args.device
    )

if __name__ == "__main__":
    main()