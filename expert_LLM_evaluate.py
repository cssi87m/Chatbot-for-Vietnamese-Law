from typing import List, Tuple
from langchain_core.documents.base import Document
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_huggingface import ChatHuggingFace
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils import EVAL_PROMPT_SPECIFIC_TYPE_OF_QUESTION

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
EVAL_MODEL = ChatOllama(
    model = "qwen2.5:7b",
    num_predict = -1)

from langchain_core.prompts import ChatPromptTemplate
def expert_evaluate(answer: str, question: str, expert_model = EVAL_MODEL) -> dict:
    """Generates a response based on retrieved document chunks."""  
    label_prompt = ChatPromptTemplate.from_template(EVAL_PROMPT_SPECIFIC_TYPE_OF_QUESTION, stream = False)
    label_chain = label_prompt | expert_model
    response = label_chain.invoke({"answer": answer,
                                   "question": question})
    # Clean output
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, response.content)

    print(f"Response content: {response.content}")
    
    if match:
        cleaned_content = match.group(1)
        response = json.loads(cleaned_content)
    else: 
        cleaned_content = response.content
        print(f"Failed to parse JSON: {cleaned_content}")
    
    return response

# # Initialize the model and tokenizer
# MODEL_NAME = "deepseek-ai/deepseek-r1-distill-14b"  # Assuming this is the HuggingFace model name
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#     device_map="auto" if torch.cuda.is_available() else None
# )

# # Set padding token if not present
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# def expert_evaluate(answer: str, question: str, expert_model=model, expert_tokenizer=tokenizer) -> dict:
#     """Generates a response based on retrieved document chunks using Transformers."""
    
#     # Format the prompt (you'll need to define EVAL_PROMPT_SPECIFIC_TYPE_OF_QUESTION)
#     # Assuming it's a template string with {answer} and {question} placeholders
#     formatted_prompt = EVAL_PROMPT_SPECIFIC_TYPE_OF_QUESTION.format(
#         answer=answer,
#         question=question
#     )
    
#     # Tokenize the input
#     inputs = expert_tokenizer(
#         formatted_prompt,
#         return_tensors="pt",
#         truncation=True,
#         max_length=2048  # Adjust based on your model's context window
#     )
    
#     # Move to GPU if available
#     if torch.cuda.is_available():
#         inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
#     # Generate response
#     with torch.no_grad():
#         outputs = expert_model.generate(
#             **inputs,
#             max_new_tokens=512,  # Equivalent to num_predict=-1 (adjust as needed)
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9,
#             pad_token_id=expert_tokenizer.eos_token_id,
#             eos_token_id=expert_tokenizer.eos_token_id
#         )
    
#     # Decode the response
#     generated_text = expert_tokenizer.decode(
#         outputs[0][len(inputs['input_ids'][0]):],  # Only get the newly generated part
#         skip_special_tokens=True
#     )
    
#     # Clean output (same logic as original)
#     pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
#     match = re.search(pattern, generated_text)
    
#     if match:
#         cleaned_content = match.group(1)
#         try:
#             response = json.loads(cleaned_content)
#         except json.JSONDecodeError:
#             print(f"Failed to parse JSON: {cleaned_content}")
#             response = {"error": "Failed to parse JSON", "raw_content": cleaned_content}
#     else:
#         cleaned_content = generated_text
#         print(f"Failed to parse JSON: {cleaned_content}")
#         response = {"error": "No JSON found", "raw_content": cleaned_content}
    
#     return response

