from typing import List, Tuple
from langchain_core.documents.base import Document
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_huggingface import ChatHuggingFace
from utils import EVAL_PROMPT_SPECIFIC_TYPE_OF_QUESTION
import json 
import re
import os 

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# os.environ["HUGGINGFACE_API_KEY"] = "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"
# llm = HuggingFaceEndpoint(
#     repo_id="google/flan-t5-xxl",
#     task="text2text-generation",
#     max_new_tokens=512,
#     do_sample=False,
#     repetition_penalty=1.03,
# )

# EVAL_MODEL = ChatHuggingFace(llm=llm, verbose=True, huggingfacehub_api_token=os.environ["HUGGINGFACE_API_KEY"])
EVAL_MODEL = ChatOllama(
    model = "deepseek-r1:14b",
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
    
    if match:
        cleaned_content = match.group(1)
        response = json.loads(cleaned_content)
    else:
        cleaned_content = response.content
        print(f"Failed to parse JSON: {cleaned_content}")
    
    return response
